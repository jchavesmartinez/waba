"""
Fuente: Meta Ads (Facebook / Instagram) por la Marketing API.

Convierte la inversion publicitaria de una cuenta de anuncios en UNA tabla
ancha y consultable, para que el bot responda cosas como "cuanto gaste en
anuncios esta semana", "que anuncio me trajo mas conversaciones de WhatsApp" o
"cual campania tiene el CPA mas caro".

GRANO DE LA TABLA. Una fila = un DIA x una ENTIDAD (anuncio, conjunto, campania
o cuenta, segun 'nivel') x una combinacion de DESGLOSE (si se pidieron
breakdowns). Es el grano mas fino que da el API con time_increment=1; cualquier
agregacion mas gruesa (semana, mes, campania) sale de un GROUP BY en SQL, que
es justo lo que el bot sabe escribir. Al reves no se puede: si se guardara ya
agregado, el detalle no se recupera nunca.

POR QUE ES ACUMULATIVA Y NO FULL REFRESH — la decision mas importante de este
archivo.

Las cifras de un dia de Meta NO son finales cuando ese dia termina. La ventana
de atribucion (7d_click / 1d_view por defecto) le sigue asignando conversiones
a un dia durante DIAS despues, y el gasto se ajusta hacia atras cuando Meta
detecta trafico invalido y lo reembolsa. O sea que el 3 de agosto leido el 4 y
leido el 15 son dos numeros distintos, y el del 15 es el bueno.

Eso descarta las dos estrategias simples:

  - Full refresh de una ventana movil de N dias BORRARIA todo el historial
    anterior a la ventana en cada corrida. Se pierde el año pasado para
    conservar el mes.
  - Append puro DUPLICARIA cada dia tantas veces como corridas lo relean, y
    ademas dejaria conviviendo la version vieja con la corregida sin forma de
    saber cual vale.

La unica correcta es releer una ventana movil y hacer UPSERT por identidad de
fila: los dias dentro de la ventana se corrigen en su lugar, los de afuera
quedan intactos. Es el mismo mecanismo que ya usan google_calendar y
zoho_imap; ver _sincronizar_meta_ads() en sync.py.

Config esperada (JSON en la columna 'config' del registro):

  {"account_id": "act_1234567890",
   "token_env": "META_ADS_TOKEN_CLIENTE_A",
   "nivel": "ad",
   "dias": 30,
   "tabla": "meta_ads"}

Opcionales:
  "breakdowns": ["age","gender"]   -> desglose (ver NOTA DE DESGLOSES abajo)
  "atribucion": ["7d_click","1d_view"]  -> ventanas de atribucion explicitas
  "incluir_estructura": true       -> segunda tabla con campanias/conjuntos/
                                      anuncios y su configuracion actual
  "campos_extra": ["cost_per_thruplay"]  -> metricas adicionales del API
  "solo_con_gasto": true           -> descarta filas de dias sin inversion
  "max_paginas": 50                -> tope de paginacion
  "api_version": "v21.0"           -> por defecto GRAPH_API_VERSION de config

EL TOKEN NO VA EN EL REGISTRO. 'token_env' es el NOMBRE de una variable de
entorno, igual que 'password_env' en zoho_imap y 'dsn_env' en el warehouse. Un
token de Meta con permiso ads_read escrito en una hoja de calculo compartida es
una fuga esperando turno, y el historial de versiones de Google lo guarda para
siempre aunque despues se borre la celda.

Se recomienda un token de SYSTEM USER del Business Manager (no caduca) con
permiso 'ads_read' y acceso asignado a esa cuenta publicitaria. Un token de
usuario normal caduca en 60 dias y la ingesta se cae un martes cualquiera sin
que nadie haya tocado nada.

NOTA DE DESGLOSES. Los breakdowns MULTIPLICAN las filas y Meta prohibe combinar
varios de ciertos grupos. Ademas, con desglose las metricas de personas unicas
(alcance, frecuencia) dejan de ser sumables entre filas: la misma persona
aparece en dos tramos de edad distintos si cumplio años, y sumar alcance por
edad da mas alcance del real. Por eso el default es SIN desglose.

CENTAVOS. El API mezcla dos escalas y esto ya costo caro en otros proyectos: el
GASTO de insights viene en unidades mayores ("12.34" son 12.34 colones) pero
los PRESUPUESTOS de la estructura vienen en la minima denominacion de la moneda
("1000" son 10.00). Solo se dividen los presupuestos; ver _a_monto_presupuesto.
"""

import hashlib
import json
import logging
import re
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

import config
from .base import (
    Fragmento,
    Source,
    describir_tabla,
    registrar_df,
)

logger = logging.getLogger("fachavi.sources.meta_ads")

GRAPH = "https://graph.facebook.com"
TIMEOUT_SEG = 60.0
DIAS_DEFECTO = 30
NIVEL_DEFECTO = "ad"
MAX_PAGINAS_DEFECTO = 50
LIMITE_POR_PAGINA = 500

NIVELES = ("account", "campaign", "adset", "ad")


# --------------------------------------------------------------------------
# Contrato estable entre el conector y los destinos acumulativos.
#
# Se declara aca, y no se infiere del DataFrame, por la misma razon que en
# zoho_imap: una ventana puede venir VACIA (una cuenta sin gasto en 30 dias es
# normal en una PYME que pausa la pauta) y un destino que adivine columnas a
# partir de un DataFrame vacio crearia una tabla sin columnas.
# --------------------------------------------------------------------------

CAMPOS_INSIGHT = (
    # identidad y contexto
    "insight_id",
    "cuenta_id",
    "cuenta_nombre",
    "moneda",
    "zona_horaria",
    "nivel",
    "entidad_id",
    "entidad_nombre",
    "campana_id",
    "campana_nombre",
    "conjunto_id",
    "conjunto_nombre",
    "anuncio_id",
    "anuncio_nombre",
    "objetivo",
    "fecha",
    "fecha_fin",
    "desglose",
    # entrega
    "impresiones",
    "alcance",
    "frecuencia",
    "gasto",
    # trafico
    "clics",
    "clics_unicos",
    "ctr",
    "ctr_unico",
    "cpc",
    "cpm",
    "cpp",
    "clics_enlace",
    "ctr_enlace",
    "costo_por_clic_enlace",
    "clics_salientes",
    "ctr_saliente",
    "vistas_pagina_destino",
    # interaccion
    "interacciones_publicacion",
    "interacciones_pagina",
    "reacciones",
    "comentarios",
    "compartidos",
    "guardados",
    # video
    "reproducciones_video",
    "video_25",
    "video_50",
    "video_75",
    "video_100",
    "thruplays",
    "tiempo_medio_video_seg",
    # resultados de negocio
    "leads",
    "compras",
    "valor_compras",
    "agregados_carrito",
    "checkouts_iniciados",
    "conversaciones_iniciadas",
    "roas_compra",
    # eficiencia (calculadas aca, no vienen del API)
    "cpa_compra",
    "costo_por_lead",
    "costo_por_conversacion",
    # calidad
    "ranking_calidad",
    "ranking_interaccion",
    "ranking_conversion",
    "configuracion_atribucion",
    # crudo: nada de lo que devuelva el API se pierde
    "acciones",
    "valores_acciones",
    "costo_por_accion",
    "raw_insight",
)

# Tipo LOGICO de cada columna. Cada destino lo traduce a su dialecto (ver
# _sql_insight en duckdb_dest.py y postgres_dest.py).
#
# Se declara una sola vez y aca, en vez de repetir dos diccionarios SQL como se
# hizo con _TIPOS_EVENTO y _TIPOS_CORREO, porque son ~60 columnas: mantener dos
# copias de esa lista a mano garantiza que tarde o temprano DuckDB y Postgres
# terminen con esquemas distintos, y el sintoma aparece meses despues, en un
# solo warehouse, en una sola columna.
_ENTERO = "entero"
_DECIMAL = "decimal"
_TEXTO = "texto"
_FECHA = "fecha"
_FECHA_HORA = "fecha_hora"

TIPOS_INSIGHT = {
    "insight_id": _TEXTO,
    "cuenta_id": _TEXTO,
    "cuenta_nombre": _TEXTO,
    "moneda": _TEXTO,
    "zona_horaria": _TEXTO,
    "nivel": _TEXTO,
    "entidad_id": _TEXTO,
    "entidad_nombre": _TEXTO,
    "campana_id": _TEXTO,
    "campana_nombre": _TEXTO,
    "conjunto_id": _TEXTO,
    "conjunto_nombre": _TEXTO,
    "anuncio_id": _TEXTO,
    "anuncio_nombre": _TEXTO,
    "objetivo": _TEXTO,
    "fecha": _FECHA,
    "fecha_fin": _FECHA,
    "desglose": _TEXTO,
    "impresiones": _ENTERO,
    "alcance": _ENTERO,
    "frecuencia": _DECIMAL,
    "gasto": _DECIMAL,
    "clics": _ENTERO,
    "clics_unicos": _ENTERO,
    "ctr": _DECIMAL,
    "ctr_unico": _DECIMAL,
    "cpc": _DECIMAL,
    "cpm": _DECIMAL,
    "cpp": _DECIMAL,
    "clics_enlace": _ENTERO,
    "ctr_enlace": _DECIMAL,
    "costo_por_clic_enlace": _DECIMAL,
    "clics_salientes": _ENTERO,
    "ctr_saliente": _DECIMAL,
    "vistas_pagina_destino": _ENTERO,
    "interacciones_publicacion": _ENTERO,
    "interacciones_pagina": _ENTERO,
    "reacciones": _ENTERO,
    "comentarios": _ENTERO,
    "compartidos": _ENTERO,
    "guardados": _ENTERO,
    "reproducciones_video": _ENTERO,
    "video_25": _ENTERO,
    "video_50": _ENTERO,
    "video_75": _ENTERO,
    "video_100": _ENTERO,
    "thruplays": _ENTERO,
    "tiempo_medio_video_seg": _DECIMAL,
    "leads": _ENTERO,
    "compras": _ENTERO,
    "valor_compras": _DECIMAL,
    "agregados_carrito": _ENTERO,
    "checkouts_iniciados": _ENTERO,
    "conversaciones_iniciadas": _ENTERO,
    "roas_compra": _DECIMAL,
    "cpa_compra": _DECIMAL,
    "costo_por_lead": _DECIMAL,
    "costo_por_conversacion": _DECIMAL,
    "ranking_calidad": _TEXTO,
    "ranking_interaccion": _TEXTO,
    "ranking_conversion": _TEXTO,
    "configuracion_atribucion": _TEXTO,
    "acciones": _TEXTO,
    "valores_acciones": _TEXTO,
    "costo_por_accion": _TEXTO,
    "raw_insight": _TEXTO,
}

CAMPOS_ESTRUCTURA = (
    "entidad_id",
    "nivel",
    "nombre",
    "estado",
    "estado_efectivo",
    "cuenta_id",
    "campana_id",
    "conjunto_id",
    "objetivo",
    "tipo_compra",
    "meta_optimizacion",
    "evento_facturacion",
    "estrategia_puja",
    "puja_monto",
    "presupuesto_diario",
    "presupuesto_total",
    "presupuesto_restante",
    "moneda",
    "inicio",
    "fin",
    "creado_en",
    "actualizado_en",
    "categorias_especiales",
    "segmentacion",
    "objeto_promocionado",
    "creativo",
    "raw_entidad",
)


# --------------------------------------------------------------------------
# Metricas que se le piden al API.
#
# La lista es conservadora A PROPOSITO. Meta deprecia campos entre versiones de
# la Graph API y pedir uno que ya no existe hace fallar la llamada ENTERA con
# error 100 — no devuelve el resto de las metricas, devuelve nada. Por eso:
#   1. aca solo van campos estables;
#   2. lo raro se agrega por cliente con "campos_extra";
#   3. si aun asi uno se cae, _pedir() lo detecta, lo saca y reintenta (ver
#      _campos_culpables), dejando una alerta en vez de una corrida perdida.
# --------------------------------------------------------------------------

CAMPOS_API = [
    "account_id", "account_name", "account_currency",
    "campaign_id", "campaign_name",
    "adset_id", "adset_name",
    "ad_id", "ad_name",
    "objective",
    "date_start", "date_stop",
    "impressions", "reach", "frequency", "spend",
    "clicks", "unique_clicks", "ctr", "unique_ctr", "cpc", "cpm", "cpp",
    "inline_link_clicks", "inline_link_click_ctr", "cost_per_inline_link_click",
    "outbound_clicks", "outbound_clicks_ctr",
    "inline_post_engagement",
    "actions", "action_values", "cost_per_action_type",
    "purchase_roas",
    "video_p25_watched_actions", "video_p50_watched_actions",
    "video_p75_watched_actions", "video_p100_watched_actions",
    "video_thruplay_watched_actions", "video_avg_time_watched_actions",
    "quality_ranking", "engagement_rate_ranking", "conversion_rate_ranking",
    "attribution_setting",
]

# action_type del API -> columna de la tabla. Lo que no este aca igual se
# conserva completo en la columna 'acciones'.
ACCIONES_A_COLUMNA = {
    "link_click": "clics_enlace",
    "landing_page_view": "vistas_pagina_destino",
    "post_engagement": "interacciones_publicacion",
    "page_engagement": "interacciones_pagina",
    "post_reaction": "reacciones",
    "comment": "comentarios",
    "post": "compartidos",
    "onsite_conversion.post_save": "guardados",
    "video_view": "reproducciones_video",
    "lead": "leads",
    "purchase": "compras",
    "add_to_cart": "agregados_carrito",
    "initiate_checkout": "checkouts_iniciados",
    # Click-to-WhatsApp. Para una PYME tica que pauta para que le escriban,
    # ESTE es el resultado que importa, no las impresiones.
    "onsite_conversion.messaging_conversation_started_7d": "conversaciones_iniciadas",
}

# Igual que arriba pero leyendo action_values (valor monetario, no conteo).
VALORES_A_COLUMNA = {
    "purchase": "valor_compras",
}

# Campos del API que son listas de {action_type, value} de una sola metrica.
VIDEO_A_COLUMNA = {
    "video_p25_watched_actions": "video_25",
    "video_p50_watched_actions": "video_50",
    "video_p75_watched_actions": "video_75",
    "video_p100_watched_actions": "video_100",
    "video_thruplay_watched_actions": "thruplays",
    "video_avg_time_watched_actions": "tiempo_medio_video_seg",
    "outbound_clicks": "clics_salientes",
    "outbound_clicks_ctr": "ctr_saliente",
}

CAMPOS_ESTRUCTURA_API = {
    "campaign": [
        "id", "name", "status", "effective_status", "objective", "buying_type",
        "bid_strategy", "daily_budget", "lifetime_budget", "budget_remaining",
        "start_time", "stop_time", "created_time", "updated_time",
        "special_ad_categories", "account_id",
    ],
    "adset": [
        "id", "name", "campaign_id", "status", "effective_status",
        "optimization_goal", "billing_event", "bid_amount", "bid_strategy",
        "daily_budget", "lifetime_budget", "budget_remaining",
        "start_time", "end_time", "created_time", "updated_time",
        "targeting", "promoted_object", "account_id",
    ],
    "ad": [
        "id", "name", "adset_id", "campaign_id", "status", "effective_status",
        "created_time", "updated_time", "creative", "account_id",
    ],
}


# --------------------------------------------------------------------------
# Helpers de conversion
# --------------------------------------------------------------------------

def limpiar_valor_meta(valor):
    """Convierte valores pandas/numpy a tipos que aceptan los drivers SQL."""
    if valor is None:
        return None
    try:
        vacio = pd.isna(valor)
        if isinstance(vacio, bool) and vacio:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(valor, "to_pydatetime"):
        try:
            return valor.to_pydatetime()
        except (TypeError, ValueError):
            pass
    if hasattr(valor, "item"):
        try:
            item = valor.item()
            return None if item is valor else limpiar_valor_meta(item)
        except (TypeError, ValueError):
            pass
    return valor


def fila_insight(serie) -> dict:
    """Proyecta una fila entrante al contrato de insights acumulativo."""
    getter = serie.get
    return {c: limpiar_valor_meta(getter(c)) for c in CAMPOS_INSIGHT}


def hash_insight(fila: dict) -> str:
    """
    Huella de contenido: si Meta no cambio nada de ese dia, no se reescribe la
    fila (solo se refresca 'visto_por_ultima_vez').

    'insight_id' se excluye a proposito: es la IDENTIDAD, no el contenido, y
    meterlo haria que toda fila pareciera distinta de si misma.
    """
    normalizado = {}
    for campo in CAMPOS_INSIGHT:
        if campo == "insight_id":
            continue
        valor = limpiar_valor_meta(fila.get(campo))
        if isinstance(valor, (datetime, date)):
            valor = valor.isoformat()
        normalizado[campo] = valor
    crudo = json.dumps(
        normalizado, ensure_ascii=False, sort_keys=True, default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(crudo.encode("utf-8")).hexdigest()


def _json_texto(valor) -> str:
    return json.dumps(
        valor if valor is not None else {},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )


def _a_numero(valor):
    """
    Todo numero del API de Meta llega como STRING ("1234", "12.34"). Un
    float() pelado sobre "" o None revienta, y dejarlo como texto haria que
    SUM(gasto) fuera una concatenacion en vez de una suma.
    """
    if valor is None or valor == "":
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _a_entero(valor):
    numero = _a_numero(valor)
    return None if numero is None else int(numero)


def _a_fecha_iso(valor):
    """date_start/date_stop vienen SIEMPRE como 'YYYY-MM-DD' (ISO, inequivoco)."""
    if not valor:
        return None
    try:
        return datetime.strptime(str(valor)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _a_fecha_hora(valor):
    """created_time y compañia vienen ISO con offset: '2026-08-03T15:00:00-0600'."""
    if not valor:
        return None
    try:
        ts = pd.Timestamp(valor)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").to_pydatetime()
    except (ValueError, TypeError):
        return None


def _a_monto_presupuesto(valor, divisor: int):
    """
    Presupuestos: minima denominacion -> unidades de la moneda.

    Meta devuelve daily_budget = "1000" para un presupuesto de 10.00. En una
    cuenta en colones eso significa que un presupuesto diario de ₡25.000 llega
    como "2500000". Guardarlo sin dividir no falla, no descarta nada y no
    dispara ninguna alerta: simplemente le dice al cliente que gasta cien veces
    mas de lo que gasta, y el numero es perfectamente plausible.

    El gasto de insights NO pasa por aca: ese ya viene en unidades mayores.
    """
    numero = _a_numero(valor)
    if numero is None:
        return None
    return numero / divisor if divisor else numero


def _suma_acciones(lista, tipo_buscado: str):
    """Busca un action_type dentro de una lista de {action_type, value}."""
    for item in lista or []:
        if str(item.get("action_type", "")) == tipo_buscado:
            return _a_numero(item.get("value"))
    return None


def _primer_valor(lista):
    """
    Campos como video_p25_watched_actions son una LISTA de un solo elemento
    ({'action_type': 'video_view', 'value': '123'}), no un numero. Es de las
    cosas que mas confunden del API: la metrica es escalar pero viaja como
    lista porque comparte estructura con 'actions'.
    """
    if isinstance(lista, list):
        return _a_numero(lista[0].get("value")) if lista else None
    return _a_numero(lista)


def _dividir(numerador, denominador):
    """Costo por resultado. Sin resultados NO es cero: es indefinido."""
    if numerador is None or not denominador:
        return None
    return round(numerador / denominador, 4)


def _texto_desglose(fila: dict, breakdowns: list) -> str:
    """
    Serializa el desglose de forma DETERMINISTA, como texto plano consultable
    ('edad=25-34;genero=female') en vez de JSON.

    Es texto y no JSON porque este valor entra en la identidad de la fila (dos
    corridas tienen que producir la misma cadena o el UPSERT duplicaria) y
    porque el SQL que genera el modelo sabe hacer `WHERE desglose = '...'` pero
    se enreda con las funciones JSON, que ademas difieren entre DuckDB y
    Postgres.
    """
    if not breakdowns:
        return ""
    partes = [f"{b}={fila.get(b, '')}" for b in sorted(breakdowns)]
    return ";".join(partes)


class MetaAdsSource(Source):
    tipo = "meta_ads"

    def cargar(self, con) -> Fragmento:
        cuenta = self._account_id()
        token = self._token()
        nivel = self._nivel()
        dias = self._dias()
        breakdowns = self._breakdowns()

        alertas = []
        meta_cuenta = self._info_cuenta(cuenta, token, alertas)
        desde = date.today() - timedelta(days=dias)
        hasta = date.today()

        crudos = self._pedir_insights(
            cuenta, token, nivel, desde, hasta, breakdowns, alertas
        )
        filas = [
            self._a_fila(c, cuenta, nivel, breakdowns, meta_cuenta)
            for c in crudos
        ]

        if self._es_verdadero(self.config.get("solo_con_gasto", False)):
            antes = len(filas)
            filas = [f for f in filas if (f.get("gasto") or 0) > 0]
            logger.info("[%s] solo_con_gasto: %d de %d filas conservadas",
                        self.fuente_id, len(filas), antes)

        if not filas:
            # No es un error: una PYME que pauso la pauta no tiene insights.
            # Pero tampoco puede pasar mudo, porque se ve IGUAL que un token
            # sin permiso sobre la cuenta.
            msg = (f"[{self.fuente_id}] Meta no devolvio NINGUNA fila para "
                   f"{cuenta} en los ultimos {dias} dias. Puede ser que no haya "
                   "habido pauta activa, o que el token no tenga acceso a esa "
                   "cuenta publicitaria. El historial ya guardado NO se toca.")
            logger.warning(msg)
            alertas.append(msg)

        df = pd.DataFrame(filas, columns=list(CAMPOS_INSIGHT))
        # inferir_tipos NO se usa: cada columna ya sale tipada de _a_fila().
        # Pasarla por el detector solo podria empeorarla — un id de campania
        # de 17 digitos se convertiria en float y perderia precision.
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], errors="coerce")

        nombre_deseado = self.config.get("tabla") or "meta_ads"
        tabla = registrar_df(con, df, nombre_deseado, self.fuente_id)
        logger.info("[meta_ads:%s] tabla %s (%d filas, nivel=%s, %d dias)",
                    self.fuente_id, tabla, len(df), nivel, dias)

        tablas = [tabla]
        if self._es_verdadero(self.config.get("incluir_estructura", False)):
            tabla_est = self._cargar_estructura(
                con, cuenta, token, meta_cuenta, alertas
            )
            if tabla_est:
                tablas.append(tabla_est)

        # Catalogo y KPIs NO se generan aca: viven en el Sheet central del
        # cliente (catalogo_cliente.py). Mientras no exista una fila para estas
        # tablas en su pestania '_catalogo', quedan invisibles para el bot por
        # la regla de fail-closed de bot/catalogo.py. Es deliberado: el gasto
        # publicitario no deberia quedar consultable por el solo hecho de
        # haberse conectado.
        return Fragmento(
            schema="\n\n".join(describir_tabla(con, t) for t in tablas),
            tablas=tablas,
            alertas=alertas,
        )

    # ---------------- config ----------------

    def _account_id(self) -> str:
        crudo = str(self.config.get("account_id", "")).strip()
        if not crudo:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (meta_ads) sin 'account_id' en "
                "config. Es el identificador de la cuenta publicitaria y va con "
                'el prefijo act_ (p.ej. {"account_id": "act_1234567890"}). Se '
                "encuentra en el Administrador de Anuncios, arriba a la "
                "izquierda, junto al nombre de la cuenta."
            )
        # Un cliente que copia el numero pelado del Administrador de Anuncios
        # produciria 404 sin ninguna pista de por que. Se arregla y se sigue.
        if not crudo.startswith("act_"):
            logger.info("[%s] account_id sin prefijo; se usa 'act_%s'.",
                        self.fuente_id, crudo)
            crudo = f"act_{crudo}"
        return crudo

    def _token(self) -> str:
        var = str(self.config.get("token_env", "")).strip()
        if not var:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': falta 'token_env' con el NOMBRE de "
                "la variable de entorno que guarda el token de Meta. El token "
                "no va escrito en el registro."
            )
        return config.secreto_de_env(
            var, para=f"La fuente '{self.fuente_id}' (meta_ads)"
        )

    def _nivel(self) -> str:
        nivel = str(self.config.get("nivel", NIVEL_DEFECTO)).strip().lower()
        if nivel not in NIVELES:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': 'nivel' invalido ('{nivel}'). "
                f"Valores validos: {', '.join(NIVELES)}."
            )
        return nivel

    def _dias(self) -> int:
        try:
            dias = int(self.config.get("dias", DIAS_DEFECTO))
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': 'dias' debe ser un entero."
            ) from e
        if dias <= 0:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': 'dias' debe ser mayor que 0."
            )
        # Meta solo conserva insights de los ultimos 37 meses.
        if dias > 1100:
            logger.warning(
                "[%s] 'dias'=%d supera el historial que conserva Meta (~37 "
                "meses); se pedira igual y el API devolvera lo que tenga.",
                self.fuente_id, dias,
            )
        return dias

    def _breakdowns(self) -> list:
        crudo = self.config.get("breakdowns") or []
        if isinstance(crudo, str):
            crudo = [p.strip() for p in crudo.split(",") if p.strip()]
        if not isinstance(crudo, list):
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': 'breakdowns' debe ser una lista."
            )
        return [str(b).strip() for b in crudo if str(b).strip()]

    def _campos(self) -> list:
        extra = self.config.get("campos_extra") or []
        if isinstance(extra, str):
            extra = [p.strip() for p in extra.split(",") if p.strip()]
        campos = list(CAMPOS_API)
        for c in extra:
            if c and c not in campos:
                campos.append(str(c).strip())
        return campos

    def _version(self) -> str:
        return str(
            self.config.get("api_version")
            or getattr(config, "GRAPH_API_VERSION", "v21.0")
        ).strip()

    @staticmethod
    def _es_verdadero(valor) -> bool:
        """bool('false') es True; por eso esto no es un bool() pelado."""
        if isinstance(valor, bool):
            return valor
        return str(valor).strip().lower() in {"1", "si", "sí", "true", "yes", "on"}

    # ---------------- Graph API ----------------

    def _info_cuenta(self, cuenta: str, token: str, alertas: list) -> dict:
        """
        Nombre, moneda y zona horaria de la cuenta.

        La ZONA HORARIA no es un adorno. Meta corta los dias en la zona de la
        CUENTA publicitaria, no en la del cliente ni en UTC. Si la cuenta quedo
        creada en America/Los_Angeles —pasa constantemente, es el default de
        muchos Business Manager— el "lunes" de Meta arranca a las 2 p.m. del
        domingo tico, y el gasto del lunes nunca le va a cuadrar al cliente
        contra su POS. Se guarda en la tabla para que la discrepancia sea
        explicable en vez de misteriosa.
        """
        datos = {}
        try:
            datos = self._pedir(
                f"{cuenta}",
                token,
                {"fields": "name,currency,timezone_name,currency_offset"},
            )
        except (RuntimeError, httpx.HTTPError) as e:
            msg = (f"[{self.fuente_id}] no se pudo leer la informacion de la "
                   f"cuenta {cuenta}: {e}. Se continua sin nombre ni moneda.")
            logger.warning(msg)
            alertas.append(msg)
            return {}

        # 'zona_esperada' vacia apaga la revision (cliente fuera de CR).
        esperada = str(
            self.config.get("zona_esperada", "America/Costa_Rica")
        ).strip()
        zona = str(datos.get("timezone_name", "")).strip()
        logger.info("[%s] la cuenta %s reporta en la zona '%s' y en %s.",
                    self.fuente_id, cuenta, zona or "(desconocida)",
                    datos.get("currency", "(moneda desconocida)"))
        if esperada and zona and zona != esperada:
            msg = (f"[{self.fuente_id}] la cuenta publicitaria corta sus dias en "
                   f"'{zona}', no en '{esperada}'. Los totales diarios NO van a "
                   "coincidir con los del POS del cliente y la diferencia va a "
                   "parecer un error de datos. Se puede cambiar en el "
                   "Administrador de Anuncios (ojo: Meta obliga a crear una "
                   "cuenta nueva para cambiar la zona) o declarar la zona real "
                   "en 'zona_esperada' para dejar de ver esta alerta.")
            logger.warning(msg)
            alertas.append(msg)
        return datos

    def _divisor_presupuesto(self, meta_cuenta: dict) -> int:
        """
        Cuantas unidades minimas tiene la moneda de la cuenta (100 para colones,
        dolares y euros; 1 para yenes y pesos chilenos).
        """
        crudo = self.config.get("divisor_presupuesto")
        if crudo:
            try:
                return max(1, int(crudo))
            except (TypeError, ValueError):
                logger.warning("[%s] 'divisor_presupuesto' invalido; se usa 100.",
                               self.fuente_id)
        offset = meta_cuenta.get("currency_offset")
        try:
            return max(1, int(offset))
        except (TypeError, ValueError):
            return 100

    def _pedir(self, ruta: str, token: str, params: dict) -> dict:
        """Un GET a la Graph API con los errores de Meta traducidos."""
        url = f"{GRAPH}/{self._version()}/{ruta}"
        with httpx.Client(timeout=TIMEOUT_SEG) as cli:
            r = cli.get(url, params={**params, "access_token": token})
        if r.status_code >= 400:
            raise RuntimeError(_traducir_error(r))
        return r.json()

    def _paginar(self, ruta: str, token: str, params: dict, alertas: list,
                 etiqueta: str) -> list:
        """
        Sigue la paginacion por cursor de la Graph API.

        Meta devuelve paging.next como una URL COMPLETA que ya lleva el cursor
        y el token adentro; se sigue tal cual en vez de reconstruirla, que es
        donde se suelen introducir bucles infinitos.
        """
        tope = int(self.config.get("max_paginas", MAX_PAGINAS_DEFECTO))
        acumulado = []
        url = f"{GRAPH}/{self._version()}/{ruta}"
        consulta = {**params, "access_token": token, "limit": LIMITE_POR_PAGINA}
        paginas = 0

        with httpx.Client(timeout=TIMEOUT_SEG, follow_redirects=True) as cli:
            while url and paginas < tope:
                r = cli.get(url, params=consulta)
                consulta = None          # el 'next' ya trae todo en la URL
                if r.status_code >= 400:
                    raise RuntimeError(_traducir_error(r))
                cuerpo = r.json()
                acumulado.extend(cuerpo.get("data", []))
                paginas += 1
                url = (cuerpo.get("paging") or {}).get("next")

        if url:
            # Mismo criterio que A-10 en api_rest: quedarse corto en silencio
            # es peor que fallar, porque el total sigue siendo plausible.
            msg = (f"[{self.fuente_id}] {etiqueta}: se alcanzo el tope de {tope} "
                   f"paginas y Meta todavia devolvia datos, o sea que FALTAN "
                   "FILAS. Subi 'max_paginas', reduci 'dias' o quita breakdowns.")
            logger.warning(msg)
            alertas.append(msg)

        return acumulado

    def _pedir_insights(self, cuenta, token, nivel, desde, hasta,
                        breakdowns, alertas) -> list:
        """
        Trae los insights diarios de la ventana, con reintento adaptativo si
        Meta rechaza algun campo.
        """
        campos = self._campos()
        intentos = 0

        while True:
            params = {
                "level": nivel,
                "fields": ",".join(campos),
                "time_range": json.dumps({
                    "since": desde.isoformat(), "until": hasta.isoformat(),
                }),
                # Una fila por dia. Sin esto Meta devuelve UN solo total del
                # periodo completo y se pierde toda la serie de tiempo.
                "time_increment": 1,
            }
            if breakdowns:
                params["breakdowns"] = ",".join(breakdowns)
            atribucion = self.config.get("atribucion")
            if atribucion:
                params["action_attribution_windows"] = json.dumps(list(atribucion))

            try:
                return self._paginar(
                    f"{cuenta}/insights", token, params, alertas, "insights"
                )
            except RuntimeError as e:
                culpables = _campos_culpables(str(e), campos)
                if not culpables or intentos >= 3:
                    raise
                intentos += 1
                campos = [c for c in campos if c not in culpables]
                msg = (f"[{self.fuente_id}] Meta rechazo el/los campo(s) "
                       f"{', '.join(sorted(culpables))} en {self._version()}. Se "
                       "reintenta sin ellos: esas columnas van a quedar VACIAS. "
                       "Probablemente se depreciaron o requieren otro permiso.")
                logger.warning(msg)
                alertas.append(msg)

    # ---------------- transformacion ----------------

    def _a_fila(self, crudo: dict, cuenta: str, nivel: str,
                breakdowns: list, meta_cuenta: dict) -> dict:
        acciones = crudo.get("actions") or []
        valores = crudo.get("action_values") or []
        costos = crudo.get("cost_per_action_type") or []

        fila = {
            "cuenta_id": crudo.get("account_id") or cuenta.replace("act_", ""),
            "cuenta_nombre": crudo.get("account_name") or meta_cuenta.get("name", ""),
            "moneda": crudo.get("account_currency") or meta_cuenta.get("currency", ""),
            "zona_horaria": meta_cuenta.get("timezone_name", ""),
            "nivel": nivel,
            "campana_id": crudo.get("campaign_id", ""),
            "campana_nombre": crudo.get("campaign_name", ""),
            "conjunto_id": crudo.get("adset_id", ""),
            "conjunto_nombre": crudo.get("adset_name", ""),
            "anuncio_id": crudo.get("ad_id", ""),
            "anuncio_nombre": crudo.get("ad_name", ""),
            "objetivo": crudo.get("objective", ""),
            "fecha": _a_fecha_iso(crudo.get("date_start")),
            "fecha_fin": _a_fecha_iso(crudo.get("date_stop")),
            "desglose": _texto_desglose(crudo, breakdowns),
            "impresiones": _a_entero(crudo.get("impressions")),
            "alcance": _a_entero(crudo.get("reach")),
            "frecuencia": _a_numero(crudo.get("frequency")),
            "gasto": _a_numero(crudo.get("spend")),
            "clics": _a_entero(crudo.get("clicks")),
            "clics_unicos": _a_entero(crudo.get("unique_clicks")),
            "ctr": _a_numero(crudo.get("ctr")),
            "ctr_unico": _a_numero(crudo.get("unique_ctr")),
            "cpc": _a_numero(crudo.get("cpc")),
            "cpm": _a_numero(crudo.get("cpm")),
            "cpp": _a_numero(crudo.get("cpp")),
            "ctr_enlace": _a_numero(crudo.get("inline_link_click_ctr")),
            "costo_por_clic_enlace": _a_numero(
                crudo.get("cost_per_inline_link_click")),
            "interacciones_publicacion": _a_entero(
                crudo.get("inline_post_engagement")),
            "roas_compra": _primer_valor(crudo.get("purchase_roas")),
            "ranking_calidad": crudo.get("quality_ranking", ""),
            "ranking_interaccion": crudo.get("engagement_rate_ranking", ""),
            "ranking_conversion": crudo.get("conversion_rate_ranking", ""),
            "configuracion_atribucion": crudo.get("attribution_setting", ""),
            "acciones": _json_texto(acciones),
            "valores_acciones": _json_texto(valores),
            "costo_por_accion": _json_texto(costos),
            # Cualquier metrica que Meta agregue, o que el cliente pida por
            # 'campos_extra' sin que exista columna dedicada, sobrevive aca.
            "raw_insight": _json_texto(crudo),
        }

        # Metricas que llegan envueltas en lista de un elemento.
        for campo_api, columna in VIDEO_A_COLUMNA.items():
            fila[columna] = _primer_valor(crudo.get(campo_api))

        for tipo, columna in ACCIONES_A_COLUMNA.items():
            fila[columna] = _suma_acciones(acciones, tipo)
        for tipo, columna in VALORES_A_COLUMNA.items():
            fila[columna] = _suma_acciones(valores, tipo)

        # Dos metricas llegan por DOS caminos: como campo directo y ademas
        # dentro de 'actions'. Se prefiere el campo directo, pero el orden
        # importa: aplicarlo ANTES del bucle de acciones lo pisaba con None
        # cada vez que ese action_type no venia en la respuesta —o sea, en toda
        # cuenta sin conversiones configuradas. El valor SI habia llegado; se
        # perdia al escribirlo.
        for columna, valor in (
            ("clics_enlace", _a_entero(crudo.get("inline_link_clicks"))),
            ("interacciones_publicacion",
             _a_entero(crudo.get("inline_post_engagement"))),
        ):
            if valor is not None:
                fila[columna] = valor

        # Eficiencia. Se calcula aca y no se le deja al bot porque un
        # SUM(gasto)/SUM(compras) por periodo es correcto, pero un
        # AVG(cpa_por_dia) NO lo es, y el modelo se equivoca en eso a menudo.
        # Teniendo la columna por fila, las dos formas quedan disponibles.
        gasto = fila.get("gasto")
        fila["cpa_compra"] = _dividir(gasto, fila.get("compras"))
        fila["costo_por_lead"] = _dividir(gasto, fila.get("leads"))
        fila["costo_por_conversacion"] = _dividir(
            gasto, fila.get("conversaciones_iniciadas"))

        entidad_id, entidad_nombre = _entidad_de(fila, nivel)
        fila["entidad_id"] = entidad_id
        fila["entidad_nombre"] = entidad_nombre
        fila["insight_id"] = _insight_id(fila)

        return _ajustar_tipos(fila)

    # ---------------- estructura (campanias / conjuntos / anuncios) --------

    def _cargar_estructura(self, con, cuenta, token, meta_cuenta,
                           alertas) -> str:
        """
        Snapshot de la configuracion actual de la cuenta.

        Los insights dicen cuanto se gasto; esto dice COMO estaba configurado:
        presupuesto diario, estado, objetivo, segmentacion, creativo. Sin esta
        tabla no se puede responder "cuales campanias estan activas ahora
        mismo" ni "cuanto tengo comprometido por dia", que es justo lo que
        pregunta el dueño de la PYME.
        """
        divisor = self._divisor_presupuesto(meta_cuenta)
        moneda = meta_cuenta.get("currency", "")
        filas = []

        for nivel, ruta in (("campaign", "campaigns"),
                            ("adset", "adsets"),
                            ("ad", "ads")):
            try:
                datos = self._paginar(
                    f"{cuenta}/{ruta}",
                    token,
                    {"fields": ",".join(CAMPOS_ESTRUCTURA_API[nivel])},
                    alertas,
                    f"estructura/{ruta}",
                )
            except (RuntimeError, httpx.HTTPError) as e:
                # Un nivel que falla no tiene por que tumbar los otros dos ni
                # la tabla de insights, que es la importante.
                msg = f"[{self.fuente_id}] no se pudo leer {ruta}: {e}"
                logger.warning(msg)
                alertas.append(msg)
                continue
            for item in datos:
                filas.append(_entidad_a_fila(nivel, item, divisor, moneda))

        df = pd.DataFrame(filas, columns=list(CAMPOS_ESTRUCTURA))
        nombre = (self.config.get("tabla") or "meta_ads") + "_estructura"
        tabla = registrar_df(con, df, nombre, self.fuente_id)
        logger.info("[meta_ads:%s] tabla %s (%d entidades)",
                    self.fuente_id, tabla, len(df))
        return tabla


# --------------------------------------------------------------------------
# Funciones sueltas (fuera de la clase para poder probarlas directo)
# --------------------------------------------------------------------------

def _ajustar_tipos(fila: dict) -> dict:
    """
    Fuerza cada columna al tipo LOGICO declarado en TIPOS_INSIGHT.

    Las metricas de conteo salen de 'actions', donde Meta manda TODO como
    string decimal ("3" pero tambien "3.0"), asi que _suma_acciones devuelve
    float. Insertar un float en una columna BIGINT no falla —los dos motores
    hacen el cast— pero deja el tipo dependiendo del motor y, peor, hace que el
    hash de contenido cambie entre 3 y 3.0 y reescriba filas identicas en cada
    corrida. Se normaliza una sola vez, aca.
    """
    ajustada = {}
    for columna in CAMPOS_INSIGHT:
        valor = fila.get(columna)
        tipo = TIPOS_INSIGHT.get(columna, _TEXTO)
        if valor is None:
            ajustada[columna] = None
        elif tipo == _ENTERO:
            try:
                ajustada[columna] = int(round(float(valor)))
            except (TypeError, ValueError):
                ajustada[columna] = None
        elif tipo == _DECIMAL:
            ajustada[columna] = _a_numero(valor)
        elif tipo == _TEXTO:
            ajustada[columna] = "" if valor is None else str(valor)
        else:
            ajustada[columna] = valor
    return ajustada


def _entidad_de(fila: dict, nivel: str) -> tuple:
    """Id y nombre de la entidad del nivel pedido, para tener columnas uniformes."""
    if nivel == "ad":
        return fila.get("anuncio_id", ""), fila.get("anuncio_nombre", "")
    if nivel == "adset":
        return fila.get("conjunto_id", ""), fila.get("conjunto_nombre", "")
    if nivel == "campaign":
        return fila.get("campana_id", ""), fila.get("campana_nombre", "")
    return fila.get("cuenta_id", ""), fila.get("cuenta_nombre", "")


def _insight_id(fila: dict) -> str:
    """
    Identidad ESTABLE de una fila, para el UPSERT.

    La clave natural es (cuenta, nivel, entidad, dia, desglose). Se hashea en
    vez de concatenarse por la misma razon que en zoho_imap: queda una sola
    columna de largo fijo, indexable, y ningun separador puede aparecer dentro
    de un componente y correr los limites (un nombre de campania con ';' o un
    valor de desglose con '|' romperian una clave concatenada a mano).

    El NOMBRE de la entidad NO entra: el cliente renombra campanias sin ningun
    cuidado, y si el nombre fuera parte de la identidad, renombrar una campania
    duplicaria todo su historial en vez de actualizarlo.
    """
    partes = [
        str(fila.get("cuenta_id") or ""),
        str(fila.get("nivel") or ""),
        str(fila.get("entidad_id") or ""),
        str(fila.get("fecha") or ""),
        str(fila.get("fecha_fin") or ""),
        str(fila.get("desglose") or ""),
    ]
    return hashlib.sha256("\0".join(partes).encode("utf-8")).hexdigest()


def _entidad_a_fila(nivel: str, item: dict, divisor: int, moneda: str) -> dict:
    """Normaliza una campania, un conjunto o un anuncio al mismo esquema."""
    fila = {
        "entidad_id": str(item.get("id", "")),
        "nivel": nivel,
        "nombre": item.get("name", ""),
        "estado": item.get("status", ""),
        "estado_efectivo": item.get("effective_status", ""),
        "cuenta_id": str(item.get("account_id", "")),
        "campana_id": str(item.get("campaign_id", "")
                          if nivel != "campaign" else item.get("id", "")),
        "conjunto_id": str(item.get("adset_id", "")
                           if nivel == "ad" else
                           (item.get("id", "") if nivel == "adset" else "")),
        "objetivo": item.get("objective", ""),
        "tipo_compra": item.get("buying_type", ""),
        "meta_optimizacion": item.get("optimization_goal", ""),
        "evento_facturacion": item.get("billing_event", ""),
        "estrategia_puja": item.get("bid_strategy", ""),
        "puja_monto": _a_monto_presupuesto(item.get("bid_amount"), divisor),
        "presupuesto_diario": _a_monto_presupuesto(
            item.get("daily_budget"), divisor),
        "presupuesto_total": _a_monto_presupuesto(
            item.get("lifetime_budget"), divisor),
        "presupuesto_restante": _a_monto_presupuesto(
            item.get("budget_remaining"), divisor),
        "moneda": moneda,
        "inicio": _a_fecha_hora(item.get("start_time")),
        "fin": _a_fecha_hora(item.get("stop_time") or item.get("end_time")),
        "creado_en": _a_fecha_hora(item.get("created_time")),
        "actualizado_en": _a_fecha_hora(item.get("updated_time")),
        "categorias_especiales": _json_texto(
            item.get("special_ad_categories", [])),
        "segmentacion": _json_texto(item.get("targeting", {})),
        "objeto_promocionado": _json_texto(item.get("promoted_object", {})),
        "creativo": _json_texto(item.get("creative", {})),
        "raw_entidad": _json_texto(item),
    }
    return {c: fila.get(c) for c in CAMPOS_ESTRUCTURA}


def _traducir_error(respuesta) -> str:
    """
    Convierte un error de la Graph API en algo accionable.

    Meta devuelve HTTP 400 para TODO —token vencido, campo inexistente, limite
    de llamadas, cuenta sin permiso— y el mensaje crudo suele ser una sola
    linea sin contexto. Sin esta traduccion, el error que llega a la bitacora
    es indistinguible entre "hay que renovar el token" y "hay que esperar una
    hora", que son dos acciones completamente distintas.
    """
    try:
        error = (respuesta.json() or {}).get("error", {})
    except (ValueError, AttributeError):
        error = {}
    codigo = error.get("code")
    subcodigo = error.get("error_subcode")
    mensaje = error.get("message", "") or respuesta.text[:300]

    if codigo == 190:
        return (f"el token de Meta no es valido o expiro (codigo 190: {mensaje}). "
                "Si es un token de usuario, caduca a los 60 dias: conviene "
                "cambiarlo por uno de System User del Business Manager, que no "
                "caduca.")
    if codigo in (4, 17, 32, 613) or subcodigo in (2446079, 1487742):
        return (f"Meta esta limitando las llamadas de esta app/cuenta (codigo "
                f"{codigo}: {mensaje}). Subi 'frescura_minutos' de la fuente "
                "para consultar menos seguido, o reduci 'dias'.")
    if codigo == 200 or codigo == 10:
        return (f"el token no tiene permiso sobre esta cuenta publicitaria "
                f"(codigo {codigo}: {mensaje}). Revisa que el System User tenga "
                "asignada la cuenta en el Business Manager y el permiso "
                "'ads_read'.")
    if codigo == 100:
        return (f"Meta rechazo la peticion (codigo 100: {mensaje}). Suele ser un "
                "campo o breakdown que no existe en esta version del API, o una "
                "combinacion de breakdowns no permitida.")
    if codigo in (1, 2):
        return (f"Meta no pudo procesar una consulta tan grande (codigo {codigo}: "
                f"{mensaje}). Reduci 'dias', quita breakdowns o usa un 'nivel' "
                "menos granular (campaign en vez de ad).")
    return f"HTTP {respuesta.status_code} de la Graph API: {mensaje}"


_RX_CAMPO = re.compile(r"[A-Za-z0-9_]+")


def _campos_culpables(mensaje: str, campos: list) -> set:
    """
    Detecta que campo(s) rechazo Meta, para reintentar sin ellos.

    El texto del error cambia de forma entre versiones ("nonexistent field
    (social_spend)", "must be a valid insights field. Got: social_spend"), asi
    que en vez de parsear una plantilla se buscan los tokens del mensaje que
    coincidan con algun campo que efectivamente PEDIMOS. Es tolerante al
    formato y no puede inventarse un campo que no estabamos mandando.
    """
    if "100" not in mensaje and "campo" not in mensaje:
        return set()
    pedidos = set(campos)
    encontrados = {t for t in _RX_CAMPO.findall(mensaje) if t in pedidos}
    # 'fields' y 'level' aparecen en muchos mensajes y no son metricas.
    return encontrados - {"fields", "level", "account_id", "date_start"}
