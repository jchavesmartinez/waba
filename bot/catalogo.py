"""
Gobernanza en tiempo de request: QUE tablas puede leer el bot.

La ingesta ya dejo en `raw_<cliente>._catalogo` una fila por tabla/columna del
cliente, con una columna `instruccion` de texto libre (p.ej. "esta tabla puede
ser usada por el bot"). Este modulo:

  1. Lee ese catalogo del warehouse (no de Google Sheets).
  2. Decide, tabla por tabla, si el bot la puede consultar, mirando la
     'instruccion'. Regla: hay tablas que el bot NO debe leer (ventas internas,
     datos de RRHH, etc.) y otras que SI. La decision es DATO, editable desde el
     Sheet del cliente, sin tocar codigo.
  3. Arma el texto de schema de SOLO las tablas permitidas, para el prompt de
     text-to-SQL. Las tablas prohibidas ni siquiera aparecen: el modelo no sabe
     que existen.

Nombre real de la tabla en Neon:
    catalogo.tabla es el nombre LOGICO ('ventas'); en el warehouse la tabla se
    llama <fuente_id>__<tabla> ('sheet_ventas__ventas').

    B-40 — esto YA NO se reconstruye a partir de 'fuente_id'. El catalogo es
    ahora consolidado por CLIENTE (una fila puede describir una tabla sin que
    esa fila sepa, ni le importe, de que fuente vino): un KPI que cruza
    'ventas' e 'inventario' de dos fuentes distintas no tiene un 'fuente_id'
    unico que reportar. En su lugar, se listan las tablas REALES que existen
    en el warehouse (warehouse_ro.listar_tablas) y se busca cual TERMINA en
    '__<tabla_logica>'. Es mas lento (una consulta extra a information_schema)
    pero es correcto sin importar cuantas fuentes tenga el cliente.
"""

import logging
from dataclasses import dataclass, field

import config
from bot import warehouse_ro

logger = logging.getLogger("fachavi.bot.catalogo")

# Columnas de linaje que agrega la ingesta a cada tabla. Utiles ("¿de cuando es
# este dato?" -> _ingestado_en) pero se marcan aparte para no confundir al modelo.
_COLS_LINAJE = {"_corrida_id", "_fuente_id", "_ingestado_en"}

# Reglas deterministas para leer la 'instruccion'. Se revisan los NEGATIVOS
# primero: "esta tabla NO debe ser usada por el bot" contiene "puede"/"bot" pero
# la intencion es prohibir, y el negativo tiene que ganar.
_NEG = (
    "no debe", "no se debe", "no usar", "no la use", "no leer", "no exponer",
    "no publicar", "no bot", "prohib", "restringid", "confidencial", "sensible",
    "uso interno", "solo interno", "interno", "privad", "oculta",
)
_POS = (
    "puede ser usada por el bot", "puede usarse por el bot", "puede usar el bot",
    "puede ser usada", "puede usar", "usar por el bot", "disponible para el bot",
    "expone", "publica", "si bot", "habilitad",
)

# A-01: valores CONTROLADOS. La coincidencia por palabras clave de abajo es un
# buen respaldo pero produce bloqueos que sorprenden: "no es confidencial, puede
# usarla el bot" bloquea por contener "confidencial", y "puede usarla el bot
# para consultas internas" bloquea por contener "interno". El sesgo va siempre
# hacia bloquear (correcto), pero genera soporte. Con un valor controlado al
# principio de la celda —"si:" / "no:" y despues el texto libre— la decision es
# inequivoca y el resto queda como documentacion.
#   Ejemplo:  si: la puede usar el bot, son las ventas publicas
#             no: nomina, es confidencial
_EXACTOS_SI = ("si", "sí", "true", "1", "x", "habilitada", "habilitado")
_EXACTOS_NO = ("no", "false", "0", "bloqueada", "bloqueado")
_PREFIJOS_SI = ("si:", "sí:")
_PREFIJOS_NO = ("no:", "no:")


def _valor_controlado(t: str):
    """
    Devuelve True/False si la celda usa un valor controlado; None si no.

    OJO con la forma del prefijo: se exige DOS PUNTOS. Aceptar un simple "no "
    al inicio seria peor que el problema que se quiere resolver, porque
    "no es confidencial, puede usarla el bot" empieza con "no " y se bloquearia
    por la forma, no por el contenido. Los dos puntos son la señal inequivoca de
    que quien escribio la celda esta declarando una decision y no redactando.
    """
    if t in _EXACTOS_NO:
        return False
    if t in _EXACTOS_SI:
        return True
    if t.startswith(_PREFIJOS_NO):
        return False
    if t.startswith(_PREFIJOS_SI):
        return True
    return None


def puede_bot(instruccion: str, etiqueta: str = "") -> bool:
    """
    True si la 'instruccion' habilita a la tabla para el bot.

    Orden de decision:
      1. Valor controlado al inicio de la celda ("si:" / "no:").  <- A-01
      2. Coincidencia por palabras clave, negativos primero (fail-closed).
      3. Vacia o ambigua -> config.BOT_PERMITIR_SIN_INSTRUCCION (default False).

    C-05: cuando el paso 3 HABILITA algo porque el modo permisivo esta prendido,
    se registra nombrando el objeto. Antes ese camino era completamente mudo:
    alguien ponia la variable en "si" para descartar una hipotesis, se olvidaba
    de bajarla, y todas las tablas sin documentar de todos los clientes quedaban
    abiertas sin una sola linea de log.
    """
    t = (instruccion or "").strip().lower()

    controlado = _valor_controlado(t) if t else None
    if controlado is not None:
        return controlado

    if t:
        if any(n in t for n in _NEG):
            # A-01: el sesgo va SIEMPRE hacia bloquear, y eso es lo correcto —
            # un bloqueo indebido lo reclama alguien y se arregla en dos
            # minutos; lo contrario es que el bot conteste una pregunta sobre
            # nomina. Pero genera soporte, asi que se deja la pista de como
            # expresarlo sin ambiguedad.
            logger.info(
                "%s queda BLOQUEADA por una palabra de la instruccion (%r). Si la "
                "intencion era habilitarla, escribi la celda como "
                "'si: <tu explicacion>' — el valor controlado gana sobre el texto.",
                etiqueta or "Una tabla del catalogo", (instruccion or "")[:70],
            )
            return False
        if any(p in t for p in _POS):
            return True

    if config.BOT_PERMITIR_SIN_INSTRUCCION:
        logger.warning(
            "MODO PERMISIVO: %s se habilita para el bot pese a que su instruccion "
            "esta %s (instruccion=%r). Esto pasa porque "
            "BOT_PERMITIR_SIN_INSTRUCCION=si.",
            etiqueta or "un objeto del catalogo",
            "vacia" if not t else "ambigua", (instruccion or "")[:60],
        )
        return True
    return False


# Alias de compatibilidad: bot/kpis.py importaba la version privada (B-29).
_puede_bot = puede_bot


@dataclass
class TablaPermitida:
    tabla_logica: str            # 'ventas'
    tabla_real: str              # 'sheet_ventas__ventas'
    fuente_id: str               # 'sheet_ventas'
    descripcion: str = ""        # de la fila columna='*'
    instruccion: str = ""
    # {columna: descripcion} de las filas por-columna del catalogo
    columnas_doc: dict = field(default_factory=dict)


@dataclass
class Contexto:
    schema_text: str                       # bloque para el prompt de nl2sql
    tablas_reales: set = field(default_factory=set)   # lista blanca para el validador
    permitidas: list = field(default_factory=list)    # [TablaPermitida]
    catalogo_tiene_instruccion: bool = True
    # B-26: distingue "el catalogo dice que ninguna tabla esta habilitada" de
    # "no pude LEER el catalogo". Las dos daban el mismo mensaje generico al
    # usuario, y son problemas completamente distintos: uno se arregla en el
    # Sheet, el otro es la base caida o un permiso mal puesto.
    error_lectura: bool = False


def nombres_habilitados(ctx: Contexto) -> list[str]:
    """Nombres lógicos, legibles y únicos de las tablas permitidas."""
    nombres = []
    vistos = set()
    for tabla in getattr(ctx, "permitidas", []) or []:
        nombre = str(getattr(tabla, "tabla_logica", "") or "").strip()
        nombre = " ".join(nombre.replace("_", " ").split())
        clave = nombre.lower()
        if nombre and clave not in vistos:
            vistos.add(clave)
            nombres.append(nombre)
    return nombres


def resumir_habilitados(ctx: Contexto, tope: int = 8) -> str:
    """Lista breve para mensajes al usuario, derivada solo del catálogo."""
    nombres = nombres_habilitados(ctx)
    if not nombres:
        return ""
    limite = max(1, int(tope))
    visibles = nombres[:limite]
    if len(nombres) > limite:
        return ", ".join(visibles) + " y otras tablas habilitadas"
    if len(visibles) == 1:
        return visibles[0]
    return ", ".join(visibles[:-1]) + " y " + visibles[-1]


def _leer_filas_catalogo(cliente: dict) -> tuple[list, bool, bool]:
    """
    Devuelve (filas, hay_columna_instruccion, hubo_error). Robusto a un
    _catalogo viejo que todavia no tenga la columna 'instruccion' (SELECT * y se
    detecta si vino).
    """
    esquema_tabla = '"_catalogo"'  # search_path ya apunta al esquema del cliente
    try:
        filas = warehouse_ro.leer_interno(cliente, f"SELECT * FROM {esquema_tabla}")
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo leer _catalogo: %s", cliente.get("cliente_id"), e)
        return [], True, True
    hay_instr = bool(filas) and ("instruccion" in filas[0])
    return filas, hay_instr, False


def resolver_tablas(cliente: dict) -> tuple[list, bool]:
    """
    Aplica la regla de gobernanza y devuelve (lista de TablaPermitida, hubo_error).
    Agrupa el catalogo por tabla LOGICA (ya no por fuente_id, ver B-40) y
    decide con la fila '*' si existe; si no hay fila '*', exige que TODAS las
    filas de esa tabla habiliten (fail-closed).
    """
    filas, hay_instr, hubo_error = _leer_filas_catalogo(cliente)
    if hubo_error:
        return [], True
    if not hay_instr:
        logger.warning(
            "[%s] el _catalogo no tiene columna 'instruccion'. Re-corre la "
            "ingesta (python sync.py --forzar) para que viaje desde el Sheet.",
            cliente.get("cliente_id"),
        )

    # B-40: se agrupa por TABLA LOGICA sola. Antes se agrupaba por
    # (fuente_id, tabla) porque cada fuente escribia su propio catalogo con su
    # propio fuente_id fiable. Con el catalogo consolidado por cliente, una
    # fila puede describir una tabla sin depender de una fuente unica (un KPI
    # que cruza dos tablas de fuentes distintas es el caso que motivo esto),
    # asi que ya no se puede -ni hace falta- agrupar por fuente.
    grupos: dict = {}
    for f in filas:
        f = {str(k).strip().lower(): ("" if v is None else str(v).strip())
             for k, v in f.items()}
        tabla = f.get("tabla", "")
        if not tabla:
            continue
        grupos.setdefault(tabla, []).append(f)

    if not grupos:
        return [], False

    # Resolver nombre logico -> nombre real preguntandole al warehouse (no
    # reconstruyendo con fuente_id, que ya no es confiable para esto).
    tablas_reales = warehouse_ro.listar_tablas(cliente)
    mapa_real = {}
    for tabla_logica in grupos:
        sufijo = f"__{tabla_logica.lower()}"
        candidatas = [t for t in tablas_reales if t.lower().endswith(sufijo)]
        if len(candidatas) == 1:
            mapa_real[tabla_logica] = candidatas[0]
        elif len(candidatas) > 1:
            # Dos fuentes distintas cargaron una tabla con el MISMO nombre
            # logico (dos hojas 'ventas' de sistemas distintos). Sin un
            # fuente_id confiable en el catalogo no hay forma de saber cual
            # de las dos describe esa fila: se rechazan las dos, fail-closed,
            # y se avisa fuerte porque el arreglo es renombrar una de las
            # tablas logicas en las fuentes (no algo que el bot pueda decidir).
            logger.error(
                "[%s] la tabla logica '%s' existe en MAS DE UNA fuente (%s). "
                "El catalogo consolidado no puede saber a cual fila describe: "
                "se bloquean las dos hasta que se le pongan nombres distintos "
                "en la config de cada fuente (p.ej. 'ventas_pos' y "
                "'ventas_ecommerce').",
                cliente.get("cliente_id"), tabla_logica, ", ".join(candidatas),
            )
        # 0 candidatas: la tabla esta documentada en el catalogo pero no
        # existe (todavia) en el warehouse -- normal si la fuente nunca
        # corrio o esta fresca y no ha escrito. No es un error, se ignora.

    permitidas = []
    for tabla, rows in grupos.items():
        tabla_real = mapa_real.get(tabla)
        if not tabla_real:
            continue

        estrella = next((r for r in rows if r.get("columna", "") in ("*", "")), None)
        if estrella is not None:
            ok = puede_bot(estrella.get("instruccion", ""), etiqueta=f"la tabla '{tabla}'")
            instr = estrella.get("instruccion", "")
            desc = estrella.get("descripcion", "")
        else:
            # A-17: sin fila de tabla completa (columna = '*'), se exige que
            # TODAS las filas de columna habiliten. Alguien puede documentar con
            # esmero las cinco columnas de una tabla, no poner la fila con
            # asterisco, y quedarse sin entender por que la tabla es invisible.
            # El mensaje al usuario ("no hay tablas habilitadas") no apunta a la
            # causa, asi que se registra explicitamente aca.
            logger.info(
                "[%s] la tabla '%s' NO tiene fila con columna='*' en el catalogo: "
                "se decide exigiendo que TODAS sus filas de columna habiliten. "
                "Agregá esa fila para documentar (y gobernar) la tabla completa.",
                cliente.get("cliente_id"), tabla,
            )
            ok = all(puede_bot(r.get("instruccion", ""),
                               etiqueta=f"{tabla}.{r.get('columna', '')}")
                     for r in rows) and bool(rows)
            instr = rows[0].get("instruccion", "")
            desc = ""

        if not ok:
            logger.info("[%s] tabla '%s' NO habilitada para el bot (instruccion=%r)",
                        cliente.get("cliente_id"), tabla, instr[:60])
            continue

        cols_doc = {r["columna"]: r.get("descripcion", "")
                    for r in rows if r.get("columna", "") not in ("*", "")}
        # fuente_id ya no se puede afirmar con certeza para una fila del
        # catalogo consolidado; se guarda vacio en vez de un valor inventado.
        permitidas.append(TablaPermitida(
            tabla_logica=tabla,
            tabla_real=tabla_real,
            fuente_id="",
            descripcion=desc,
            instruccion=instr,
            columnas_doc=cols_doc,
        ))

    return permitidas, False


def construir_contexto(cliente: dict) -> Contexto:
    """
    Arma el Contexto para nl2sql: texto de schema de las tablas permitidas
    (columnas reales de information_schema + descripcion del catalogo) y la
    lista blanca de nombres reales para el validador.
    """
    permitidas, hubo_error = resolver_tablas(cliente)
    if not permitidas:
        return Contexto(schema_text="", tablas_reales=set(), permitidas=[],
                        error_lectura=hubo_error)

    reales = {t.tabla_real for t in permitidas}
    cols_por_tabla = warehouse_ro.listar_columnas(cliente, reales) \
        if hasattr(warehouse_ro, "listar_columnas") else {}

    bloques = []
    for t in permitidas:
        cols = cols_por_tabla.get(t.tabla_real, [])
        lineas_col = []
        for col, tipo in cols:
            if col in _COLS_LINAJE:
                doc = "linaje (columna tecnica de la ingesta)"
            else:
                doc = t.columnas_doc.get(col, "")
            lineas_col.append(f"    - {col} ({tipo}){(': ' + doc) if doc else ''}")
        encabezado = f"Tabla {t.tabla_real}"
        if t.descripcion:
            encabezado += f" — {t.descripcion}"
        bloques.append(encabezado + "\n" + "\n".join(lineas_col))

    return Contexto(
        schema_text="\n\n".join(bloques),
        tablas_reales=reales,
        permitidas=permitidas,
    )
