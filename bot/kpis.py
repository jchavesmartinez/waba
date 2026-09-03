"""
Capa semantica: KPIs predefinidos y planificador de decision.

La ingesta deja en `<esquema>._kpis` una fila por KPI (definido en el tab '_kpis'
del Sheet del cliente): nombre, descripcion, preguntas de ejemplo, la formula SQL
canonica, tabla(s) que usa, dimensiones, unidad, supuestos y minimo de datos.

Este modulo:
  1. Lee esos KPIs del warehouse (robusto: si la tabla no existe, no hay KPIs).
  2. Con la pregunta del usuario + el catalogo de KPIs + el esquema real, decide
     UNA de cuatro acciones (planificar):
       - usar_kpi     : un KPI calza claro -> se arma el SQL desde su formula
                        canonica (definicion unica, no improvisada).
       - sql_libre    : no hay KPI pero la pregunta se responde con las tablas
                        habilitadas -> cae al text-to-SQL de siempre (hibrido).
       - pedir_contexto: ambigua o falta un parametro -> se PREGUNTA antes de
                        responder (ej: "¿mejor por unidades o por ingreso?").
       - retar        : un KPI aplica pero no se cumple su minimo de datos o el
                        supuesto haria el numero engañoso -> se ADVIERTE.

El SQL que arma el planificador para 'usar_kpi' pasa igual por el validador de
nl2sql y por la ejecucion de solo-lectura: la capa semantica no saltea la
seguridad, solo mejora la definicion de la consulta.
"""

import json
import logging
import re
import unicodedata
from datetime import date, timedelta

import config
import llm
import sqlglot
from sqlglot import exp
from bot.nl2sql import contexto_temporal
from bot import catalogo, warehouse_ro

logger = logging.getLogger("fachavi.bot.kpis")

def cargar_kpis(cliente: dict) -> list:
    """
    Lee los KPIs habilitados de <esquema>._kpis. Lista vacia si no hay tabla,
    esta apagado, o falla (best-effort: el bot sigue sin capa semantica).
    """
    if not config.BOT_KPIS:
        return []
    try:
        filas = warehouse_ro.leer_interno(cliente, 'SELECT * FROM "_kpis"')
    except Exception as e:  # noqa: BLE001
        logger.info("[%s] sin tabla _kpis (%s)", cliente.get("cliente_id"), e)
        return []
    candidatos = {}
    for f in filas:
        f = {str(k).strip().lower(): ("" if v is None else str(v).strip())
             for k, v in f.items()}
        if not f.get("kpi"):
            continue
        # B-29: antes se importaba catalogo._puede_bot (privada). Si esa
        # funcion cambiaba de nombre o de semantica, la gobernanza de los KPIs
        # se rompia en silencio. Ahora se usa la publica.
        if not catalogo.puede_bot(f.get("instruccion", ""),
                                  etiqueta=f"el KPI '{f.get('kpi')}'"):
            continue
        nombre = f["kpi"]
        # El catálogo consolidado del cliente es la definición de mayor
        # prioridad. El modelo semántico global queda como fallback y no debe
        # sobrescribir una fórmula específica del cliente.
        fuente = f.get("fuente_id", "").lower()
        prioridad = 0 if fuente in {"_cliente", "metadata_cliente_a"} else 1
        anterior = candidatos.get(nombre)
        if anterior is None or prioridad < anterior[0]:
            candidatos[nombre] = (prioridad, f)
    return [f for _, f in candidatos.values()]


# B-31: con muchos KPIs definidos el prompt se vuelve caro y el reconocimiento
# empeora por exceso de opciones. Antes se cortaban por POSICION y eso escondia
# los KPIs del final del Sheet. Ahora se eligen por relevancia contra la pregunta.
_TOPE_KPIS_EN_PROMPT = 16
_STOPWORDS = {
    "como", "cual", "cuales", "cuanto", "cuantos", "dame", "decime",
    "donde", "este", "esta", "estos", "estas", "para", "quiero", "tengo",
    "total", "todos", "todas", "mostrar", "muestra", "necesito", "mes",
}


def _tokens(texto: str) -> set[str]:
    normal = unicodedata.normalize("NFKD", str(texto or ""))
    normal = normal.encode("ascii", "ignore").decode("ascii").lower()
    return {
        t for t in re.findall(r"[a-z0-9_]+", normal)
        if len(t) >= 4 and t not in _STOPWORDS
    }


def _es_ajuste_presentacion(pregunta: str) -> bool:
    """Detecta seguimientos que solo cambian columnas o forma de salida."""
    normal = unicodedata.normalize("NFKD", str(pregunta or ""))
    texto = normal.encode("ascii", "ignore").decode("ascii").lower()
    return bool(re.search(
        r"\b(?:no (?:pongas|incluyas|muestres)|nada mas|la respuesta es|"
        r"las columnas son|los campos son)\b",
        texto,
    ))


def _tiene_periodo_explicito(pregunta: str) -> bool:
    """True si la consulta exige adaptar el SQL a una fecha o periodo."""
    normal = unicodedata.normalize("NFKD", str(pregunta or ""))
    texto = normal.encode("ascii", "ignore").decode("ascii").lower()
    meses = (
        "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
        "octubre|noviembre|diciembre"
    )
    return bool(re.search(
        rf"\b(?:hoy|ayer|anteayer|esta\s+semana|este\s+mes|este\s+ano|"
        rf"semana\s+pasada|mes\s+pasado|ano\s+pasado|ultimos?\s+\d+\s+"
        rf"(?:dias?|semanas?|meses?)|{meses}|20\d{{2}})\b|"
        rf"\b\d{{1,2}}[/.-]\d{{1,2}}(?:[/.-]\d{{2,4}})?\b",
        texto,
    ))


def _seleccionar_kpis(kpis: list, pregunta: str) -> list:
    """Elige KPIs relevantes sin depender del orden de filas en metadata."""
    if not _TOPE_KPIS_EN_PROMPT or len(kpis) <= _TOPE_KPIS_EN_PROMPT:
        return list(kpis)

    consulta = _tokens(pregunta)
    puntuados = []
    for pos, kpi in enumerate(kpis):
        principales = _tokens(" ".join((
            kpi.get("kpi", ""), kpi.get("nombre", ""),
            kpi.get("preguntas_ejemplo", ""),
        )))
        secundarios = _tokens(" ".join((
            kpi.get("descripcion", ""), kpi.get("dimensiones", ""),
            kpi.get("supuestos", ""),
        )))
        score = 3 * len(consulta & principales) + len(consulta & secundarios)
        puntuados.append((score, -pos, kpi))

    if not any(score for score, _, _ in puntuados):
        return list(kpis[:_TOPE_KPIS_EN_PROMPT])
    puntuados.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [k for _, _, k in puntuados[:_TOPE_KPIS_EN_PROMPT]]


def _kpis_texto(kpis: list, pregunta: str = "") -> str:
    seleccionados = _seleccionar_kpis(kpis, pregunta)
    if len(seleccionados) < len(kpis):
        logger.info(
            "se seleccionaron %d de %d KPIs por relevancia para la pregunta",
            len(seleccionados), len(kpis),
        )
    kpis = seleccionados
    bloques = []
    for k in kpis:
        campos = [f"kpi: {k.get('kpi','')}", f"nombre: {k.get('nombre','')}"]
        if k.get("descripcion"): campos.append(f"descripcion: {k['descripcion']}")
        if k.get("preguntas_ejemplo"): campos.append(f"preguntas_ejemplo: {k['preguntas_ejemplo']}")
        if k.get("formula_sql"): campos.append(f"formula_sql: {k['formula_sql']}")
        if k.get("tabla"): campos.append(f"tabla: {k['tabla']}")
        if k.get("dimensiones"): campos.append(f"dimensiones: {k['dimensiones']}")
        if k.get("unidad"): campos.append(f"unidad: {k['unidad']}")
        if k.get("supuestos"): campos.append(f"supuestos: {k['supuestos']}")
        if k.get("minimo_datos"): campos.append(f"minimo_datos: {k['minimo_datos']}")
        bloques.append("- " + "\n  ".join(campos))
    return "\n".join(bloques)


def _mapa_nombres(ctx) -> str:
    """Texto 'nombre_logico -> nombre_real' para que el SQL use el real."""
    pares = [
        f"{t.tabla_logica} -> {t.tabla_real}"
        for t in getattr(ctx, "permitidas", [])
    ]
    return "\n".join(pares)


def sql_canonico(kpi: dict, ctx) -> str:
    """
    Materializa la formula declarada en metadata con nombres fisicos reales.

    El modelo elige QUE KPI corresponde, pero no vuelve a escribir su formula.
    El reemplazo se hace sobre el AST para no tocar alias, columnas o literales.
    """
    formula = str(kpi.get("formula_sql", "")).strip().rstrip(";")
    if not formula:
        raise ValueError(f"el KPI '{kpi.get('kpi', '')}' no tiene formula_sql")

    arbol = sqlglot.parse_one(formula, read="postgres")
    mapa = {p.tabla_logica.lower(): p.tabla_real for p in ctx.permitidas}
    reales = {str(t).lower() for t in ctx.tablas_reales}
    ctes = {str(c.alias_or_name).lower() for c in arbol.find_all(exp.CTE)}

    for tabla in arbol.find_all(exp.Table):
        if tabla.db:
            raise ValueError("formula KPI con esquema explicito no permitida")
        nombre = tabla.name.lower()
        if nombre in ctes or nombre in reales:
            continue
        real = mapa.get(nombre)
        if not real:
            raise ValueError(
                f"la tabla logica '{tabla.name}' del KPI no esta habilitada"
            )
        tabla.set("this", exp.to_identifier(real))

    return arbol.sql(dialect="postgres")


_FILTROS_SALIDA = {
    "linea_id": ("linea_id", "linea_presupuesto_id", "linea_presupuestaria_id"),
    "concepto": ("concepto",),
    "categoria": ("categoria",),
    "moneda": ("moneda",),
}

_PARAMETRO_PERIODO_INICIO = "{{periodo_inicio}}"
_PARAMETRO_PERIODO_FIN = "{{periodo_fin}}"


def admite_periodo_parametrizado(sql: str) -> bool:
    """Indica si una fórmula KPI declara el contrato de período seguro."""
    texto = str(sql or "")
    return (
        _PARAMETRO_PERIODO_INICIO in texto
        and _PARAMETRO_PERIODO_FIN in texto
    )


def _rango_periodo(periodo: dict | None) -> tuple[str, str] | None:
    inicio = str((periodo or {}).get("inicio", "")).strip()
    fin = str((periodo or {}).get("fin_exclusivo", "")).strip()
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", inicio) and re.fullmatch(
        r"20\d{2}-\d{2}-\d{2}", fin,
    ):
        return inicio, fin
    fin_inclusivo = str((periodo or {}).get("fin_inclusivo", "")).strip()
    if not (
        re.fullmatch(r"20\d{2}-\d{2}-\d{2}", inicio)
        and re.fullmatch(r"20\d{2}-\d{2}-\d{2}", fin_inclusivo)
    ):
        return None
    try:
        return inicio, (date.fromisoformat(fin_inclusivo) + timedelta(days=1)).isoformat()
    except ValueError:
        return None


def parametrizar_sql(sql: str, filtros: dict | None,
                     periodo: dict | None = None) -> tuple[str, dict]:
    """Acota un KPI canonico usando dimensiones de un turno verificado.

    La formula KPI no se reescribe ni se entrega al modelo. Se envuelve como
    subconsulta y se agregan comparaciones AST contra columnas que la propia
    formula proyecta. Los valores vienen del resultado anterior ejecutado.
    """
    if not filtros:
        filtros = {}
    rango = _rango_periodo(periodo)
    tiene_parametros = (
        _PARAMETRO_PERIODO_INICIO in sql or _PARAMETRO_PERIODO_FIN in sql
    )
    if tiene_parametros:
        if not rango:
            raise ValueError("el KPI requiere un período explícito válido")
        inicio, fin = rango
        sql = sql.replace(_PARAMETRO_PERIODO_INICIO, inicio).replace(
            _PARAMETRO_PERIODO_FIN, fin,
        )
    arbol = sqlglot.parse_one(sql, read="postgres")
    aplicados = {}
    if tiene_parametros:
        aplicados["periodo"] = dict(periodo or {})
    inicio = str((periodo or {}).get("inicio", ""))
    if not tiene_parametros and re.fullmatch(r"20\d{2}-\d{2}-\d{2}", inicio):
        reemplazo = exp.Cast(
            this=exp.Literal.string(inicio),
            to=exp.DataType.build("DATE"),
        )
        if any(isinstance(n, exp.CurrentDate) for n in arbol.walk()):
            arbol = arbol.transform(
                lambda nodo: reemplazo.copy()
                if isinstance(nodo, exp.CurrentDate) else nodo
            )
            aplicados["periodo"] = periodo
    salidas = {str(n).lower(): str(n) for n in arbol.named_selects}
    condiciones = []
    # La llave estable basta por si sola y evita filtros redundantes por nombre.
    claves = ["linea_id", "concepto", "categoria", "moneda"]
    for clave in claves:
        valor = filtros.get(clave)
        if valor in (None, ""):
            continue
        columna = next((salidas[a] for a in _FILTROS_SALIDA[clave] if a in salidas), None)
        if not columna:
            continue
        referencia = exp.column(columna, table="_kpi")
        normalizada = exp.Lower(this=exp.Trim(this=exp.Cast(
            this=referencia, to=exp.DataType.build("TEXT"))))
        condiciones.append(exp.EQ(
            this=normalizada,
            expression=exp.Literal.string(str(valor).strip().lower()),
        ))
        aplicados[clave] = valor
        # Una linea_id identifica el concepto de forma estable; no hace falta
        # combinarla con etiquetas que pueden haber cambiado de capitalizacion.
        if clave == "linea_id":
            break
    if not condiciones:
        return arbol.sql(dialect="postgres"), aplicados
    condicion = condiciones[0]
    for otra in condiciones[1:]:
        condicion = exp.and_(condicion, otra)
    envuelta = exp.select("*").from_(arbol.subquery("_kpi")).where(condicion)
    return envuelta.sql(dialect="postgres"), aplicados


_SISTEMA = (
    "Sos el planificador de un bot de datos por WhatsApp. Con la PREGUNTA del "
    "usuario, un catalogo de KPIS predefinidos y el ESQUEMA real de tablas, "
    "decidis UNA accion y devolves SOLO un JSON (sin markdown, sin ```), con la "
    "forma:\n"
    '{"relacion":"nueva|seguimiento|modificacion|ambigua",'
    '"heredar_filtros":[],"filtros_actuales":{},'
    '"heredar_periodo":false,"heredar_kpi":false,'
    '"accion":"usar_kpi|sql_libre|pedir_contexto|retar",'
    '"kpi":"","sql":"","mensaje":""}\n'
    "Si completa 'mensaje', use español profesional, cordial y breve; trate al "
    "usuario de usted y no use jerga ni localismos.\n"
    "Reglas:\n"
    "- relacion describe como se conecta la pregunta ACTUAL con el historial: "
    "nueva si se entiende por si misma sin reutilizar la intencion anterior; "
    "seguimiento si continua el mismo analisis; modificacion si conserva parte "
    "del analisis pero cambia dimensiones, periodo o filtros; ambigua si no se "
    "puede decidir con seguridad. Juzgalo por el significado completo, no por "
    "una palabra aislada ni por si la frase trae tema y periodo.\n"
    "- La informacion explicita del mensaje actual manda. En heredar_filtros "
    "inclui SOLO nombres de filtros del ultimo Estado verificado que realmente "
    "deban conservarse y que el usuario no haya sustituido. Valores permitidos: "
    "linea_id, concepto, categoria y moneda. Nunca inventes valores.\n"
    "- filtros_actuales contiene SOLO filtros escritos explicitamente en la "
    "pregunta actual, con claves linea_id, concepto, categoria o moneda. "
    "Ejemplo: 'gastos por comercio de alimentacion' usa "
    "{\"categoria\":\"Alimentacion\"}. No pongas filtros que no aparecen "
    "en la pregunta, ni copies valores del historial: esos van exclusivamente "
    "en heredar_filtros.\n"
    "- heredar_periodo/heredar_kpi indican si hacen falta el periodo o KPI del "
    "ultimo Estado verificado. En una pregunta nueva deben ser false y "
    "heredar_filtros debe estar vacio.\n"
    "- Un pronombre se resuelve dentro de la pregunta actual cuando sea natural: "
    "en 'gasto en alimentacion y como se compara contra su presupuesto', 'su' "
    "se refiere a alimentacion, no automaticamente al resultado anterior.\n"
    "- Si la relacion es ambigua y cambia materialmente el resultado, usa "
    "pedir_contexto con UNA pregunta breve.\n"
    "- usar_kpi: un KPI del catalogo calza claro y tenes los parametros. Pone su "
    "id exacto en 'kpi' y deja 'sql' VACIO. La aplicacion ejecutara la formula_sql "
    "canonica; vos NO la copies, adaptes ni reescribas.\n"
    "- pedir_contexto: si calzan VARIOS KPIs o falta un parametro clave (ej: piden "
    "'el mejor' y hay KPI por unidades y por ingreso; o 'crecimiento' sin periodo). "
    "En 'mensaje' hace UNA pregunta corta para desambiguar. No inventes la respuesta.\n"
    "- retar: si un KPI aplica pero su 'minimo_datos' no se cumple o su 'supuesto' "
    "haria el numero engañoso. En 'mensaje' advertí el riesgo en una linea y ofrecé "
    "como acotarlo. Mejor retar que dar un numero de mentira.\n"
    "- sql_libre: NO hay KPI que calce PERO la pregunta se responde claramente con "
    "las tablas del esquema (un agregado o filtro simple). Dejá 'sql' vacio; otro "
    "paso lo genera.\n"
    "- Si no hay KPI y la pregunta es ambigua o no alcanza con las tablas: "
    "pedir_contexto.\n"
    "- Nunca uses tablas o columnas que no esten en el esquema. Ante duda entre "
    "usar_kpi y sql_libre, preferí usar_kpi.\n"
    "\n"
    "REGLAS PARA NO SER PESADO (criticas):\n"
    "0. NUNCA le preguntes al usuario que dia es hoy, en que año estamos ni cual "
    "es la fecha actual: la tenes arriba. Preguntar eso hace que el bot parezca "
    "roto —el usuario sabe que el sistema conoce la fecha— y ademas no era lo que "
    "faltaba. 'Cuantas ventas hubo hoy' esta COMPLETA: ejecutala.\n"
    "1. Preguntá UNA sola vez como maximo. Si en la conversacion reciente YA "
    "preguntaste por contexto (o ya se ofrecieron opciones), NO vuelvas a "
    "preguntar: elegí un default sensato y EJECUTA (usar_kpi/sql_libre).\n"
    "2. Si el usuario dice 'como veas', 'vos decidí', 'dale', 'lo que sea', 'da "
    "igual' o similar, es LUZ VERDE: tomá el default y ejecutá. Nunca respondas a "
    "eso con otra pregunta.\n"
    "3. Interpretá la respuesta del usuario CONTRA lo que vos ofreciste recien. Si "
    "ofreciste opciones y responde 'las 3', 'todas', 'si', 'esas', se refiere a "
    "ESAS opciones que diste, no a algo nuevo. No reinventes el significado.\n"
    "4. Ofrecé SOLO dimensiones que existan de verdad: las de la lista "
    "'dimensiones' del KPI y que esten en el esquema. NUNCA inventes una dimension "
    "('cliente', 'sucursal', etc.) si no hay columna/tabla para eso. Antes de "
    "nombrar una dimension, verificá que exista.\n"
    "5. Cuando preguntes, UNA sola pregunta y corta. Nada de listar 4 alternativas "
    "ni encadenar '¿y en que periodo? ¿y por que?'. El default primero, la pregunta "
    "solo si de verdad no se puede seguir sin ella.\n"
    "6. Si un KPI trae 'default' en sus supuestos, asumilo y decilo; no lo "
    "preguntes.\n"
    "7. Ante un agregado que pueda dividir por cero (un total sobre un promedio "
    "que puede ser 0), preferi el desglose por la dimension mas fina disponible "
    "antes que el total.\n"
    "8. Para gasto o ventas sin periodo explicito, usa el MES ACTUAL como default "
    "y ejecuta; no preguntes el periodo. Decilo luego en la respuesta.\n"
    "9. En filtros de texto usa comparaciones sin distinguir mayusculas ni "
    "espacios (LOWER(TRIM(...)) o ILIKE).\n"
    "10. 'Cuales son esas', 'mostramelas' o 'dame el detalle' significa listar "
    "las filas que formaron el resultado anterior: conserva exactamente su "
    "categoria, periodo y filtros. El ESQUEMA actual es la unica verdad sobre "
    "tablas disponibles; ignora negativas antiguas del historial."
    "\n11. Si el usuario solo pide quitar, conservar o reordenar columnas del "
    "resultado anterior, conserva la misma metrica, periodo y filtros. No le "
    "vuelvas a preguntar si queria plan o ejecucion: ya lo definio antes."
)

# B-30: la regla especifica de 'runway_inventario' vivia escrita AQUI, en el
# prompt del sistema. Era logica de negocio de UN cliente dentro del codigo del
# producto: no escala a diez clientes y obliga a desplegar para cambiarla. Ahora
# el default se declara como DATO, en la columna 'supuestos' del tab '_kpis' del
# Sheet del cliente:
#
#   kpi              | supuestos
#   runway_inventario| default: por producto (evita el divide-por-cero del total)
#
# La regla 6 de arriba es la version generica que lee ese dato.

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "relacion": {
            "type": "string",
            "enum": ["nueva", "seguimiento", "modificacion", "ambigua"],
        },
        "heredar_filtros": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["linea_id", "concepto", "categoria", "moneda"],
            },
            "uniqueItems": True,
        },
        "filtros_actuales": {
            "type": "object",
            "properties": {
                "linea_id": {"type": "string"},
                "concepto": {"type": "string"},
                "categoria": {"type": "string"},
                "moneda": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "heredar_periodo": {"type": "boolean"},
        "heredar_kpi": {"type": "boolean"},
        "accion": {
            "type": "string",
            "enum": ["usar_kpi", "sql_libre", "pedir_contexto", "retar"],
        },
        "kpi": {"type": "string"},
        "sql": {"type": "string"},
        "mensaje": {"type": "string"},
    },
    "required": [
        "relacion", "heredar_filtros", "filtros_actuales", "heredar_periodo", "heredar_kpi",
        "accion", "kpi", "sql", "mensaje",
    ],
    "additionalProperties": False,
}


def _plan_sql_libre() -> dict:
    return {
        "relacion": "nueva",
        "heredar_filtros": [],
        "filtros_actuales": {},
        "heredar_periodo": False,
        "heredar_kpi": False,
        "accion": "sql_libre",
        "kpi": "",
        "sql": "",
        "mensaje": "",
    }


def _historial_compacto(historial) -> str:
    """Contexto semantico reciente sin respuestas ni SQL desproporcionados."""
    if not historial:
        return ""
    tope_turnos = max(int(getattr(config, "BOT_PLAN_HISTORIAL_TURNOS", 6)), 0)
    tope_chars = max(int(getattr(config, "BOT_PLAN_HISTORIAL_MAX_CHARS", 6000)), 0)
    if not tope_turnos or not tope_chars:
        return ""

    prefijo = "Conversacion reciente:\n"
    sufijo = "\n\n"
    tope_util = max(tope_chars - len(prefijo) - len(sufijo), 0)
    if not tope_util:
        return ""
    etq = {"user": "Usuario", "assistant": "Asistente"}
    bloques = []
    # Se preservan los turnos mas recientes. Cada respuesta se limita antes de
    # agregar el estado compacto para que una tabla larga no desplace el dato
    # semantico importante.
    por_turno = max(min(tope_util // tope_turnos, 1200), 240)
    for turno in list(historial)[-tope_turnos:]:
        contenido = " ".join(str(turno.get("contenido", "") or "").split())
        if len(contenido) > por_turno:
            contenido = contenido[:por_turno] + " ..."
        bloque = f"{etq.get(turno.get('rol'), turno.get('rol'))}: {contenido}"
        estado = turno.get("estado") if turno.get("rol") == "assistant" else None
        if isinstance(estado, dict) and estado:
            bloque += (
                "\nEstado verificado: "
                + json.dumps({
                    "kpi": estado.get("kpi", ""),
                    "filtros": estado.get("filtros", {}),
                    "periodo": estado.get("periodo", {}),
                }, ensure_ascii=False, separators=(",", ":"))
            )
        bloques.append(bloque)

    # El corte se hace desde el inicio de los turnos mas antiguos; nunca se
    # adjuntan los SQL historicos, que ya viven en Neon y no ayudan a resolver
    # si una frase es seguimiento.
    while bloques and len("\n".join(bloques)) > tope_util:
        if len(bloques) > 1:
            bloques.pop(0)
        else:
            bloques[0] = bloques[0][:tope_util]
    return prefijo + "\n".join(bloques) + sufijo


def planificar(pregunta: str, kpis: list, ctx, historial=None) -> dict:
    """
    Devuelve el dict de decision. Si algo falla o no hay KPIs, cae a
    {'accion':'sql_libre'} para no bloquear el camino de datos de siempre.
    """
    if not config.BOT_KPIS or not kpis:
        return _plan_sql_libre()

    hist = _historial_compacto(historial)

    pregunta_relevancia = pregunta
    if historial and _es_ajuste_presentacion(pregunta):
        # La frase actual puede contener solo nombres de columnas y no las
        # palabras del KPI original. Recuperamos la ultima pregunta sustantiva
        # para que el preselector no esconda el KPI usado en el turno anterior.
        for turno in reversed(historial):
            anterior = turno.get("contenido", "")
            if turno.get("rol") == "user" and not _es_ajuste_presentacion(anterior):
                pregunta_relevancia = f"{anterior}\n{pregunta}"
                break

    contenido = (
        f"{contexto_temporal()}\n\n"
        f"{hist}"
        f"KPIS disponibles:\n{_kpis_texto(kpis, pregunta_relevancia)}\n\n"
        f"ESQUEMA real (tablas y columnas que existen):\n{ctx.schema_text}\n\n"
        f"Mapa de nombres (logico -> real; usa el real en el SQL):\n{_mapa_nombres(ctx)}\n\n"
        f"Pregunta del usuario:\n{pregunta}\n\n"
        "Devolve SOLO el JSON."
    )
    try:
        resp = llm.generar_texto(
            config.BOT_MODELO_KPIS,
            contenido,
            max_tokens=700,
            system=_SISTEMA,
            thinking_level="low",
            response_schema=_PLAN_SCHEMA,
        )
        plan = _parsear(resp.texto)
        # Una fecha explícita no invalida un KPI: el responder la aplica con el
        # contrato {{periodo_inicio}}/{{periodo_fin}} de la fórmula canónica.
        # Solo se evita heredar un período anterior distinto.
        if _tiene_periodo_explicito(pregunta):
            plan["heredar_periodo"] = False
        if plan["accion"] != "usar_kpi":
            return plan

        elegido = next(
            (k for k in kpis
             if str(k.get("kpi", "")).strip().lower() == plan["kpi"].lower()),
            None,
        )
        if elegido is None:
            logger.warning("el planificador eligio un KPI inexistente: %r", plan["kpi"])
            plan.update(accion="sql_libre", kpi="", sql="", mensaje="")
            return plan
        try:
            plan["sql"] = sql_canonico(elegido, ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning("KPI '%s' sin SQL canonico ejecutable: %s", plan["kpi"], e)
            plan.update(accion="sql_libre", kpi="", sql="", mensaje="")
            return plan
        return plan
    except Exception as e:  # noqa: BLE001
        logger.warning("planificador KPI fallo (%s); cae a sql_libre", e)
        return _plan_sql_libre()


def _parsear(texto: str) -> dict:
    """Extrae el JSON de la respuesta, tolerante a cercas y ruido."""
    t = (texto or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return _plan_sql_libre()
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return _plan_sql_libre()
    accion = str(d.get("accion", "")).strip().lower()
    if accion not in ("usar_kpi", "sql_libre", "pedir_contexto", "retar"):
        accion = "sql_libre"
    relacion = str(d.get("relacion", "nueva")).strip().lower()
    if relacion not in ("nueva", "seguimiento", "modificacion", "ambigua"):
        relacion = "nueva"
    permitidos = {"linea_id", "concepto", "categoria", "moneda"}
    heredados = []
    for clave in d.get("heredar_filtros", []) or []:
        clave = str(clave).strip().lower()
        if clave in permitidos and clave not in heredados:
            heredados.append(clave)
    if relacion == "nueva":
        heredados = []
    filtros_actuales = {}
    for clave, valor in (d.get("filtros_actuales") or {}).items():
        clave = str(clave).strip().lower()
        valor = str(valor).strip()
        if clave in permitidos and valor:
            filtros_actuales[clave] = valor
    return {
        "relacion": relacion,
        "heredar_filtros": heredados,
        "filtros_actuales": filtros_actuales,
        "heredar_periodo": (
            bool(d.get("heredar_periodo", False)) if relacion != "nueva" else False
        ),
        "heredar_kpi": (
            bool(d.get("heredar_kpi", False)) if relacion != "nueva" else False
        ),
        "accion": accion,
        "kpi": str(d.get("kpi", "")).strip(),
        "sql": str(d.get("sql", "")).strip(),
        "mensaje": str(d.get("mensaje", "")).strip(),
    }
