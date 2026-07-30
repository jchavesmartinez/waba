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

import config
from bot import catalogo, warehouse_ro

logger = logging.getLogger("fachavi.bot.kpis")

_cliente = None


def _anthropic():
    global _cliente
    if _cliente is None:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("Falta ANTHROPIC_API_KEY para el bot.")
        _cliente = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _cliente


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
    out = []
    for f in filas:
        f = {str(k).strip().lower(): ("" if v is None else str(v).strip())
             for k, v in f.items()}
        if not f.get("kpi"):
            continue
        if not catalogo._puede_bot(f.get("instruccion", "")):
            continue
        out.append(f)
    return out


def _kpis_texto(kpis: list) -> str:
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
    pares = [f"{t.tabla_logica} -> {t.tabla_real}" for t in ctx.permitidas]
    return "\n".join(pares)


_SISTEMA = (
    "Sos el planificador de un bot de datos por WhatsApp. Con la PREGUNTA del "
    "usuario, un catalogo de KPIS predefinidos y el ESQUEMA real de tablas, "
    "decidis UNA accion y devolves SOLO un JSON (sin markdown, sin ```), con la "
    "forma:\n"
    '{"accion":"usar_kpi|sql_libre|pedir_contexto|retar","kpi":"","sql":"","mensaje":""}\n'
    "Reglas:\n"
    "- usar_kpi: un KPI del catalogo calza claro y tenes los parametros. Escribi "
    "en 'sql' UN SELECT de solo lectura usando la 'formula_sql' de ese KPI como "
    "definicion canonica (respetala), adaptando dimension/periodo/filtros que pida "
    "el usuario. Usa los NOMBRES REALES de tabla del mapa. Poné el id en 'kpi'.\n"
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
    "usar_kpi y sql_libre, preferí usar_kpi."
)


def planificar(pregunta: str, kpis: list, ctx, historial=None) -> dict:
    """
    Devuelve el dict de decision. Si algo falla o no hay KPIs, cae a
    {'accion':'sql_libre'} para no bloquear el camino de datos de siempre.
    """
    if not config.BOT_KPIS or not kpis:
        return {"accion": "sql_libre", "kpi": "", "sql": "", "mensaje": ""}

    hist = ""
    if historial:
        etq = {"user": "Usuario", "assistant": "Asistente"}
        hist = "Conversacion reciente:\n" + "\n".join(
            f"{etq.get(t['rol'], t['rol'])}: {t['contenido']}" for t in historial[-6:]
        ) + "\n\n"

    contenido = (
        f"{hist}"
        f"KPIS disponibles:\n{_kpis_texto(kpis)}\n\n"
        f"ESQUEMA real (tablas y columnas que existen):\n{ctx.schema_text}\n\n"
        f"Mapa de nombres (logico -> real; usa el real en el SQL):\n{_mapa_nombres(ctx)}\n\n"
        f"Pregunta del usuario:\n{pregunta}\n\n"
        "Devolve SOLO el JSON."
    )
    try:
        resp = _anthropic().messages.create(
            model=config.BOT_MODELO_KPIS,
            max_tokens=700,
            system=_SISTEMA,
            messages=[{"role": "user", "content": contenido}],
        )
        txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _parsear(txt)
    except Exception as e:  # noqa: BLE001
        logger.warning("planificador KPI fallo (%s); cae a sql_libre", e)
        return {"accion": "sql_libre", "kpi": "", "sql": "", "mensaje": ""}


def _parsear(texto: str) -> dict:
    """Extrae el JSON de la respuesta, tolerante a cercas y ruido."""
    t = (texto or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {"accion": "sql_libre", "kpi": "", "sql": "", "mensaje": ""}
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {"accion": "sql_libre", "kpi": "", "sql": "", "mensaje": ""}
    accion = str(d.get("accion", "")).strip().lower()
    if accion not in ("usar_kpi", "sql_libre", "pedir_contexto", "retar"):
        accion = "sql_libre"
    return {
        "accion": accion,
        "kpi": str(d.get("kpi", "")).strip(),
        "sql": str(d.get("sql", "")).strip(),
        "mensaje": str(d.get("mensaje", "")).strip(),
    }
