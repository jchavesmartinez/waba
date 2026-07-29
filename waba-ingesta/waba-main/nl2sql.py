"""
Motor del bot con clasificacion de intencion.

Ante un mensaje, primero clasifica:
  - SALUDO        -> responde amable + que puede hacer
  - GOVERNANCE    -> responde desde el catalogo (metadata: origen, dueño, etc.)
  - DATOS         -> genera SQL, ejecuta, redacta (flujo original)
  - FUERA         -> explica amablemente que solo maneja datos y governance

Asi el bot deja de responder "no puedo" a todo lo que no sea SQL.
"""

import logging
import re

from anthropic import Anthropic

import config
from loader import get_data

logger = logging.getLogger("fachavi.nl2sql")

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else None

_PROHIBIDAS = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "copy", "pragma", "install", "load", "export", "truncate",
)


# ---------- Clasificacion de intencion ----------

def clasifica(pregunta: str, schema: str, catalogo: str) -> str:
    """Devuelve una de: SALUDO, GOVERNANCE, DATOS, FUERA."""
    prompt = f"""Clasifica el mensaje del usuario en UNA categoria. Responde SOLO con la palabra.

Categorias:
- SALUDO: saludos o cortesias ("hola", "buenas", "gracias", "como estas").
- GOVERNANCE: preguntas sobre la metadata de los datos: de que sistema vienen, que
  significa una columna, quien es el dueño del dato, cada cuanto se actualizan, como
  se llama la tabla, de donde sale la informacion.
- DATOS: preguntas que se responden consultando los datos (totales, sumas, conteos,
  comparaciones, rankings, filtros sobre las ventas u otras tablas).
- FUERA: cualquier otra cosa no relacionada con estos datos ni su governance.

Tablas disponibles:
{schema}

Mensaje del usuario: {pregunta}

Categoria:"""
    resp = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    cat = resp.content[0].text.strip().upper()
    for c in ("SALUDO", "GOVERNANCE", "DATOS", "FUERA"):
        if c in cat:
            return c
    return "DATOS"  # por defecto intenta datos


# ---------- Respuestas por tipo ----------

def responde_saludo() -> str:
    return (
        "¡Hola! Soy el asistente de datos de FACHAVI. Puedo ayudarte a consultar "
        "informacion de tus datos (por ejemplo: \"cuanto se vendio este mes\", "
        "\"quien vendio mas\", \"ventas por producto\") y tambien contarte sobre el "
        "origen y la gobernanza de esos datos (de que sistema vienen, que significa "
        "cada campo, quien es el responsable). ¿Que te gustaria saber?"
    )


def responde_governance(pregunta: str, catalogo: str, historial: str = "") -> str:
    if not catalogo.strip():
        return (
            "Todavia no hay catalogo de datos documentado para estas fuentes. "
            "Contacta a FACHAVI para documentar el origen, el significado de los "
            "campos y los responsables de cada dato."
        )
    prompt = f"""El usuario pregunta sobre la metadata/governance de los datos.

Catalogo de datos (metadata documentada):
{catalogo}

{historial}

Pregunta: {pregunta}

Responde en espaniol de Costa Rica, claro y breve para WhatsApp, usando SOLO la
informacion del catalogo. Si el catalogo no cubre lo que preguntan, decilo con
honestidad. No inventes datos de origen ni dueños que no esten en el catalogo."""
    resp = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def responde_fuera() -> str:
    return (
        "Puedo ayudarte con consultas sobre tus datos (ventas, totales, comparaciones) "
        "y con la gobernanza de esos datos (origen, significado de los campos, "
        "responsables). Esa pregunta se sale un poco de eso. ¿Te ayudo con alguna "
        "consulta de tus datos?"
    )


# ---------- Flujo de DATOS (SQL) ----------

def _prompt_sql(pregunta: str, schema: str, historial: str) -> str:
    return f"""Eres un generador de SQL para DuckDB. Convierte la pregunta del usuario en UNA consulta SQL.

Reglas estrictas:
- Devuelve SOLO la consulta SQL, sin explicaciones, sin markdown, sin ```.
- Usa unicamente sentencias SELECT (o WITH ... SELECT). Nunca modifiques datos.
- Usa solo las tablas y columnas del esquema. No inventes nombres.
- Si la pregunta no se puede responder con estas tablas, devuelve exactamente: NO_SQL
- Para fechas usa funciones de DuckDB (CURRENT_DATE, date_trunc, etc.).

Esquema de la base de datos:
{schema}

{historial}

Pregunta del usuario: {pregunta}

SQL:"""


def genera_sql(pregunta: str, schema: str, historial: str = "") -> str:
    resp = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": _prompt_sql(pregunta, schema, historial)}],
    )
    sql = resp.content[0].text.strip()
    return sql.replace("```sql", "").replace("```", "").strip()


def valida_sql(sql: str) -> bool:
    low = sql.lower()
    if low == "no_sql":
        return False
    if not (low.startswith("select") or low.startswith("with")):
        return False
    for palabra in _PROHIBIDAS:
        if re.search(rf"\b{palabra}\b", low):
            return False
    if sql.strip().rstrip(";").count(";") > 0:
        return False
    return True


def ejecuta_sql(con, sql: str):
    if "limit" not in sql.lower():
        sql = sql.rstrip(";") + f" LIMIT {config.MAX_RESULT_ROWS}"
    cur = con.execute(sql)
    columnas = [d[0] for d in cur.description]
    filas = cur.fetchmany(config.MAX_RESULT_ROWS)
    return columnas, filas


def redacta(pregunta: str, columnas, filas) -> str:
    datos = f"Columnas: {columnas}\nFilas: {filas}"
    prompt = f"""El usuario pregunto: "{pregunta}"

Estos son los datos que devolvio la consulta:
{datos}

Redacta una respuesta clara y breve en espaniol de Costa Rica, para WhatsApp.
- Responde directo, sin tecnicismos ni mencionar SQL.
- Si hay montos de dinero, formatea con separadores de miles.
- Si no hay filas, deci que no se encontraron datos para esa consulta.
- No inventes datos que no esten en las filas."""
    resp = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _flujo_datos(pregunta, con, schema, historial):
    sql = genera_sql(pregunta, schema, historial)
    logger.info("SQL generado: %s", sql)
    if not valida_sql(sql):
        return (
            "Esa consulta de datos no la pude armar bien. Proba reformularla, "
            "por ejemplo: \"cuanto se vendio este mes\" o \"ventas por producto\"."
        )
    try:
        columnas, filas = ejecuta_sql(con, sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error ejecutando SQL: %s", e)
        return "Tuve un problema consultando los datos. Proba reformular la pregunta."
    return redacta(pregunta, columnas, filas)


# ---------- Punto de entrada ----------

def responder(pregunta: str, cliente: dict, historial: str = "") -> str:
    if _client is None:
        return "El bot no tiene configurada la API de Claude todavia."

    try:
        con, schema, catalogo = get_data(cliente)
    except RuntimeError as e:
        motivo = str(e)
        if "SIN_FUENTES" in motivo:
            return (
                "Este cliente todavia no tiene ninguna fuente de datos activa. "
                "Contacta a FACHAVI para activar al menos una fuente."
            )
        if "SIN_DATOS" in motivo:
            return (
                "Las fuentes de datos activas no devolvieron tablas consultables "
                "en este momento. Contacta a FACHAVI para revisarlas."
            )
        logger.exception("Error cargando datos: %s", e)
        return "No pude acceder a los datos en este momento. Probemos de nuevo en un ratito."
    except Exception as e:  # noqa: BLE001
        logger.exception("Error cargando datos: %s", e)
        return "No pude acceder a los datos en este momento. Probemos de nuevo en un ratito."

    intencion = clasifica(pregunta, schema, catalogo)
    logger.info("Intencion: %s", intencion)

    if intencion == "SALUDO":
        return responde_saludo()
    if intencion == "GOVERNANCE":
        return responde_governance(pregunta, catalogo, historial)
    if intencion == "FUERA":
        return responde_fuera()
    return _flujo_datos(pregunta, con, schema, historial)
