"""
Motor text-to-SQL con Claude. Ahora recibe el spreadsheet_id del cliente
(lo resuelve main.py a partir de quien escribe) y consulta ESE Sheet.
"""

import logging
import re

from anthropic import Anthropic

import config
from sheets import get_connection

logger = logging.getLogger("fachavi.nl2sql")

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else None

_PROHIBIDAS = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "copy", "pragma", "install", "load", "export", "truncate",
)


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


def responder(pregunta: str, spreadsheet_id: str, historial: str = "") -> str:
    """Pregunta en lenguaje natural -> respuesta, consultando el Sheet del cliente."""
    if _client is None:
        return "El bot no tiene configurada la API de Claude todavia."

    try:
        con, schema = get_connection(spreadsheet_id)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error cargando datos: %s", e)
        return "No pude acceder a los datos en este momento. Probemos de nuevo en un ratito."

    sql = genera_sql(pregunta, schema, historial)
    logger.info("SQL generado: %s", sql)

    if not valida_sql(sql):
        return (
            "Esa pregunta no la puedo responder con los datos que tengo, "
            "o no es una consulta valida. Proba reformularla."
        )

    try:
        columnas, filas = ejecuta_sql(con, sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error ejecutando SQL: %s", e)
        return "Tuve un problema consultando los datos. Proba reformular la pregunta."

    return redacta(pregunta, columnas, filas)
