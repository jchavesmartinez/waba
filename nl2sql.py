"""
Motor text-to-SQL con Claude.

Flujo:
  1. genera_sql()      -> Claude convierte la pregunta + esquema en SQL (solo SELECT)
  2. valida_sql()      -> guarda de seguridad: rechaza cualquier cosa que no sea lectura
  3. ejecuta_sql()     -> corre el SQL en DuckDB, con tope de filas
  4. redacta()         -> Claude convierte el resultado en una respuesta natural
  5. responder()       -> orquesta todo lo anterior
"""

import logging

from anthropic import Anthropic

import config
from sheets import get_connection

logger = logging.getLogger("fachavi-sql.nl2sql")

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else None

# Palabras prohibidas: si aparecen, no es una consulta de solo lectura.
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
- Para fechas usa las funciones de DuckDB (ej: CURRENT_DATE, date_trunc, etc.).

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
    # Por si el modelo puso cercas de codigo a pesar de la instruccion
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def valida_sql(sql: str) -> bool:
    """True si el SQL es una lectura segura."""
    low = sql.lower()
    if low == "no_sql":
        return False
    # Debe empezar como consulta de lectura
    if not (low.startswith("select") or low.startswith("with")):
        return False
    # No debe contener verbos de escritura/administracion
    # (se chequea con limites de palabra para no marcar, p.ej., "created_at")
    import re
    for palabra in _PROHIBIDAS:
        if re.search(rf"\b{palabra}\b", low):
            return False
    # Una sola sentencia (evita '; DROP ...')
    if sql.strip().rstrip(";").count(";") > 0:
        return False
    return True


def ejecuta_sql(con, sql: str):
    """Ejecuta y devuelve (columnas, filas) con tope de seguridad."""
    # Si no trae LIMIT, le agregamos uno para no traer millones de filas.
    if "limit" not in sql.lower():
        sql = sql.rstrip(";") + f" LIMIT {config.MAX_RESULT_ROWS}"
    cur = con.execute(sql)
    columnas = [d[0] for d in cur.description]
    filas = cur.fetchmany(config.MAX_RESULT_ROWS)
    return columnas, filas


def redacta(pregunta: str, columnas, filas) -> str:
    """Claude convierte el resultado tabular en una respuesta en espaniol."""
    datos = f"Columnas: {columnas}\nFilas: {filas}"
    prompt = f"""El usuario pregunto: "{pregunta}"

Estos son los datos que devolvio la consulta:
{datos}

Redacta una respuesta clara y breve en espaniol de Costa Rica, para WhatsApp.
- Responde directo, sin tecnicismos ni mencionar SQL.
- Si hay numeros de dinero, formatea con separadores de miles.
- Si no hay filas, deci que no se encontraron datos para esa consulta.
- No inventes datos que no esten en las filas."""
    resp = _client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def responder(pregunta: str, historial: str = "") -> str:
    """Punto de entrada: pregunta en lenguaje natural -> respuesta en lenguaje natural."""
    if _client is None:
        return "El bot no tiene configurada la API de Claude todavia."

    try:
        con, schema = get_connection()
    except Exception as e:  # noqa: BLE001
        logger.exception("Error cargando datos: %s", e)
        return "No pude acceder a los datos en este momento. Probemos de nuevo en un ratito."

    sql = genera_sql(pregunta, schema, historial)
    logger.info("SQL generado: %s", sql)

    if not valida_sql(sql):
        return (
            "Esa pregunta no la puedo responder con los datos que tengo, "
            "o no es una consulta valida. Probá reformularla."
        )

    try:
        columnas, filas = ejecuta_sql(con, sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error ejecutando SQL: %s", e)
        return "Tuve un problema consultando los datos. Probá reformular la pregunta."

    return redacta(pregunta, columnas, filas)
