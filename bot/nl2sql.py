"""
Text-to-SQL: convierte la pregunta del cliente en UN SELECT de solo lectura,
restringido a las tablas que el catalogo habilito, y luego redacta la respuesta.

Dos capas de seguridad, no una:
  1. El prompt SOLO incluye el schema de las tablas permitidas. El modelo no
     sabe que existen las demas, asi que no puede pedirlas.
  2. El SQL generado se VALIDA con sqlglot antes de ejecutarse:
       - exactamente una sentencia,
       - que sea SELECT / WITH ... SELECT / UNION,
       - sin ningun nodo de escritura o DDL,
       - toda tabla fisica referenciada debe estar en la lista blanca,
       - sin calificar por esquema (no se permite "otro_esquema.tabla").
     Si no valida, no se ejecuta.

La ejecucion en si vive en bot/warehouse_ro.py (transaccion READ ONLY).
"""

import logging
import re

import sqlglot
from sqlglot import exp

import config

logger = logging.getLogger("fachavi.bot.nl2sql")

# Cliente Anthropic perezoso: solo se crea si de verdad se usa (asi importar el
# paquete no exige la key, util para tests).
_cliente = None


def _anthropic():
    global _cliente
    if _cliente is None:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("Falta ANTHROPIC_API_KEY para el bot.")
        _cliente = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _cliente


_SISTEMA_SQL = (
    "Sos un generador de SQL para PostgreSQL. Traduces la pregunta del usuario a "
    "UNA sola consulta SELECT de solo lectura.\n"
    "Reglas estrictas:\n"
    "- Usa UNICAMENTE las tablas y columnas del esquema que se te da. No inventes "
    "  ni asumas otras.\n"
    "- Nombra las tablas SIN prefijo de esquema (el search_path ya esta puesto).\n"
    "- Prohibido INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE u otra "
    "  escritura. Solo SELECT.\n"
    "- Si la pregunta no se puede responder con estas tablas, devolve exactamente: "
    "  SELECT 'NO_RESPONDIBLE' AS nota;\n"
    "- Devolve SOLO el SQL, sin explicacion, sin markdown, sin ```."
)


def _extraer_sql(texto: str) -> str:
    """Limpia cercas de markdown y ruido; deja el SQL pelado."""
    t = (texto or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip().rstrip(";").strip()


def generar_sql(pregunta: str, schema_text: str,
                correccion: str = "", sql_previo: str = "") -> str:
    """Le pide a Claude el SELECT. `correccion` se usa en el reintento."""
    partes = [
        f"Esquema disponible (unicas tablas que existen para vos):\n\n{schema_text}\n",
        f"Pregunta del usuario:\n{pregunta}\n",
    ]
    if correccion:
        partes.append(
            f"El intento anterior fue rechazado por el validador: {correccion}\n"
            f"SQL rechazado:\n{sql_previo}\n"
            "Corregilo respetando las reglas."
        )
    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_SQL,
        max_tokens=600,
        system=_SISTEMA_SQL,
        messages=[{"role": "user", "content": "\n".join(partes)}],
    )
    texto = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _extraer_sql(texto)


# --- Validador -------------------------------------------------------------

_PROHIBIDOS = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Grant, exp.Command,
)


def validar_sql(sql: str, tablas_permitidas) -> tuple[bool, str]:
    """
    Devuelve (ok, motivo). Ver reglas en el docstring del modulo.
    `tablas_permitidas` = set de nombres reales (p.ej. {'sheet_ventas__ventas'}).
    """
    permit = {str(t).strip().lower() for t in tablas_permitidas}
    if not sql:
        return False, "SQL vacio."

    try:
        arboles = [a for a in sqlglot.parse(sql, read="postgres") if a is not None]
    except Exception as e:  # noqa: BLE001
        return False, f"no parseable: {e}"

    if len(arboles) != 1:
        return False, "debe ser exactamente una sentencia."
    arbol = arboles[0]

    if not isinstance(arbol, (exp.Select, exp.Union, exp.Subquery, exp.With)):
        return False, "solo se permite SELECT."

    for nodo in arbol.walk():
        if isinstance(nodo, _PROHIBIDOS):
            return False, f"operacion no permitida ({type(nodo).__name__})."

    # Nombres de CTE: son alias, no tablas fisicas; se ignoran en el chequeo.
    ctes = {c.alias_or_name.lower() for c in arbol.find_all(exp.CTE)}

    for tabla in arbol.find_all(exp.Table):
        if tabla.db:  # tiene esquema explicito -> no se permite calificar
            return False, f"no se permite calificar por esquema: {tabla.db}"
        nombre = tabla.name.lower()
        if nombre in ctes:
            continue
        if nombre not in permit:
            return False, f"tabla no permitida: {nombre}"

    return True, ""


# --- Redaccion de la respuesta --------------------------------------------

_SISTEMA_RESP = (
    "Sos el asistente de datos de una empresa, respondiendo por WhatsApp. A partir "
    "de la pregunta y del resultado de la consulta, escribi UNA respuesta breve, "
    "clara y en español (tico, natural). Sin markdown pesado. Si el resultado viene "
    "vacio, decilo con naturalidad. No inventes datos que no esten en el resultado."
)


def _tabla_texto(columnas, filas, tope=30) -> str:
    if not filas:
        return "(sin filas)"
    lineas = [" | ".join(str(c) for c in columnas)]
    for f in filas[:tope]:
        lineas.append(" | ".join("" if v is None else str(v) for v in f))
    if len(filas) > tope:
        lineas.append(f"... (+{len(filas) - tope} filas)")
    return "\n".join(lineas)


def redactar_respuesta(pregunta: str, columnas, filas) -> str:
    """Convierte el resultado del SELECT en una respuesta de WhatsApp."""
    # Caso trivial: una sola celda -> se responde directo, sin gastar tokens.
    if len(filas) == 1 and len(columnas) == 1:
        valor = filas[0][0]
        if str(valor) == "NO_RESPONDIBLE":
            return ("No puedo responder eso con los datos habilitados para este "
                    "chat. Preguntame sobre ventas o inventario.")

    contenido = (
        f"Pregunta:\n{pregunta}\n\n"
        f"Resultado de la consulta ({len(filas)} filas):\n{_tabla_texto(columnas, filas)}"
    )
    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_RESPUESTA,
        max_tokens=500,
        system=_SISTEMA_RESP,
        messages=[{"role": "user", "content": contenido}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
