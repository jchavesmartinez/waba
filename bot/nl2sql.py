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


# B-24: era 600 fijo. Una consulta con varios JOIN y un CTE se pasa, sale
# truncada y el motivo del rechazo confunde. Configurable.
_MAX_TOKENS_SQL = int(getattr(config, "BOT_MAX_TOKENS_SQL", 1200))


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


def _historial_texto(historial) -> str:
    """Formatea el historial como bloque de texto (para el prompt de SQL)."""
    if not historial:
        return ""
    etiqueta = {"user": "Usuario", "assistant": "Asistente"}
    lineas = [f"{etiqueta.get(t['rol'], t['rol'])}: {t['contenido']}"
              for t in historial]
    return "\n".join(lineas)


def _historial_a_messages(historial) -> list:
    """
    Convierte el historial en mensajes user/assistant para la API, saneado:
    tiene que empezar en 'user' y alternar. Descarta turnos que rompan eso.
    """
    msgs = []
    for t in historial or []:
        rol = t.get("rol")
        if rol not in ("user", "assistant"):
            continue
        if not msgs and rol != "user":
            continue  # no puede arrancar con assistant
        if msgs and msgs[-1]["role"] == rol:
            continue  # sin dos del mismo rol seguidos
        msgs.append({"role": rol, "content": t["contenido"]})
    return msgs


def generar_sql(pregunta: str, schema_text: str,
                correccion: str = "", sql_previo: str = "",
                historial=None) -> str:
    """Le pide a Claude el SELECT. `correccion` se usa en el reintento."""
    partes = [
        f"Esquema disponible (unicas tablas que existen para vos):\n\n{schema_text}\n",
    ]
    # El historial ayuda a resolver referencias ("y de proveedores?", "y ayer?").
    # OJO: es solo contexto para entender la pregunta; NO amplia el esquema. Las
    # unicas tablas que existen son las del bloque de arriba.
    hist = _historial_texto(historial)
    if hist:
        partes.append(
            "Conversacion reciente (contexto para interpretar la pregunta; NO "
            f"agrega tablas ni columnas):\n{hist}\n"
        )
    partes.append(f"Pregunta actual del usuario:\n{pregunta}\n")
    if correccion:
        partes.append(
            f"El intento anterior fue rechazado por el validador: {correccion}\n"
            f"SQL rechazado:\n{sql_previo}\n"
            "Corregilo respetando las reglas."
        )
    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_SQL,
        max_tokens=_MAX_TOKENS_SQL,
        system=_SISTEMA_SQL,
        messages=[{"role": "user", "content": "\n".join(partes)}],
    )
    # B-24: si el modelo se quedo sin tokens, el SQL viene CORTADO y el
    # validador lo rechaza con "no parseable", un motivo que no tiene nada que
    # ver con la causa real. Se registra para no perder una tarde ahi.
    if getattr(resp, "stop_reason", "") == "max_tokens":
        logger.warning(
            "El SQL se trunco por el tope de %d tokens: la consulta va a quedar "
            "incompleta y el validador la va a rechazar por 'no parseable'. "
            "Subi BOT_MAX_TOKENS_SQL si esto se repite.", _MAX_TOKENS_SQL,
        )
    texto = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _extraer_sql(texto)


# --- Validador -------------------------------------------------------------

_PROHIBIDOS = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Grant, exp.Command,
)

# A-15: PostgreSQL permite CREAR una tabla con una sentencia que empieza con
# SELECT ("SELECT ... INTO nueva_tabla"). Sintacticamente ES un Select, asi que
# pasaba los controles de arriba. Lo detenia la transaccion READ ONLY —la
# segunda capa, que hizo exactamente su trabajo— pero una defensa en profundidad
# funciona porque cada capa se mantiene sana, no porque la otra tape.
_PALABRA_INTO = re.compile(r"(?i)\binto\b")
_LITERAL = re.compile(r"'(?:[^']|'')*'")


def _sin_literales(sql: str) -> str:
    """Quita los literales de texto para que un '%into%' en un LIKE no de un
    falso positivo en los chequeos por palabra."""
    return _LITERAL.sub("''", sql or "")


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

    # A-15: dos chequeos, el del arbol y el textual. sqlglot representa el INTO
    # como propiedad del Select segun la version/dialecto, asi que no se confia
    # en una sola forma de detectarlo.
    if arbol.args.get("into") is not None:
        return False, "no se permite SELECT ... INTO."
    if _PALABRA_INTO.search(_sin_literales(sql)):
        return False, "no se permite la palabra INTO en la consulta."

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
    "vacio, decilo con naturalidad. No inventes datos que no esten en el resultado.\n"
    "IMPORTANTE — que NO podes hacer:\n"
    "- Cada mensaje se responde por separado, con el resultado que se te da en ESTE "
    "turno. No podes dejar una consulta 'pendiente' ni ejecutar algo 'despues'.\n"
    "- Por eso NUNCA ofrezcas 'revisar', 'consultar', 'buscar', 'averiguar' ni "
    "prometas traer un dato mas tarde, y no preguntes '¿querES que lo consulte?'. "
    "No tenes acciones diferidas.\n"
    "- Si el usuario quiere OTRO dato, invitalo a pedirlo directamente (ej: "
    "'preguntame por el producto que menos vendio') y se consultara en el momento.\n"
    "- Responde SOLO lo que la pregunta de este turno pide y que este en el "
    "resultado. Si el dato pedido no aparece en el resultado, decilo claro; no lo "
    "inventes ni prometas ir por el.\n"
    "- Se te da la CONSULTA SQL que produjo el resultado. Interpreta las filas "
    "segun lo que esa consulta calculo (fijate en ORDER BY ASC/DESC, MIN/MAX, "
    "los filtros). NO copies la etiqueta ni la muletilla de respuestas anteriores: "
    "si la consulta ordeno ascendente para traer el MENOR, no digas 'el mayor'. "
    "La pregunta de este turno manda, no el formato del turno pasado.\n"
    "- Nunca muestres el SQL ni hables de tablas/columnas tecnicas; hablale al "
    "usuario en terminos de negocio."
)


def tabla_texto(columnas, filas, tope=30) -> str:
    if not filas:
        return "(sin filas)"
    lineas = [" | ".join(str(c) for c in columnas)]
    for f in filas[:tope]:
        lineas.append(" | ".join("" if v is None else str(v) for v in f))
    if len(filas) > tope:
        lineas.append(f"... (+{len(filas) - tope} filas)")
    return "\n".join(lineas)


def redactar_respuesta(pregunta: str, columnas, filas, historial=None, sql="") -> str:
    """Convierte el resultado del SELECT en una respuesta de WhatsApp."""
    # Caso trivial: una sola celda -> se responde directo, sin gastar tokens.
    if len(filas) == 1 and len(columnas) == 1:
        valor = filas[0][0]
        if str(valor) == "NO_RESPONDIBLE":
            return ("No puedo responder eso con los datos habilitados para este "
                    "chat. Preguntame sobre ventas o inventario.")

    # Se incluye el SQL para que el redactor sepa QUE se calculo (ASC/DESC,
    # agregaciones) y no mal-etiquete un resultado de una pregunta eliptica.
    bloque_sql = f"Consulta SQL ejecutada (para que la interpretes; NO la muestres):\n{sql}\n\n" if sql else ""
    contenido = (
        f"Pregunta de este turno:\n{pregunta}\n\n"
        f"{bloque_sql}"
        f"Resultado de la consulta ({len(filas)} filas):\n{tabla_texto(columnas, filas)}"
    )
    # El historial va como turnos previos, para que la respuesta tenga
    # continuidad ("como te decia", "de esos 3 productos..."). El turno actual
    # es el ultimo mensaje 'user'.
    messages = _historial_a_messages(historial)
    messages.append({"role": "user", "content": contenido})
    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_RESPUESTA,
        max_tokens=500,
        system=_SISTEMA_RESP,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


# B-20: bot/responder.py usaba _tabla_texto (privada de este modulo) en su modo
# de emergencia. Ahora la funcion es publica; el alias queda por compatibilidad.
_tabla_texto = tabla_texto
