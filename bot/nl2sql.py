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
from datetime import date

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


def _contexto_temporal() -> str:
    """
    La fecha de HOY, para el prompt de SQL.

    POR QUE. Sin esto el modelo no tiene forma de saber en que año vive, y una
    pregunta tan comun como "cuanto se vendio el 2 de enero" lo obliga a
    inventar el año. Inventa el de su entrenamiento, la consulta sale con
    '2025-01-02', devuelve cero filas y el bot informa —con toda honestidad—
    que no hay datos. El usuario ve un bot que le niega ventas que existen y a
    los dos mensajes se contradice, porque la pregunta siguiente no llevaba año
    y esa si funciono.

    No es un problema de capacidad del modelo: adivinar el año sin referencia
    temporal es imposible por definicion, y un modelo mas caro adivina igual.
    """
    hoy = date.today()
    dias = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
    return (
        f"Fecha de hoy: {hoy.isoformat()} ({dias[hoy.weekday()]}). "
        f"Año en curso: {hoy.year}.\n"
        "- Para fechas relativas ('hoy', 'ayer', 'este mes', 'el trimestre "
        "pasado') usa CURRENT_DATE y aritmetica de intervalos, no fechas fijas.\n"
        "- Si el usuario da un dia y un mes SIN año ('el 2 de enero'), NO asumas "
        "el año en curso a ciegas: los datos pueden ser de otro periodo. Resolve "
        "el año contra la propia tabla, p.ej. filtrando por mes y dia y dejando "
        "que el año salga del dato:\n"
        "    WHERE EXTRACT(MONTH FROM fecha) = 1 AND EXTRACT(DAY FROM fecha) = 2\n"
        "  e inclui la fecha completa en el SELECT para que la respuesta pueda "
        "decir de que año es.\n"
        "- Las columnas de fecha pueden ser timestamp. Para comparar contra un "
        "dia usa fecha::date, no igualdad directa contra un texto."
    )


def generar_sql(pregunta: str, schema_text: str,
                correccion: str = "", sql_previo: str = "",
                historial=None) -> str:
    """Le pide a Claude el SELECT. `correccion` se usa en el reintento."""
    partes = [
        f"{_contexto_temporal()}\n",
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
    "usuario en terminos de negocio.\n"
    "\n"
    "FORMATO DE LA RESPUESTA — reglas duras:\n"
    "- PROHIBIDO dibujar graficos en texto: nada de barras con caracteres, ejes "
    "con | y _, puntos, bloques ▇, ni 'arte ASCII' de ningun tipo. En el celular "
    "se desalinea y queda ilegible. Si el dato pide una visual, NO la dibujes: "
    "deci en una linea que se lo podes mandar como grafico si lo pide.\n"
    "- PROHIBIDO volcar tablas largas en el mensaje. Hasta 8 filas se pueden "
    "listar en texto simple; de ahi en adelante, resumi (el total, el maximo, el "
    "minimo, la tendencia) e invitalo a pedir el archivo.\n"
    "- SI PODES mandar archivos: grafico (imagen), Excel y PDF. NUNCA digas que "
    "no podes generarlos, ni que los copie a mano, ni que los pida 'por otro "
    "medio'. Si los quiere, solo tiene que pedirlos en este mismo chat: "
    "'graficame eso', 'pasamelo en Excel', 'mandame el reporte en PDF'.\n"
    "- No inventes la MONEDA ni la unidad. Si el resultado trae numeros pelados, "
    "presentalos sin simbolo o con el que aparezca en los datos; no asumas pesos, "
    "dolares ni colones.\n"
    "\n"
    "CUANDO EL RESULTADO VIENE VACIO — importa mucho:\n"
    "- Cero filas significa que NADA COINCIDIO CON ESE FILTRO, no que el dato no "
    "exista. Casi siempre el filtro es el que esta mal (un año que el usuario no "
    "dijo, un nombre escrito distinto, un periodo fuera del rango cargado).\n"
    "- Deci 'no encontre registros para <lo que se filtro>', nunca 'no hay ventas "
    "ese dia' ni 'no aparece en el sistema'. La diferencia no es de tono: la "
    "segunda afirma algo sobre la realidad del negocio que el resultado no "
    "respalda, y el usuario le cree.\n"
    "- Si la pregunta traia una fecha sin año o un nombre parcial, decilo: es la "
    "causa mas probable y le da al usuario algo concreto que corregir."
)


# Caracteres con los que un modelo dibuja un grafico en texto. Una linea de un
# eje ("2M |  |  |  |") es casi pura simbologia; una linea de prosa o una fila de
# tabla ("| 2026-01-01 | 944600 |") tiene letras y digitos.
_CHARS_ARTE = set("|_-─│┤├┼╎▇█▄▁▂▃▅▆•·*+^ \t")


def _es_linea_de_arte(linea: str) -> bool:
    l = linea.rstrip()
    if len(l) < 8:
        return False
    simbolos = sum(1 for c in l if c in _CHARS_ARTE)
    return simbolos / len(l) >= 0.8


def _densidad_alfabetica(linea: str) -> float:
    l = linea.strip()
    if not l:
        return 1.0
    return sum(1 for c in l if c.isalpha()) / len(l)


def limpiar_arte_ascii(texto: str) -> tuple[str, bool]:
    """
    Saca los graficos dibujados en texto que el modelo improvisa.
    Devuelve (texto_limpio, hubo_arte).

    POR QUE ESTA CAPA. El prompt ya lo prohibe, pero un prompt no es una
    garantia: el mismo modelo que respeta la regla diez veces la rompe a la
    once, y el sintoma es un mensaje que en el celular del cliente se ve como un
    borron de barras desalineadas. Detectarlo es barato y determinista, asi que
    no hay razon para confiar solo en el prompt.

    El booleano es lo mas util que devuelve: que el modelo HAYA intentado
    dibujar un grafico es la mejor señal de que este resultado pide una visual.
    bot/responder.py lo usa para mandar el grafico DE VERDAD (ver ahi).

    Se exige un BLOQUE (3 lineas o mas): una linea suelta con muchos guiones es
    un separador legitimo, tres seguidas ya son un eje.
    """
    if not texto:
        return texto, False
    lineas = texto.split("\n")
    marcas = [_es_linea_de_arte(l) for l in lineas]

    salida, i, hubo = [], 0, False
    while i < len(lineas):
        if marcas[i]:
            j = i
            while j < len(lineas) and marcas[j]:
                j += 1
            if j - i >= 3:                 # bloque: es un grafico
                hubo = True
                i = j
                continue
        salida.append(lineas[i])
        i += 1

    if not hubo:
        return texto, False

    # Segunda pasada: las etiquetas del eje ("Ene 1 5 10 15 20 25 30 Feb...")
    # quedan huerfanas despues de sacar el bloque, separadas por una linea en
    # blanco, y sueltas no dicen nada. Se van las lineas casi sin letras. Solo
    # se hace cuando YA se detecto arte, para no tocar mensajes normales.
    salida = [l for l in salida
              if not (len(l.strip()) >= 8 and _densidad_alfabetica(l) < 0.25)]

    limpio = re.sub(r"\n{3,}", "\n\n", "\n".join(salida)).strip()
    logger.info("Se quito un grafico ASCII de la respuesta redactada.")
    return limpio, True


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
    texto = "".join(b.text for b in resp.content
                    if getattr(b, "type", "") == "text").strip()
    # OJO: aca NO se limpia el arte ASCII. Lo hace bot/responder.py, que ademas
    # necesita SABER si lo hubo para mandar el grafico de verdad en su lugar.
    return texto


# B-20: bot/responder.py usaba _tabla_texto (privada de este modulo) en su modo
# de emergencia. Ahora la funcion es publica; el alias queda por compatibilidad.
_tabla_texto = tabla_texto
