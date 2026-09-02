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
import unicodedata
from datetime import date, datetime
from decimal import Decimal

import sqlglot
from sqlglot import exp

import config
import llm
from bot.tiempo import fecha_local, nombre_zona

logger = logging.getLogger("fachavi.bot.nl2sql")

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
    "- Para filtros de texto escritos por el usuario (categorias, conceptos, "
    "comercios, productos), compara sin distinguir mayusculas, espacios NI "
    "tildes. PostgreSQL puede no tener unaccent: usa TRANSLATE(LOWER(TRIM(columna)), "
    "'áéíóúüñ', 'aeiouun') y aplica el mismo TRANSLATE al literal.\n"
    "- Una pregunta de COMPOSICION ('que gastos conforman X', 'cuales son esos "
    "movimientos') sin periodo NO significa mes actual: lista el historial que "
    "coincide con el concepto y muestra su fecha. Solo aplica un mes cuando el "
    "usuario lo pida.\n"
    "- Si pregunta por gasto/ventas sin indicar periodo, usa el MES ACTUAL como "
    "default y deja que la respuesta lo diga; no pidas aclaracion.\n"
    "- En seguimientos como 'cuales son esas', 'mostramelas' o 'y el detalle', "
    "reutiliza la categoria, periodo y demas filtros del resultado anterior y "
    "devuelve las filas de detalle que lo componen. El esquema actual manda sobre "
    "cualquier afirmacion vieja del historial acerca de acceso a datos.\n"
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
    lineas = []
    for t in historial:
        linea = f"{etiqueta.get(t['rol'], t['rol'])}: {t['contenido']}"
        if t.get("rol") == "assistant" and t.get("sql"):
            linea += f"\nSQL que produjo esa respuesta: {t['sql']}"
        lineas.append(linea)
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


def contexto_temporal() -> str:
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
    hoy = fecha_local()
    dias = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
    return (
        f"Fecha de hoy: {hoy.isoformat()} ({dias[hoy.weekday()]}). "
        f"Año en curso: {hoy.year}.\n"
        f"Zona horaria del negocio: {nombre_zona()}.\n"
        "- Para fechas relativas ('hoy', 'ayer', 'este mes', 'el trimestre "
        "pasado') usa CURRENT_DATE y aritmetica de intervalos. La sesion de "
        "PostgreSQL ya usa la zona horaria del negocio.\n"
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
    """Le pide a Gemini el SELECT. `correccion` se usa en el reintento."""
    partes = [
        f"{contexto_temporal()}\n",
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
    resp = llm.generar_texto(
        config.BOT_MODELO_SQL,
        "\n".join(partes),
        max_tokens=_MAX_TOKENS_SQL,
        system=_SISTEMA_SQL,
        thinking_level="low",
    )
    # B-24: si el modelo se quedo sin tokens, el SQL viene CORTADO y el
    # validador lo rechaza con "no parseable", un motivo que no tiene nada que
    # ver con la causa real. Se registra para no perder una tarde ahi.
    if resp.truncada:
        logger.warning(
            "El SQL se trunco por el tope de %d tokens: la consulta va a quedar "
            "incompleta y el validador la va a rechazar por 'no parseable'. "
            "Subi BOT_MAX_TOKENS_SQL si esto se repite.", _MAX_TOKENS_SQL,
        )
    return _extraer_sql(resp.texto)


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

    tablas_fisicas = []
    for tabla in arbol.find_all(exp.Table):
        if tabla.db:  # tiene esquema explicito -> no se permite calificar
            return False, f"no se permite calificar por esquema: {tabla.db}"
        nombre = tabla.name.lower()
        if nombre in ctes:
            continue
        tablas_fisicas.append(nombre)
        if nombre not in permit:
            return False, f"tabla no permitida: {nombre}"

    # Un SELECT de texto inventado pasa todas las validaciones de seguridad y
    # produce respuestas absurdas como "no tengo contexto" sin consultar nada.
    # El unico SELECT sin tabla permitido es el sentinel contractual.
    if not tablas_fisicas:
        limpio = re.sub(r"\s+", " ", (sql or "").strip().rstrip(";")).lower()
        if limpio != "select 'no_respondible' as nota":
            return False, "la consulta debe leer una tabla; no se aceptan respuestas literales."

    return True, ""


def validar_granularidad(pregunta: str, sql: str) -> tuple[bool, str]:
    """Impide ejecutar una agrupación temporal distinta de la solicitada."""
    texto = _normalizar_para_columnas(pregunta)
    consulta = str(sql or "").lower()
    pide_mes = bool(re.search(
        r"\b(?:mes\s+a\s+mes|por\s+mes|mensual(?:mente)?)\b", texto,
    ))
    pide_semana = bool(re.search(
        r"\b(?:semana\s+a\s+semana|por\s+semana|semanal(?:mente)?)\b", texto,
    ))
    usa_semana = bool(re.search(
        r"date_trunc\s*\(\s*['\"]week['\"]|\bsemana\b|\bas\s+sem\b", consulta,
    ))
    usa_mes = bool(re.search(
        r"date_trunc\s*\(\s*['\"]month['\"]|\bmensual\b|\bas\s+mes\b", consulta,
    ))
    if pide_mes and usa_semana:
        return False, "El usuario pidió resultados mes a mes, no por semana."
    if pide_semana and usa_mes:
        return False, "El usuario pidió resultados semana a semana, no por mes."
    return True, ""


# --- Redaccion de la respuesta --------------------------------------------

_SISTEMA_RESP = (
    "Es el asistente de datos de una empresa y responde por WhatsApp. A partir "
    "de la pregunta y del resultado de la consulta, escriba UNA respuesta breve, "
    "clara y en español profesional y cordial. Trate al usuario de usted. No use "
    "jerga ni localismos. Sin markdown pesado. Si el "
    "resultado está vacío, indíquelo claramente. No invente datos.\n"
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


def _normalizar_para_columnas(texto: str) -> str:
    normal = unicodedata.normalize("NFKD", str(texto or ""))
    return normal.encode("ascii", "ignore").decode("ascii").lower()


def _aliases_columna(columna: str) -> set[str]:
    """Formas habituales en que un usuario nombra una columna de resultado."""
    nombre = _normalizar_para_columnas(columna).replace("_", " ").strip()
    aliases = {nombre}
    especiales = {
        "linea id": {"linea id", "id de linea", "identificador de linea"},
        "categoria": {"categoria", "categorias"},
        "concepto": {"concepto", "conceptos", "rubro", "rubros"},
        "titular": {"titular", "persona"},
        "presupuesto": {"presupuesto", "presupuestado"},
        "gastado": {"gastado", "gasto"},
        "disponible": {"disponible", "saldo"},
    }
    aliases.update(especiales.get(nombre, set()))
    if "porcentaje" in nombre or re.search(r"(^| )pct($| )", nombre):
        aliases.update({"porcentaje", "porcentaje consumido", "pct", "%"})
    return aliases


def _posicion_alias(texto: str, aliases: set[str]):
    posiciones = []
    for alias in aliases:
        if alias == "%":
            pos = texto.find(alias)
        else:
            encontrado = re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", texto,
            )
            pos = encontrado.start() if encontrado else -1
        if pos >= 0:
            posiciones.append(pos)
    return min(posiciones) if posiciones else None


def proyectar_columnas_solicitadas(pregunta: str, columnas, filas):
    """
    Respeta pedidos explicitos de presentacion sin alterar SQL ni valores.

    La deteccion es conservadora: un simple "solo" no basta si parece expresar
    un filtro de datos en vez de una lista de columnas.
    """
    texto = _normalizar_para_columnas(pregunta)
    if not columnas:
        return list(columnas), list(filas), False

    marcadores = list(re.finditer(
        r"\b(?:solo|solamente|unicamente|la respuesta es|las columnas son|"
        r"los campos son)\b",
        texto,
    ))
    segmento = texto[marcadores[-1].end():] if marcadores else ""
    segmento = re.split(r"\bnada mas\b", segmento, maxsplit=1)[0]
    enfatico = bool(re.search(
        r"\b(?:nada mas|la respuesta es|las columnas son|los campos son|"
        r"no (?:pongas|incluyas|muestres))\b",
        texto,
    ))

    solicitadas = []
    if segmento:
        for indice, columna in enumerate(columnas):
            posicion = _posicion_alias(segmento, _aliases_columna(columna))
            if posicion is not None:
                solicitadas.append((posicion, indice))

    # "Solo categoria Vivienda" suele ser un filtro. Exigimos dos columnas,
    # salvo que el usuario diga expresamente respuesta/columnas/nada mas.
    if solicitadas and (enfatico or len(solicitadas) >= 2):
        indices = [indice for _, indice in sorted(solicitadas)]
    else:
        indices = list(range(len(columnas)))

    # Tambien se admite una exclusion aislada: "sin linea id".
    excluidos = set()
    for indice, columna in enumerate(columnas):
        for alias in _aliases_columna(columna):
            if alias == "%":
                continue
            patron = (
                rf"(?:\bsin\s+|\bomite\s+|\bquita\s+|"
                rf"\bno\s+(?:pongas|incluyas|muestres)\s+(?:el\s+|la\s+)?){re.escape(alias)}\b"
            )
            if re.search(patron, texto):
                excluidos.add(indice)
                break
    indices = [i for i in indices if i not in excluidos]

    cambio = indices != list(range(len(columnas)))
    if not cambio or not indices:
        return list(columnas), list(filas), False
    return (
        [columnas[i] for i in indices],
        [tuple(fila[i] for i in indices) for fila in filas],
        True,
    )


def _nombre_legible(nombre) -> str:
    return str(nombre or "dato").strip().replace("_", " ").capitalize()


def _numero_decimal(valor):
    if isinstance(valor, bool) or not isinstance(valor, (int, float, Decimal)):
        return None
    return Decimal(str(valor))


def _formatear_numero(valor) -> str:
    numero = _numero_decimal(valor)
    if numero is None:
        return str(valor)
    if numero == numero.to_integral_value():
        return f"{int(numero):,}".replace(",", ".")
    ingles = f"{numero:,.2f}"
    return ingles.translate(str.maketrans({",": ".", ".": ","})).rstrip("0").rstrip(",")


def _formatear_valor(valor, columna: str = "", unidad: str = "") -> str:
    if valor is None:
        return "—"
    if isinstance(valor, datetime):
        return valor.strftime("%Y-%m-%d %H:%M")
    if isinstance(valor, date):
        return valor.isoformat()
    numero = _numero_decimal(valor)
    if numero is None:
        return str(valor)

    nombre = str(columna or "").lower()
    unidad_l = str(unidad or "").lower()
    texto = _formatear_numero(numero)
    if any(x in nombre for x in ("pct", "porcentaje", "porcentual", "%")):
        return f"{texto}%"
    es_conteo = any(x in nombre for x in (
        "cantidad", "unidades", "movimientos", "registros", "pendientes",
    ))
    if not es_conteo and "colon" in unidad_l:
        return f"₡{texto}"
    if not es_conteo and ("dolar" in unidad_l or "usd" in unidad_l):
        return f"USD {texto}"
    return texto


def _presupuesto_formateado(columnas, filas, unidad: str):
    """Formato exacto para los dos KPIs de plan presupuestario."""
    idx = {str(c).strip().lower(): i for i, c in enumerate(columnas)}
    if not {"categoria", "mensual", "total_general"}.issubset(idx):
        return None

    total = filas[0][idx["total_general"]] if filas else 0
    lineas = [f"*Presupuesto mensual total: {_formatear_valor(total, 'mensual', unidad)}*"]
    tiene_concepto = "concepto" in idx
    tiene_total_categoria = "total_categoria" in idx

    if not tiene_concepto:
        for fila in filas:
            categoria = fila[idx["categoria"]]
            mensual = fila[idx["mensual"]]
            lineas.append(
                f"• *{categoria}:* {_formatear_valor(mensual, 'mensual', unidad)}"
            )
        return "\n".join(lineas)

    grupos = []
    posiciones = {}
    for fila in filas:
        categoria = str(fila[idx["categoria"]])
        if categoria not in posiciones:
            posiciones[categoria] = len(grupos)
            grupos.append([categoria, [], None])
        grupo = grupos[posiciones[categoria]]
        grupo[1].append(fila)
        if tiene_total_categoria:
            grupo[2] = fila[idx["total_categoria"]]

    for categoria, detalles, total_categoria in grupos:
        if total_categoria is None:
            # Nunca sumar en el redactor: si SQL no trajo el subtotal, se omite.
            lineas.append(f"\n*{categoria}*")
        else:
            lineas.append(
                f"\n*{categoria} — "
                f"{_formatear_valor(total_categoria, 'mensual', unidad)}*"
            )
        for fila in detalles:
            concepto = fila[idx["concepto"]]
            mensual = fila[idx["mensual"]]
            lineas.append(
                f"  • {concepto}: {_formatear_valor(mensual, 'mensual', unidad)}"
            )
    return "\n".join(lineas)


def _resultado_compacto(columnas, filas, unidad: str, max_caracteres: int | None = None) -> str:
    """Tabla compacta para una lista de columnas pedida expresamente."""
    lineas = ["*" + " | ".join(_nombre_legible(c) for c in columnas) + "*"]
    usados = 0
    for fila in filas:
        linea = "• " + " | ".join(
            _formatear_valor(valor, columna, unidad)
            for columna, valor in zip(columnas, fila)
        )
        reserva = 80
        if max_caracteres and len("\n".join(lineas)) + len(linea) + reserva > max_caracteres:
            break
        lineas.append(linea)
        lineas.append("")
        usados += 1
    if usados < len(filas):
        lineas.append(
            f"Hay {len(filas) - usados} registros más. Solicite el Excel para verlos completos."
        )
    return "\n".join(lineas).rstrip()


def _buscar_columna(idx, *fragmentos):
    """Devuelve la primera columna cuyo nombre contiene uno de los fragmentos."""
    for nombre, posicion in idx.items():
        if any(fragmento in nombre for fragmento in fragmentos):
            return posicion
    return None


def _fecha_corta(valor) -> str:
    """Fecha breve y natural para listas que se leen en un teléfono."""
    if not isinstance(valor, (date, datetime)):
        return str(valor)
    meses = (
        "ene", "feb", "mar", "abr", "may", "jun",
        "jul", "ago", "sep", "oct", "nov", "dic",
    )
    return f"{valor.day} {meses[valor.month - 1]}"


def _resultado_movimientos(columnas, filas, unidad: str, tope: int | None = None):
    """Presenta cargos/movimientos sin repetir metadatos en cada renglón."""
    idx = {str(c).strip().lower(): i for i, c in enumerate(columnas)}
    i_desc = _buscar_columna(idx, "descripcion", "descripción", "comercio")
    i_monto = _buscar_columna(idx, "monto", "importe")
    i_fecha = _buscar_columna(idx, "fecha")
    if None in (i_desc, i_monto, i_fecha):
        return None

    i_categoria = _buscar_columna(idx, "categoria", "categoría")
    i_titular = _buscar_columna(idx, "titular")
    i_total = _buscar_columna(idx, "total_general", "monto_total", "total_monto")

    def valor_comun(posicion):
        if posicion is None:
            return ""
        valores = {str(fila[posicion]).strip() for fila in filas if fila[posicion] not in (None, "")}
        return next(iter(valores)) if len(valores) == 1 else ""

    categoria = valor_comun(i_categoria)
    titular = valor_comun(i_titular)
    titulo = " · ".join(x for x in (categoria, titular) if x) or "Movimientos"
    sustantivo = "movimiento" if len(filas) == 1 else "movimientos"
    resumen = f"{len(filas)} {sustantivo}"
    if i_total is not None:
        resumen += f" · *{_formatear_valor(filas[0][i_total], columnas[i_total], unidad)}*"

    lineas = [f"💳 *{titulo}*", resumen, ""]
    for fila in filas[:tope] if tope else filas:
        descripcion = str(fila[i_desc]).strip()
        monto = _formatear_valor(fila[i_monto], columnas[i_monto], unidad)
        fecha = _fecha_corta(fila[i_fecha])
        lineas.append(f"• *{descripcion}* — {monto} · {fecha}")
        lineas.append("")
    if tope and len(filas) > tope:
        restantes = len(filas) - tope
        lineas.extend(("", f"Hay {restantes} movimientos más. Solicite el Excel para verlos completos."))
    return "\n".join(lineas).rstrip()


def _resultado_serie_temporal(columnas, filas, unidad: str):
    """Lista compacta para variaciones mensuales o semanales."""
    idx = {str(c).strip().lower(): i for i, c in enumerate(columnas)}
    i_variacion = _buscar_columna(
        idx, "variacion_pct", "variacion_porcentual", "variacion porcentual",
        "crecimiento_pct",
    )
    i_mes = next((p for n, p in idx.items() if n == "mes"), None)
    i_anio = next((p for n, p in idx.items() if n in ("anio", "año")), None)
    i_fecha = _buscar_columna(idx, "fecha", "semana", "periodo")
    if i_variacion is None or (i_mes is None and i_fecha is None):
        return None

    i_total = _buscar_columna(idx, "total_ventas", "total ventas", "monto_total", "total")
    i_diferencia = _buscar_columna(idx, "diferencia_monto", "diferencia monto")
    meses = ("Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic")

    def periodo(fila):
        if i_mes is not None:
            try:
                etiqueta = meses[int(fila[i_mes]) - 1]
            except (TypeError, ValueError, IndexError):
                etiqueta = str(fila[i_mes])
            if i_anio is not None:
                etiqueta += f" {int(fila[i_anio])}"
            return etiqueta
        valor = fila[i_fecha]
        if isinstance(valor, (date, datetime)):
            return f"{meses[valor.month - 1]} {valor.year}"
        return str(valor)

    lineas = ["📊 *Variación por período*", f"{len(filas)} períodos", ""]
    for fila in filas:
        variacion = _formatear_valor(
            fila[i_variacion], columnas[i_variacion], unidad,
        )
        total = (
            _formatear_valor(fila[i_total], columnas[i_total], unidad)
            if i_total is not None else ""
        )
        encabezado = f"• *{periodo(fila)}*"
        if total:
            encabezado += f" — {total}"
        lineas.append(encabezado)
        detalle = f"  Variación: {variacion}"
        if i_diferencia is not None and fila[i_diferencia] is not None:
            detalle += (
                f" · Diferencia: "
                f"{_formatear_valor(fila[i_diferencia], columnas[i_diferencia], unidad)}"
            )
        lineas.append(detalle)
        lineas.append("")
    return "\n".join(lineas).rstrip()


def _resultado_lista(columnas, filas, unidad: str, tope: int | None = None):
    """Formato móvil común para cualquier tabla con varias filas."""
    idx = {str(c).strip().lower(): i for i, c in enumerate(columnas)}
    i_fecha = _buscar_columna(idx, "fecha")
    i_principal = _buscar_columna(
        idx, "producto", "concepto", "nombre", "descripcion", "descripción",
        "comercio", "cliente", "sucursal", "categoria", "categoría",
    )
    titulo = _nombre_legible(columnas[i_principal]) if i_principal is not None else "Resultados"
    sustantivo = "registro" if len(filas) == 1 else "registros"
    lineas = [f"📋 *{titulo}*", f"{len(filas)} {sustantivo}", ""]

    seleccion = filas[:tope] if tope else filas
    for numero, fila in enumerate(seleccion, 1):
        encabezado = (
            f"Registro {numero}" if i_principal is None else
            _formatear_valor(fila[i_principal], columnas[i_principal], unidad)
        )
        extras = []
        for posicion, (columna, valor) in enumerate(zip(columnas, fila)):
            if posicion == i_principal:
                continue
            if posicion == i_fecha:
                extras.append(_fecha_corta(valor))
            else:
                extras.append(
                    f"{_nombre_legible(columna)}: "
                    f"{_formatear_valor(valor, columna, unidad)}"
                )
        lineas.append(f"• *{encabezado}*")
        if extras:
            lineas.append("  " + " · ".join(extras))
        lineas.append("")
    if tope and len(filas) > tope:
        lineas.extend(("", f"Se omitieron {len(filas) - tope} registros."))
    return "\n".join(lineas).rstrip()


_PEDIDO_ANALITICO = re.compile(
    r"\b(?:analiz[aeá]|analisis|análisis|patrones?|tendencias?|anomali[áa]s?|"
    r"hallazgos?|insights?|aumentan?|suben?|crecen?|bajan?|disminuyen?|caen?|"
    r"evoluci[oó]n|comportamiento|compar[ae]|versus|por qu[eé])\b",
    re.IGNORECASE,
)


def _es_pedido_analitico(pregunta: str) -> bool:
    return bool(_PEDIDO_ANALITICO.search(str(pregunta or "")))


def _datos_para_analisis(columnas, filas, max_caracteres: int = 16000) -> str:
    """Serializa resultados exactos con un límite de entrada predecible."""
    lineas = ["\t".join(str(c) for c in columnas)]
    for fila in filas:
        linea = "\t".join(
            _formatear_valor(valor, columna) for columna, valor in zip(columnas, fila)
        )
        if len("\n".join(lineas)) + len(linea) + 1 > max_caracteres:
            break
        lineas.append(linea)
    return "\n".join(lineas)


def _redactar_analisis(pregunta: str, columnas, filas, sql: str = "") -> str:
    """Pide conclusiones al modelo usando solo cifras ya calculadas por SQL."""
    sistema = (
        "Es un analista de datos que responde por WhatsApp en español claro. "
        "Responda primero la conclusión concreta que pide el usuario y después "
        "presente de 2 a 5 hallazgos breves respaldados con cifras exactas del "
        "resultado. Destaque tendencia, cambios relevantes y anomalías solo si "
        "los datos las demuestran. Distinga hechos de hipótesis. No invente "
        "causas, métricas, monedas ni datos. No copie todas las filas, no muestre "
        "SQL y no sugiera pedir Excel. Use formato ligero de WhatsApp. Si el "
        "resultado no permite llegar a una conclusión, dígalo claramente. "
        "Use asteriscos simples únicamente en títulos cortos; no encierre cifras "
        "ni oraciones completas entre asteriscos y nunca use asteriscos dobles."
    )
    contenido = (
        f"Pregunta: {pregunta}\n\n"
        f"Resultado exacto de la consulta:\n{_datos_para_analisis(columnas, filas)}\n\n"
        f"Contexto de cálculo (no mostrar al usuario): {sql}"
    )
    respuesta = llm.generar_texto(
        config.BOT_MODELO_RESPUESTA,
        contenido,
        system=sistema,
        max_tokens=1200,
        thinking_level="minimal",
    )
    if respuesta.truncada:
        respuesta = llm.generar_texto(
            config.BOT_MODELO_RESPUESTA,
            contenido,
            system=sistema + " Entregue una respuesta completa y muy concisa.",
            max_tokens=2400,
            thinking_level="minimal",
        )
    if respuesta.truncada:
        raise RuntimeError("El análisis fue truncado dos veces por el modelo.")
    return respuesta.texto.strip().replace("**", "*")


def _analisis_tendencia_local(columnas, filas, unidad: str = "") -> str | None:
    """Conclusión numérica segura cuando el redactor analítico no responde."""
    if len(filas) < 2:
        return None
    nombres = [str(c).strip().lower() for c in columnas]
    candidatos = []
    for i, nombre in enumerate(nombres):
        if any(x in nombre for x in ("pct", "porcentaje", "anterior", "cantidad")):
            continue
        if any(x in nombre for x in ("total_ventas", "total ventas", "monto", "total", "ventas")):
            candidatos.append(i)
    i_valor = next(
        (i for i in candidatos if all(_numero_decimal(f[i]) is not None for f in filas)),
        None,
    )
    if i_valor is None:
        return None

    valores = [_numero_decimal(f[i_valor]) for f in filas]
    subidas = sum(b > a for a, b in zip(valores, valores[1:]))
    bajadas = sum(b < a for a, b in zip(valores, valores[1:]))
    iguales = len(valores) - 1 - subidas - bajadas
    primero, ultimo = valores[0], valores[-1]
    if subidas and bajadas:
        conclusion = "No hay una subida o bajada constante: las ventas fluctúan."
    elif subidas and not bajadas:
        conclusion = "Las ventas muestran una tendencia ascendente."
    elif bajadas and not subidas:
        conclusion = "Las ventas muestran una tendencia descendente."
    else:
        conclusion = "Las ventas se mantuvieron sin cambios."

    comparacion = ""
    if primero:
        cambio = (ultimo - primero) / abs(primero) * Decimal("100")
        signo = "+" if cambio > 0 else ""
        comparacion = f" En el período completo, el cambio fue de {signo}{_formatear_numero(cambio)}%."
    lineas = [f"📈 *{conclusion}*", comparacion.strip(), ""]
    lineas.append(f"• Subieron en {subidas} períodos y bajaron en {bajadas}.")
    if iguales:
        lineas.append(f"• Se mantuvieron iguales en {iguales} períodos.")
    lineas.append(
        f"• Primer valor: {_formatear_valor(primero, columnas[i_valor], unidad)}; "
        f"último valor: {_formatear_valor(ultimo, columnas[i_valor], unidad)}."
    )
    return "\n".join(x for x in lineas if x)


def redactar_resultado_exacto(columnas, filas, unidad: str = "", tope: int | None = None,
                              compacto: bool = False) -> str:
    """Presenta las celdas ejecutadas sin pedirle aritmetica ni datos al LLM."""
    if not filas:
        return "No encontré registros para ese filtro."

    if compacto:
        return _resultado_compacto(columnas, filas, unidad)

    presupuesto = _presupuesto_formateado(columnas, filas, unidad)
    if presupuesto:
        return presupuesto

    movimientos = _resultado_movimientos(columnas, filas, unidad, tope)
    if movimientos:
        return movimientos

    serie = _resultado_serie_temporal(columnas, filas, unidad)
    if serie:
        return serie

    if len(filas) == 1:
        partes = [
            f"*{_nombre_legible(col)}:* {_formatear_valor(valor, col, unidad)}"
            for col, valor in zip(columnas, filas[0])
        ]
        return "\n".join(partes)

    # Todas las tablas comparten la misma presentación móvil; no se limita la
    # cantidad aquí porque WhatsApp divide el texto largo en varios mensajes.
    return _resultado_lista(columnas, filas, unidad, tope)


def es_resultado_no_respondible(columnas, filas) -> bool:
    """True cuando el SELECT devuelve el marcador interno de rechazo."""
    return (
        len(filas) == 1
        and len(columnas) == 1
        and str(columnas[0]).strip().lower() == "nota"
        and str(filas[0][0]).strip() == "NO_RESPONDIBLE"
    )


def redactar_respuesta(pregunta: str, columnas, filas, historial=None, sql="",
                       unidad: str = "", temas_habilitados: str = "") -> str:
    """
    Convierte el SELECT en texto sin permitir una segunda interpretacion numerica.

    El LLM ya participo al entender la pregunta y elegir/generar el SQL. Una vez
    que PostgreSQL respondio, volver a pedirle que sume o resuma introduce una
    segunda fuente de verdad: era la causa de que el Excel estuviera correcto y
    el mensaje no. La salida textual ahora usa exactamente las mismas filas que
    el archivo.
    """
    if es_resultado_no_respondible(columnas, filas):
        temas = temas_habilitados or "los datos habilitados para su empresa"
        return (
            "No puedo responder esa consulta con los datos habilitados para "
            f"este chat. Puede consultarme sobre {temas}."
        )
    if filas and _es_pedido_analitico(pregunta):
        try:
            return _redactar_analisis(pregunta, columnas, filas, sql=sql)
        except Exception as e:  # noqa: BLE001
            logger.warning("No se pudo redactar el análisis con IA: %s", e)
            tendencia = _analisis_tendencia_local(columnas, filas, unidad)
            if tendencia:
                return tendencia
    columnas, filas, compacto = proyectar_columnas_solicitadas(
        pregunta, columnas, filas,
    )
    return redactar_resultado_exacto(
        columnas, filas, unidad=unidad, compacto=compacto,
    )


# B-20: bot/responder.py usaba _tabla_texto (privada de este modulo) en su modo
# de emergencia. Ahora la funcion es publica; el alias queda por compatibilidad.
_tabla_texto = tabla_texto
