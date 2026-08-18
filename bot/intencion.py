"""
Clasificador de intencion del bot.

Corre ANTES del text-to-SQL y decide que hacer con el mensaje:

  - "datos"  : pide informacion que sale de la base -> sigue al text-to-SQL.
  - "meta"   : es sobre la propia conversacion o el bot ("por que dijiste X",
               "que te pregunte", "cuales productos me mencionaste", "que podes
               hacer") -> se responde con el HISTORIAL, sin tocar la base.
  - "saludo" : saludo o cortesia sin pedido ("hola", "gracias") -> respuesta fija.

Asi dejamos de mandar TODO al generador de SQL. Antes, una pregunta sobre la
charla terminaba en "no puedo responder con los datos habilitados" (el modelo no
podia armar un SELECT para algo que no era de datos), o peor, inventaba un query
con literales sacados del historial. Ahora eso se responde donde corresponde.

Regla de oro: ante la duda entre 'datos' y 'meta', se elige 'datos'. Es
preferible intentar una consulta de mas que tratar una pregunta real de datos
como si fuera charla.
"""

import logging
import re
import unicodedata

import config
import llm

logger = logging.getLogger("fachavi.bot.intencion")

# Atajo sin LLM para saludos/cortesias obvias: evita gastar una llamada.
_SALUDOS = {
    "hola", "holaa", "buenas", "buenos dias", "buenos días", "buenas tardes",
    "buenas noches", "hey", "ey", "que tal", "qué tal", "gracias", "muchas gracias",
    "ok", "oka", "dale", "listo", "perfecto", "chao", "adios", "adiós",
}

_ETIQUETAS = {"datos", "meta", "saludo"}

_SISTEMA_CLASIF = (
    "Clasificas el ULTIMO mensaje del usuario de un bot de datos por WhatsApp en "
    "UNA de tres categorias. Respondes SOLO la etiqueta, en minuscula, sin nada mas.\n"
    "- datos: pide informacion que sale de la base de datos del negocio (ventas, "
    "inventario, productos, montos, cantidades, proveedores, comparaciones, "
    "totales, calculos, tendencias, crecimiento, runway, ticket, promedios, "
    "desglose, top, ranking). SIEMPRE es 'datos' si el mensaje pide CALCULAR, "
    "CONSULTAR, MOSTRAR, DAME, CUANTO, CUANTOS, CUANTAS o cualquier operacion "
    "sobre los datos, AUNQUE sea un seguimiento de la conversacion.\n"
    "- meta: es SOLO sobre la CONVERSACION o el BOT en si, SIN pedir datos nuevos "
    "ni calculos. Ej: que dijiste antes, por que lo dijiste, que te pregunte, que "
    "me mencionaste, resumime lo que hablamos, como lo calculaste, que podes hacer, "
    "no entendi, repetilo. TAMBIEN va aca una pregunta EXCLUSIVAMENTE sobre la "
    "FECHA U HORA ACTUAL ('que dia es hoy', 'en que año estamos', 'que hora es'). "
    "En cambio, 'ventas de hoy', 'gastos de ayer' o cualquier dato filtrado por "
    "fecha SIEMPRE es datos.\n"
    "- saludo: saludo o cortesia sin ningun pedido (hola, buenas, gracias, ok). "
    "Solo esto: si el mensaje pregunta ALGO, no es saludo.\n"
    "Ante la duda entre 'datos' y 'meta', SIEMPRE 'datos'. Solo usa 'meta' cuando "
    "el usuario NO pide ningun dato nuevo ni calculo."
)

_SISTEMA_META = (
    "Es el asistente de datos de una empresa y responde por WhatsApp en español "
    "profesional, claro, cordial y breve. Trate al usuario de usted. No use jerga "
    "ni localismos. No imite el tono informal de mensajes anteriores. El usuario "
    "hizo una pregunta "
    "sobre la CONVERSACION o sobre el bot, no sobre la base de datos. Responda "
    "USANDO SOLO el historial proporcionado.\n"
    "\n"
    "REGLA SOBRE LOS DATOS EN EL HISTORIAL:\n"
    "- Los numeros y resultados que el Asistente dio antes son solamente "
    "afirmaciones del historial: pueden contener un error de interpretacion. No "
    "los declares verdaderos si el usuario los cuestiona.\n"
    "- Si preguntan COMO se calculo algo, expliquelo con la "
    "logica de negocio (ej: 'dividi el stock entre el promedio diario de ventas "
    "de los ultimos 30 dias'). Puede citar los numeros ya mencionados. No diga que "
    "no los tiene ni que fueron inventados.\n"
    "- No puede CREAR numeros nuevos que no esten en el historial. "
    "Si solicitan un dato que ni el asistente ni el usuario mencionaron antes "
    "(por ejemplo: piden "
    "quincenal y solo hay semanal), NO recalcule ni interpole: indique que ese dato "
    "no se ha consultado y que debe solicitarse directamente.\n"
    "- Resumen: citar el historial como historial = OK. Defender una cifra no "
    "verificada o inventar datos nuevos = PROHIBIDO.\n"
    "\n"
    "- Si preguntan qué puede hacer, explique brevemente que responde consultas sobre los "
    "datos de ventas/inventario que tengan habilitados, con un par de ejemplos. "
    "Mencione que además puede enviar el resultado como GRAFICO, EXCEL o PDF.\n"
    "- NUNCA diga que no puede generar archivos ni que los copien a mano: si el "
    "usuario pide un Excel, un grafico o un PDF, lo unico que tiene que hacer es "
    "pedirlo en este chat y se genera en el momento.\n"
    "- No dibuje graficos en texto (barras con caracteres, ejes con | y _). En el "
    "celular quedan ilegibles. Ofrezca el grafico como imagen.\n"
    "- No prometa 'revisarlo despues': no tiene acciones diferidas."
)


def _normalizar(texto: str) -> str:
    normal = unicodedata.normalize("NFKD", str(texto or ""))
    return normal.encode("ascii", "ignore").decode("ascii").lower()


def _hay_sql_previo(historial) -> bool:
    return any(
        turno.get("rol") == "assistant" and str(turno.get("sql", "")).strip()
        for turno in historial or []
    )


def _pregunta_solo_reloj(texto: str) -> bool:
    return bool(re.fullmatch(
        r"\s*(?:que\s+)?(?:dia|fecha|hora|ano)\s+(?:es\s+)?(?:hoy|actual)\s*",
        texto,
    )) or texto.strip() in {"en que ano estamos", "que hora es"}


def _referencia_temporal(texto: str) -> bool:
    meses = (
        "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
        "octubre|noviembre|diciembre"
    )
    return bool(re.search(
        rf"\b(?:hoy|ayer|anteayer|este\s+mes|esta\s+semana|este\s+ano|"
        rf"ultimos?\s+\d+\s+dias?|{meses}|20\d{{2}})\b|\b\d{{1,2}}[/.-]\d{{1,2}}",
        texto,
    ))


def _menciona_datos(texto: str) -> bool:
    return bool(re.search(
        r"\b(?:ventas?|vendid[oa]s?|facturacion|ingresos?|gastos?|presupuesto|"
        r"inventario|stock|productos?|transacciones?|montos?|saldos?|deudas?|"
        r"clientes?|proveedores?|categorias?|conceptos?|utilidad)\b",
        texto,
    ))

# Verbos de accion que SIEMPRE son 'datos', aunque el contexto parezca meta.
# Atajo sin LLM: evita que "calcula el crecimiento quincenal" vaya a meta.
_VERBOS_DATOS = (
    "calcula", "calculá", "calcular", "consulta", "consultá", "muestra",
    "mostrá", "mostrame", "dame", "dime", "decime", "cuanto", "cuánto",
    "cuantos", "cuántos", "cuantas", "cuántas", "cuales", "cuáles",
    "compara", "compará", "desglosá", "desglose", "lista", "listá",
    # Pedidos de archivo: "graficame las ventas" o "exportá el inventario" son
    # consultas de datos con otro envoltorio. Sin esto el clasificador los podia
    # mandar a 'meta' (suenan a pedido sobre el bot) y el usuario recibia una
    # explicacion en vez de su grafico.
    "grafica", "graficá", "graficame", "graficar", "grafique", "grafiquem",
    "exporta", "exportá", "exportar", "pasame", "mandame", "mandá", "enviame",
    "envia", "enviá", "genera", "generá", "generar", "descarga", "descargá",
)


def _historial_texto(historial, tope=8) -> str:
    if not historial:
        return "(sin conversacion previa)"
    etq = {"user": "Usuario", "assistant": "Asistente"}
    ult = historial[-tope:]
    return "\n".join(f"{etq.get(t['rol'], t['rol'])}: {t['contenido']}" for t in ult)


def clasificar(pregunta: str, historial=None) -> str:
    """Devuelve 'datos' | 'meta' | 'saludo'. Ante error o duda, 'datos'."""
    if not config.BOT_INTENCION:
        return "datos"

    limpia = re.sub(r"[?!.\s]+$", "", _normalizar(pregunta).strip())
    if limpia in _SALUDOS:
        return "saludo"

    # Preguntar por el reloj/calendario es conversacion; filtrar datos por una
    # fecha es una consulta. Esta distincion no se deja al modelo porque "hoy"
    # aparecia en ambos casos y enviaba ventas reales al camino sin warehouse.
    if _pregunta_solo_reloj(limpia):
        return "meta"
    if _menciona_datos(limpia):
        return "datos"

    # Una objecion a una cifra/categoria debe VOLVER A LOS DATOS. Antes iba a
    # 'meta', que por diseño no consulta el warehouse: el bot intentaba explicar
    # su propio error usando el mismo texto equivocado y podia hasta negar el
    # mensaje inmediatamente anterior.
    reclamos_dato = (
        "calculo esta mal", "numero esta mal", "te equivocaste",
        "esta equivocado", "por que agregaste", "por que pusiste",
        "verifica el dato", "revisa el dato", "esa cifra", "ese monto",
        "no es la total", "no es el total", "estas seguro",
    )
    if any(frase in limpia for frase in reclamos_dato):
        return "datos"

    # Respuestas cortas a una pregunta del bot ("este mes", "en total") y
    # referencias al ultimo resultado ("esas", "el detalle") son continuacion
    # de una consulta, no charla sobre el bot. Mandarlas a 'meta' pierde el hilo
    # justo cuando el usuario esta dando el parametro que se le pidio.
    palabras = limpia.split()
    if historial and _hay_sql_previo(historial):
        if _referencia_temporal(limpia):
            return "datos"
        # Correcciones minimas como "3?" se refieren al resultado numerico
        # anterior. En meta Gemini intentaba completar la cifra de memoria.
        if re.fullmatch(r"[\d.,]+", limpia):
            return "datos"

    if historial and len(palabras) <= 8:
        ultimo_asistente = next(
            (str(t.get("contenido", "")) for t in reversed(historial)
             if t.get("rol") == "assistant"), "")
        referencias = ("esas", "esos", "estas", "estos", "detalle",
                       "en total", "este mes", "ultimos 30")
        if ultimo_asistente and (
                ultimo_asistente.rstrip().endswith("?")
                or any(r in limpia for r in referencias)):
            return "datos"

    # Atajo: si el mensaje empieza con un verbo de accion sobre datos, es
    # SIEMPRE 'datos' — sin gastar una llamada al LLM. Esto evita que "calcula
    # el crecimiento quincenal" se clasifique como 'meta' por contexto.
    if palabras and any(palabras[0].startswith(v) for v in _VERBOS_DATOS):
        return "datos"

    try:
        contenido = (
            f"Conversacion reciente:\n{_historial_texto(historial)}\n\n"
            f"Ultimo mensaje del usuario:\n{pregunta}\n\n"
            "Etiqueta (datos/meta/saludo):"
        )
        resp = llm.generar_texto(
            config.BOT_MODELO_INTENCION,
            contenido,
            # Aunque la salida sea una sola etiqueta, Gemini contabiliza su
            # presupuesto de salida completo. Un margen pequeno evita cortar
            # la palabra "saludo" o "datos".
            max_tokens=20,
            system=_SISTEMA_CLASIF,
            thinking_level="minimal",
        )
        txt = resp.texto
        etiqueta = txt.strip().lower().split()[0] if txt.strip() else "datos"
        etiqueta = re.sub(r"[^a-z]", "", etiqueta)
        if etiqueta not in _ETIQUETAS:
            logger.info("clasificador devolvio %r; se asume 'datos'", txt[:40])
            return "datos"
        return etiqueta
    except Exception as e:  # noqa: BLE001
        logger.warning("clasificador fallo (%s); se asume 'datos'", e)
        return "datos"


def responder_conversacional(pregunta: str, historial=None) -> str:
    """Responde una pregunta 'meta' usando SOLO el historial. Sin tocar la base."""
    from bot.nl2sql import (_historial_a_messages, contexto_temporal,
                            limpiar_arte_ascii)
    messages = _historial_a_messages(historial)
    messages.append({"role": "user", "content": pregunta})
    sistema = _SISTEMA_META + "\n\n" + contexto_temporal()
    resp = llm.generar_texto(
        config.BOT_MODELO_RESPUESTA,
        messages,
        max_tokens=1000,
        # Una respuesta conversacional breve no necesita razonamiento extenso.
        # "low" consumia el presupuesto antes de terminar el texto visible.
        thinking_level="minimal",
        system=sistema,
    )
    if resp.truncada:
        logger.warning(
            "respuesta meta truncada (%s); se reintenta con mayor presupuesto",
            getattr(resp, "finish_reason", "MAX_TOKENS"),
        )
        resp = llm.generar_texto(
            config.BOT_MODELO_RESPUESTA,
            messages,
            max_tokens=3000,
            thinking_level="minimal",
            system=(
                sistema
                + "\n\nEntregue una respuesta completa. No termine a mitad de "
                "una palabra, cifra u oracion."
            ),
        )
    if resp.truncada:
        logger.error(
            "respuesta meta truncada tambien en el reintento (%s)",
            getattr(resp, "finish_reason", "MAX_TOKENS"),
        )
        return (
            "No pude completar la respuesta. Por favor, formule nuevamente "
            "la consulta."
        )
    texto = resp.texto.strip()
    return limpiar_arte_ascii(texto)[0]
