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

import config

logger = logging.getLogger("fachavi.bot.intencion")

_cliente = None


def _anthropic():
    global _cliente
    if _cliente is None:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("Falta ANTHROPIC_API_KEY para el bot.")
        _cliente = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _cliente


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
    "no entendi, repetilo. TAMBIEN va aca cualquier pregunta sobre la FECHA U HORA "
    "ACTUAL ('que dia es hoy', 'en que año estamos', 'que hora es'): no sale de la "
    "base del negocio, pero es una pregunta real y se responde.\n"
    "- saludo: saludo o cortesia sin ningun pedido (hola, buenas, gracias, ok). "
    "Solo esto: si el mensaje pregunta ALGO, no es saludo.\n"
    "Ante la duda entre 'datos' y 'meta', SIEMPRE 'datos'. Solo usa 'meta' cuando "
    "el usuario NO pide ningun dato nuevo ni calculo."
)

_SISTEMA_META = (
    "Sos el asistente de datos de una empresa, respondiendo por WhatsApp en español "
    "(tico, natural, breve). El usuario hizo una pregunta sobre la CONVERSACION o "
    "sobre vos, no sobre la base de datos. Responde USANDO SOLO el historial que se "
    "te da.\n"
    "\n"
    "REGLA SOBRE LOS DATOS EN EL HISTORIAL:\n"
    "- Los numeros y resultados que VOS (Asistente) diste antes son solamente "
    "afirmaciones del historial: pueden contener un error de interpretacion. No "
    "los declares verdaderos si el usuario los cuestiona.\n"
    "- Si te preguntan COMO se calculo algo que vos diste, explicalo con la "
    "logica de negocio (ej: 'dividi el stock entre el promedio diario de ventas "
    "de los ultimos 30 dias'). Podes citar los numeros que diste. No digas que "
    "no los tenes ni que fueron inventados.\n"
    "- Lo que NO podes hacer es CREAR numeros nuevos que no esten en el historial. "
    "Si te piden un dato que ni vos ni el usuario mencionaron antes (ej: piden "
    "quincenal y solo hay semanal), NO recalcules ni interpoles: decí que ese dato "
    "no se ha consultado y que lo pregunten directamente.\n"
    "- Resumen: citar el historial como historial = OK. Defender una cifra no "
    "verificada o inventar datos nuevos = PROHIBIDO.\n"
    "\n"
    "- Si preguntan que podes hacer, explica breve que respondes consultas sobre los "
    "datos de ventas/inventario que tengan habilitados, con un par de ejemplos. "
    "Mencionales que ademas podes mandar el resultado como GRAFICO, EXCEL o PDF "
    "si lo piden ('graficame las ventas por mes', 'pasamelo en Excel').\n"
    "- NUNCA digas que no podes generar archivos ni que los copien a mano: si el "
    "usuario pide un Excel, un grafico o un PDF, lo unico que tiene que hacer es "
    "pedirlo en este chat y se genera en el momento.\n"
    "- No dibujes graficos en texto (barras con caracteres, ejes con | y _). En el "
    "celular quedan ilegibles. Ofrecé el grafico como imagen en su lugar.\n"
    "- No prometas 'revisarlo despues': no tenes acciones diferidas."
)

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

    limpia = re.sub(r"[¿?¡!.,\s]+$", "", (pregunta or "").strip().lower())
    if limpia in _SALUDOS:
        return "saludo"

    # Una objecion a una cifra/categoria debe VOLVER A LOS DATOS. Antes iba a
    # 'meta', que por diseño no consulta el warehouse: el bot intentaba explicar
    # su propio error usando el mismo texto equivocado y podia hasta negar el
    # mensaje inmediatamente anterior.
    reclamos_dato = (
        "calculo esta mal", "calculo está mal", "numero esta mal",
        "número está mal", "te equivocaste", "esta equivocado",
        "está equivocado", "por que agregaste", "por qué agregaste",
        "por que pusiste", "por qué pusiste", "verifica el dato",
        "verificá el dato", "revisa el dato", "revisá el dato",
    )
    if any(frase in limpia for frase in reclamos_dato):
        return "datos"

    # Respuestas cortas a una pregunta del bot ("este mes", "en total") y
    # referencias al ultimo resultado ("esas", "el detalle") son continuacion
    # de una consulta, no charla sobre el bot. Mandarlas a 'meta' pierde el hilo
    # justo cuando el usuario esta dando el parametro que se le pidio.
    palabras = limpia.split()
    if historial and len(palabras) <= 8:
        ultimo_asistente = next(
            (str(t.get("contenido", "")) for t in reversed(historial)
             if t.get("rol") == "assistant"), "")
        referencias = ("esas", "esos", "estas", "estos", "detalle",
                       "en total", "este mes", "ultimos 30", "últimos 30")
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
        resp = _anthropic().messages.create(
            model=config.BOT_MODELO_INTENCION,
            max_tokens=5,
            system=_SISTEMA_CLASIF,
            messages=[{"role": "user", "content": contenido}],
        )
        txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
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
    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_RESPUESTA,
        max_tokens=400,
        # La fecha va en el system para que "¿qué día es hoy?" tenga respuesta.
        # Antes esa pregunta caia en 'saludo' y devolvia el mensaje de bienvenida.
        system=_SISTEMA_META + "\n\n" + contexto_temporal(),
        messages=messages,
    )
    texto = "".join(b.text for b in resp.content
                    if getattr(b, "type", "") == "text").strip()
    return limpiar_arte_ascii(texto)[0]
