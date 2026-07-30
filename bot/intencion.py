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
    "totales, calculos, tendencias).\n"
    "- meta: es sobre la CONVERSACION o el BOT, no sobre la base. Ej: que dijiste "
    "antes, por que, que te pregunte, que me mencionaste, resumime lo que hablamos, "
    "que podes hacer, no entendi, repetilo.\n"
    "- saludo: saludo o cortesia sin ningun pedido (hola, buenas, gracias, ok).\n"
    "Ante la duda entre 'datos' y 'meta', responde 'datos'."
)

_SISTEMA_META = (
    "Sos el asistente de datos de una empresa, respondiendo por WhatsApp en español "
    "(tico, natural, breve). El usuario hizo una pregunta sobre la CONVERSACION o "
    "sobre vos, no sobre la base de datos. Responde USANDO SOLO el historial que se "
    "te da.\n"
    "- No inventes datos, cifras ni nombres que no esten en el historial.\n"
    "- Si te piden un dato que no aparece en la conversacion, deciles que lo "
    "pregunten directamente y lo consultas en el momento. No prometas 'revisarlo "
    "despues': no tenes acciones diferidas.\n"
    "- Si preguntan que podes hacer, explica breve que respondes consultas sobre los "
    "datos de ventas/inventario que tengan habilitados, con un par de ejemplos."
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
    from bot.nl2sql import _historial_a_messages  # reusa el saneo de turnos
    messages = _historial_a_messages(historial)
    messages.append({"role": "user", "content": pregunta})
    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_RESPUESTA,
        max_tokens=400,
        system=_SISTEMA_META,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
