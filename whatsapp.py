"""
Cliente SALIENTE de la WhatsApp Cloud API (Meta / Graph API).

Con Twilio la respuesta viajaba en el mismo HTTP response (TwiML). Con Meta el
webhook solo confirma recepcion (200) y la respuesta al usuario se manda con
una llamada aparte a:

    POST https://graph.facebook.com/{version}/{phone_number_id}/messages
    Authorization: Bearer {WHATSAPP_TOKEN}

Esta llamada solo es libre (texto suelto, sin plantilla) DENTRO de la ventana de
24 h desde el ultimo mensaje del usuario. Como aca siempre respondemos a un
mensaje entrante, estamos dentro de esa ventana y un mensaje de tipo 'text'
alcanza.
"""

import logging

import httpx

import config

logger = logging.getLogger("fachavi.bot.whatsapp")

_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def _url() -> str:
    return (
        f"https://graph.facebook.com/{config.GRAPH_API_VERSION}"
        f"/{config.WHATSAPP_PHONE_NUMBER_ID}/messages"
    )


def _recortar(texto: str) -> str:
    """WhatsApp corta a 4096 chars; recortamos antes para no perder el aviso."""
    tope = config.WHATSAPP_MAX_CHARS
    if len(texto) <= tope:
        return texto
    corte = tope - 1
    return texto[:corte].rstrip() + "…"


def enviar_texto(numero_destino: str, texto: str) -> bool:
    """
    Manda un mensaje de texto por la Cloud API. Devuelve True si Meta lo acepto.

    No relanza excepciones: un fallo de envio se loguea y se traga, porque esto
    corre en un BackgroundTask y ya le devolvimos 200 a Meta. Si tirara, el
    error quedaria huerfano sin cambiar nada del lado del webhook.
    """
    if not (config.WHATSAPP_TOKEN and config.WHATSAPP_PHONE_NUMBER_ID):
        logger.error("Faltan WHATSAPP_TOKEN o WHATSAPP_PHONE_NUMBER_ID; no se envia.")
        return False

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": numero_destino,
        "type": "text",
        "text": {"preview_url": False, "body": _recortar(texto)},
    }
    headers = {
        "Authorization": f"Bearer {config.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        r = httpx.post(_url(), json=payload, headers=headers, timeout=_TIMEOUT)
    except httpx.HTTPError as e:
        logger.exception("Error de red enviando a WhatsApp (%s): %s", numero_destino, e)
        return False

    if r.status_code >= 400:
        # El cuerpo de error de Meta trae el motivo (token vencido, numero fuera
        # de la ventana de 24 h, phone_number_id malo, etc.). Va al log completo.
        logger.error(
            "Meta rechazo el envio a %s [%s]: %s",
            numero_destino, r.status_code, r.text[:500],
        )
        return False

    logger.info("Enviado a %s (%s)", numero_destino, r.status_code)
    return True
