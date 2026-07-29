"""Envio de mensajes por la WhatsApp Cloud API (Graph API)."""

import logging
import httpx

import config

logger = logging.getLogger("fachavi-sql.whatsapp")


def send_text_message(to_number: str, text: str) -> None:
    if not config.PHONE_NUMBER_ID or not config.WHATSAPP_TOKEN:
        logger.error("Faltan PHONE_NUMBER_ID o WHATSAPP_TOKEN.")
        return

    url = f"https://graph.facebook.com/{config.GRAPH_API_VERSION}/{config.PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {config.WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"body": text[:4000]},  # WhatsApp corta mensajes muy largos
    }
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=20.0)
        resp.raise_for_status()
        logger.info("Mensaje enviado a %s", to_number)
    except httpx.HTTPStatusError as e:
        logger.error("Error Graph API (%s): %s", e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        logger.error("Error de red: %s", e)
