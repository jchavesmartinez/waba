"""
Webhook de WhatsApp (Meta Cloud API / Graph API) para el bot de datos.

A diferencia de Twilio (que mandaba form-urlencoded y esperaba TwiML de vuelta),
la Cloud API funciona en DOS tiempos:

  1. GET  /webhook  -> handshake de verificacion. Meta manda hub.mode,
     hub.verify_token y hub.challenge; si el token coincide, devolvemos el
     challenge en texto plano y Meta da por validado el endpoint.

  2. POST /webhook  -> el mensaje entra como JSON anidado:
         entry[].changes[].value.messages[].text.body   (texto)
         entry[].changes[].value.messages[].from        (numero, solo digitos)
     Respondemos 200 de una y mandamos la respuesta al usuario con una llamada
     SALIENTE aparte (bot/whatsapp.py), en un BackgroundTask para no dejar a
     Meta esperando el text-to-SQL (si tardamos, Meta reintenta el webhook).

La resolucion numero->cliente, la seleccion de tablas por catalogo y el
text-to-SQL viven en bot/responder.py, igual que antes. Aca solo cambia el
transporte.

Correr local:
    uvicorn bot.app:app --reload --port 8000
Exponer con cloudflared/ngrok y en el panel de Meta (WhatsApp > Configuration)
pegar la URL .../webhook y el mismo WHATSAPP_VERIFY_TOKEN.
"""

import hashlib
import hmac
import logging
from collections import OrderedDict

from fastapi import BackgroundTasks, FastAPI, Request, Response

import config
from bot import whatsapp
from bot.responder import responder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fachavi.bot.app")

app = FastAPI(title="FACHAVI — WhatsApp bot (Meta Cloud API)")

_SALUDO = "Hola 👋 Mandame tu consulta sobre los datos (ventas, inventario…)."
_SOLO_TEXTO = "Por ahora solo entiendo mensajes de texto. Escribime tu consulta 🙂"

# Dedup: Meta reintenta el webhook si no ve un 200 a tiempo, y puede repetir un
# mensaje. Guardamos los ultimos ids vistos para no contestar (ni gastar LLM)
# dos veces. En memoria alcanza para un solo proceso; si se escala a varios
# workers habria que mover esto a Redis.
_VISTOS: "OrderedDict[str, None]" = OrderedDict()
_VISTOS_TOPE = 1000


def _ya_procesado(msg_id: str) -> bool:
    if not msg_id:
        return False
    if msg_id in _VISTOS:
        return True
    _VISTOS[msg_id] = None
    if len(_VISTOS) > _VISTOS_TOPE:
        _VISTOS.popitem(last=False)
    return False


def _firma_valida(cuerpo: bytes, firma_header: str) -> bool:
    """
    Valida X-Hub-Signature-256 (HMAC-SHA256 del cuerpo crudo con el app secret).
    Si no hay WHATSAPP_APP_SECRET configurado, no se valida (modo dev).
    """
    if not config.WHATSAPP_APP_SECRET:
        return True
    if not firma_header or not firma_header.startswith("sha256="):
        return False
    esperado = hmac.new(
        config.WHATSAPP_APP_SECRET.encode("utf-8"), cuerpo, hashlib.sha256
    ).hexdigest()
    recibido = firma_header.split("=", 1)[1]
    return hmac.compare_digest(esperado, recibido)


def _atender(numero: str, texto: str) -> None:
    """Corre en background: arma la respuesta y la manda por la Cloud API."""
    try:
        respuesta = responder(numero, texto) if texto else _SALUDO
    except Exception as e:  # noqa: BLE001
        logger.exception("Error generando respuesta para %s: %s", numero, e)
        respuesta = "Tuve un problema procesando tu consulta. Probá de nuevo."
    whatsapp.enviar_texto(numero, respuesta)


@app.get("/salud")
def salud():
    return {"ok": True}


@app.get("/webhook")
def verificar(request: Request):
    """Handshake de verificacion del webhook (Meta lo llama una sola vez)."""
    params = request.query_params
    modo = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")

    if modo == "subscribe" and token == config.WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verificado por Meta.")
        return Response(content=challenge, media_type="text/plain")

    logger.warning("Verificacion de webhook fallida (token no coincide).")
    return Response(content="forbidden", status_code=403)


@app.post("/webhook")
async def webhook(request: Request, tareas: BackgroundTasks):
    cuerpo = await request.body()

    if not _firma_valida(cuerpo, request.headers.get("X-Hub-Signature-256", "")):
        logger.warning("Firma X-Hub-Signature-256 invalida; se descarta el POST.")
        return Response(status_code=403)

    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        logger.warning("Cuerpo del webhook no es JSON valido; se ignora.")
        return Response(status_code=200)  # 200 igual: no queremos reintentos

    # Estructura: object=whatsapp_business_account -> entry[] -> changes[] ->
    # value{ messages[], statuses[], ... }. Los 'statuses' (entregado/leido) y
    # cualquier value sin 'messages' se ignoran en silencio.
    for entry in data.get("entry", []):
        for cambio in entry.get("changes", []):
            valor = cambio.get("value", {})
            for msg in valor.get("messages", []) or []:
                msg_id = msg.get("id", "")
                if _ya_procesado(msg_id):
                    logger.info("Mensaje repetido %s; se omite.", msg_id)
                    continue

                numero = (msg.get("from") or "").strip()
                if not numero:
                    continue

                tipo = msg.get("type")
                if tipo == "text":
                    texto = (msg.get("text", {}).get("body") or "").strip()
                    tareas.add_task(_atender, numero, texto)
                else:
                    # Imagen, audio, ubicacion, etc.: avisamos que solo texto.
                    logger.info("Mensaje tipo '%s' de %s; no es texto.", tipo, numero)
                    tareas.add_task(whatsapp.enviar_texto, numero, _SOLO_TEXTO)

    # Meta solo quiere un 200 rapido; el envio real va por BackgroundTask.
    return Response(status_code=200)
