"""
FACHAVI SQL Bot (multi-cliente) - Webhook de WhatsApp.

Un solo numero de WhatsApp atiende a varios clientes. Segun QUIEN escribe
(su numero), el bot resuelve a que cliente pertenece y consulta SU Sheet de datos.
Numeros no registrados reciben un mensaje pidiendo contactar a FACHAVI.

Flujo:
  Meta -> POST /webhook (200 al instante)
       -> en segundo plano:
            1. resolver(numero) -> cliente + spreadsheet_id  (o no autorizado)
            2. recuperar contexto de conversacion
            3. Claude genera SQL -> ejecuta en el Sheet del cliente -> Claude redacta
            4. enviar por WhatsApp
"""

import logging

from fastapi import FastAPI, Request, Response, BackgroundTasks

import config
import memory
from registry import resolver
from whatsapp import send_text_message
from nl2sql import responder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fachavi")

app = FastAPI(title="FACHAVI SQL Bot (multi-cliente)")
memory.init_db()


@app.get("/")
def health():
    return {"status": "ok", "service": "fachavi-sql-multi"}


@app.get("/webhook")
def verify_webhook(request: Request):
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == config.VERIFY_TOKEN
    ):
        return Response(content=params.get("hub.challenge"), media_type="text/plain")
    return Response(content="Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    try:
        value = data["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if not messages:
            return Response(status_code=200)

        message = messages[0]
        from_number = message["from"]

        if message.get("type") == "text":
            user_text = message["text"]["body"]
            background_tasks.add_task(procesar, from_number, user_text)
        else:
            background_tasks.add_task(
                send_text_message,
                from_number,
                "Por ahora solo entiendo preguntas escritas. Escribime tu consulta.",
            )
    except (KeyError, IndexError) as e:
        logger.error("Evento no parseable: %s", e)

    return Response(status_code=200)


def procesar(from_number: str, user_text: str):
    """Resuelve el cliente y corre el pipeline; responde por WhatsApp."""
    try:
        cliente = resolver(from_number)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error resolviendo cliente: %s", e)
        send_text_message(from_number, "Tuve un problema tecnico. Proba de nuevo en un ratito.")
        return

    # Numero no registrado -> mensaje para contactar a FACHAVI
    if cliente is None:
        send_text_message(from_number, config.MSG_NO_AUTORIZADO)
        return

    # La memoria se separa por (cliente, usuario) para no mezclar contextos.
    mem_key = f"{cliente['cliente_id']}:{from_number}"

    try:
        historial = memory.contexto(mem_key)
        respuesta = responder(user_text, cliente, historial)
        memory.guardar(mem_key, "user", user_text)
        memory.guardar(mem_key, "assistant", respuesta)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error en el pipeline: %s", e)
        respuesta = "Uy, tuve un problema procesando tu consulta. Probemos de nuevo."

    send_text_message(from_number, respuesta)
