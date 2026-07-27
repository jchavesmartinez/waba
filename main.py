"""
FACHAVI SQL Bot - Webhook de WhatsApp con motor text-to-SQL sobre Google Sheets.

Flujo de un mensaje:
  Meta -> POST /webhook -> se responde 200 al instante
       -> en segundo plano: recupera contexto -> Claude genera SQL
       -> ejecuta en DuckDB (datos de Sheets) -> Claude redacta -> envia por WhatsApp
"""

import logging

from fastapi import FastAPI, Request, Response, BackgroundTasks

import config
import memory
from whatsapp import send_text_message
from nl2sql import responder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fachavi-sql")

app = FastAPI(title="FACHAVI SQL Bot")
memory.init_db()


@app.get("/")
def health():
    return {"status": "ok", "service": "fachavi-sql-bot"}


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
            return Response(status_code=200)  # update de estado, no un mensaje

        message = messages[0]
        from_number = message["from"]

        if message.get("type") == "text":
            user_text = message["text"]["body"]
            background_tasks.add_task(procesar, from_number, user_text)
        else:
            background_tasks.add_task(
                send_text_message,
                from_number,
                "Por ahora solo entiendo preguntas escritas. Escribime tu consulta sobre las ventas.",
            )
    except (KeyError, IndexError) as e:
        logger.error("Evento no parseable: %s", e)

    return Response(status_code=200)


def procesar(from_number: str, user_text: str):
    """Corre el pipeline completo y responde por WhatsApp."""
    try:
        historial = memory.contexto(from_number)
        respuesta = responder(user_text, historial)
        memory.guardar(from_number, "user", user_text)
        memory.guardar(from_number, "assistant", respuesta)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error en el pipeline: %s", e)
        respuesta = "Uy, tuve un problema procesando tu consulta. Probemos de nuevo."

    send_text_message(from_number, respuesta)
