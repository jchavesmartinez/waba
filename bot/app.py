"""
Webhook de WhatsApp (Twilio) para el bot de datos.

Twilio hace POST application/x-www-form-urlencoded al recibir un mensaje, con
(entre otros) estos campos:
    From = 'whatsapp:+50683919244'   -> numero de quien escribe
    Body = 'cuanto vendimos ayer'    -> texto del mensaje

Se responde con TwiML (<Response><Message>...). La resolucion numero->cliente,
la seleccion de tablas por catalogo y el text-to-SQL viven en bot/responder.py.

Correr local:
    uvicorn bot.app:app --reload --port 8000
Exponer con ngrok/cloudflared y pegar la URL /webhook en la consola de Twilio.
"""

import logging
from xml.sax.saxutils import escape

from fastapi import FastAPI, Form, Response

from bot.responder import responder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="FACHAVI — WhatsApp bot")


@app.get("/salud")
def salud():
    return {"ok": True}


@app.post("/webhook")
def webhook(From: str = Form(default=""), Body: str = Form(default="")):
    numero = From.replace("whatsapp:", "").strip()
    texto = (Body or "").strip()

    if not texto:
        respuesta = "Hola 👋 Mandame tu consulta sobre los datos (ventas, inventario…)."
    else:
        respuesta = responder(numero, texto)

    twiml = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        f"<Response><Message>{escape(respuesta)}</Message></Response>"
    )
    return Response(content=twiml, media_type="application/xml")
