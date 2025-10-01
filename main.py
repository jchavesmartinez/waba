import os, json
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
import httpx

app = FastAPI()

# === Config ===
VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN", "wabita123")
WABA_TOKEN       = os.getenv("WABA_TOKEN")          # Token de WhatsApp (EAA...)
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")      # TU API KEY de OpenAI (ponla en Render)
GRAPH_VER        = os.getenv("GRAPH_VER", "v22.0")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # o "gpt-4.1-mini"

# --- Health ---
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

# --- Verificación del webhook (GET con hub.challenge) ---
@app.get("/webhook", response_class=PlainTextResponse)
async def verify(
    hub_mode: str | None        = Query(None, alias="hub.mode"),
    hub_challenge: str | None   = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None= Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return hub_challenge or ""
    raise HTTPException(status_code=403, detail="Verification failed")

# --- Recepción de eventos (POST) ---
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("Incoming:", json.dumps(data, indent=2, ensure_ascii=False))

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            phone_number_id = value.get("metadata", {}).get("phone_number_id")

            # mensajes entrantes
            for msg in (value.get("messages") or []):
                if msg.get("type") != "text":
                    continue
                user_text = msg.get("text", {}).get("body", "")
                to_msisdn = msg.get("from")

                # 1) Llamar a OpenAI con el texto del usuario
                reply_text = await ask_chatgpt(user_text)

                # 2) Responder por WhatsApp con la salida de OpenAI
                await send_whatsapp_text(phone_number_id, to_msisdn, reply_text)

    return {"status": "ok"}

# --- OpenAI (Responses API / Chat) ---
async def ask_chatgpt(user_text: str) -> str:
    """
    Llama a OpenAI y devuelve la respuesta de texto del modelo.
    Usa el Responses API / chat (modelos gpt-4o-mini / gpt-4.1-mini, etc.).
    """
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY faltante")
        return "Lo siento, no estoy configurado para responder en este momento."

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "Responde siempre en español, claro y útil."},
            {"role": "user",   "content": user_text}
        ]
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError:
            print("OpenAI error:", r.status_code, r.text)
            return "Hubo un problema al generar la respuesta."

        data = r.json()
        # La Responses API devuelve texto en data["output"][0]["content"][0]["text"] (según docs)
        # Hacemos parse defensivo:
        try:
            parts = data.get("output", [])
            if parts and "content" in parts[0]:
                contents = parts[0]["content"]
                if contents and "text" in contents[0]:
                    return contents[0]["text"].strip()
        except Exception as e:
            print("Parse error:", e)

        # Fallback genérico
        return "No pude entender la respuesta del modelo."

# --- WhatsApp Cloud API ---
async def send_whatsapp_text(phone_number_id: str, to_msisdn: str, text: str):
    if not (phone_number_id and to_msisdn and WABA_TOKEN):
        print("Faltan phone_number_id / to_msisdn / WABA_TOKEN")
        return
    url = f"https://graph.facebook.com/{GRAPH_VER}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {WABA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "text",
        "text": {"body": text[:4000]},  # límite de WhatsApp
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=headers, json=payload)
        print("WhatsApp send:", r.status_code, r.text)
