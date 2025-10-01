import os, json
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
import httpx

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "wabita123")
WABA_TOKEN = os.getenv("WABA_TOKEN")
GRAPH_VER = "v22.0"


@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"


@app.get("/webhook", response_class=PlainTextResponse)
async def verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return hub_challenge or ""
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("Incoming:", json.dumps(data, indent=2, ensure_ascii=False))

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            phone_number_id = value.get("metadata", {}).get("phone_number_id")
            for msg in (value.get("messages") or []):
                to_msisdn = msg.get("from")
                await send_reply(phone_number_id, to_msisdn, "que dice el perro")

    return {"status": "ok"}


async def send_reply(phone_number_id: str, to_msisdn: str, text: str):
    if not (phone_number_id and to_msisdn and WABA_TOKEN):
        print("Missing phone_number_id / to / WABA_TOKEN")
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
        "text": {"body": text},
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=headers, json=payload)
        print("WhatsApp send:", r.status_code, r.text)
