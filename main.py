import os, json, sqlite3, time
from contextlib import contextmanager
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
import httpx

app = FastAPI()

# === Config ===
VERIFY_TOKEN   = os.getenv("VERIFY_TOKEN", "wabita123")
WABA_TOKEN     = os.getenv("WABA_TOKEN")               # Token de WhatsApp (EAA…)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")           # Tu API key de OpenAI
GRAPH_VER      = os.getenv("GRAPH_VER", "v22.0")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# === Persistencia (SQLite simple) ===
DB_PATH = os.getenv("DB_PATH", "data.db")

@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield con
        con.commit()
    finally:
        con.close()

def init_db():
    with db() as con:
        # historial por usuario (wa_id)
        con.execute("""
        CREATE TABLE IF NOT EXISTS chat_history(
            user_id TEXT,
            role    TEXT,         -- 'user' | 'assistant' | 'system'
            content TEXT,
            ts      INTEGER
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS ix_hist_user_ts ON chat_history(user_id, ts)")
init_db()

def save_msg(user_id: str, role: str, content: str):
    with db() as con:
        con.execute(
            "INSERT INTO chat_history(user_id, role, content, ts) VALUES (?,?,?,?)",
            (user_id, role, content, int(time.time()))
        )

def get_recent_history(user_id: str, max_chars: int = 6000, max_turns: int = 10):
    """
    Devuelve mensajes recientes para ese usuario, recortando para no exceder límites.
    (Aproximación por caracteres; suficiente para la mayoría de casos.)
    """
    with db() as con:
        cur = con.execute(
            "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, max_turns * 2)
        )
        rows = cur.fetchall()

    rows = rows[::-1]  # cronológico
    # recorte simple por caracteres
    total = 0
    kept = []
    for role, content in reversed(rows):
        total += len(content)
        if total > max_chars:
            break
        kept.append({"role": role, "content": content})
    kept.reverse()
    return kept


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
                user_id   = msg.get("from")  # wa_id del usuario

                # 1) Pedir a OpenAI con HISTORIAL de ese usuario
                reply_text = await ask_chatgpt_with_history(user_id, user_text)

                # 2) Responder por WhatsApp con la salida de OpenAI
                await send_whatsapp_text(phone_number_id, user_id, reply_text)

    return {"status": "ok"}


# --- OpenAI (Responses API) con historial ---
async def ask_chatgpt_with_history(user_id: str, new_user_text: str) -> str:
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY faltante")
        return "Lo siento, no estoy configurado para responder en este momento."

    # Recuperar historial reciente de ese usuario
    history = get_recent_history(user_id)

    # Construir mensajes: system + historial + mensaje actual
    messages = [{"role": "system", "content": "Eres un asistente útil. Responde siempre en español, claro y conciso."}]
    messages.extend(history)
    messages.append({"role": "user", "content": new_user_text})

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": OPENAI_MODEL, "input": messages}

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError:
            print("OpenAI error:", r.status_code, r.text)
            # Guarda el turno del usuario aunque falle, por trazabilidad
            save_msg(user_id, "user", new_user_text)
            return "Hubo un problema al generar la respuesta."

        data = r.json()

        # Intentar extraer el texto (formato de Responses API)
        reply = "No pude entender la respuesta del modelo."
        try:
            parts = data.get("output", [])
            if parts and "content" in parts[0]:
                contents = parts[0]["content"]
                if contents and "text" in contents[0]:
                    reply = (contents[0]["text"] or "").strip()
        except Exception as e:
            print("Parse error:", e)

    # Guardar el turno completo (user + assistant)
    save_msg(user_id, "user", new_user_text)
    save_msg(user_id, "assistant", reply)
    return reply


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
