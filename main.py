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
        con.execute("""
        CREATE TABLE IF NOT EXISTS chat_history(
            user_id TEXT,
            role    TEXT,
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
    with db() as con:
        cur = con.execute(
            "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, max_turns * 2)
        )
        rows = cur.fetchall()

    rows = rows[::-1]
    total = 0
    kept = []
    for role, content in reversed(rows):
        total += len(content)
        if total > max_chars:
            break
        kept.append({"role": role, "content": content})
    kept.reverse()
    return kept



SYSTEM_PROMPT = """
Te llamas **Sofía Soler**, asesora inmobiliaria costarricense, cálida y profesional, de la agencia 506BOX PROPERTY NERDS.
Objetivo: calificar al cliente y llevarlo a un siguiente paso claro (agendar visita o pasar a un asesor humano).

Estilo:
- Tono cercano, respetuoso, con chispa y profesionalismo latino.
- Frases cortas, claras; evita tecnicismos innecesarios.
- Siempre respondes en español de Costa Rica.

Flujo conversacional (estricto, en este orden, UNA pregunta de calificación a la vez):
1) Saludo breve + UNA pregunta de calificación.
2) Identificar: intención (compra/alquiler); zona preferida; zonas cercanas aceptables; tipo de propiedad; presupuesto y moneda;
   habitaciones/baños; metros aproximados; forma de pago/financiamiento (si compra); ventana de visita; mascotas; parqueos.
3) Si la zona está en GAM (Escazú, Santa Ana, Rohrmoser, Sabana, Heredia, Curridabat, etc.), sugiere sutilmente alternativas cercanas.
4) Propón el siguiente paso (agendar visita o derivar a un asesor humano), confirmando día/horario preferido.

Reglas:
- Máximo 1–2 preguntas por mensaje.
- No des asesoría legal/tributaria/financiera; sugiere consultar a un profesional cuando corresponda.
- No prometas precios, tasas ni disponibilidad; usa lenguaje condicional (“podemos explorar”, “podría estar disponible”).
- Mantén empatía ante objeciones; ofrece opciones.
- Si falta información clave, prioriza preguntarla antes de enviar listados.
- Si el cliente pide contacto humano, ofrece pasar con un asesor y pide ventana horaria y medio de contacto.

"""



@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK"

@app.get("/webhook", response_class=PlainTextResponse)
async def verify(
    hub_mode: str | None        = Query(None, alias="hub.mode"),
    hub_challenge: str | None   = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None= Query(None, alias="hub.verify_token"),
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
                if msg.get("type") != "text":
                    continue

                user_text = msg.get("text", {}).get("body", "")
                user_id   = msg.get("from")  

                reply_text = await ask_chatgpt_with_history(user_id, user_text)
                await send_whatsapp_text(phone_number_id, user_id, reply_text)

    return {"status": "ok"}

async def ask_chatgpt_with_history(user_id: str, new_user_text: str) -> str:
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY faltante")
        return "Lo siento, no estoy configurado para responder en este momento."

    history = get_recent_history(user_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
            save_msg(user_id, "user", new_user_text)
            return "Hubo un problema al generar la respuesta."

        data = r.json()
        reply = "No pude entender la respuesta del modelo."
        try:
            parts = data.get("output", [])
            if parts and "content" in parts[0]:
                contents = parts[0]["content"]
                if contents and "text" in contents[0]:
                    reply = (contents[0]["text"] or "").strip()
        except Exception as e:
            print("Parse error:", e)

    save_msg(user_id, "user", new_user_text)
    save_msg(user_id, "assistant", reply)
    return reply

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
        "text": {"body": text[:4000]},
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=headers, json=payload)
        print("WhatsApp send:", r.status_code, r.text)
