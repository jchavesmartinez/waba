# filename: app.py
import os, json, sqlite3, time, asyncio
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
        # Mensajes pendientes para agrupar por debounce
        con.execute("""
        CREATE TABLE IF NOT EXISTS pending_msgs(
            user_id   TEXT,
            content   TEXT,
            ts        INTEGER,
            processed INTEGER DEFAULT 0
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS ix_pending_user ON pending_msgs(user_id)")
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
    # Seleccionamos desde el final hacia atrás, respetando max_chars
    for role, content in reversed(rows):
        total += len(content)
        if total > max_chars:
            break
        kept.append({"role": role, "content": content})
    kept.reverse()
    return kept

def enqueue_pending(user_id: str, content: str):
    now = int(time.time())
    with db() as con:
        # Guardamos en historial de inmediato (para contexto del modelo)
        con.execute(
            "INSERT INTO chat_history(user_id, role, content, ts) VALUES (?,?,?,?)",
            (user_id, "user", content, now)
        )
        # Acumulamos como pendiente para la ráfaga
        con.execute(
            "INSERT INTO pending_msgs(user_id, content, ts, processed) VALUES (?,?,?,0)",
            (user_id, content, now)
        )

def fetch_unprocessed(user_id: str):
    with db() as con:
        cur = con.execute(
            "SELECT content, ts FROM pending_msgs WHERE user_id=? AND processed=0 ORDER BY ts ASC",
            (user_id,)
        )
        rows = cur.fetchall()
    return rows

def mark_processed(user_id: str):
    with db() as con:
        con.execute(
            "UPDATE pending_msgs SET processed=1 WHERE user_id=? AND processed=0",
            (user_id,)
        )

SYSTEM_PROMPT = """
Te llamas Sofía Soler, asesora inmobiliaria costarricense, cálida y profesional, de la agencia 506BOX PROPERTY NERDS.
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
- Siempre al iniciar la conversación presentarse diciendo tu nombre ( Sofia ) y también preguntar el nombre del cliente
- Hablar en "usted" y nunca en "tu"
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

# === Debounce infra (30s por usuario) ===
DEBOUNCE_SECONDS = 5
DEBOUNCE_TASKS: dict[str, asyncio.Task] = {}
PHONE_ID_CACHE: dict[str, str] = {}  # último phone_number_id por usuario

async def debounce_fire(user_id: str, phone_number_id: str):
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)
        # Tomamos todos los mensajes aún no procesados
        pending = fetch_unprocessed(user_id)
        if not pending:
            return

        # Construimos el prompt con el historial reciente (ya incluye los mensajes del cliente)
        history = get_recent_history(user_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)

        # También agregamos un bloque explicando al modelo que integre los últimos N mensajes
        # (opcional, ayuda a forzar integración)
        joined = "\n".join(f"- {c}" for c, _ in pending)
        messages.append({
            "role": "user",
            "content": f"Integra y responde en UN solo mensaje considerando estos mensajes recientes:\n{joined}"
        })

        reply = await call_openai(messages)
        if not reply:
            reply = "Hubo un problema al generar la respuesta."

        # Guardamos y enviamos
        save_msg(user_id, "assistant", reply)
        await send_whatsapp_text(phone_number_id, user_id, reply)

        # Marcamos procesados los pendientes
        mark_processed(user_id)

    finally:
        # Limpiamos el task del usuario si sigue siendo el mismo
        t = DEBOUNCE_TASKS.get(user_id)
        if t and t.done():
            DEBOUNCE_TASKS.pop(user_id, None)

async def schedule_debounce(user_id: str, phone_number_id: str):
    # Cacheamos el phone_number_id (por si no lo mandan igual después)
    PHONE_ID_CACHE[user_id] = phone_number_id
    # Si ya había un temporizador, lo cancelamos y recreamos
    old = DEBOUNCE_TASKS.get(user_id)
    if old and not old.done():
        old.cancel()
    task = asyncio.create_task(debounce_fire(user_id, PHONE_ID_CACHE[user_id]))
    DEBOUNCE_TASKS[user_id] = task

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

                if not user_text or not user_id:
                    continue

                # Encolamos el mensaje y reprogramamos el temporizador de 30s
                enqueue_pending(user_id, user_text)
                await schedule_debounce(user_id, phone_number_id)

    return {"status": "ok"}

async def call_openai(messages: list[dict]) -> str:
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY faltante")
        return ""
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": OPENAI_MODEL, "input": messages}

    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError:
            print("OpenAI error:", r.status_code, r.text)
            return ""
        data = r.json()
        try:
            parts = data.get("output", [])
            if parts and "content" in parts[0]:
                contents = parts[0]["content"]
                if contents and "text" in contents[0]:
                    return (contents[0]["text"] or "").strip()
        except Exception as e:
            print("Parse error:", e)
            return ""
    return ""

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
