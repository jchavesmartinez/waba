# filename: app.py
import os, json, sqlite3, time, asyncio, traceback, base64
from contextlib import contextmanager
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
import httpx

app = FastAPI()

# === Config ===
VERIFY_TOKEN   = os.getenv("VERIFY_TOKEN", "wabita123")
WABA_TOKEN     = os.getenv("WABA_TOKEN")                 # Token de WhatsApp (EAA…)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")             # Tu API key de OpenAI
GRAPH_VER      = os.getenv("GRAPH_VER", "v22.0")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # modelo chat/visión
ASR_MODEL      = os.getenv("ASR_MODEL", "whisper-1")       # o "gpt-4o-transcribe"

# Tamaños máximos de seguridad
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10MB
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))  # 25MB

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
    """Devuelve en orden cronológico, limitado por turns y por chars."""
    with db() as con:
        cur = con.execute(
            "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY ts ASC",
            (user_id,)
        )
        rows = cur.fetchall()

    rows = rows[-(max_turns*2):]

    total = 0
    kept = []
    for role, content in reversed(rows):
        if total + len(content) > max_chars and kept:
            break
        kept.append({"role": role, "content": content})
        total += len(content)
    kept.reverse()
    return kept

def enqueue_pending(user_id: str, content: str):
    now = int(time.time())
    with db() as con:
        con.execute(
            "INSERT INTO chat_history(user_id, role, content, ts) VALUES (?,?,?,?)",
            (user_id, "user", content, now)
        )
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
        return cur.fetchall()

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
3) Si la zona está en GAM (Escazú, Santa Ana, Rohrmoser, Sabana, Heredia, Curridabat, etc.), sugiera sutilmente alternativas cercanas.
4) Proponga el siguiente paso (agendar visita o derivar a un asesor humano), confirmando día/horario preferido.

Reglas:
- Máximo 1–2 preguntas por mensaje.
- No dé asesoría legal/tributaria/financiera; sugiera consultar a un profesional cuando corresponda.
- No prometa precios, tasas ni disponibilidad; use lenguaje condicional (“podemos explorar”, “podría estar disponible”).
- Mantenga empatía ante objeciones; ofrezca opciones.
- Si falta información clave, priorice preguntarla antes de enviar listados.
- Si el cliente pide contacto humano, ofrezca pasar con un asesor y pida ventana horaria y medio de contacto.
- Siempre al iniciar la conversación preséntese (Sofía) y pregunte el nombre del cliente.
- Hable en "usted" y nunca en "tú".
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
DEBOUNCE_SECONDS = 30
DEBOUNCE_TASKS: dict[str, asyncio.Task] = {}
PHONE_ID_CACHE: dict[str, str] = {}  # último phone_number_id por usuario

async def debounce_fire(user_id: str):
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)

        pending = fetch_unprocessed(user_id)
        if not pending:
            print(f"[debounce] No pending for {user_id}")
            return

        history = get_recent_history(user_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)

        joined = "\n".join(f"- {c}" for c, _ in pending)
        messages.append({
            "role": "user",
            "content": f"Integre y responda en UN solo mensaje considerando estos mensajes recientes:\n{joined}"
        })

        reply = await call_openai_chat(messages)
        if not reply:
            reply = "Disculpe, tuve un inconveniente técnico al contestar. ¿Podría confirmarme su intención (compra o alquiler) y la zona de interés?"

        save_msg(user_id, "assistant", reply)

        phone_number_id = PHONE_ID_CACHE.get(user_id)
        if not phone_number_id:
            print(f"[send] phone_number_id ausente para {user_id}. No se envía.")
        else:
            await send_whatsapp_text(phone_number_id, user_id, reply)

        mark_processed(user_id)

    except asyncio.CancelledError:
        print(f"[debounce] Cancelled for {user_id} (reinicio del timer)")
        raise
    except Exception:
        print("[debounce] Unexpected error:\n", traceback.format_exc())
    finally:
        t = DEBOUNCE_TASKS.get(user_id)
        if t and t.done():
            DEBOUNCE_TASKS.pop(user_id, None)

async def schedule_debounce(user_id: str, phone_number_id: str | None):
    if phone_number_id:
        PHONE_ID_CACHE[user_id] = phone_number_id

    old = DEBOUNCE_TASKS.get(user_id)
    if old and not old.done():
        old.cancel()
    task = asyncio.create_task(debounce_fire(user_id))
    DEBOUNCE_TASKS[user_id] = task

# ========= WhatsApp Cloud API (media helpers) =========
async def wa_get_media_url(media_id: str) -> str | None:
    """GET /{media_id} => {url} (requiere Bearer)"""
    if not (media_id and WABA_TOKEN):
        return None
    url = f"https://graph.facebook.com/{GRAPH_VER}/{media_id}"
    headers = {"Authorization": f"Bearer {WABA_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, headers=headers)
            if r.status_code >= 400:
                print("[wa_media] meta id lookup error:", r.status_code, r.text[:500])
                return None
            data = r.json()
            return data.get("url")
    except Exception:
        print("[wa_media] Exception lookup:\n", traceback.format_exc())
        return None

async def wa_download_media(media_url: str, max_bytes: int) -> bytes | None:
    """GET binarios desde media_url con Bearer; limita tamaño."""
    if not (media_url and WABA_TOKEN):
        return None
    headers = {"Authorization": f"Bearer {WABA_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.get(media_url, headers=headers)
            if r.status_code >= 400:
                print("[wa_media] download error:", r.status_code, r.text[:500])
                return None
            content = r.content
            if len(content) > max_bytes:
                print(f"[wa_media] file too large: {len(content)} bytes > {max_bytes}")
                return None
            return content
    except Exception:
        print("[wa_media] Exception download:\n", traceback.format_exc())
        return None

# ========= OpenAI helpers: chat + ASR + vision =========
async def call_openai_chat(messages: list[dict]) -> str:
    if not OPENAI_API_KEY:
        print("[openai] OPENAI_API_KEY faltante")
        return ""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.4}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                print("[openai] HTTP", r.status_code, r.text[:1000])
                return ""
            data = r.json()
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            if not content:
                print("[openai] Respuesta sin contenido:", json.dumps(data)[:1000])
            return content
    except Exception:
        print("[openai] Exception:\n", traceback.format_exc())
        return ""

async def openai_transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """Transcribe con whisper-1 (o gpt-4o-transcribe)."""
    if not (OPENAI_API_KEY and audio_bytes):
        return ""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    # multipart/form-data
    files = {
        "file": (filename, audio_bytes, "application/octet-stream"),
        "model": (None, ASR_MODEL),
        "response_format": (None, "text"),
        # "language": (None, "es"),  # opcional
        # "temperature": (None, "0.2"),
    }
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(url, headers=headers, files=files)
            if r.status_code >= 400:
                print("[asr] HTTP", r.status_code, r.text[:1000])
                return ""
            return r.text.strip()
    except Exception:
        print("[asr] Exception:\n", traceback.format_exc())
        return ""

async def openai_describe_image(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """
    Describe/extrae datos de una imagen con modelo multimodal.
    Enviamos la imagen como data URL base64 en el mensaje del usuario.
    """
    if not (OPENAI_API_KEY and image_bytes):
        return ""
    # generar data URL
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Eres un asistente que ayuda a un asesor inmobiliario a entender imágenes enviadas por clientes."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Analice brevemente esta imagen. Si es una propiedad, extraiga lo útil para calificar (zona visible, tipo de inmueble, "
                        "habitaciones/baños aparentes, estado, acabados, indicios de rango de precio). Responda en español de Costa Rica y sea conciso."
                    )},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        "temperature": 0.3
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                print("[vision] HTTP", r.status_code, r.text[:1000])
                return ""
            data = r.json()
            content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            return content
    except Exception:
        print("[vision] Exception:\n", traceback.format_exc())
        return ""

# ========= WhatsApp sender =========
async def send_whatsapp_text(phone_number_id: str, to_msisdn: str, text: str):
    if not WABA_TOKEN:
        print("[wa] WABA_TOKEN faltante; no se envía")
        return
    if not phone_number_id:
        print(f"[wa] phone_number_id faltante (user {to_msisdn}); no se envía")
        return
    if not to_msisdn:
        print("[wa] to_msisdn faltante; no se envía")
        return

    url = f"https://graph.facebook.com/{GRAPH_VER}/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {WABA_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "text",
        "text": {"body": text[:4000]},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            print("[wa] send status:", r.status_code, r.text[:500])
    except Exception:
        print("[wa] Exception:\n", traceback.format_exc())

# ========= Webhook principal =========
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("Incoming:", json.dumps(data, indent=2, ensure_ascii=False))

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {}) or {}
            metadata = value.get("metadata", {}) or {}
            phone_number_id = metadata.get("phone_number_id")

            msgs = value.get("messages") or []
            for msg in msgs:
                user_id = msg.get("from")
                if not user_id:
                    continue

                # == TEXT ==
                if msg.get("type") == "text":
                    user_text = (msg.get("text", {}) or {}).get("body", "") or ""
                    if user_text.strip():
                        enqueue_pending(user_id, user_text)
                        await schedule_debounce(user_id, phone_number_id)
                    continue

                # == AUDIO / VOICE ==
                if msg.get("type") == "audio":
                    audio = msg.get("audio") or {}
                    media_id = audio.get("id")
                    mime = audio.get("mime_type") or "audio/ogg"
                    if not media_id:
                        continue

                    media_url = await wa_get_media_url(media_id)
                    audio_bytes = await wa_download_media(media_url, MAX_AUDIO_BYTES) if media_url else None
                    transcript = ""
                    if audio_bytes:
                        # WhatsApp voz suele ser OGG/OPUS
                        filename = "audio.ogg" if "ogg" in mime or "opus" in mime else "audio.m4a"
                        transcript = await openai_transcribe_audio(audio_bytes, filename=filename)
                    if transcript:
                        enqueue_pending(user_id, f"[Audio transcrito]: {transcript}")
                    else:
                        enqueue_pending(user_id, "[Audio recibido]: (no se pudo transcribir)")

                    await schedule_debounce(user_id, phone_number_id)
                    continue

                # == IMAGE ==
                if msg.get("type") == "image":
                    image = msg.get("image") or {}
                    media_id = image.get("id")
                    mime = image.get("mime_type") or "image/jpeg"
                    if not media_id:
                        continue

                    media_url = await wa_get_media_url(media_id)
                    img_bytes = await wa_download_media(media_url, MAX_IMAGE_BYTES) if media_url else None
                    caption = ""
                    if img_bytes:
                        caption = await openai_describe_image(img_bytes, mime=mime)
                    if caption:
                        enqueue_pending(user_id, f"[Imagen analizada]: {caption}")
                    else:
                        enqueue_pending(user_id, "[Imagen recibida]: (no se pudo analizar)")

                    await schedule_debounce(user_id, phone_number_id)
                    continue

                # (Opcional) otros tipos: document, sticker, video, etc.
                # Podés agregar ramas similares: descargar -> extraer texto -> encolar.

    return {"status": "ok"}
