"""
Configuracion central. Todo sale de variables de entorno (Render las inyecta).
"""

import os

# --- WhatsApp / Meta ---
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "cambia-esto")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID", "")
GRAPH_API_VERSION = os.environ.get("GRAPH_API_VERSION", "v21.0")

# --- Anthropic (Claude) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Haiku: rapido y barato, ideal para alto volumen. El sufijo de fecha es obligatorio.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# --- Google Sheets ---
# El JSON del service account se pasa como variable de entorno (texto completo),
# no como archivo, para que sea facil de configurar en Render.
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
# ID del libro de Google Sheets (lo que va entre /d/ y /edit en la URL).
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "")

# --- Cache de datos ---
# Segundos que se mantienen los datos en memoria antes de releer la hoja.
# 60s da frescura "casi en tiempo real" sin golpear la API en cada mensaje.
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "60"))

# --- Guardas de ejecucion SQL ---
# Tope de filas que se devuelven de una consulta (evita respuestas gigantes).
MAX_RESULT_ROWS = int(os.environ.get("MAX_RESULT_ROWS", "100"))

# --- Memoria de conversacion ---
MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", "memory.db")
MEMORY_MAX_TURNS = int(os.environ.get("MEMORY_MAX_TURNS", "6"))
