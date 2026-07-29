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
# Un solo service account lee TODO: el Sheet maestro y los Sheets de datos.
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")

# OJO: ahora este es el ID del SHEET MAESTRO (registro de clientes/usuarios),
# NO el de los datos. Los IDs de datos de cada cliente viven en el maestro.
MASTER_SPREADSHEET_ID = os.environ.get("MASTER_SPREADSHEET_ID", "")

# --- Warehouse (Fase 3: aterrizaje de la data cruda) ---
# duckdb  -> archivo local  (/data/fachavi.duckdb)
#            o MotherDuck   (md:fachavi?motherduck_token=XXX)  <- mismo codigo
# postgres -> Supabase / Neon / Postgres administrado
WAREHOUSE_TIPO = os.environ.get("WAREHOUSE_TIPO", "duckdb")
WAREHOUSE_DSN = os.environ.get("WAREHOUSE_DSN", "fachavi.duckdb")

# --- Cache ---
# Datos de clientes: releer maximo cada N segundos (frescura casi tiempo real).
DATA_CACHE_TTL = int(os.environ.get("DATA_CACHE_TTL", "60"))
# Registro (clientes/usuarios): cambia poco; se cachea mas tiempo.
REGISTRY_CACHE_TTL = int(os.environ.get("REGISTRY_CACHE_TTL", "300"))

# --- Guardas de ejecucion SQL ---
MAX_RESULT_ROWS = int(os.environ.get("MAX_RESULT_ROWS", "100"))

# --- Memoria de conversacion (efimera; ok si se pierde en redeploy) ---
MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", "memory.db")
MEMORY_MAX_TURNS = int(os.environ.get("MEMORY_MAX_TURNS", "6"))

# --- Mensaje para numeros no registrados ---
MSG_NO_AUTORIZADO = os.environ.get(
    "MSG_NO_AUTORIZADO",
    "Hola. Tu numero no esta registrado para consultar datos todavia. "
    "Por favor contacta a FACHAVI para darte de alta y empezar a usar el bot.",
)
