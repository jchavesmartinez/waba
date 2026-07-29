"""
Configuracion central. Todo sale de variables de entorno (Render las inyecta).
"""

import os
import re

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


def dsn_de_cliente(cliente: dict) -> str:
    """
    Resuelve a QUE warehouse va cada cliente. Permite un proyecto de Neon por
    cliente (aislamiento fisico) sin poner credenciales en el Sheet maestro.

    Orden de resolucion:
      1. La variable nombrada en la columna 'dsn_env' de la pestania 'clientes'
         (p.ej. dsn_env = NEON_FERRETERIA_A)
      2. Por convencion: WAREHOUSE_DSN_<CLIENTE_ID en mayusculas>
      3. WAREHOUSE_DSN global (todos los clientes en el mismo proyecto,
         separados por esquema)

    Asi se puede empezar con un proyecto compartido e ir migrando clientes a su
    propio proyecto de a uno, agregando una variable de entorno. Sin tocar
    codigo y sin que el secreto pase nunca por Google Sheets.
    """
    nombrada = str(cliente.get("dsn_env", "")).strip()
    if nombrada:
        valor = os.environ.get(nombrada, "").strip()
        if valor:
            return valor
        raise RuntimeError(
            f"El cliente '{cliente.get('cliente_id')}' declara dsn_env='{nombrada}' "
            "pero esa variable de entorno no existe o esta vacia."
        )

    cid = re.sub(r"[^0-9a-zA-Z_]", "_", str(cliente.get("cliente_id", "")).strip()).upper()
    por_convencion = os.environ.get(f"WAREHOUSE_DSN_{cid}", "").strip()
    if por_convencion:
        return por_convencion

    return WAREHOUSE_DSN

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
