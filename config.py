"""
Configuracion central de la INGESTA (Fase 2). Todo sale de variables de entorno.

Nota: este repo es solo la capa de ingesta. El bot de WhatsApp se reconstruira
aparte y leera del warehouse, NUNCA de los sistemas fuente.
"""

import os
import re

# --- Google (service account) ---
# El mismo service account lee el Sheet MAESTRO (registro de clientes/fuentes)
# y los Sheets de DATOS de cada cliente.
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")

# ID del SHEET MAESTRO (registro), NO el de datos.
MASTER_SPREADSHEET_ID = os.environ.get("MASTER_SPREADSHEET_ID", "")

# --- Warehouse (Fase 3: aterrizaje de la data cruda) ---
# duckdb   -> archivo local (dev) o MotherDuck (md:...)
# postgres -> Neon / Supabase / Postgres administrado
WAREHOUSE_TIPO = os.environ.get("WAREHOUSE_TIPO", "duckdb")
WAREHOUSE_DSN = os.environ.get("WAREHOUSE_DSN", "fachavi.duckdb")

# --- Cache del registro ---
# El registro (clientes/fuentes) cambia poco; se relee cada N segundos.
REGISTRY_CACHE_TTL = int(os.environ.get("REGISTRY_CACHE_TTL", "300"))


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
