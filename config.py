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


# ==========================================================================
# BOT DE WHATSAPP (capa de LECTURA). La ingesta no usa nada de esto; son
# variables opcionales que solo el bot (paquete bot/) necesita. Se dejan aca
# para tener una sola superficie de configuracion.
# ==========================================================================

# Clave de la API de Anthropic (text-to-SQL + redaccion de la respuesta).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Modelo para GENERAR el SQL. Haiku alcanza de sobra para esto y es barato.
BOT_MODELO_SQL = os.environ.get("BOT_MODELO_SQL", "claude-haiku-4-5-20251001")

# Modelo para REDACTAR la respuesta en lenguaje natural a partir de las filas.
BOT_MODELO_RESPUESTA = os.environ.get("BOT_MODELO_RESPUESTA", "claude-haiku-4-5-20251001")

# Tope duro de filas que el bot trae del warehouse por consulta.
BOT_MAX_FILAS = int(os.environ.get("BOT_MAX_FILAS", "200"))

# Corta consultas que se pasen de tiempo (proteccion del warehouse).
BOT_TIMEOUT_MS = int(os.environ.get("BOT_TIMEOUT_MS", "8000"))

# Politica de gobernanza: ¿que hace el bot con una tabla cuya 'instruccion'
# viene VACIA o ambigua en el catalogo? False = no la usa (fail-closed, la
# opcion segura). Ponelo en True solo si preferis que "sin instruccion" = abierta.
BOT_PERMITIR_SIN_INSTRUCCION = (
    os.environ.get("BOT_PERMITIR_SIN_INSTRUCCION", "no").strip().lower()
    in ("si", "sí", "true", "1", "yes")
)

# --------------------------------------------------------------------------
# Memoria conversacional del bot.
# Guarda los turnos (pregunta/respuesta) por numero en un esquema APARTE (_bot)
# del MISMO Neon del cliente, para dar continuidad ("y de proveedores?") y
# recordar incluso dias atras. Es escritura, asi que NO pasa por la via de
# solo-lectura: vive en su propia tabla y jamas toca los datos del cliente.
# --------------------------------------------------------------------------
BOT_MEMORIA = os.environ.get("BOT_MEMORIA", "si").strip().lower() in (
    "si", "sí", "true", "1", "yes"
)
# Cuantos turnos (user+assistant cuentan como 2) se cargan como contexto.
BOT_MEMORIA_MAX_TURNOS = int(os.environ.get("BOT_MEMORIA_MAX_TURNOS", "20"))
# Ventana de recencia: solo se recuerda lo hablado en las ultimas N horas.
# 72 = tres dias. Subilo si queres memoria mas larga.
BOT_MEMORIA_VENTANA_HORAS = int(os.environ.get("BOT_MEMORIA_VENTANA_HORAS", "72"))
# Retencion: se borran los turnos mas viejos que esto (gobernanza/privacidad).
BOT_MEMORIA_TTL_DIAS = int(os.environ.get("BOT_MEMORIA_TTL_DIAS", "30"))

# --------------------------------------------------------------------------
# Clasificador de intencion: rutea cada mensaje a datos / meta / saludo antes
# del text-to-SQL. Apagalo (no) para volver a mandar TODO al generador de SQL.
# --------------------------------------------------------------------------
BOT_INTENCION = os.environ.get("BOT_INTENCION", "si").strip().lower() in (
    "si", "sí", "true", "1", "yes"
)
BOT_MODELO_INTENCION = os.environ.get(
    "BOT_MODELO_INTENCION", "claude-haiku-4-5-20251001"
)

# --------------------------------------------------------------------------
# WhatsApp Cloud API (Meta / Graph API) — transporte del bot.
# Reemplaza a Twilio: el mensaje entra como JSON en el webhook y la respuesta
# se manda con una llamada SALIENTE aparte al Graph API (no se responde en el
# mismo HTTP response como hacia TwiML).
# --------------------------------------------------------------------------

# Token de acceso de la app de WhatsApp (Bearer). En produccion usa un
# System User token permanente; el temporal del panel dura 24 h.
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")

# ID del NUMERO de telefono (phone_number_id), NO el WABA id ni el numero en si.
# Es lo que va en la URL de envio: /{version}/{phone_number_id}/messages.
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")

# Token que VOS inventas y pegas en el panel de Meta al configurar el webhook.
# Meta lo devuelve en el GET de verificacion (hub.verify_token) y debe coincidir.
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")

# App secret de la app de Meta. Si esta seteado, se valida la firma
# X-Hub-Signature-256 de cada POST (recomendado). Vacio = no se valida (dev).
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")

# Version del Graph API. Se puede subir sin tocar codigo.
GRAPH_API_VERSION = os.environ.get("GRAPH_API_VERSION", "v21.0")

# Tope de caracteres del cuerpo de un mensaje de texto de WhatsApp (limite de
# Meta = 4096). Si la respuesta se pasa, se recorta antes de enviar.
WHATSAPP_MAX_CHARS = int(os.environ.get("WHATSAPP_MAX_CHARS", "4096"))
