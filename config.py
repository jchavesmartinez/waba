"""
Configuracion central del repo. Todo sale de variables de entorno.

El repo tiene DOS mitades y este archivo configura las dos (B-39):
  - la INGESTA (sync.py, sources/, warehouse/), que corre como cron y escribe;
  - el BOT de WhatsApp (bot/), que corre como servicio web y solo LEE del
    warehouse — nunca de los sistemas fuente.

Una sola superficie de configuracion: si algo se puede ajustar por variable de
entorno, esta en este archivo. Ningun otro modulo lee os.environ directamente.
"""

import logging
import os
import re

logger = logging.getLogger("fachavi.config")


def _es_si(valor: str) -> bool:
    return str(valor).strip().lower() in ("si", "sí", "true", "1", "yes")


def secreto_de_env(nombre_variable: str, para: str = "") -> str:
    """
    Lee un secreto de una variable de entorno declarada POR NOMBRE en la hoja
    maestra (C-04).

    La regla del proyecto es que el Sheet nunca lleva credenciales: lleva el
    NOMBRE de la variable que las guarda. Este helper es el mismo mecanismo que
    ya usaba dsn_de_cliente(), extraido para que lo puedan reusar los conectores
    (sources/postgres.py, sources/api_rest.py).

    Falla explicito si la variable no existe: caer en silencio a "sin
    credencial" produce un error confuso mucho mas adelante.
    """
    nombre = str(nombre_variable).strip()
    if not nombre:
        return ""
    valor = os.environ.get(nombre, "").strip()
    if not valor:
        raise RuntimeError(
            f"{para or 'La configuracion'} declara '{nombre}' como variable de "
            "entorno con el secreto, pero esa variable no existe o esta vacia "
            "en el servidor."
        )
    return valor

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

# --- Retencion de la bitacora _meta.sync_corridas (A-04) ---
# Se registra una fila por fuente y por corrida, incluidas las omitidas. Con 4
# fuentes y 96 corridas diarias son ~140.000 filas al año. 0 = no purgar nunca.
SYNC_RETENCION_DIAS = int(os.environ.get("SYNC_RETENCION_DIAS", "180"))

# --- Formato de fecha por defecto de las fuentes (A-07) ---
# "dia_primero" (convencion tica: 07/01/2026 = 7 de enero) o "mes_primero"
# (convencion anglosajona: 07/01/2026 = 1 de julio). Cada fuente lo puede pisar
# con {"formato_fecha": "mes_primero"} en su columna config.
FORMATO_FECHA = os.environ.get("FORMATO_FECHA", "dia_primero").strip().lower()


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
        return secreto_de_env(
            nombrada, para=f"El cliente '{cliente.get('cliente_id')}'"
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

# Tope de tokens para GENERAR el SQL. 600 truncaba consultas con varios JOIN o
# un CTE, y el validador las rechazaba con "no parseable" — un motivo que no
# apunta a la causa real (B-24).
BOT_MAX_TOKENS_SQL = int(os.environ.get("BOT_MAX_TOKENS_SQL", "1200"))

# Tope duro de filas que el bot trae del warehouse por consulta.
BOT_MAX_FILAS = int(os.environ.get("BOT_MAX_FILAS", "200"))

# Corta consultas que se pasen de tiempo (proteccion del warehouse).
BOT_TIMEOUT_MS = int(os.environ.get("BOT_TIMEOUT_MS", "8000"))

# Politica de gobernanza: ¿que hace el bot con una tabla cuya 'instruccion'
# viene VACIA o ambigua en el catalogo? False = no la usa (fail-closed, la
# opcion segura). Ponelo en True solo si preferis que "sin instruccion" = abierta.
BOT_PERMITIR_SIN_INSTRUCCION = _es_si(
    os.environ.get("BOT_PERMITIR_SIN_INSTRUCCION", "no")
)

# --- Topes de consumo del bot (A-13, A-14) ---
# Cada mensaje gasta entre 3 y 5 llamadas al modelo. Sin tope, un usuario
# entusiasta —o alguien que descubrio el webhook— se traduce directo en factura.
# 0 en cualquiera de los dos = sin limite (no recomendado en produccion).
BOT_MAX_MSJ_POR_MINUTO = int(os.environ.get("BOT_MAX_MSJ_POR_MINUTO", "12"))
BOT_MAX_MSJ_POR_DIA = int(os.environ.get("BOT_MAX_MSJ_POR_DIA", "300"))

# --------------------------------------------------------------------------
# Memoria conversacional del bot.
# Guarda los turnos (pregunta/respuesta) por numero en un esquema APARTE (_bot)
# del MISMO Neon del cliente, para dar continuidad ("y de proveedores?") y
# recordar incluso dias atras. Es escritura, asi que NO pasa por la via de
# solo-lectura: vive en su propia tabla y jamas toca los datos del cliente.
# --------------------------------------------------------------------------
BOT_MEMORIA = _es_si(os.environ.get("BOT_MEMORIA", "si"))
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
BOT_INTENCION = _es_si(os.environ.get("BOT_INTENCION", "si"))
BOT_MODELO_INTENCION = os.environ.get(
    "BOT_MODELO_INTENCION", "claude-haiku-4-5-20251001"
)

# --------------------------------------------------------------------------
# Capa semantica de KPIs: usa metricas predefinidas del tab '_kpis' del Sheet
# (persistidas en <esquema>._kpis). Apagala (no) para responder solo con
# text-to-SQL libre, sin KPIs ni la logica de pedir contexto / retar.
# --------------------------------------------------------------------------
BOT_KPIS = _es_si(os.environ.get("BOT_KPIS", "si"))
BOT_MODELO_KPIS = os.environ.get("BOT_MODELO_KPIS", "claude-haiku-4-5-20251001")

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

# App secret de la app de Meta. Se usa para validar la firma
# X-Hub-Signature-256 de cada POST entrante.
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")

# C-08: antes, si WHATSAPP_APP_SECRET venia vacia el bot aceptaba CUALQUIER POST
# sin avisar nada. Un despliegue donde alguien se olvido de configurarla se veia
# identico a uno correcto — hasta que alguien lo aprovechara. Ahora el modo
# inseguro requiere una decision EXPLICITA: sin secreto y sin este permiso, el
# webhook rechaza todo y el arranque grita.
# Poné "si" SOLO para desarrollo local.
BOT_PERMITIR_SIN_FIRMA = _es_si(os.environ.get("BOT_PERMITIR_SIN_FIRMA", "no"))

# Version del Graph API. Se puede subir sin tocar codigo.
GRAPH_API_VERSION = os.environ.get("GRAPH_API_VERSION", "v21.0")

# Tope de caracteres del cuerpo de un mensaje de texto de WhatsApp (limite de
# Meta = 4096). Si la respuesta se pasa, se recorta antes de enviar.
WHATSAPP_MAX_CHARS = int(os.environ.get("WHATSAPP_MAX_CHARS", "4096"))


# --------------------------------------------------------------------------
# Verificacion de arranque del BOT (C-05, C-08, B-37).
# La llama bot/app.py al levantar. Un valor por defecto inseguro es aceptable;
# uno SILENCIOSO no. Todo lo que baje la seguridad tiene que dejar rastro en el
# log de arranque, que es lo primero que se mira cuando algo pasa.
# --------------------------------------------------------------------------

def revisar_arranque_bot() -> list:
    """Devuelve la lista de advertencias; las loguea en nivel alto."""
    avisos = []

    if not WHATSAPP_APP_SECRET:
        if BOT_PERMITIR_SIN_FIRMA:
            avisos.append(
                "MODO SIN FIRMA ACTIVO (BOT_PERMITIR_SIN_FIRMA=si y sin "
                "WHATSAPP_APP_SECRET): cualquiera que conozca la URL del webhook "
                "puede suplantar a un numero registrado. Solo para desarrollo."
            )
        else:
            avisos.append(
                "FALTA WHATSAPP_APP_SECRET: no se puede validar la firma de Meta, "
                "asi que TODOS los POST al webhook seran rechazados. Configura el "
                "app secret en el panel, o BOT_PERMITIR_SIN_FIRMA=si para local."
            )

    if BOT_PERMITIR_SIN_INSTRUCCION:
        avisos.append(
            "MODO PERMISIVO ACTIVO (BOT_PERMITIR_SIN_INSTRUCCION=si): TODAS las "
            "tablas sin instruccion de gobernanza quedan consultables por el bot, "
            "en TODOS los clientes. Si esto no fue deliberado, ponelo en 'no'."
        )

    if not ANTHROPIC_API_KEY:
        avisos.append("FALTA ANTHROPIC_API_KEY: el bot no va a poder responder.")

    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID):
        avisos.append(
            "FALTAN WHATSAPP_TOKEN o WHATSAPP_PHONE_NUMBER_ID: el bot recibe "
            "mensajes pero no puede contestar."
        )
    elif len(WHATSAPP_TOKEN) < 100:
        # B-37: el token temporal del panel de Meta dura 24 h y es notoriamente
        # mas corto que el permanente de System User. Heuristica, no certeza.
        avisos.append(
            "WHATSAPP_TOKEN parece el token TEMPORAL del panel de Meta (vence en "
            "24 h). Para produccion genera uno permanente de System User."
        )

    for a in avisos:
        logger.warning("ARRANQUE DEL BOT — %s", a)
    if not avisos:
        logger.info("Arranque del bot: configuracion de seguridad OK.")
    return avisos
