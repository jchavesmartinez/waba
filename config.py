"""
Configuracion central del repo. Todo sale de variables de entorno.

El repo tiene DOS mitades y este archivo configura las dos (B-39):
  - la INGESTA (sync.py, sources/, warehouse/), que corre como cron y escribe;
  - el BOT de WhatsApp (bot/), que corre como servicio web y solo LEE del
    warehouse — nunca de los sistemas fuente.

Una sola superficie de configuracion: si algo se puede ajustar por variable de
entorno, esta en este archivo. Ningun otro modulo lee os.environ directamente. prueba
"""

import json
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


def secreto_de_mapa(nombre_variable: str, clave: str, para: str = "") -> str:
    """
    Lee UN secreto de una variable de entorno que contiene un OBJETO JSON de
    {clave: secreto}. Es la version "muchos en una" de secreto_de_env().

        META_ADS_TOKENS = {"cliente_a":"EAAG...","cliente_b":"EAAH..."}

    y en el registro cada fuente declara nada mas su clave:

        {"tokens_env":"META_ADS_TOKENS","token_ref":"cliente_a"}

    POR QUE EXISTE ESTO. La regla del proyecto (C-04) es que el Sheet lleva el
    NOMBRE de la variable, nunca el secreto. Con una variable POR CLIENTE eso
    significa entrar al panel de Render cada vez que se da de alta uno nuevo, y
    esa friccion es exactamente lo que empuja a la tentacion de pegar el token
    en la hoja "solo por esta vez". Un mapa deja el alta como una fila del
    Sheet, sin tocar el servidor, y el secreto sigue viviendo donde debe.

    QUE NO HACE. No cifra nada ni reemplaza a un gestor de secretos: sigue
    siendo una variable de entorno del servidor. Lo que resuelve es la
    FRICCION operativa, que es lo que en la practica termina rompiendo la
    regla.

    NUNCA se registra ni se devuelve el contenido en un mensaje de error. Las
    claves disponibles SI se listan cuando la buscada no aparece, porque son
    identificadores de cliente (no secretos) y sin ellas depurar un typo
    obliga a ir a mirar la variable a mano.
    """
    nombre = str(nombre_variable).strip()
    clave = str(clave).strip()
    quien = para or "La configuracion"
    if not nombre or not clave:
        return ""

    crudo = os.environ.get(nombre, "").strip()
    if not crudo:
        raise RuntimeError(
            f"{quien} declara '{nombre}' como la variable de entorno con el "
            "mapa de secretos, pero esa variable no existe o esta vacia en el "
            "servidor."
        )

    try:
        mapa = json.loads(crudo)
    except json.JSONDecodeError as e:
        # A proposito NO se incluye 'crudo' en el mensaje: el error terminaria
        # en la bitacora del warehouse con todos los tokens adentro.
        raise RuntimeError(
            f"La variable '{nombre}' no contiene un JSON valido ({e}). Se "
            'espera un objeto {"cliente_a":"secreto","cliente_b":"secreto"}. '
            "Revisa que no haya comillas curvas ni una coma de mas al final."
        ) from e

    if not isinstance(mapa, dict):
        raise RuntimeError(
            f"La variable '{nombre}' debe contener un OBJETO JSON de "
            f"clave->secreto, no un {type(mapa).__name__}."
        )

    valor = str(mapa.get(clave, "") or "").strip()
    if not valor:
        disponibles = ", ".join(sorted(str(k) for k in mapa)) or "(ninguna)"
        raise RuntimeError(
            f"{quien} busca la clave '{clave}' en la variable '{nombre}', pero "
            f"esa clave no esta o esta vacia. Claves disponibles: {disponibles}."
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

# Cuando los DATOS contradicen a FORMATO_FECHA, ¿a quien se le hace caso?
#
# Un valor con el dia mayor a 12 ('16/01/2026') solo admite una lectura, asi que
# no es una heuristica: es prueba de que la declaracion esta mal. En "si" (por
# defecto) la ingesta usa la evidencia y deja una alerta. En "no" respeta lo
# declarado igual — util solo si sabes que tu fuente tiene basura y preferis
# controlarla a mano. Cualquiera de las dos avisa; el silencio no es opcion.
FECHA_AUTOCORREGIR = _es_si(os.environ.get("FECHA_AUTOCORREGIR", "si"))

# --- Convencion numerica por defecto de las fuentes ---
# "coma_decimal" (convencion tica: ₡5.000 son cinco mil, el punto es separador
# de miles) o "punto_decimal" (anglosajona: 5.000 son cinco). Cada fuente lo
# puede pisar con {"formato_numero": "punto_decimal"} en su columna config.
# Equivocarse aca multiplica o divide TODOS los montos por mil, en silencio.
FORMATO_NUMERO = os.environ.get("FORMATO_NUMERO", "coma_decimal").strip().lower()


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

# --- Gemini / Vertex AI: proveedor unico de IA del repositorio ---
# "vertex" reutiliza GOOGLE_CREDENTIALS_JSON (recomendado en produccion).
# "api_key" usa GEMINI_API_KEY y sirve como alternativa para desarrollo.
GEMINI_BACKEND = os.environ.get("GEMINI_BACKEND", "vertex").strip().lower()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Vacio = usa project_id del JSON del service account.
GEMINI_PROJECT_ID = os.environ.get("GEMINI_PROJECT_ID", "")
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "global")
# Render y Neon suelen operar en UTC. "Hoy" y "ayer" deben resolverse en la
# zona civil del negocio, no en la del servidor.
BOT_TIMEZONE = (
    os.environ.get("BOT_TIMEZONE", "America/Costa_Rica").strip()
    or "America/Costa_Rica"
)
GEMINI_TIMEOUT_SEGUNDOS = int(os.environ.get("GEMINI_TIMEOUT_SEGUNDOS", "90"))

# Gemini Flash reemplaza Haiku en todas las tareas. Se conservan variables por
# funcion para poder subir solo SQL/KPIs a un modelo mayor sin tocar las demas.
_GEMINI_MODELO_DEFAULT = os.environ.get("GEMINI_MODELO", "gemini-3.6-flash")
BOT_MODELO_SQL = os.environ.get("BOT_MODELO_SQL", _GEMINI_MODELO_DEFAULT)
# Solo para conversacion sin datos. Las cifras que devuelve SQL se formatean
# deterministicamente y no pasan de nuevo por el modelo.
BOT_MODELO_RESPUESTA = os.environ.get(
    "BOT_MODELO_RESPUESTA", _GEMINI_MODELO_DEFAULT
)

# Tope de tokens para GENERAR el SQL. 600 truncaba consultas con varios JOIN o
# un CTE, y el validador las rechazaba con "no parseable" — un motivo que no
# apunta a la causa real (B-24).
BOT_MAX_TOKENS_SQL = int(os.environ.get("BOT_MAX_TOKENS_SQL", "1200"))
# El planificador semantico solo necesita la conversacion reciente y el ultimo
# estado verificado. El historial completo sigue en Neon, pero no se envia al
# LLM en cada mensaje.
BOT_PLAN_HISTORIAL_TURNOS = int(
    os.environ.get("BOT_PLAN_HISTORIAL_TURNOS", "6")
)
BOT_PLAN_HISTORIAL_MAX_CHARS = int(
    os.environ.get("BOT_PLAN_HISTORIAL_MAX_CHARS", "6000")
)

# Tope duro de filas que el bot trae del warehouse por consulta.
BOT_MAX_FILAS = int(os.environ.get("BOT_MAX_FILAS", "500"))

# Corta consultas que se pasen de tiempo (proteccion del warehouse).
BOT_TIMEOUT_MS = int(os.environ.get("BOT_TIMEOUT_MS", "8000"))

# Auditoria de consultas en los logs de Render. El SQL se escribe en una sola
# linea junto con un id, origen, duracion y cantidad de filas. Puede contener
# literales de negocio; apaguelo si su politica de logs no permite esos datos.
BOT_LOG_SQL = _es_si(os.environ.get("BOT_LOG_SQL", "si"))
BOT_LOG_SQL_MAX_CHARS = int(os.environ.get("BOT_LOG_SQL_MAX_CHARS", "20000"))

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
    "BOT_MODELO_INTENCION", _GEMINI_MODELO_DEFAULT
)

# --------------------------------------------------------------------------
# Capa semantica de KPIs: usa metricas predefinidas del tab '_kpis' del Sheet
# (persistidas en <esquema>._kpis). Apagala (no) para responder solo con
# text-to-SQL libre, sin KPIs ni la logica de pedir contexto / retar.
# --------------------------------------------------------------------------
BOT_KPIS = _es_si(os.environ.get("BOT_KPIS", "si"))
BOT_MODELO_KPIS = os.environ.get("BOT_MODELO_KPIS", _GEMINI_MODELO_DEFAULT)
# Critico final opcional. Las validaciones matematicas siempre corren en codigo;
# esta llamada solo puede bloquear una respuesta incoherente, nunca reescribir
# cifras. Se deja apagada porque agrega costo y latencia y no sustituye contratos.
BOT_CRITICO_RESPUESTAS = _es_si(os.environ.get("BOT_CRITICO_RESPUESTAS", "no"))

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

# El PIE de una imagen o un documento NO son 4096: son 1024. Mandar mas hace que
# Meta rechace el mensaje entero, no que lo recorte.
WHATSAPP_MAX_CAPTION = int(os.environ.get("WHATSAPP_MAX_CAPTION", "1024"))


# --------------------------------------------------------------------------
# ADJUNTOS: el bot responde con graficos (PNG), Excel, CSV o PDF cuando el
# usuario los pide ("graficame las ventas", "pasame eso en Excel").
#
# El archivo se arma con el MISMO resultado del SELECT que ya se ejecuto y ya
# paso por el catalogo: no hay una segunda consulta ni una via alterna a los
# datos. La gobernanza no cambia — cambia el envoltorio.
# --------------------------------------------------------------------------

# Interruptor general. En "no", todo se responde en texto como antes.
BOT_ADJUNTOS = _es_si(os.environ.get("BOT_ADJUNTOS", "si"))

# Tope de filas para un export (Excel/CSV). Es DISTINTO de BOT_MAX_FILAS: ese
# protege la memoria del proceso y el tamaño del prompt de redaccion, y ninguna
# de las dos cosas aplica a un archivo, que no pasa por el modelo. 5.000 filas
# son ~250 KB de xlsx. El freno real del warehouse sigue siendo BOT_TIMEOUT_MS.
BOT_ADJUNTO_MAX_FILAS = int(os.environ.get("BOT_ADJUNTO_MAX_FILAS", "5000"))

# Cuantas barras/puntos entran en un grafico antes de que deje de leerse en un
# celular. Si el resultado trae mas, se recorta al top N (o a los N periodos mas
# recientes si el eje es temporal) y el titulo lo dice.
BOT_ADJUNTO_MAX_BARRAS = int(os.environ.get("BOT_ADJUNTO_MAX_BARRAS", "25"))

# El tope equivalente para SERIES DE TIEMPO, que se dibujan como linea. Una
# linea con 90 puntos se lee perfecto; recortarla a 25 mutila justo la tendencia
# que el usuario queria ver ("ventas por dia" de un mes son 30 puntos).
BOT_ADJUNTO_MAX_PUNTOS = int(os.environ.get("BOT_ADJUNTO_MAX_PUNTOS", "90"))

# Filas que entran en la tabla del PDF (un reporte de 3.000 filas no es un
# reporte). Si hay mas, el PDF lo aclara e invita a pedir el Excel.
BOT_ADJUNTO_PDF_MAX_FILAS = int(os.environ.get("BOT_ADJUNTO_PDF_MAX_FILAS", "60"))

# Peso maximo del adjunto. Meta acepta 5 MB de imagen y 100 MB de documento;
# el tope propio es mas bajo a proposito: un archivo de 20 MB por WhatsApp en
# una red movil tica es una descarga que el cliente no va a completar.
BOT_ADJUNTO_MAX_MB = float(os.environ.get("BOT_ADJUNTO_MAX_MB", "4.5"))

# Un 200 de /messages solo significa "aceptado". Si el webhook posterior trae
# status=failed, se reutiliza el media_id y se intenta enviar una vez más. El
# reclamo vive en Neon para que dos workers no dupliquen el reintento.
BOT_ADJUNTO_REINTENTOS_ENTREGA = int(os.environ.get(
    "BOT_ADJUNTO_REINTENTOS_ENTREGA", "1",
))
BOT_ADJUNTO_ENTREGAS_TTL_DIAS = int(os.environ.get(
    "BOT_ADJUNTO_ENTREGAS_TTL_DIAS", "30",
))

# PDF: agrega reportlab al build. Ponelo en "no" si no lo vas a usar y preferis
# un contenedor mas liviano; los pedidos de PDF se sirven como Excel.
BOT_ADJUNTO_PDF = _es_si(os.environ.get("BOT_ADJUNTO_PDF", "si"))

# CSV: la lista de MIME de documento que acepta la Cloud API no incluye
# text/csv, asi que el envio puede volver con un 400. Por defecto un pedido de
# CSV se sirve como .xlsx, que si esta soportado y ademas se abre mejor en el
# celular. Ponelo en "si" solo si probaste que tu numero lo acepta.
BOT_ADJUNTO_CSV = _es_si(os.environ.get("BOT_ADJUNTO_CSV", "no"))

# Adjuntos ENTRANTES: si el usuario manda una foto o un archivo CON pie de
# mensaje, se usa ese pie como pregunta en vez de contestar "solo entiendo
# texto". El archivo NO se procesa (ver la nota en bot/app.py).
BOT_MEDIA_ENTRANTE = _es_si(os.environ.get("BOT_MEDIA_ENTRANTE", "si"))

# --------------------------------------------------------------------------
# CORREO SALIENTE: cada numero conecta su propia cuenta mediante Google OAuth.
# Los archivos se guardan temporalmente en el esquema privado _bot y solo
# salen tras una confirmacion separada del mismo numero.
# --------------------------------------------------------------------------
BOT_EMAIL = _es_si(os.environ.get("BOT_EMAIL", "no"))
APP_PUBLIC_URL = (
    os.environ.get("APP_PUBLIC_URL", "").strip().rstrip("/")
    or os.environ.get("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
)
APP_TERMS_URL = os.environ.get("APP_TERMS_URL", "").strip()
APP_PRIVACY_URL = os.environ.get("APP_PRIVACY_URL", "").strip()
APP_TERMS_VERSION = os.environ.get("APP_TERMS_VERSION", "2026-08").strip()
GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
GOOGLE_OAUTH_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
OAUTH_TOKEN_KEY = os.environ.get("OAUTH_TOKEN_KEY", "").strip()
GOOGLE_OAUTH_REDIRECT_URI = (
    f"{APP_PUBLIC_URL}/oauth/google/callback" if APP_PUBLIC_URL else ""
)
OAUTH_ENLACE_TTL_MINUTOS = int(os.environ.get("OAUTH_ENLACE_TTL_MINUTOS", "10"))
OAUTH_HTTP_TIMEOUT_SEGUNDOS = int(os.environ.get("OAUTH_HTTP_TIMEOUT_SEGUNDOS", "30"))
BOT_EMAIL_ARTEFACTO_TTL_HORAS = int(os.environ.get(
    "BOT_EMAIL_ARTEFACTO_TTL_HORAS", "72",
))
BOT_EMAIL_CONFIRMACION_MINUTOS = int(os.environ.get(
    "BOT_EMAIL_CONFIRMACION_MINUTOS", "30",
))
BOT_EMAIL_MAX_ADJUNTO_MB = float(os.environ.get(
    "BOT_EMAIL_MAX_ADJUNTO_MB", "25",
))
BOT_EMAIL_TIMEOUT_SEGUNDOS = int(os.environ.get(
    "BOT_EMAIL_TIMEOUT_SEGUNDOS", "30",
))
BOT_EMAIL_MAX_POR_HORA = int(os.environ.get("BOT_EMAIL_MAX_POR_HORA", "10"))
BOT_EMAIL_MAX_POR_DIA = int(os.environ.get("BOT_EMAIL_MAX_POR_DIA", "50"))

# --------------------------------------------------------------------------
# NOTAS DE VOZ ENTRANTES: se descargan desde Meta, se transcriben y el texto
# resultante entra al MISMO flujo de permisos, memoria, KPIs y text-to-SQL.
# El audio original solo existe en memoria y en un directorio temporal durante
# la conversion de OGG/Opus; no se conserva en el warehouse.
# --------------------------------------------------------------------------

# Interruptor independiente del de captions/archivos. Se deja apagado por
# defecto para no consumir una API por accidente en instalaciones existentes;
# render.yaml lo activa explicitamente para este servicio.
BOT_AUDIO_ENTRANTE = _es_si(os.environ.get("BOT_AUDIO_ENTRANTE", "no"))

BOT_AUDIO_MODELO = os.environ.get(
    "BOT_AUDIO_MODELO", _GEMINI_MODELO_DEFAULT
)

# Las consultas habladas suelen durar segundos. Este tope evita que una nota
# accidentalmente larguisima genere una factura o bloquee el worker. La API
# admite solicitudes inline de menos de 20 MB; el codigo deja margen adicional.
BOT_AUDIO_MAX_MB = float(os.environ.get("BOT_AUDIO_MAX_MB", "5"))
BOT_AUDIO_TIMEOUT_SEGUNDOS = int(
    os.environ.get("BOT_AUDIO_TIMEOUT_SEGUNDOS", "90")
)

# Contexto de vocabulario, no una orden para contestar. Ayuda a conservar
# montos, fechas y nombres propios frecuentes en consultas empresariales.
BOT_AUDIO_PROMPT = os.environ.get(
    "BOT_AUDIO_PROMPT",
    "Consulta breve en espanol de Costa Rica sobre ventas, gastos, presupuesto "
    "o inventario. Conserva exactamente montos, fechas y nombres propios.",
)


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

    if GEMINI_BACKEND == "vertex":
        if not GOOGLE_CREDENTIALS_JSON:
            avisos.append(
                "GEMINI_BACKEND=vertex pero falta GOOGLE_CREDENTIALS_JSON: "
                "Gemini no podra autenticar contra Vertex AI."
            )
        elif not GEMINI_PROJECT_ID:
            try:
                proyecto_json = json.loads(GOOGLE_CREDENTIALS_JSON).get(
                    "project_id", ""
                )
            except Exception:  # noqa: BLE001
                proyecto_json = ""
            if not proyecto_json:
                avisos.append(
                    "GEMINI_BACKEND=vertex pero no hay GEMINI_PROJECT_ID ni "
                    "project_id util en GOOGLE_CREDENTIALS_JSON."
                )
    elif GEMINI_BACKEND == "api_key":
        if not GEMINI_API_KEY:
            avisos.append(
                "GEMINI_BACKEND=api_key pero falta GEMINI_API_KEY: el bot no "
                "va a poder responder."
            )
    else:
        avisos.append(
            f"GEMINI_BACKEND invalido: '{GEMINI_BACKEND}'. Usa vertex o api_key."
        )

    try:
        from google import genai as _genai  # noqa: F401
    except ImportError:
        avisos.append(
            "Falta google-genai: ninguna funcion del modelo va a operar. "
            "Reinstala requirements-bot.txt."
        )

    for variable, modelo in (
        ("BOT_MODELO_SQL", BOT_MODELO_SQL),
        ("BOT_MODELO_RESPUESTA", BOT_MODELO_RESPUESTA),
        ("BOT_MODELO_INTENCION", BOT_MODELO_INTENCION),
        ("BOT_MODELO_KPIS", BOT_MODELO_KPIS),
        ("BOT_AUDIO_MODELO", BOT_AUDIO_MODELO),
    ):
        if not str(modelo or "").strip().lower().startswith("gemini-"):
            avisos.append(
                f"{variable}='{modelo}' no es un modelo Gemini. Revisa una "
                "variable heredada del despliegue anterior."
            )

    # Los adjuntos dependen de librerias que estan en requirements-bot.txt. Si
    # el build quedo viejo, el import falla RECIEN cuando alguien pide un
    # grafico —o sea, en produccion, frente al cliente, con un mensaje generico.
    # Mejor gritarlo al arrancar, que es donde se mira.
    if BOT_ADJUNTOS:
        faltan = []
        for lib, para in (("matplotlib", "gráficos"),
                          ("xlsxwriter", "Excel"),
                          *((("reportlab", "PDF"),) if BOT_ADJUNTO_PDF else ())):
            try:
                __import__(lib)
            except ImportError:
                faltan.append(f"{lib} ({para})")
        if faltan:
            avisos.append(
                "BOT_ADJUNTOS=si pero faltan librerias: " + ", ".join(faltan) +
                ". Los pedidos de archivo van a fallar. Reinstala "
                "requirements-bot.txt o pone BOT_ADJUNTOS=no."
            )

    if BOT_AUDIO_ENTRANTE:
        try:
            __import__("imageio_ffmpeg")
        except ImportError:
            avisos.append(
                "BOT_AUDIO_ENTRANTE=si pero falta imageio-ffmpeg: WhatsApp "
                "entrega las notas en OGG/Opus y no se podran convertir. "
                "Reinstala requirements-bot.txt."
            )

    if BOT_EMAIL:
        faltan_oauth = [nombre for nombre, valor in (
            ("APP_PUBLIC_URL", APP_PUBLIC_URL),
            ("APP_TERMS_URL", APP_TERMS_URL),
            ("APP_PRIVACY_URL", APP_PRIVACY_URL),
            ("GOOGLE_OAUTH_CLIENT_ID", GOOGLE_OAUTH_CLIENT_ID),
            ("GOOGLE_OAUTH_CLIENT_SECRET", GOOGLE_OAUTH_CLIENT_SECRET),
            ("OAUTH_TOKEN_KEY", OAUTH_TOKEN_KEY),
        ) if not valor]
        if faltan_oauth:
            avisos.append(
                "BOT_EMAIL=si pero faltan variables de Google OAuth: "
                + ", ".join(faltan_oauth) + "."
            )
        if OAUTH_TOKEN_KEY and len(OAUTH_TOKEN_KEY) < 32:
            avisos.append(
                "OAUTH_TOKEN_KEY tiene menos de 32 caracteres; use un secreto "
                "aleatorio largo para cifrar refresh tokens."
            )
        try:
            __import__("cryptography")
        except ImportError:
            avisos.append(
                "BOT_EMAIL=si pero falta cryptography: no se pueden cifrar los "
                "tokens OAuth. Reinstala requirements-bot.txt."
            )

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
