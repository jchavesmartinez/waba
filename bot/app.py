"""
Webhook de WhatsApp (Meta Cloud API / Graph API) para el bot de datos.

A diferencia de Twilio (que mandaba form-urlencoded y esperaba TwiML de vuelta),
la Cloud API funciona en DOS tiempos:

  1. GET  /webhook  -> handshake de verificacion. Meta manda hub.mode,
     hub.verify_token y hub.challenge; si el token coincide, devolvemos el
     challenge en texto plano y Meta da por validado el endpoint.

  2. POST /webhook  -> el mensaje entra como JSON anidado:
         entry[].changes[].value.messages[].text.body   (texto)
         entry[].changes[].value.messages[].from        (numero, solo digitos)
     Respondemos 200 de una y mandamos la respuesta al usuario con una llamada
     SALIENTE aparte (bot/whatsapp.py), en un BackgroundTask para no dejar a
     Meta esperando el text-to-SQL (si tardamos, Meta reintenta el webhook).

La resolucion numero->cliente, la seleccion de tablas por catalogo y el
text-to-SQL viven en bot/responder.py, igual que antes. Aca solo cambia el
transporte.

Correr local:
    uvicorn bot.app:app --reload --port 8000
Exponer con cloudflared/ngrok y en el panel de Meta (WhatsApp > Configuration)
pegar la URL .../webhook y el mismo WHATSAPP_VERIFY_TOKEN.
"""

import hashlib
import hmac
import logging
import threading
import time
from collections import OrderedDict, deque
from html import escape
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

import config
import registry
from bot import audio, correo, entregas, whatsapp
from bot.responder import responder
from bot.salida import Respuesta

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fachavi.bot.app")

app = FastAPI(title="FACHAVI — WhatsApp bot (Meta Cloud API)")

# C-05 / C-08 / B-37: se revisa la configuracion de seguridad AL ARRANCAR y se
# deja en el log en nivel alto. Un default inseguro es discutible; uno
# silencioso no lo es. Esto es lo que convierte "alguien dejo el modo permisivo
# prendido hace tres meses" en algo que se ve en dos segundos.
_AVISOS_ARRANQUE = config.revisar_arranque_bot()

_SALUDO = "Hola. Envíe su consulta sobre los datos habilitados para su empresa."
_MEDIA_NO_SOPORTADO = (
    "Entiendo mensajes de texto y notas de voz, pero todavía no leo el contenido "
    "de fotos o archivos adjuntos. Escriba la consulta o agréguela como "
    "pie del archivo."
)
_AUDIO_NO_DISPONIBLE = (
    "No pude procesar notas de voz en este momento. Escriba la consulta y la "
    "atiendo igual."
)
_AUDIO_MUY_GRANDE = (
    "La nota de voz es demasiado grande. Envíe una nota más corta o escriba "
    "la consulta."
)
_AUDIO_NO_ENTENDIDO = (
    "No logré entender esa nota de voz. Intente grabarla nuevamente, más cerca "
    "del micrófono, o escriba la consulta."
)
_NO_REGISTRADO = (
    "Tu número no está registrado para consultar datos. "
    "Contacte a la persona que administra este servicio."
)
_MUY_RAPIDO = (
    "Está enviando mensajes con demasiada frecuencia. Inténtelo nuevamente "
    "en un minuto."
)

# Dedup: Meta reintenta el webhook si no ve un 200 a tiempo, y puede repetir un
# mensaje. Guardamos los ultimos ids vistos para no contestar (ni gastar LLM)
# dos veces. En memoria alcanza para un solo proceso; si se escala a varios
# workers habria que mover esto a Redis (A-12).
_VISTOS: "OrderedDict[str, None]" = OrderedDict()
_VISTOS_TOPE = 1000

# A-13: limite de frecuencia por numero. Un numero registrado podia mandar cien
# mensajes en un minuto y cada uno gasta entre 3 y 5 llamadas al modelo. Igual
# que el dedup, esto vive en el proceso: con un solo worker alcanza, y con
# varios cada uno aplica su propia cuota (mas laxo, nunca mas estricto).
_HISTORIAL_ENVIOS: "OrderedDict[str, deque]" = OrderedDict()
_LOCK = threading.Lock()


def _ya_procesado(msg_id: str) -> bool:
    if not msg_id:
        return False
    with _LOCK:
        if msg_id in _VISTOS:
            return True
        _VISTOS[msg_id] = None
        if len(_VISTOS) > _VISTOS_TOPE:
            _VISTOS.popitem(last=False)
    return False


def _pasa_limite(numero: str) -> bool:
    """False si el numero se paso de config.BOT_MAX_MSJ_POR_MINUTO (A-13)."""
    tope = int(config.BOT_MAX_MSJ_POR_MINUTO or 0)
    if tope <= 0:
        return True
    ahora = time.time()
    with _LOCK:
        cola = _HISTORIAL_ENVIOS.setdefault(numero, deque())
        while cola and ahora - cola[0] > 60:
            cola.popleft()
        if len(cola) >= tope:
            return False
        cola.append(ahora)
        # Higiene de memoria: no guardar numeros inactivos para siempre.
        if len(_HISTORIAL_ENVIOS) > 5000:
            _HISTORIAL_ENVIOS.popitem(last=False)
    return True


def _firma_valida(cuerpo: bytes, firma_header: str) -> bool:
    """
    Valida X-Hub-Signature-256 (HMAC-SHA256 del cuerpo crudo con el app secret).

    C-08: antes, si no habia WHATSAPP_APP_SECRET esta funcion devolvia True y
    TODO pasaba — o sea, el valor por defecto era "no validar nada", y nada
    avisaba. Cualquiera que descubriera la URL del webhook podia hacerse pasar
    por Meta, suplantar un numero registrado, extraer sus datos y consumir la
    cuenta del proveedor de IA. Ahora, sin secreto, se rechaza; el modo inseguro
    requiere poner BOT_PERMITIR_SIN_FIRMA=si a proposito (para desarrollo local).
    """
    if not config.WHATSAPP_APP_SECRET:
        if config.BOT_PERMITIR_SIN_FIRMA:
            return True
        logger.error(
            "POST rechazado: no hay WHATSAPP_APP_SECRET para validar la firma de "
            "Meta. Configuralo en el panel (o BOT_PERMITIR_SIN_FIRMA=si en local)."
        )
        return False
    if not firma_header or not firma_header.startswith("sha256="):
        return False
    esperado = hmac.new(
        config.WHATSAPP_APP_SECRET.encode("utf-8"), cuerpo, hashlib.sha256
    ).hexdigest()
    recibido = firma_header.split("=", 1)[1]
    # compare_digest tarda lo mismo sin importar donde este la primera
    # diferencia: comparar con == filtraria la firma correcta por tiempo.
    return hmac.compare_digest(esperado, recibido)


def _atender(numero: str, texto: str, numero_origen: str = "") -> None:
    """
    Corre en background: arma la respuesta y la manda por la Cloud API.

    `numero_origen` es el phone_number_id por el que ENTRO el mensaje. Se
    responde por ese mismo numero: si alguien le escribe al numero de prueba,
    le contesta el de prueba, no el de produccion.
    """
    try:
        respuesta = responder(numero, texto) if texto else Respuesta(_SALUDO)
    except Exception as e:  # noqa: BLE001
        logger.exception("Error generando respuesta para %s: %s", numero, e)
        respuesta = Respuesta("Tuve un problema procesando su consulta. "
                              "Inténtelo nuevamente.")

    # Primero el texto y despues el archivo, en ese orden y en mensajes
    # separados. WhatsApp permite un caption en la imagen, pero son 1024 chars
    # contra 4096 y el celular lo muestra colapsado bajo un "ver mas": la
    # respuesta se leeria peor por ahorrarse un mensaje.
    if respuesta.texto:
        whatsapp.enviar_texto(numero, respuesta.texto, numero_origen)

    try:
        cliente = registry.resolver(numero) if respuesta.adjuntos else None
    except Exception as e:  # noqa: BLE001
        logger.warning("No se pudo resolver cliente para seguir adjuntos: %s", e)
        cliente = None
    for adj in respuesta.adjuntos:
        envio = whatsapp.enviar_adjunto(numero, adj, numero_origen)
        if not envio:
            # Subir o mandar el archivo fallo, pero el texto ya salio. Se avisa
            # para que el usuario no quede esperando un archivo que no llega.
            whatsapp.enviar_texto(
                numero,
                "No pude enviarte el archivo (problema con WhatsApp, no con los "
                "datos). Pedímelo de nuevo en un momento.",
                numero_origen,
            )
            continue
        if not cliente:
            logger.warning(
                "Adjunto %s aceptado, pero no se pudo resolver el cliente para "
                "seguir su entrega.", adj.nombre,
            )
            continue
        pendiente = entregas.registrar(
            cliente, numero, numero_origen,
            envio.message_id, envio.media_id,
            adj.tipo, adj.nombre, adj.mime,
        )
        if pendiente:
            _reintentar_entrega(cliente, pendiente)


def _reintentar_entrega(cliente: dict, pendiente: entregas.Reintento) -> None:
    """Ejecuta el único reintento reclamado transaccionalmente en Neon."""
    nuevo_id = whatsapp.reintentar_adjunto(
        pendiente.numero, pendiente.media_id, pendiente.tipo,
        pendiente.nombre, pendiente.phone_number_id,
    )
    if not nuevo_id:
        entregas.vincular_reintento(
            cliente, pendiente.message_id, "",
            error="Meta rechazó el reintento inmediato del adjunto",
        )
        whatsapp.enviar_texto(
            pendiente.numero,
            f"No pude entregar el archivo {pendiente.nombre}. "
            "Inténtelo nuevamente en un momento.",
            pendiente.phone_number_id,
        )
        return

    entregas.vincular_reintento(
        cliente, pendiente.message_id, nuevo_id,
    )
    segundo = entregas.registrar(
        cliente, pendiente.numero, pendiente.phone_number_id,
        nuevo_id, pendiente.media_id, pendiente.tipo,
        pendiente.nombre, pendiente.mime,
        intentos=pendiente.intentos + 1,
        reintento_de=pendiente.message_id,
    )
    # En la carrera extrema donde el segundo status llegó antes del registro,
    # registrar() puede devolver ya el fallo reclamado. No debería haber un
    # tercer envío: la política permite solo un reintento.
    if segundo:
        logger.warning("Se descartó un reintento adicional de %s", nuevo_id)


def _procesar_estado_salida(estado: dict, numero_origen: str = "") -> None:
    """Procesa sent/delivered/read/failed recibidos de Meta."""
    numero = str(estado.get("recipient_id", "") or "").strip()
    if not numero:
        logger.warning("Status de WhatsApp sin recipient_id: %s", estado)
        return
    try:
        cliente = registry.resolver(numero)
    except Exception as e:  # noqa: BLE001
        logger.warning("No se pudo resolver cliente para status de %s: %s",
                       numero, e)
        return
    if not cliente:
        logger.warning("Status de adjunto para número no registrado: %s", numero)
        return

    resultado = entregas.actualizar_estado(cliente, estado)
    if not resultado:
        return
    if resultado.reintento:
        _reintentar_entrega(cliente, resultado.reintento)
    elif resultado.fallo_final:
        whatsapp.enviar_texto(
            resultado.numero,
            f"No pude entregar el archivo {resultado.nombre} después de "
            "reintentarlo. Solicítelo nuevamente en un momento.",
            numero_origen,
        )


def _atender_audio(numero: str, media_id: str, mime_webhook: str = "",
                   numero_origen: str = "") -> None:
    """Descarga, transcribe y entrega una nota al mismo flujo que el texto."""
    # Se valida ANTES de descargar o llamar a Gemini: un numero ajeno no debe
    # poder convertir el webhook en una llave para consumir la API.
    try:
        registrado = bool(registry.resolver(numero))
    except Exception as e:  # noqa: BLE001
        logger.exception("No se pudo validar el numero antes del audio: %s", e)
        whatsapp.enviar_texto(numero, _AUDIO_NO_DISPONIBLE, numero_origen)
        return
    if not registrado:
        logger.info("Nota de voz de numero no registrado: %s", numero)
        whatsapp.enviar_texto(numero, _NO_REGISTRADO, numero_origen)
        return

    contenido, mime_descarga = whatsapp.descargar_media(media_id)
    if not contenido:
        whatsapp.enviar_texto(numero, _AUDIO_NO_DISPONIBLE, numero_origen)
        return

    try:
        texto = audio.transcribir(contenido, mime_descarga or mime_webhook)
    except audio.AudioDemasiadoGrande:
        whatsapp.enviar_texto(numero, _AUDIO_MUY_GRANDE, numero_origen)
        return
    except audio.AudioNoEntendido:
        whatsapp.enviar_texto(numero, _AUDIO_NO_ENTENDIDO, numero_origen)
        return
    except audio.ErrorAudio as e:
        logger.warning("No se pudo transcribir la nota de %s: %s", numero, e)
        whatsapp.enviar_texto(numero, _AUDIO_NO_DISPONIBLE, numero_origen)
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("Error inesperado procesando audio de %s: %s", numero, e)
        whatsapp.enviar_texto(numero, _AUDIO_NO_DISPONIBLE, numero_origen)
        return

    # No se escribe el texto en el log: puede contener datos sensibles. La
    # consulta transcrita se guarda en memoria solo si responder() ya lo hacia
    # con las consultas escritas.
    logger.info("Nota de voz de %s transcrita (%d caracteres).", numero,
                len(texto))
    _atender(numero, texto, numero_origen)


@app.get("/salud")
def salud():
    """Estado del servicio + advertencias de configuracion (sin filtrar secretos)."""
    return {"ok": True, "advertencias": _AVISOS_ARRANQUE}


@app.get("/webhook")
def verificar(request: Request):
    """Handshake de verificacion del webhook (Meta lo llama una sola vez)."""
    params = request.query_params
    modo = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")

    if modo == "subscribe" and token == config.WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verificado por Meta.")
        return Response(content=challenge, media_type="text/plain")

    logger.warning("Verificacion de webhook fallida (token no coincide).")
    return Response(content="forbidden", status_code=403)


@app.get("/oauth/google/iniciar", response_class=HTMLResponse)
def oauth_google_iniciar(token: str = ""):
    """Aviso propio de privacidad/terminos antes de entrar al consentimiento Google."""
    if not token or not correo.validar_enlace_oauth(token):
        return HTMLResponse(
            "<h1>Enlace inválido o vencido</h1><p>Solicite uno nuevo desde WhatsApp.</p>",
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    html = f"""
    <!doctype html><html lang="es"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Conectar Gmail con Fachavi</title></head>
    <body style="font-family:system-ui;max-width:620px;margin:48px auto;padding:0 20px;line-height:1.5">
      <h1>Conectar Gmail con Fachavi</h1>
      <p>Fachavi solicitará únicamente permiso para enviar los correos que usted
      confirme desde WhatsApp. No podrá leer ni eliminar mensajes de su bandeja.</p>
      <p>El refresh token se almacenará cifrado y puede revocar la conexión en
      cualquier momento escribiendo <b>desconectar mi correo</b>.</p>
      <form method="post" action="/oauth/google/autorizar">
        <input type="hidden" name="token" value="{escape(token)}">
        <label><input type="checkbox" name="acepto" value="si" required>
        Acepto los <a href="{escape(config.APP_TERMS_URL)}" target="_blank">Términos</a>
        y la <a href="{escape(config.APP_PRIVACY_URL)}" target="_blank">Política de privacidad</a>.</label>
        <p><button type="submit" style="background:#1769e0;color:white;border:0;
        padding:12px 18px;border-radius:8px;font-weight:600">Continuar con Google</button></p>
      </form>
      <small>Este enlace es personal, de un solo uso y vence pronto.</small>
    </body></html>
    """
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.post("/oauth/google/autorizar")
async def oauth_google_autorizar(request: Request):
    formulario = parse_qs((await request.body()).decode("utf-8", errors="replace"))
    token = (formulario.get("token") or [""])[0]
    acepto = (formulario.get("acepto") or [""])[0]
    if acepto != "si":
        return HTMLResponse(
            "<h1>Debe aceptar los términos para continuar</h1>",
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    destino = correo.url_autorizacion_google(token)
    if not destino:
        return HTMLResponse(
            "<h1>Enlace inválido o vencido</h1><p>Solicite uno nuevo desde WhatsApp.</p>",
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    return RedirectResponse(destino, status_code=302,
                            headers={"Cache-Control": "no-store"})


@app.get("/oauth/google/callback", response_class=HTMLResponse)
def oauth_google_callback(state: str = "", code: str = "", error: str = ""):
    if error or not state or not code:
        return HTMLResponse(
            "<h1>Conexión cancelada</h1><p>No se concedió acceso a Gmail.</p>",
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    try:
        resultado = correo.completar_oauth_google(state, code)
        whatsapp.enviar_texto(
            resultado.numero,
            f"✅ Gmail conectado: {resultado.correo}. Ya puede pedirme que envíe un archivo por correo.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo completar Google OAuth: %s", exc)
        return HTMLResponse(
            "<h1>No se pudo conectar Gmail</h1><p>El enlace pudo vencer o Google no concedió el permiso. Solicite otro desde WhatsApp.</p>",
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    return HTMLResponse(
        f"<h1>Gmail conectado</h1><p>La cuenta <b>{escape(resultado.correo)}</b> quedó conectada. Puede cerrar esta ventana y volver a WhatsApp.</p>",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/webhook")
async def webhook(request: Request, tareas: BackgroundTasks):
    cuerpo = await request.body()

    if not _firma_valida(cuerpo, request.headers.get("X-Hub-Signature-256", "")):
        logger.warning("Firma X-Hub-Signature-256 invalida; se descarta el POST.")
        return Response(status_code=403)

    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        logger.warning("Cuerpo del webhook no es JSON valido; se ignora.")
        return Response(status_code=200)  # 200 igual: no queremos reintentos

    # Estructura: object=whatsapp_business_account -> entry[] -> changes[] ->
    # value{ messages[], statuses[], ... }. Los statuses confirman de forma
    # asíncrona si un adjunto fue enviado, entregado, leído o falló.
    for entry in data.get("entry", []):
        for cambio in entry.get("changes", []):
            valor = cambio.get("value", {})
            # De que numero NUESTRO llego el mensaje. Se usa para responder por
            # el mismo, en vez de por el de la variable de entorno.
            origen = (valor.get("metadata") or {}).get("phone_number_id", "")
            for estado in valor.get("statuses", []) or []:
                tareas.add_task(_procesar_estado_salida, estado, origen)
            for msg in valor.get("messages", []) or []:
                msg_id = msg.get("id", "")
                if _ya_procesado(msg_id):
                    logger.info("Mensaje repetido %s; se omite.", msg_id)
                    continue

                numero = (msg.get("from") or "").strip()
                if not numero:
                    continue

                if not _pasa_limite(numero):
                    logger.warning(
                        "Limite de frecuencia alcanzado para %s (%d msj/min); "
                        "no se procesa.", numero, config.BOT_MAX_MSJ_POR_MINUTO,
                    )
                    tareas.add_task(whatsapp.enviar_texto, numero, _MUY_RAPIDO,
                                    origen)
                    continue

                tipo = msg.get("type")
                if tipo == "text":
                    texto = (msg.get("text", {}).get("body") or "").strip()
                    tareas.add_task(_atender, numero, texto, origen)
                    continue

                if tipo == "audio":
                    info_audio = msg.get("audio") or {}
                    media_id = str(info_audio.get("id", "") or "").strip()
                    mime = str(info_audio.get("mime_type", "") or "").strip()
                    if config.BOT_AUDIO_ENTRANTE and media_id:
                        tareas.add_task(
                            _atender_audio, numero, media_id, mime, origen
                        )
                    else:
                        tareas.add_task(
                            whatsapp.enviar_texto,
                            numero,
                            _AUDIO_NO_DISPONIBLE,
                            origen,
                        )
                    continue

                # El usuario mando una foto o un archivo CON pie de mensaje
                # ("mirá esto, ¿cuánto suma?"). El archivo no se procesa —eso es
                # otro proyecto: parsear un Excel que llega por chat necesita
                # validacion de esquema, deteccion de tipos y una decision de
                # gobernanza sobre donde aterriza. Pero el caption SI es una
                # pregunta de verdad, y tratarla como "solo entiendo texto" es
                # perder una consulta que el bot podia contestar perfectamente.
                caption = ((msg.get(tipo) or {}).get("caption") or "").strip()
                if caption and config.BOT_MEDIA_ENTRANTE:
                    logger.info("Mensaje tipo '%s' de %s con caption; se usa el "
                                "caption como pregunta.", tipo, numero)
                    tareas.add_task(_atender, numero, caption, origen)
                    continue

                # Imagen, ubicacion, etc.: explicamos que el contenido no se lee.
                logger.info("Mensaje tipo '%s' de %s; no es texto.", tipo, numero)
                tareas.add_task(whatsapp.enviar_texto, numero,
                                _MEDIA_NO_SOPORTADO, origen)

    # Meta solo quiere un 200 rapido; el envio real va por BackgroundTask.
    return Response(status_code=200)
