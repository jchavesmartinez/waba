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

from fastapi import BackgroundTasks, FastAPI, Request, Response

import config
import registry
from bot import audio, whatsapp
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

    for adj in respuesta.adjuntos:
        if not whatsapp.enviar_adjunto(numero, adj, numero_origen):
            # Subir o mandar el archivo fallo, pero el texto ya salio. Se avisa
            # para que el usuario no quede esperando un archivo que no llega.
            whatsapp.enviar_texto(
                numero,
                "No pude enviarte el archivo (problema con WhatsApp, no con los "
                "datos). Pedímelo de nuevo en un momento.",
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
    # value{ messages[], statuses[], ... }. Los 'statuses' (entregado/leido) y
    # cualquier value sin 'messages' se ignoran en silencio.
    for entry in data.get("entry", []):
        for cambio in entry.get("changes", []):
            valor = cambio.get("value", {})
            # De que numero NUESTRO llego el mensaje. Se usa para responder por
            # el mismo, en vez de por el de la variable de entorno.
            origen = (valor.get("metadata") or {}).get("phone_number_id", "")
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
