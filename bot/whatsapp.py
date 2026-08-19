"""
Cliente SALIENTE de la WhatsApp Cloud API (Meta / Graph API).

Con Twilio la respuesta viajaba en el mismo HTTP response (TwiML). Con Meta el
webhook solo confirma recepcion (200) y la respuesta al usuario se manda con
una llamada aparte a:

    POST https://graph.facebook.com/{version}/{phone_number_id}/messages
    Authorization: Bearer {WHATSAPP_TOKEN}

Esta llamada solo es libre (texto suelto, sin plantilla) DENTRO de la ventana de
24 h desde el ultimo mensaje del usuario. Como aca siempre respondemos a un
mensaje entrante, estamos dentro de esa ventana y un mensaje de tipo 'text'
alcanza.

ADJUNTOS (imagenes y documentos). Mandar un archivo son DOS llamadas, no una:

    1. POST /{phone_number_id}/media   (multipart)  -> devuelve un media_id
    2. POST /{phone_number_id}/messages  con {"type":"image","image":{"id":...}}

La alternativa es mandar un LINK publico en vez del media_id, pero eso obliga a
hostear el archivo en una URL abierta a internet — o sea, datos de ventas de un
cliente colgados sin autenticacion. Con el upload el archivo vive en la
infraestructura de Meta, se entrega por el mismo canal del chat y se borra a los
30 dias. Es una llamada mas y es el camino correcto.

Limites de Meta que importan aca:
  - imagen: 5 MB, jpeg/png
  - documento: 100 MB (pdf, xlsx, docx, pptx, txt)
  - caption: 1024 caracteres (el texto suelto son 4096)
"""

import logging
import mimetypes
import time

import httpx

import config

logger = logging.getLogger("fachavi.bot.whatsapp")

_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
# Subir un archivo tarda mucho mas que mandar un JSON de 200 bytes; con 15 s un
# Excel grande desde un contenedor frio se cae por timeout y el usuario no
# recibe nada.
_TIMEOUT_MEDIA = httpx.Timeout(60.0, connect=10.0)

# B-19: un fallo de envio se registraba y se descartaba. El usuario nunca
# recibia nada y ni siquiera sabia que su pregunta se habia procesado (y se
# habia pagado). Los fallos tipicos —Neon/Meta con un hipo de red, un 429, un
# 5xx— se resuelven solos en segundos, asi que un par de reintentos con espera
# creciente recupera la mayoria. Un 4xx que no sea 429 no se reintenta: token
# vencido o numero fuera de la ventana de 24 h no mejoran esperando.
_REINTENTOS = 3
_ESPERA_BASE_SEG = 1.5


def _url(recurso: str = "messages", numero_origen: str = "") -> str:
    """
    URL del endpoint para el numero DESDE el que se responde.

    `numero_origen` es el phone_number_id que Meta manda en el webhook
    (value.metadata.phone_number_id): el numero por el que ENTRO el mensaje.
    Sin este parametro, el bot respondia siempre por el numero de la variable
    de entorno, sin importar por cual le hubieran escrito — asi que un mensaje
    al numero de prueba salia contestado por el de produccion. Ademas de ser
    confuso, eso deja sin sandbox: cada prueba toca la reputacion del numero
    que ve el cliente.

    Cae a la variable de entorno cuando no viene, para no romper las llamadas
    que no nacen de un webhook.
    """
    pnid = numero_origen or config.WHATSAPP_PHONE_NUMBER_ID
    return (
        f"https://graph.facebook.com/{config.GRAPH_API_VERSION}"
        f"/{pnid}/{recurso}"
    )


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.WHATSAPP_TOKEN}"}


def _hay_credenciales() -> bool:
    if config.WHATSAPP_TOKEN and config.WHATSAPP_PHONE_NUMBER_ID:
        return True
    logger.error("Faltan WHATSAPP_TOKEN o WHATSAPP_PHONE_NUMBER_ID; no se envia.")
    return False


def _tope_caption() -> int:
    """El pie de una imagen/documento son 1024 chars, no 4096 como el texto."""
    return int(getattr(config, "WHATSAPP_MAX_CAPTION", 1024))


def _recortar(texto: str, tope: int | None = None) -> str:
    """WhatsApp corta a 4096 chars; recortamos antes para no perder el aviso."""
    tope = tope or config.WHATSAPP_MAX_CHARS
    if len(texto) <= tope:
        return texto
    corte = tope - 1
    return texto[:corte].rstrip() + "…"


def _fragmentar_texto(texto: str, tope: int | None = None) -> list[str]:
    """Divide sin perder contenido, priorizando los saltos entre registros."""
    tope = tope or config.WHATSAPP_MAX_CHARS
    pendientes = str(texto or "").splitlines(keepends=True)
    fragmentos = []
    actual = ""
    for linea in pendientes:
        while len(linea) > tope:
            if actual:
                fragmentos.append(actual.rstrip())
                actual = ""
            corte = linea.rfind(" ", 0, tope + 1)
            corte = corte if corte > 0 else tope
            fragmentos.append(linea[:corte].rstrip())
            linea = linea[corte:].lstrip()
        if actual and len(actual) + len(linea) > tope:
            fragmentos.append(actual.rstrip())
            actual = ""
        actual += linea
    if actual.strip():
        fragmentos.append(actual.rstrip())
    return fragmentos or [""]


def _post_mensaje(payload: dict, descripcion: str,
                  numero_origen: str = "") -> bool:
    """
    Manda un payload a /messages con la politica de reintentos de arriba.

    Es el bucle que antes vivia dentro de enviar_texto(); se extrajo para que
    imagen y documento no lo dupliquen. Si mañana se cambia el backoff o se le
    agrega telemetria, se toca en un solo lugar.
    """
    headers = {**_headers(), "Content-Type": "application/json"}
    destino = payload.get("to", "?")

    for intento in range(1, _REINTENTOS + 1):
        try:
            r = httpx.post(_url("messages", numero_origen), json=payload,
                           headers=headers, timeout=_TIMEOUT)
        except httpx.HTTPError as e:
            logger.warning("Error de red enviando %s a WhatsApp (%s), intento %d/%d: %s",
                           descripcion, destino, intento, _REINTENTOS, e)
            if intento == _REINTENTOS:
                logger.error("No se pudo enviar %s a %s tras %d intentos; el usuario "
                             "no va a recibir respuesta.", descripcion, destino,
                             _REINTENTOS)
                return False
            time.sleep(_ESPERA_BASE_SEG * intento)
            continue

        if r.status_code < 400:
            logger.info("Enviado %s a %s (%s)", descripcion, destino, r.status_code)
            return True

        # El cuerpo de error de Meta trae el motivo (token vencido, numero fuera
        # de la ventana de 24 h, phone_number_id malo, etc.). Va al log completo.
        recuperable = r.status_code == 429 or r.status_code >= 500
        logger.error(
            "Meta rechazo el envio de %s a %s [%s]%s: %s",
            descripcion, destino, r.status_code,
            f" (intento {intento}/{_REINTENTOS})" if recuperable else "",
            r.text[:500],
        )
        if not recuperable or intento == _REINTENTOS:
            return False
        time.sleep(_ESPERA_BASE_SEG * intento)

    return False


def enviar_texto(numero_destino: str, texto: str,
                 numero_origen: str = "") -> bool:
    """
    Manda un mensaje de texto por la Cloud API. Devuelve True si Meta lo acepto.

    No relanza excepciones: un fallo de envio se loguea y se traga, porque esto
    corre en un BackgroundTask y ya le devolvimos 200 a Meta. Si tirara, el
    error quedaria huerfano sin cambiar nada del lado del webhook.
    """
    if not _hay_credenciales():
        return False

    fragmentos = _fragmentar_texto(texto)
    for numero, fragmento in enumerate(fragmentos, 1):
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": numero_destino,
            "type": "text",
            "text": {"preview_url": False, "body": fragmento},
        }
        descripcion = f"texto {numero}/{len(fragmentos)}"
        if not _post_mensaje(payload, descripcion, numero_origen):
            return False
    return True


# --- Adjuntos: subir y mandar --------------------------------------------

def subir_media(contenido: bytes, nombre: str, mime: str,
                numero_origen: str = "") -> str:
    """
    Sube un archivo a /{phone_number_id}/media y devuelve el media_id ("" si falla).

    El media_id sirve para MANDAR el archivo y vale 30 dias.
    """
    if not _hay_credenciales():
        return ""

    tope_mb = float(getattr(config, "BOT_ADJUNTO_MAX_MB", 4.5))
    mb = len(contenido) / (1024 * 1024)
    if mb > tope_mb:
        logger.error("Adjunto '%s' pesa %.1f MB y el tope es %.1f MB; no se sube.",
                     nombre, mb, tope_mb)
        return ""

    files = {
        # El tercer elemento de la tupla (el MIME) NO es opcional: sin el, httpx
        # manda application/octet-stream y Meta responde 400 diciendo que el tipo
        # no esta soportado — un error que despista, porque el archivo esta bien.
        "file": (nombre, contenido, mime or "application/octet-stream"),
        "messaging_product": (None, "whatsapp"),
        "type": (None, mime),
    }

    for intento in range(1, _REINTENTOS + 1):
        try:
            r = httpx.post(_url("media", numero_origen), files=files,
                           headers=_headers(), timeout=_TIMEOUT_MEDIA)
        except httpx.HTTPError as e:
            logger.warning("Error de red subiendo '%s', intento %d/%d: %s",
                           nombre, intento, _REINTENTOS, e)
            if intento == _REINTENTOS:
                return ""
            time.sleep(_ESPERA_BASE_SEG * intento)
            continue

        if r.status_code < 400:
            media_id = (r.json() or {}).get("id", "")
            if media_id:
                logger.info("Subido '%s' (%.0f KB) -> media_id %s",
                            nombre, len(contenido) / 1024, media_id)
                return media_id
            logger.error("Meta acepto la subida de '%s' pero no devolvio id: %s",
                         nombre, r.text[:300])
            return ""

        recuperable = r.status_code == 429 or r.status_code >= 500
        logger.error("Meta rechazo la subida de '%s' [%s]: %s",
                     nombre, r.status_code, r.text[:500])
        if not recuperable or intento == _REINTENTOS:
            return ""
        time.sleep(_ESPERA_BASE_SEG * intento)

    return ""


def enviar_imagen(numero_destino: str, media_id: str, caption: str = "",
                  numero_origen: str = "") -> bool:
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": numero_destino,
        "type": "image",
        "image": {"id": media_id},
    }
    if caption:
        payload["image"]["caption"] = _recortar(caption, _tope_caption())
    return _post_mensaje(payload, "imagen", numero_origen)


def enviar_documento(numero_destino: str, media_id: str, nombre: str,
                     caption: str = "", numero_origen: str = "") -> bool:
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": numero_destino,
        "type": "document",
        # 'filename' es lo que ve el usuario y con lo que se guarda el archivo.
        # Sin el, WhatsApp muestra el media_id: un chorro de digitos sin
        # extension que en el celular ni siquiera abre con la app correcta.
        "document": {"id": media_id, "filename": nombre},
    }
    if caption:
        payload["document"]["caption"] = _recortar(caption, _tope_caption())
    return _post_mensaje(payload, "documento", numero_origen)


def enviar_adjunto(numero_destino: str, adjunto,
                   numero_origen: str = "") -> bool:
    """
    Sube y manda un bot.salida.Adjunto de una. Devuelve True si llego.

    Si la subida falla no se intenta mandar el mensaje: sin media_id no hay nada
    que enviar y el POST devolveria un 400 que no explica la causa real.
    """
    media_id = subir_media(adjunto.contenido, adjunto.nombre, adjunto.mime,
                           numero_origen)
    if not media_id:
        return False
    if adjunto.tipo == "image":
        return enviar_imagen(numero_destino, media_id, adjunto.caption,
                             numero_origen)
    return enviar_documento(numero_destino, media_id, adjunto.nombre,
                            adjunto.caption, numero_origen)


def descargar_media(media_id: str) -> tuple[bytes, str]:
    """
    Baja un archivo que MANDO el usuario. Devuelve (contenido, mime); (b"", "")
    si falla.

    Tambien son dos pasos: GET /{media_id} devuelve una URL temporal, y esa URL
    hay que pedirla CON el mismo Bearer (es la parte que se olvida y da 401).

    Se usa para notas de voz y para cualquier futura recepcion de adjuntos.
    """
    if not _hay_credenciales() or not media_id:
        return b"", ""
    try:
        r = httpx.get(
            f"https://graph.facebook.com/{config.GRAPH_API_VERSION}/{media_id}",
            headers=_headers(), timeout=_TIMEOUT,
        )
        r.raise_for_status()
        info = r.json() or {}
        url = info.get("url", "")
        mime = info.get("mime_type", "")
        if not url:
            return b"", ""
        r2 = httpx.get(url, headers=_headers(), timeout=_TIMEOUT_MEDIA)
        r2.raise_for_status()
        return r2.content, mime or (mimetypes.guess_type(url)[0]
                                    or "application/octet-stream")
    except Exception as e:  # noqa: BLE001
        logger.error("No se pudo descargar el media %s: %s", media_id, e)
        return b"", ""
