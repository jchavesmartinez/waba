"""Correo saliente confirmado usando la cuenta Google de cada usuario.

El numero registrado inicia OAuth desde WhatsApp, Google concede solamente
``gmail.send`` y el refresh token se cifra antes de persistirlo en ``_bot``.
Los archivos generados siguen aislados por (cliente_id, numero), no se publican
en URLs y solo salen tras una confirmacion separada del mismo numero.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import mimetypes
import re
import secrets
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from email.message import EmailMessage
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import create_engine, text

import config
import registry
from bot.salida import Adjunto, Respuesta

logger = logging.getLogger("fachavi.bot.correo")

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"
_SCOPES_GOOGLE = ("openid", "email", "profile", GMAIL_SEND_SCOPE)

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63})(?![\w.-])",
    re.IGNORECASE,
)
_PEDIDO_RE = re.compile(
    r"\b(?:envi[ae]\w*|mand[ae]\w*|reenvi[ae]\w*|comparti\w*|correo|e-?mail)\b",
    re.IGNORECASE,
)
_ARCHIVO_RE = re.compile(
    r"\b(?:archivo|adjunto|pdf|excel|xlsx|csv|grafico|grafica|imagen|documento|reporte|informe)\b",
    re.IGNORECASE,
)
_GENERAR_RE = re.compile(
    r"\b(?:crea|crear|cree|genera|generar|genere|haz|haga|arma|armar|arme|"
    r"prepara|preparar|prepare)\w*\b",
    re.IGNORECASE,
)
_ENVIAR_RE = re.compile(
    r"\b(?:envi|mand|reenvi|compart)\w*\b",
    re.IGNORECASE,
)
_CONECTAR_RE = re.compile(
    r"\b(?:conectar|conecta|conecte|conectame|vincular|vincula|vincule|vinculame)\b.*"
    r"\b(?:correo|gmail|google|email)\b|"
    r"\b(?:correo|gmail|google|email)\b.*\b(?:conectar|conecta|conecte|vincular|vincula|vincule)\b",
    re.IGNORECASE,
)
_DESCONECTAR_RE = re.compile(
    r"\b(?:desconecta|desconectar|desvincula|desvincular|revoca|revocar)\w*\b.*"
    r"\b(?:correo|gmail|google|email)\b",
    re.IGNORECASE,
)
_ESTADO_CONEXION_RE = re.compile(
    r"\b(?:que|cual|estado|tengo)\b.*\b(?:correo|gmail|email)\b.*"
    r"\b(?:conectado|vinculado|conexion|cuenta)\b|"
    r"\b(?:correo|gmail|email)\b.*\b(?:conectado|vinculado)\b",
    re.IGNORECASE,
)
_ESTADO_ENVIO_RE = re.compile(
    r"\b(?:estado|salio|enviado|envio|correo)\b.*\b(?:correo|envio|enviado|salio)\b",
    re.IGNORECASE,
)
_CONFIRMAR = {"si", "sí", "si enviar", "sí enviar", "confirmo", "enviar", "envialo", "envíalo"}
_CANCELAR = {"no", "cancelar", "cancela", "no enviar", "olvidalo", "olvídalo"}

_engines: dict = {}
_tablas_listas: set = set()
_lock = threading.Lock()

_DDL = (
    'CREATE SCHEMA IF NOT EXISTS "_bot"',
    """
    CREATE TABLE IF NOT EXISTS "_bot".artefactos (
        id BIGSERIAL PRIMARY KEY, cliente_id TEXT NOT NULL, numero TEXT NOT NULL,
        nombre TEXT NOT NULL, mime TEXT NOT NULL, contenido BYTEA NOT NULL,
        sha256 TEXT NOT NULL, creado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
        expira_en TIMESTAMPTZ NOT NULL
    )
    """,
    'CREATE INDEX IF NOT EXISTS ix_artefactos_cliente_numero_fecha ON "_bot".artefactos (cliente_id, numero, creado_en DESC)',
    """
    CREATE TABLE IF NOT EXISTS "_bot".conexiones_email (
        id BIGSERIAL PRIMARY KEY, cliente_id TEXT NOT NULL, numero TEXT NOT NULL,
        proveedor TEXT NOT NULL DEFAULT 'google', provider_user_id TEXT NOT NULL,
        correo TEXT NOT NULL, refresh_token_cifrado TEXT NOT NULL,
        scopes TEXT NOT NULL, estado TEXT NOT NULL DEFAULT 'activa',
        terminos_version TEXT NOT NULL DEFAULT '',
        conectado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
        actualizado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
        ultimo_uso_en TIMESTAMPTZ, revocado_en TIMESTAMPTZ,
        UNIQUE (cliente_id, numero, proveedor)
    )
    """,
    'CREATE INDEX IF NOT EXISTS ix_conexiones_email_cliente_numero ON "_bot".conexiones_email (cliente_id, numero, proveedor)',
    """
    CREATE TABLE IF NOT EXISTS "_bot".oauth_sesiones (
        state_hash TEXT PRIMARY KEY, cliente_id TEXT NOT NULL, numero TEXT NOT NULL,
        proveedor TEXT NOT NULL DEFAULT 'google', code_verifier TEXT NOT NULL,
        terminos_version TEXT NOT NULL DEFAULT '', terminos_aceptados_en TIMESTAMPTZ,
        creado_en TIMESTAMPTZ NOT NULL DEFAULT now(), expira_en TIMESTAMPTZ NOT NULL,
        usado_en TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "_bot".envios_email (
        id TEXT PRIMARY KEY, cliente_id TEXT NOT NULL, numero TEXT NOT NULL,
        artefacto_id BIGINT NOT NULL, remitente TEXT NOT NULL DEFAULT '',
        destinatario TEXT NOT NULL, asunto TEXT NOT NULL, cuerpo TEXT NOT NULL,
        nombre_archivo TEXT NOT NULL, mime TEXT NOT NULL,
        estado TEXT NOT NULL DEFAULT 'pendiente', provider_id TEXT NOT NULL DEFAULT '',
        error TEXT NOT NULL DEFAULT '', creado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
        confirmado_en TIMESTAMPTZ, actualizado_en TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    'ALTER TABLE "_bot".envios_email ADD COLUMN IF NOT EXISTS remitente TEXT NOT NULL DEFAULT \'\'',
    'CREATE INDEX IF NOT EXISTS ix_envios_email_cliente_numero_fecha ON "_bot".envios_email (cliente_id, numero, creado_en DESC)',
    'CREATE UNIQUE INDEX IF NOT EXISTS ux_envios_email_provider ON "_bot".envios_email (provider_id) WHERE provider_id <> \'\'',
)


@dataclass(frozen=True)
class Artefacto:
    id: int
    nombre: str
    mime: str
    contenido: bytes


@dataclass(frozen=True)
class Conexion:
    correo: str
    refresh_token_cifrado: str
    scopes: str
    provider_user_id: str = ""


@dataclass(frozen=True)
class Borrador:
    id: str
    remitente: str
    destinatario: str
    asunto: str
    cuerpo: str
    artefacto: Artefacto


@dataclass(frozen=True)
class OAuthCompletado:
    cliente: dict
    numero: str
    correo: str


def _normalizar(valor: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", valor or "")
                   if not unicodedata.combining(c)).lower()


def _engine(cliente: dict):
    dsn = config.dsn_de_cliente(cliente)
    if dsn not in _engines:
        _engines[dsn] = create_engine(dsn, pool_pre_ping=True)
    return _engines[dsn]


def _asegurar_tablas(cliente: dict) -> None:
    dsn = config.dsn_de_cliente(cliente)
    if dsn in _tablas_listas:
        return
    with _lock:
        if dsn in _tablas_listas:
            return
        with _engine(cliente).begin() as cx:
            cx.execute(text("SELECT pg_advisory_xact_lock(hashtext('fachavi_bot_correo'))"))
            for sentencia in _DDL:
                cx.execute(text(sentencia))
        _tablas_listas.add(dsn)


def _oauth_configurado() -> bool:
    return bool(config.APP_PUBLIC_URL and config.GOOGLE_OAUTH_CLIENT_ID
                and config.GOOGLE_OAUTH_CLIENT_SECRET and config.OAUTH_TOKEN_KEY
                and config.APP_TERMS_URL and config.APP_PRIVACY_URL)


def _b64url(datos: bytes) -> str:
    return base64.urlsafe_b64encode(datos).decode().rstrip("=")


def _b64url_decode(valor: str) -> bytes:
    return base64.urlsafe_b64decode(valor + "=" * (-len(valor) % 4))


def _clave_derivada(etiqueta: bytes) -> bytes:
    return hashlib.sha256(etiqueta + b":" + config.OAUTH_TOKEN_KEY.encode()).digest()


def _cifrar_token(token: str, cliente_id: str, numero: str) -> str:
    nonce = secrets.token_bytes(12)
    aad = f"{cliente_id}:{numero}:google".encode()
    cifrado = AESGCM(_clave_derivada(b"refresh-token")).encrypt(nonce, token.encode(), aad)
    return _b64url(nonce + cifrado)


def _descifrar_token(valor: str, cliente_id: str, numero: str) -> str:
    datos = _b64url_decode(valor)
    aad = f"{cliente_id}:{numero}:google".encode()
    return AESGCM(_clave_derivada(b"refresh-token")).decrypt(datos[:12], datos[12:], aad).decode()


def _crear_state(cliente_id: str) -> str:
    cid = _b64url(cliente_id.encode())
    base = f"{cid}.{secrets.token_urlsafe(24)}"
    firma = hmac.new(_clave_derivada(b"oauth-state"), base.encode(), hashlib.sha256).digest()
    return f"{base}.{_b64url(firma)}"


def _cliente_desde_state(state: str) -> dict | None:
    try:
        cid_b64, nonce, firma_b64 = state.split(".", 2)
        base = f"{cid_b64}.{nonce}"
        esperada = hmac.new(_clave_derivada(b"oauth-state"), base.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(esperada, _b64url_decode(firma_b64)):
            return None
        cliente_id = _b64url_decode(cid_b64).decode()
        return next((c for c in registry.listar_clientes()
                     if c.get("cliente_id") == cliente_id), None)
    except Exception:  # noqa: BLE001
        return None


def _state_hash(state: str) -> str:
    return hashlib.sha256(state.encode()).hexdigest()


def crear_enlace_conexion(cliente: dict, numero: str) -> str:
    if not _oauth_configurado():
        return ""
    _asegurar_tablas(cliente)
    state = _crear_state(cliente["cliente_id"])
    verifier = secrets.token_urlsafe(64)[:128]
    with _engine(cliente).begin() as cx:
        cx.execute(text("""
            INSERT INTO "_bot".oauth_sesiones
                (state_hash, cliente_id, numero, code_verifier, expira_en)
            VALUES (:hash, :cid, :num, :verifier,
                    now() + make_interval(mins => :mins))
        """), {"hash": _state_hash(state), "cid": cliente["cliente_id"],
                 "num": numero, "verifier": verifier,
                 "mins": config.OAUTH_ENLACE_TTL_MINUTOS})
        cx.execute(text('DELETE FROM "_bot".oauth_sesiones WHERE expira_en < now()'))
    return f"{config.APP_PUBLIC_URL.rstrip('/')}/oauth/google/iniciar?{urlencode({'token': state})}"


def _sesion_oauth(cliente: dict, state: str, exigir_aceptada: bool = False):
    _asegurar_tablas(cliente)
    extra = " AND terminos_aceptados_en IS NOT NULL" if exigir_aceptada else ""
    with _engine(cliente).connect() as cx:
        return cx.execute(text("""
            SELECT numero, code_verifier, terminos_version
            FROM "_bot".oauth_sesiones
            WHERE state_hash=:hash AND cliente_id=:cid
              AND expira_en > now() AND usado_en IS NULL
        """ + extra + " LIMIT 1"),
        {"hash": _state_hash(state), "cid": cliente["cliente_id"]}).fetchone()


def validar_enlace_oauth(state: str) -> bool:
    cliente = _cliente_desde_state(state)
    return bool(cliente and _sesion_oauth(cliente, state))


def url_autorizacion_google(state: str) -> str:
    cliente = _cliente_desde_state(state)
    if not cliente:
        return ""
    sesion = _sesion_oauth(cliente, state)
    if not sesion:
        return ""
    verifier = sesion[1]
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    with _engine(cliente).begin() as cx:
        cx.execute(text("""
            UPDATE "_bot".oauth_sesiones
            SET terminos_version=:version, terminos_aceptados_en=now()
            WHERE state_hash=:hash AND usado_en IS NULL
        """), {"version": config.APP_TERMS_VERSION, "hash": _state_hash(state)})
    parametros = {
        "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
        "redirect_uri": config.GOOGLE_OAUTH_REDIRECT_URI,
        "response_type": "code", "scope": " ".join(_SCOPES_GOOGLE),
        "access_type": "offline", "include_granted_scopes": "true",
        "prompt": "consent select_account", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(parametros)


def completar_oauth_google(state: str, code: str) -> OAuthCompletado:
    cliente = _cliente_desde_state(state)
    if not cliente:
        raise ValueError("enlace invalido")
    sesion = _sesion_oauth(cliente, state, exigir_aceptada=True)
    if not sesion:
        raise ValueError("enlace vencido o ya utilizado")
    numero, verifier, terminos_version = sesion
    token_resp = httpx.post("https://oauth2.googleapis.com/token", data={
        "code": code, "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": config.GOOGLE_OAUTH_CLIENT_SECRET,
        "redirect_uri": config.GOOGLE_OAUTH_REDIRECT_URI,
        "grant_type": "authorization_code", "code_verifier": verifier,
    }, timeout=config.OAUTH_HTTP_TIMEOUT_SEGUNDOS)
    token_resp.raise_for_status()
    tokens = token_resp.json()
    access_token = str(tokens.get("access_token", ""))
    if not access_token:
        raise RuntimeError("Google no devolvio access_token")
    perfil_resp = httpx.get(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=config.OAUTH_HTTP_TIMEOUT_SEGUNDOS,
    )
    perfil_resp.raise_for_status()
    perfil = perfil_resp.json()
    correo = str(perfil.get("email", "")).strip().lower()
    provider_user_id = str(perfil.get("sub", "")).strip()
    if not correo or not provider_user_id or not perfil.get("email_verified", False):
        raise RuntimeError("Google no devolvio una identidad de correo verificada")
    scopes = str(tokens.get("scope", ""))
    if GMAIL_SEND_SCOPE not in scopes.split():
        raise RuntimeError("el usuario no concedio el permiso gmail.send")
    refresh_token = str(tokens.get("refresh_token", ""))
    with _engine(cliente).begin() as cx:
        anterior = cx.execute(text("""
            SELECT provider_user_id, refresh_token_cifrado
            FROM "_bot".conexiones_email
            WHERE cliente_id=:cid AND numero=:num AND proveedor='google'
            FOR UPDATE
        """), {"cid": cliente["cliente_id"], "num": numero}).fetchone()
        if not refresh_token and anterior and anterior[0] == provider_user_id:
            refresh_cifrado = anterior[1]
        elif refresh_token:
            refresh_cifrado = _cifrar_token(refresh_token, cliente["cliente_id"], numero)
        else:
            raise RuntimeError("Google no devolvio refresh_token; revoque el acceso anterior e intentelo nuevamente")
        cx.execute(text("""
            INSERT INTO "_bot".conexiones_email
                (cliente_id, numero, proveedor, provider_user_id, correo,
                 refresh_token_cifrado, scopes, estado, terminos_version)
            VALUES (:cid, :num, 'google', :puid, :correo, :token, :scopes,
                    'activa', :terminos)
            ON CONFLICT (cliente_id, numero, proveedor) DO UPDATE SET
                provider_user_id=EXCLUDED.provider_user_id,
                correo=EXCLUDED.correo,
                refresh_token_cifrado=EXCLUDED.refresh_token_cifrado,
                scopes=EXCLUDED.scopes, estado='activa',
                terminos_version=EXCLUDED.terminos_version,
                conectado_en=now(), actualizado_en=now(), revocado_en=NULL
        """), {"cid": cliente["cliente_id"], "num": numero,
                 "puid": provider_user_id, "correo": correo,
                 "token": refresh_cifrado, "scopes": scopes,
                 "terminos": terminos_version or config.APP_TERMS_VERSION})
        cx.execute(text('UPDATE "_bot".oauth_sesiones SET usado_en=now() WHERE state_hash=:hash'),
                   {"hash": _state_hash(state)})
    return OAuthCompletado(cliente, numero, correo)


def _conexion(cliente: dict, numero: str) -> Conexion | None:
    _asegurar_tablas(cliente)
    with _engine(cliente).connect() as cx:
        fila = cx.execute(text("""
            SELECT correo, refresh_token_cifrado, scopes, provider_user_id
            FROM "_bot".conexiones_email
            WHERE cliente_id=:cid AND numero=:num AND proveedor='google'
              AND estado='activa' LIMIT 1
        """), {"cid": cliente["cliente_id"], "num": numero}).fetchone()
    return Conexion(*fila) if fila else None


def desconectar_google(cliente: dict, numero: str) -> bool:
    conexion = _conexion(cliente, numero)
    if not conexion:
        return False
    try:
        refresh = _descifrar_token(conexion.refresh_token_cifrado,
                                   cliente["cliente_id"], numero)
        httpx.post("https://oauth2.googleapis.com/revoke", params={"token": refresh},
                   timeout=config.OAUTH_HTTP_TIMEOUT_SEGUNDOS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Google no confirmo revocacion: %s", cliente.get("cliente_id"), exc)
    with _engine(cliente).begin() as cx:
        cx.execute(text("""
            UPDATE "_bot".conexiones_email
            SET estado='revocada', refresh_token_cifrado='', revocado_en=now(),
                actualizado_en=now()
            WHERE cliente_id=:cid AND numero=:num AND proveedor='google'
        """), {"cid": cliente["cliente_id"], "num": numero})
        cx.execute(text("""
            UPDATE "_bot".envios_email SET estado='cancelado', actualizado_en=now()
            WHERE cliente_id=:cid AND numero=:num AND estado='pendiente'
        """), {"cid": cliente["cliente_id"], "num": numero})
    return True


def es_pedido(texto_usuario: str) -> bool:
    t = _normalizar(texto_usuario)
    return bool(_PEDIDO_RE.search(t) and (_EMAIL_RE.search(t) or _ARCHIVO_RE.search(t)))


def es_generacion_y_correo(texto_usuario: str) -> bool:
    """Detecta "crea el PDF ... y envialo a ..." en un solo turno."""
    t = _normalizar(texto_usuario)
    generar = _GENERAR_RE.search(t)
    if not generar or not _ARCHIVO_RE.search(t) or not _EMAIL_RE.search(t):
        return False
    return bool(_ENVIAR_RE.search(t, generar.end()))


def pregunta_para_generar(texto_usuario: str) -> str:
    """Quita la instruccion de email para que text-to-SQL reciba solo los datos."""
    t = _normalizar(texto_usuario)
    generar = _GENERAR_RE.search(t)
    enviar = _ENVIAR_RE.search(t, generar.end()) if generar else None
    if not enviar:
        return texto_usuario
    pregunta = texto_usuario[:enviar.start()].rstrip(" ,;:-")
    pregunta = re.sub(r"\s+\b(?:y|luego|despues)\s*$", "", pregunta,
                      flags=re.IGNORECASE).rstrip(" ,;:-")
    return pregunta or texto_usuario


def _extension_pedida(texto_usuario: str) -> str:
    t = _normalizar(texto_usuario)
    if "pdf" in t:
        return ".pdf"
    if "excel" in t or "xlsx" in t:
        return ".xlsx"
    if "csv" in t:
        return ".csv"
    if any(x in t for x in ("grafico", "grafica", "imagen", "png")):
        return ".png"
    return ""


def _extraer_campos(texto_usuario: str, nombre_archivo: str) -> tuple[str, str, str]:
    coincidencia = _EMAIL_RE.search(texto_usuario or "")
    destino = coincidencia.group(1).lower() if coincidencia else ""
    asunto = "Reporte solicitado"
    cuerpo = "Hola,\n\nAdjunto el archivo solicitado.\n\nSaludos."
    m_asunto = re.search(
        r"\basunto\s*[:=-]?\s*[\"“]?(.+?)[\"”]?(?=\s+(?:y\s+)?(?:texto|mensaje|cuerpo|agrega|agregue|añade|anade)\b|$)",
        texto_usuario or "", re.IGNORECASE)
    if m_asunto:
        asunto = m_asunto.group(1).strip(" .,:;\"“”")[:200] or asunto
    m_cuerpo = re.search(
        r"\b(?:texto|mensaje|cuerpo|agrega|agregue|añade|anade)\s*[:=-]?\s*[\"“]?(.+?)[\"”]?\s*$",
        texto_usuario or "", re.IGNORECASE)
    if m_cuerpo:
        cuerpo = m_cuerpo.group(1).strip(" \"“”")[:10000] or cuerpo
    if asunto == "Reporte solicitado" and nombre_archivo:
        asunto = "Reporte: " + nombre_archivo.rsplit(".", 1)[0].replace("_", " ")
    return destino, asunto, cuerpo


def guardar_artefactos(cliente: dict, numero: str, adjuntos: list[Adjunto]) -> None:
    if not config.BOT_EMAIL or not adjuntos:
        return
    try:
        _asegurar_tablas(cliente)
        sentencia = text("""
            INSERT INTO "_bot".artefactos
                (cliente_id, numero, nombre, mime, contenido, sha256, expira_en)
            VALUES (:cid, :num, :nombre, :mime, :contenido, :sha,
                    now() + make_interval(hours => :horas))
        """)
        with _engine(cliente).begin() as cx:
            for adj in adjuntos:
                if adj.tamano_mb <= config.BOT_EMAIL_MAX_ADJUNTO_MB:
                    cx.execute(sentencia, {"cid": cliente["cliente_id"], "num": numero,
                        "nombre": adj.nombre, "mime": adj.mime, "contenido": adj.contenido,
                        "sha": hashlib.sha256(adj.contenido).hexdigest(),
                        "horas": config.BOT_EMAIL_ARTEFACTO_TTL_HORAS})
            cx.execute(text('DELETE FROM "_bot".artefactos WHERE expira_en < now()'))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] no se pudieron conservar artefactos: %s", cliente.get("cliente_id"), exc)


def _ultimo_artefacto(cliente: dict, numero: str, extension: str = "") -> Artefacto | None:
    _asegurar_tablas(cliente)
    filtro = " AND lower(nombre) LIKE :ext" if extension else ""
    consulta = text("""
        SELECT id, nombre, mime, contenido FROM "_bot".artefactos
        WHERE cliente_id=:cid AND numero=:num AND expira_en > now()
    """ + filtro + " ORDER BY creado_en DESC, id DESC LIMIT 1")
    params = {"cid": cliente["cliente_id"], "num": numero}
    if extension:
        params["ext"] = "%" + extension.lower()
    with _engine(cliente).connect() as cx:
        fila = cx.execute(consulta, params).fetchone()
    return Artefacto(int(fila[0]), fila[1], fila[2], bytes(fila[3])) if fila else None


def _crear_borrador(cliente: dict, numero: str, conexion: Conexion,
                    artefacto: Artefacto, destinatario: str,
                    asunto: str, cuerpo: str) -> Borrador:
    envio_id = str(uuid.uuid4())
    with _engine(cliente).begin() as cx:
        cx.execute(text("""
            UPDATE "_bot".envios_email SET estado='cancelado', actualizado_en=now()
            WHERE cliente_id=:cid AND numero=:num AND estado='pendiente'
        """), {"cid": cliente["cliente_id"], "num": numero})
        cx.execute(text("""
            INSERT INTO "_bot".envios_email
                (id, cliente_id, numero, artefacto_id, remitente, destinatario,
                 asunto, cuerpo, nombre_archivo, mime)
            VALUES (:id, :cid, :num, :aid, :rem, :dest, :asunto, :cuerpo,
                    :nombre, :mime)
        """), {"id": envio_id, "cid": cliente["cliente_id"], "num": numero,
                 "aid": artefacto.id, "rem": conexion.correo, "dest": destinatario,
                 "asunto": asunto, "cuerpo": cuerpo, "nombre": artefacto.nombre,
                 "mime": artefacto.mime})
    return Borrador(envio_id, conexion.correo, destinatario, asunto, cuerpo, artefacto)


def _pendiente(cliente: dict, numero: str) -> Borrador | None:
    _asegurar_tablas(cliente)
    with _engine(cliente).connect() as cx:
        fila = cx.execute(text("""
            SELECT e.id, e.remitente, e.destinatario, e.asunto, e.cuerpo,
                   a.id, a.nombre, a.mime, a.contenido
            FROM "_bot".envios_email e JOIN "_bot".artefactos a ON a.id=e.artefacto_id
            WHERE e.cliente_id=:cid AND e.numero=:num AND e.estado='pendiente'
              AND e.creado_en > now() - make_interval(mins => :mins)
              AND a.expira_en > now()
            ORDER BY e.creado_en DESC LIMIT 1
        """), {"cid": cliente["cliente_id"], "num": numero,
                 "mins": config.BOT_EMAIL_CONFIRMACION_MINUTOS}).fetchone()
    if not fila:
        return None
    return Borrador(fila[0], fila[1], fila[2], fila[3], fila[4],
                    Artefacto(int(fila[5]), fila[6], fila[7], bytes(fila[8])))


def _reclamar(cliente: dict, borrador: Borrador) -> tuple[bool, str]:
    cid = cliente["cliente_id"]
    with _engine(cliente).begin() as cx:
        cx.execute(text("SELECT pg_advisory_xact_lock(hashtext(:clave))"),
                   {"clave": f"fachavi-email:{cid}:{borrador.remitente}"})
        conteos = cx.execute(text("""
            SELECT count(*) FILTER (WHERE confirmado_en > now() - interval '1 hour'),
                   count(*) FILTER (WHERE confirmado_en > now() - interval '1 day')
            FROM "_bot".envios_email
            WHERE cliente_id=:cid AND remitente=:rem AND confirmado_en IS NOT NULL
        """), {"cid": cid, "rem": borrador.remitente}).fetchone()
        if config.BOT_EMAIL_MAX_POR_HORA > 0 and conteos[0] >= config.BOT_EMAIL_MAX_POR_HORA:
            return False, "hora"
        if config.BOT_EMAIL_MAX_POR_DIA > 0 and conteos[1] >= config.BOT_EMAIL_MAX_POR_DIA:
            return False, "dia"
        resultado = cx.execute(text("""
            UPDATE "_bot".envios_email SET estado='enviando', confirmado_en=now(), actualizado_en=now()
            WHERE id=:id AND estado='pendiente'
        """), {"id": borrador.id})
    return resultado.rowcount == 1, ""


def _access_token(cliente: dict, numero: str, conexion: Conexion) -> str:
    refresh = _descifrar_token(conexion.refresh_token_cifrado, cliente["cliente_id"], numero)
    respuesta = httpx.post("https://oauth2.googleapis.com/token", data={
        "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": config.GOOGLE_OAUTH_CLIENT_SECRET,
        "refresh_token": refresh, "grant_type": "refresh_token",
    }, timeout=config.OAUTH_HTTP_TIMEOUT_SEGUNDOS)
    if respuesta.status_code in (400, 401):
        with _engine(cliente).begin() as cx:
            cx.execute(text("""
                UPDATE "_bot".conexiones_email SET estado='requiere_reconexion', actualizado_en=now()
                WHERE cliente_id=:cid AND numero=:num AND proveedor='google'
            """), {"cid": cliente["cliente_id"], "num": numero})
    respuesta.raise_for_status()
    token = str(respuesta.json().get("access_token", ""))
    if not token:
        raise RuntimeError("Google no devolvio access_token")
    return token


def _mensaje_mime(borrador: Borrador) -> str:
    mensaje = EmailMessage()
    mensaje["From"], mensaje["To"] = borrador.remitente, borrador.destinatario
    mensaje["Subject"] = borrador.asunto
    mensaje.set_content(borrador.cuerpo)
    mime = borrador.artefacto.mime or mimetypes.guess_type(borrador.artefacto.nombre)[0]
    if not mime or "/" not in mime:
        mime = "application/octet-stream"
    maintype, subtype = mime.split("/", 1)
    mensaje.add_attachment(borrador.artefacto.contenido, maintype=maintype,
                           subtype=subtype, filename=borrador.artefacto.nombre)
    return _b64url(mensaje.as_bytes())


def _enviar_google(cliente: dict, numero: str, conexion: Conexion,
                   borrador: Borrador) -> tuple[bool, str, bool]:
    """Devuelve (ok, detalle/id, ambiguo). No reintenta un timeout ambiguo."""
    intento_iniciado = False
    try:
        access = _access_token(cliente, numero, conexion)
        intento_iniciado = True
        respuesta = httpx.post(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            headers={"Authorization": f"Bearer {access}"},
            json={"raw": _mensaje_mime(borrador)},
            timeout=config.BOT_EMAIL_TIMEOUT_SEGUNDOS)
        respuesta.raise_for_status()
        message_id = str(respuesta.json().get("id", ""))
        if not message_id:
            raise RuntimeError("Gmail acepto la solicitud sin devolver message id")
        with _engine(cliente).begin() as cx:
            cx.execute(text("""
                UPDATE "_bot".envios_email SET estado='enviado', provider_id=:pid,
                    error='', actualizado_en=now() WHERE id=:id
            """), {"id": borrador.id, "pid": message_id})
            cx.execute(text("""
                UPDATE "_bot".conexiones_email SET ultimo_uso_en=now(), actualizado_en=now()
                WHERE cliente_id=:cid AND numero=:num AND proveedor='google'
            """), {"cid": cliente["cliente_id"], "num": numero})
        return True, message_id, False
    except httpx.TimeoutException as exc:
        detalle = str(exc)[:500]
        estado = "incierto" if intento_iniciado else "fallido"
        with _engine(cliente).begin() as cx:
            cx.execute(text("""
                UPDATE "_bot".envios_email SET estado=:estado, error=:error,
                    actualizado_en=now() WHERE id=:id
            """), {"id": borrador.id, "estado": estado, "error": detalle})
        return False, detalle, intento_iniciado
    except Exception as exc:  # noqa: BLE001
        detalle = str(exc)[:500]
        with _engine(cliente).begin() as cx:
            cx.execute(text("""
                UPDATE "_bot".envios_email SET estado='fallido', error=:error,
                    actualizado_en=now() WHERE id=:id
            """), {"id": borrador.id, "error": detalle})
        logger.warning("[%s] fallo Gmail %s: %s", cliente.get("cliente_id"), borrador.id, detalle)
        return False, detalle, False


def _cancelar(cliente: dict, borrador: Borrador) -> None:
    with _engine(cliente).begin() as cx:
        cx.execute(text("UPDATE \"_bot\".envios_email SET estado='cancelado', actualizado_en=now() WHERE id=:id AND estado='pendiente'"), {"id": borrador.id})


def _estado_ultimo(cliente: dict, numero: str) -> str | None:
    _asegurar_tablas(cliente)
    with _engine(cliente).connect() as cx:
        fila = cx.execute(text("""
            SELECT remitente, destinatario, nombre_archivo, estado FROM "_bot".envios_email
            WHERE cliente_id=:cid AND numero=:num AND estado <> 'pendiente'
            ORDER BY creado_en DESC LIMIT 1
        """), {"cid": cliente["cliente_id"], "num": numero}).fetchone()
    if not fila:
        return None
    etiquetas = {"enviado": "aceptado y enviado por Gmail",
                 "incierto": "sin confirmacion definitiva; revise la carpeta Enviados",
                 "fallido": "fallido antes de recibir confirmacion de Gmail",
                 "cancelado": "cancelado"}
    return f"Correo desde {fila[0]} a {fila[1]} con {fila[2]}: {etiquetas.get(fila[3], fila[3])}."


def procesar_mensaje(cliente: dict, numero: str, texto_usuario: str) -> Respuesta | None:
    if not config.BOT_EMAIL:
        return None
    normalizado = " ".join((texto_usuario or "").strip().lower().rstrip(".! ").split())
    sin_tildes = _normalizar(normalizado)
    if _CONECTAR_RE.search(sin_tildes):
        enlace = crear_enlace_conexion(cliente, numero)
        if not enlace:
            return Respuesta("La conexión con Google todavía no está configurada en el servidor.")
        return Respuesta(
            "Para conectar su Gmail de forma segura, abra este enlace:\n\n"
            f"{enlace}\n\nEl enlace vence en {config.OAUTH_ENLACE_TTL_MINUTOS} minutos y solo puede usarse una vez. "
            "Google solicitará únicamente permiso para enviar correos; el bot no podrá leer su bandeja.")
    if _DESCONECTAR_RE.search(sin_tildes):
        try:
            desconectado = desconectar_google(cliente, numero)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] no se pudo desconectar Google: %s", cliente.get("cliente_id"), exc)
            return Respuesta("No pude desconectar el correo en este momento. Inténtelo nuevamente.")
        return Respuesta("Listo. Desconecté su cuenta de Google y eliminé la credencial guardada."
                         if desconectado else "No hay una cuenta de Google conectada a este número.")
    if _ESTADO_CONEXION_RE.search(sin_tildes):
        try:
            conexion = _conexion(cliente, numero)
        except Exception:  # noqa: BLE001
            conexion = None
        return Respuesta(f"La cuenta conectada a este número es {conexion.correo}."
                         if conexion else "Este número no tiene una cuenta de Google conectada.")
    pendiente = None
    if normalizado in (_CONFIRMAR | _CANCELAR):
        try:
            pendiente = _pendiente(cliente, numero)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] borrador email no disponible: %s", cliente.get("cliente_id"), exc)
    if pendiente and normalizado in _CANCELAR:
        _cancelar(cliente, pendiente)
        return Respuesta("Entendido. Cancelé el correo; no se envió nada.")
    if pendiente and normalizado in _CONFIRMAR:
        conexion = _conexion(cliente, numero)
        if not conexion or conexion.correo != pendiente.remitente:
            return Respuesta("La cuenta de Google ya no está conectada. Escriba *conectar mi correo* para continuar.")
        reclamado, limite = _reclamar(cliente, pendiente)
        if not reclamado:
            if limite:
                periodo = "hora" if limite == "hora" else "día"
                return Respuesta(f"Se alcanzó el límite de correos por {periodo} para esta cuenta. No envié nada.")
            return Respuesta("Ese correo ya fue procesado; no lo envié nuevamente.")
        ok, _, ambiguo = _enviar_google(cliente, numero, conexion, pendiente)
        if ok:
            return Respuesta(f"Listo. Gmail aceptó y envió el correo desde {conexion.correo} a {pendiente.destinatario} con el archivo {pendiente.artefacto.nombre}.")
        if ambiguo:
            return Respuesta("Gmail no devolvió una confirmación definitiva. Para evitar duplicados no lo envié nuevamente; revise la carpeta Enviados de su cuenta.")
        return Respuesta("No pude enviar el correo. No se creó un segundo envío; revise la conexión de Google.")
    if _ESTADO_ENVIO_RE.search(sin_tildes):
        try:
            estado = _estado_ultimo(cliente, numero)
        except Exception:  # noqa: BLE001
            estado = None
        return Respuesta(estado or "Todavía no hay correos enviados desde esta conversación.")
    if not es_pedido(texto_usuario):
        return None
    try:
        conexion = _conexion(cliente, numero)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] conexion email no disponible: %s", cliente.get("cliente_id"), exc)
        conexion = None
    if not conexion:
        return Respuesta("Este número no tiene un Gmail conectado. Escriba *conectar mi correo* para autorizarlo.")
    if not _EMAIL_RE.search(texto_usuario or ""):
        return Respuesta("¿A qué dirección de correo desea enviarlo? Escríbala completa, por favor.")
    try:
        artefacto = _ultimo_artefacto(cliente, numero, _extension_pedida(texto_usuario))
        if not artefacto:
            return Respuesta("No encuentro un archivo reciente de ese tipo en esta conversación. Pídame primero que genere el PDF, Excel o gráfico.")
        destinatario, asunto, cuerpo = _extraer_campos(texto_usuario, artefacto.nombre)
        borrador = _crear_borrador(cliente, numero, conexion, artefacto,
                                   destinatario, asunto, cuerpo)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] no se pudo preparar correo: %s", cliente.get("cliente_id"), exc)
        return Respuesta("No pude preparar el correo en este momento. Inténtelo nuevamente.")
    cuerpo_vista = borrador.cuerpo.replace("\n", " ")
    if len(cuerpo_vista) > 240:
        cuerpo_vista = cuerpo_vista[:237] + "..."
    return Respuesta(
        "Confirme el envío:\n\n"
        f"De: {borrador.remitente}\nPara: {borrador.destinatario}\n"
        f"Asunto: {borrador.asunto}\nArchivo: {borrador.artefacto.nombre}\n"
        f"Mensaje: {cuerpo_vista}\n\nResponda *sí* para enviar o *cancelar* para descartarlo.")
