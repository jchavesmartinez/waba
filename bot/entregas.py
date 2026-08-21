"""Persistencia y máquina de estados para adjuntos salientes de WhatsApp.

La respuesta HTTP del endpoint /messages solo confirma que Meta aceptó el
mensaje. La entrega real llega después, en ``value.statuses[]`` del webhook.
Este módulo guarda ambos momentos en Neon y reclama de forma transaccional un
único reintento cuando Meta informa ``failed``.

Todas las funciones son best-effort: un problema de telemetría nunca debe
impedir que el bot intente enviar la respuesta al usuario.
"""

from dataclasses import dataclass
import json
import logging
import threading

from sqlalchemy import create_engine, text

import config

logger = logging.getLogger("fachavi.bot.entregas")

_engines: dict = {}
_tablas_listas: set = set()
_lock = threading.Lock()

_ESTADOS = {"aceptado": 0, "sent": 1, "delivered": 2, "read": 3}

_DDL = (
    'CREATE SCHEMA IF NOT EXISTS "_bot"',
    """
    CREATE TABLE IF NOT EXISTS "_bot".envios_adjuntos (
        message_id             TEXT PRIMARY KEY,
        cliente_id            TEXT NOT NULL DEFAULT '',
        numero                 TEXT NOT NULL DEFAULT '',
        phone_number_id        TEXT NOT NULL DEFAULT '',
        media_id               TEXT NOT NULL DEFAULT '',
        tipo                   TEXT NOT NULL DEFAULT '',
        nombre                 TEXT NOT NULL DEFAULT '',
        mime                   TEXT NOT NULL DEFAULT '',
        estado                 TEXT NOT NULL DEFAULT 'aceptado',
        intentos               INTEGER NOT NULL DEFAULT 0,
        reintento_de           TEXT NOT NULL DEFAULT '',
        reintento_message_id   TEXT NOT NULL DEFAULT '',
        reintento_solicitado_en TIMESTAMPTZ,
        fallo_notificado_en    TIMESTAMPTZ,
        error                  TEXT NOT NULL DEFAULT '',
        creado_en              TIMESTAMPTZ NOT NULL DEFAULT now(),
        actualizado_en         TIMESTAMPTZ NOT NULL DEFAULT now(),
        entregado_en           TIMESTAMPTZ,
        leido_en               TIMESTAMPTZ
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_envios_adjuntos_cliente_fecha
        ON "_bot".envios_adjuntos (cliente_id, creado_en DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_envios_adjuntos_estado
        ON "_bot".envios_adjuntos (estado, actualizado_en DESC)
    """,
)


@dataclass(frozen=True)
class Reintento:
    message_id: str
    numero: str
    phone_number_id: str
    media_id: str
    tipo: str
    nombre: str
    mime: str
    intentos: int


@dataclass(frozen=True)
class ResultadoEstado:
    reintento: Reintento | None = None
    fallo_final: bool = False
    numero: str = ""
    nombre: str = ""


def _engine(cliente: dict):
    dsn = config.dsn_de_cliente(cliente)
    if dsn not in _engines:
        _engines[dsn] = create_engine(dsn, pool_pre_ping=True)
        logger.info("engine de entregas abierto para '%s'",
                    cliente.get("cliente_id"))
    return _engines[dsn]


def _asegurar_tabla(cliente: dict) -> None:
    dsn = config.dsn_de_cliente(cliente)
    if dsn in _tablas_listas:
        return
    with _lock:
        if dsn in _tablas_listas:
            return
        with _engine(cliente).begin() as cx:
            cx.execute(text(
                "SELECT pg_advisory_xact_lock("
                "hashtext('fachavi_bot_envios_adjuntos'))"
            ))
            for sentencia in _DDL:
                cx.execute(text(sentencia))
            existe = cx.execute(text(
                "SELECT to_regclass('\"_bot\".envios_adjuntos')"
            )).scalar()
        if not existe:
            raise RuntimeError("no se pudo crear _bot.envios_adjuntos")
        _tablas_listas.add(dsn)
        logger.info("tabla de seguimiento de adjuntos lista para '%s'",
                    cliente.get("cliente_id"))


def _error_texto(estado: dict) -> str:
    errores = estado.get("errors") or []
    if not errores:
        return ""
    try:
        return json.dumps(errores, ensure_ascii=False, default=str)[:4000]
    except Exception:  # noqa: BLE001
        return str(errores)[:4000]


def _fila_a_reintento(fila) -> Reintento:
    return Reintento(
        message_id=fila[0], numero=fila[1], phone_number_id=fila[2],
        media_id=fila[3], tipo=fila[4], nombre=fila[5], mime=fila[6],
        intentos=int(fila[7] or 0),
    )


def _reclamar_si_corresponde(cx, message_id: str):
    """Marca el reintento dentro del lock de fila y devuelve sus datos."""
    fila = cx.execute(text(
        """
        SELECT message_id, numero, phone_number_id, media_id, tipo, nombre,
               mime, intentos, estado, reintento_solicitado_en
        FROM "_bot".envios_adjuntos
        WHERE message_id = :mid
        FOR UPDATE
        """
    ), {"mid": message_id}).fetchone()
    if not fila:
        return None
    max_reintentos = int(getattr(
        config, "BOT_ADJUNTO_REINTENTOS_ENTREGA", 1,
    ))
    metadata_lista = all((fila[1], fila[2], fila[3], fila[4]))
    if (fila[8] != "failed" or not metadata_lista
            or int(fila[7] or 0) > max_reintentos or fila[9] is not None):
        return None
    cx.execute(text(
        """
        UPDATE "_bot".envios_adjuntos
        SET reintento_solicitado_en = now(), actualizado_en = now()
        WHERE message_id = :mid
        """
    ), {"mid": message_id})
    return _fila_a_reintento(fila)


def registrar(cliente: dict, numero: str, phone_number_id: str,
              message_id: str, media_id: str, tipo: str, nombre: str,
              mime: str, intentos: int = 1,
              reintento_de: str = "") -> Reintento | None:
    """Registra un envío aceptado y devuelve un fallo temprano reclamado."""
    if not message_id:
        return None
    try:
        _asegurar_tabla(cliente)
        with _engine(cliente).begin() as cx:
            cx.execute(text(
                """
                INSERT INTO "_bot".envios_adjuntos
                    (message_id, cliente_id, numero, phone_number_id, media_id,
                     tipo, nombre, mime, estado, intentos, reintento_de)
                VALUES
                    (:mid, :cid, :num, :pnid, :media, :tipo, :nombre, :mime,
                     'aceptado', :intentos, :padre)
                ON CONFLICT (message_id) DO UPDATE SET
                    cliente_id = EXCLUDED.cliente_id,
                    numero = EXCLUDED.numero,
                    phone_number_id = EXCLUDED.phone_number_id,
                    media_id = EXCLUDED.media_id,
                    tipo = EXCLUDED.tipo,
                    nombre = EXCLUDED.nombre,
                    mime = EXCLUDED.mime,
                    intentos = EXCLUDED.intentos,
                    reintento_de = EXCLUDED.reintento_de,
                    actualizado_en = now()
                """
            ), {
                "mid": message_id, "cid": cliente.get("cliente_id", ""),
                "num": numero, "pnid": phone_number_id,
                "media": media_id, "tipo": tipo, "nombre": nombre,
                "mime": mime, "intentos": intentos, "padre": reintento_de,
            })
            ttl = int(getattr(config, "BOT_ADJUNTO_ENTREGAS_TTL_DIAS", 30))
            if ttl > 0:
                cx.execute(text(
                    """
                    DELETE FROM "_bot".envios_adjuntos
                    WHERE cliente_id = :cid
                      AND creado_en < now() - make_interval(days => :dias)
                    """
                ), {"cid": cliente.get("cliente_id", ""), "dias": ttl})
            pendiente = _reclamar_si_corresponde(cx, message_id)
        logger.info("adjunto '%s' aceptado por Meta: %s", nombre, message_id)
        return pendiente
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo registrar el adjunto %s: %s",
                       cliente.get("cliente_id"), message_id, e)
        return None


def actualizar_estado(cliente: dict, estado: dict) -> ResultadoEstado | None:
    """Persiste un status y reclama un reintento o aviso final, una sola vez."""
    message_id = str(estado.get("id", "") or "").strip()
    nuevo = str(estado.get("status", "") or "").strip().lower()
    if not message_id or nuevo not in {"sent", "delivered", "read", "failed"}:
        return None
    error = _error_texto(estado)
    try:
        _asegurar_tabla(cliente)
        with _engine(cliente).begin() as cx:
            fila = cx.execute(text(
                """
                SELECT estado FROM "_bot".envios_adjuntos
                WHERE message_id = :mid FOR UPDATE
                """
            ), {"mid": message_id}).fetchone()
            if not fila:
                # Meta manda statuses para texto y archivos por el mismo
                # webhook. Si el ID no está registrado, no es un adjunto que
                # este módulo deba seguir.
                return None
            else:
                actual = str(fila[0] or "aceptado")
                # No degradar read/delivered por webhooks atrasados. failed es
                # terminal salvo que ya conste una entrega real.
                debe_actualizar = (
                    nuevo == "failed" and actual not in {"delivered", "read"}
                ) or (
                    nuevo != "failed" and actual != "failed"
                    and _ESTADOS.get(nuevo, -1) >= _ESTADOS.get(actual, -1)
                )
                if debe_actualizar:
                    cx.execute(text(
                        """
                        UPDATE "_bot".envios_adjuntos
                        SET estado = :estado,
                            error = CASE WHEN :error = '' THEN error ELSE :error END,
                            actualizado_en = now(),
                            entregado_en = CASE WHEN :estado IN ('delivered','read')
                                THEN COALESCE(entregado_en, now()) ELSE entregado_en END,
                            leido_en = CASE WHEN :estado = 'read'
                                THEN COALESCE(leido_en, now()) ELSE leido_en END
                        WHERE message_id = :mid
                        """
                    ), {"mid": message_id, "estado": nuevo, "error": error})
            pendiente = _reclamar_si_corresponde(cx, message_id)
            final = None
            if nuevo == "failed" and pendiente is None:
                final = cx.execute(text(
                    """
                    SELECT numero, nombre, intentos, fallo_notificado_en
                    FROM "_bot".envios_adjuntos
                    WHERE message_id = :mid
                    FOR UPDATE
                    """
                ), {"mid": message_id}).fetchone()
                max_reintentos = int(getattr(
                    config, "BOT_ADJUNTO_REINTENTOS_ENTREGA", 1,
                ))
                if (not final or not final[0]
                        or int(final[2] or 0) <= max_reintentos
                        or final[3] is not None):
                    final = None
                else:
                    cx.execute(text(
                        """
                        UPDATE "_bot".envios_adjuntos
                        SET fallo_notificado_en = now(), actualizado_en = now()
                        WHERE message_id = :mid
                        """
                    ), {"mid": message_id})
        logger.info("estado de adjunto %s: %s", message_id, nuevo)
        return ResultadoEstado(
            reintento=pendiente,
            fallo_final=bool(final),
            numero=str(final[0]) if final else "",
            nombre=str(final[1]) if final else "",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo guardar estado %s de %s: %s",
                       cliente.get("cliente_id"), nuevo, message_id, e)
        return None


def vincular_reintento(cliente: dict, original_id: str,
                       nuevo_id: str, error: str = "") -> None:
    """Deja trazabilidad del segundo mensaje o del fallo al reintentarlo."""
    try:
        _asegurar_tabla(cliente)
        with _engine(cliente).begin() as cx:
            cx.execute(text(
                """
                UPDATE "_bot".envios_adjuntos
                SET reintento_message_id = :nuevo,
                    error = CASE WHEN :error = '' THEN error ELSE :error END,
                    actualizado_en = now()
                WHERE message_id = :original
                """
            ), {"original": original_id, "nuevo": nuevo_id,
                "error": str(error or "")[:4000]})
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo vincular reintento de %s: %s",
                       cliente.get("cliente_id"), original_id, e)
