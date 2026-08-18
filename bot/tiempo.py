"""Reloj local compartido por el bot y las consultas al warehouse."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import config


@lru_cache(maxsize=4)
def _zona(nombre: str) -> ZoneInfo:
    try:
        return ZoneInfo(nombre)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            f"Zona horaria invalida en BOT_TIMEZONE: {nombre!r}."
        ) from exc


def nombre_zona() -> str:
    return str(
        getattr(config, "BOT_TIMEZONE", "America/Costa_Rica") or ""
    ).strip()


def fecha_local(ahora: datetime | None = None) -> date:
    """Fecha civil del negocio, independiente de la zona UTC de Render."""
    zona = _zona(nombre_zona())
    instante = ahora.astimezone(zona) if ahora is not None else datetime.now(zona)
    return instante.date()
