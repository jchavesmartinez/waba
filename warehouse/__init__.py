"""
Registro de destinos (warehouses) + factory.

Mismo patron que sources/: el tipo se elige por variable de entorno y el job de
sync no sabe cual esta usando.

    WAREHOUSE_TIPO=duckdb    WAREHOUSE_DSN=/data/fachavi.duckdb
    WAREHOUSE_TIPO=duckdb    WAREHOUSE_DSN=md:fachavi?motherduck_token=XXX
    WAREHOUSE_TIPO=postgres  WAREHOUSE_DSN=postgresql://...
"""

import logging

from .base import Destino, Corrida, nombre_esquema, nombre_tabla, agregar_trazabilidad

logger = logging.getLogger("fachavi.warehouse")

_REGISTRO = {}


def registrar(cls):
    if not getattr(cls, "tipo", None) or cls.tipo == "base":
        raise ValueError(f"El destino {cls} debe definir un 'tipo' propio.")
    _REGISTRO[cls.tipo] = cls
    return cls


def crear_destino(tipo: str, dsn: str = "") -> Destino:
    tipo = (tipo or "").strip().lower()

    # Guarda anti-error-silencioso: si el DSN es de Postgres pero el tipo quedo
    # en 'duckdb' (el default, porque falto definir WAREHOUSE_TIPO), la ingesta
    # escribiria a un archivo local efimero y reportaria "OK" sin que nada
    # llegue al warehouse de verdad. Mejor fallar fuerte y temprano.
    if dsn.startswith(("postgres://", "postgresql://")) and tipo != "postgres":
        raise RuntimeError(
            f"WAREHOUSE_DSN es de Postgres pero WAREHOUSE_TIPO='{tipo}'. "
            "Defini WAREHOUSE_TIPO=postgres (probablemente falta esa variable "
            "de entorno y se esta usando el default 'duckdb')."
        )
    if tipo == "postgres" and not dsn:
        raise RuntimeError("WAREHOUSE_TIPO=postgres pero falta WAREHOUSE_DSN.")

    if tipo not in _REGISTRO:
        raise RuntimeError(
            f"Tipo de warehouse desconocido: '{tipo}'. Disponibles: {tipos_disponibles()}"
        )
    return _REGISTRO[tipo](dsn)


def tipos_disponibles() -> list:
    return sorted(_REGISTRO)


# DuckDB es el destino por defecto de la INGESTA, pero el BOT (que solo lee de
# Neon) no lo necesita. Se registra si esta instalado; asi importar warehouse.*
# desde el bot no obliga a traer duckdb.
try:  # noqa: SIM105
    from .duckdb_dest import DuckDBDestino  # noqa: E402
    registrar(DuckDBDestino)
except Exception as e:  # noqa: BLE001
    logger.info("Destino 'duckdb' no disponible en este entorno: %s", e)

# Postgres es opcional: necesita sqlalchemy + driver instalados.
try:  # noqa: SIM105
    from .postgres_dest import PostgresDestino  # noqa: E402
    registrar(PostgresDestino)
except Exception as e:  # noqa: BLE001
    logger.info("Destino 'postgres' no disponible (falta sqlalchemy/driver): %s", e)

logger.info("Destinos registrados: %s", tipos_disponibles())
