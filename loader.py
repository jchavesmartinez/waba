"""
Orquestador de datos por cliente.

Reemplaza al viejo sheets.py. En vez de leer UN Google Sheet, arma UNA sola
conexion DuckDB combinando TODAS las fuentes ACTIVAS del cliente. Cada fuente
carga sus tablas en la misma conexion, asi se pueden cruzar datos entre fuentes
(p.ej. ventas de Sheets con inventario de Postgres) en la misma consulta.

Prende/apaga una fuente = cambiar su 'activo' en el registro. Sin tocar codigo.

Devuelve (con, schema, catalogo), cacheado por cliente_id con TTL.
"""

import time
import logging

import duckdb

import config
from sources import crear_fuente

logger = logging.getLogger("fachavi.loader")

# Cache por cliente_id: {cid: {"ts", "con", "schema", "catalogo"}}
_cache = {}


def _build(cliente: dict):
    """Crea la DuckDB del cliente cargando cada fuente activa."""
    cid = cliente.get("cliente_id", "?")
    fuentes = [f for f in cliente.get("fuentes", []) if f.get("activo", True)]

    if not fuentes:
        raise RuntimeError("SIN_FUENTES")

    con = duckdb.connect(database=":memory:")
    schema_parts = []
    catalogo_parts = []
    tablas_totales = []
    errores = []

    for f in fuentes:
        fuente_id = f.get("fuente_id", "fuente")
        tipo = f.get("tipo", "")
        cfg = f.get("config", {})
        try:
            fuente = crear_fuente(tipo, fuente_id, cfg)
            frag = fuente.cargar(con)
            if frag.schema:
                schema_parts.append(frag.schema)
            if frag.catalogo:
                catalogo_parts.append(frag.catalogo)
            tablas_totales += frag.tablas
            logger.info("[%s] fuente '%s' (%s): %d tablas",
                        cid, fuente_id, tipo, len(frag.tablas))
        except Exception as e:  # noqa: BLE001
            # Una fuente rota no debe tumbar al resto del cliente.
            logger.exception("[%s] fuente '%s' (%s) fallo: %s", cid, fuente_id, tipo, e)
            errores.append(fuente_id)

    if not tablas_totales:
        # Ninguna fuente aporto datos utilizables.
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError("SIN_DATOS")

    schema = "\n\n".join(schema_parts)
    catalogo = "\n".join(catalogo_parts)  # puede quedar "" si nadie documento
    return con, schema, catalogo


def get_data(cliente: dict):
    """
    Devuelve (con, schema, catalogo) para el cliente, con cache TTL por
    cliente_id. 'catalogo' puede ser "" si ninguna fuente activa esta documentada.
    """
    cid = cliente.get("cliente_id", "?")
    ahora = time.time()
    entry = _cache.get(cid)
    if entry and (ahora - entry["ts"]) < config.DATA_CACHE_TTL:
        return entry["con"], entry["schema"], entry["catalogo"]

    if entry and entry.get("con") is not None:
        try:
            entry["con"].close()
        except Exception:  # noqa: BLE001
            pass

    con, schema, catalogo = _build(cliente)
    _cache[cid] = {"ts": ahora, "con": con, "schema": schema, "catalogo": catalogo}
    return con, schema, catalogo


def invalidar(cliente_id: str = None):
    """Fuerza recarga: borra la cache de un cliente (o de todos si es None)."""
    if cliente_id is None:
        _cache.clear()
    else:
        _cache.pop(cliente_id, None)
