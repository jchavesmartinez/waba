"""
Ingesta de los Sheets de DATOS -> DuckDB, por cliente.

Cada cliente tiene su propio spreadsheet_id (sale del registro). Este modulo
mantiene un cache de conexiones DuckDB, una por spreadsheet_id, cada una con su
TTL. Asi el bot consulta los datos del cliente correcto sin mezclar.

Cada pestania del Sheet de datos se vuelve una tabla de DuckDB (multiples tablas).
"""

import time
import logging
import re

import duckdb
import pandas as pd

import config
from gclient import abrir_libro

logger = logging.getLogger("fachavi.sheets")

# Cache por spreadsheet_id: {sid: {"ts": epoch, "con": duckdb_con, "schema": str}}
_cache = {}


def _clean_name(name: str) -> str:
    n = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip().lower())
    if not n or n[0].isdigit():
        n = "t_" + n
    return n


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        serie = df[col]
        limpio = (
            serie.astype(str)
            .str.replace(r"[,\s₡$]", "", regex=True)
            .str.replace(r"^$", "nan", regex=True)
        )
        num = pd.to_numeric(limpio, errors="coerce")
        if num.notna().sum() >= max(1, int(0.8 * len(serie))):
            df[col] = num
            continue
        fecha = pd.to_datetime(serie, errors="coerce", dayfirst=True)
        if fecha.notna().sum() >= max(1, int(0.8 * len(serie))):
            df[col] = fecha
            continue
        df[col] = serie.astype(str)
    return df


def _build(spreadsheet_id: str):
    """Lee todas las pestanias del Sheet de datos de un cliente -> DuckDB."""
    libro = abrir_libro(spreadsheet_id)
    con = duckdb.connect(database=":memory:")
    schema_parts = []

    for ws in libro.worksheets():
        registros = ws.get_all_values()
        if not registros or len(registros) < 2:
            continue
        headers = registros[0]
        df = pd.DataFrame(registros[1:], columns=headers)
        df.columns = [_clean_name(c) for c in df.columns]
        df = _coerce_types(df)

        tabla = _clean_name(ws.title)
        con.register(f"_df_{tabla}", df)
        con.execute(f"CREATE TABLE {tabla} AS SELECT * FROM _df_{tabla}")
        con.unregister(f"_df_{tabla}")

        cols = con.execute(f"DESCRIBE {tabla}").fetchall()
        cols_txt = ", ".join(f"{c[0]} ({c[1]})" for c in cols)
        muestra = con.execute(f"SELECT * FROM {tabla} LIMIT 2").fetchall()
        schema_parts.append(
            f"Tabla: {tabla}\nColumnas: {cols_txt}\nEjemplo de filas: {muestra}"
        )
        logger.info("[%s] tabla %s (%d filas)", spreadsheet_id[:8], tabla, len(df))

    if not schema_parts:
        raise RuntimeError("El Sheet de datos no tiene pestanias con datos.")

    return con, "\n\n".join(schema_parts)


def get_connection(spreadsheet_id: str):
    """Devuelve (con, schema) para el Sheet de datos de un cliente, con cache TTL."""
    ahora = time.time()
    entry = _cache.get(spreadsheet_id)
    if entry and (ahora - entry["ts"]) < config.DATA_CACHE_TTL:
        return entry["con"], entry["schema"]

    if entry and entry.get("con") is not None:
        try:
            entry["con"].close()
        except Exception:  # noqa: BLE001
            pass

    con, schema = _build(spreadsheet_id)
    _cache[spreadsheet_id] = {"ts": ahora, "con": con, "schema": schema}
    return con, schema
