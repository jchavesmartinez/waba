"""
Ingesta de Google Sheets -> DuckDB.

- Usa un service account (privado, no expone la hoja).
- Cada pestania (worksheet) del libro se convierte en una TABLA de DuckDB.
  Asi ya queda listo para "multiples tablas": agregas pestanias y aparecen solas.
- Cachea los datos en memoria por CACHE_TTL_SECONDS para no releer la hoja en
  cada mensaje (los datos cambian todo el dia, pero 60s de frescura alcanza).
- Infiera tipos (numeros y fechas) para que el SQL pueda sumar, filtrar por
  fecha, etc.
"""

import json
import time
import logging
import re

import duckdb
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

import config

logger = logging.getLogger("fachavi-sql.sheets")

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Cache simple en memoria: {"ts": <epoch>, "con": <duckdb con>, "schema": <str>}
_cache = {"ts": 0.0, "con": None, "schema": ""}


def _clean_table_name(name: str) -> str:
    """Convierte el nombre de una pestania en un nombre de tabla SQL valido."""
    n = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip().lower())
    if not n or n[0].isdigit():
        n = "t_" + n
    return n


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Los valores de Sheets llegan como texto. Intentamos convertir cada columna
    a numero o fecha; si no se puede, se queda como texto.
    """
    for col in df.columns:
        serie = df[col]

        # Intento numerico (quita separadores de miles y simbolos de moneda comunes)
        limpio = (
            serie.astype(str)
            .str.replace(r"[,\s₡$]", "", regex=True)
            .str.replace(r"^$", "nan", regex=True)
        )
        num = pd.to_numeric(limpio, errors="coerce")
        if num.notna().sum() >= max(1, int(0.8 * len(serie))):
            df[col] = num
            continue

        # Intento de fecha
        fecha = pd.to_datetime(serie, errors="coerce", dayfirst=True)
        if fecha.notna().sum() >= max(1, int(0.8 * len(serie))):
            df[col] = fecha
            continue

        # Se queda como texto
        df[col] = serie.astype(str)
    return df


def _build_connection():
    """Lee todas las pestanias del libro y arma una conexion DuckDB en memoria."""
    if not config.GOOGLE_CREDENTIALS_JSON or not config.SPREADSHEET_ID:
        raise RuntimeError("Faltan GOOGLE_CREDENTIALS_JSON o SPREADSHEET_ID.")

    creds_info = json.loads(config.GOOGLE_CREDENTIALS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=_SCOPES)
    gc = gspread.authorize(creds)
    libro = gc.open_by_key(config.SPREADSHEET_ID)

    con = duckdb.connect(database=":memory:")
    schema_parts = []

    for ws in libro.worksheets():
        registros = ws.get_all_values()
        if not registros or len(registros) < 2:
            continue  # pestania vacia o solo encabezado

        headers = registros[0]
        filas = registros[1:]
        df = pd.DataFrame(filas, columns=headers)
        # Normaliza nombres de columna
        df.columns = [_clean_table_name(c) for c in df.columns]
        df = _coerce_types(df)

        tabla = _clean_table_name(ws.title)
        con.register(f"_df_{tabla}", df)
        con.execute(f"CREATE TABLE {tabla} AS SELECT * FROM _df_{tabla}")
        con.unregister(f"_df_{tabla}")

        # Descripcion del esquema para el prompt de Claude
        cols_desc = con.execute(f"DESCRIBE {tabla}").fetchall()
        cols_txt = ", ".join(f"{c[0]} ({c[1]})" for c in cols_desc)
        muestra = con.execute(f"SELECT * FROM {tabla} LIMIT 2").fetchall()
        schema_parts.append(
            f"Tabla: {tabla}\nColumnas: {cols_txt}\nEjemplo de filas: {muestra}"
        )
        logger.info("Tabla cargada: %s (%d filas)", tabla, len(df))

    if not schema_parts:
        raise RuntimeError("No se encontraron tablas con datos en el libro.")

    schema = "\n\n".join(schema_parts)
    return con, schema


def get_connection():
    """
    Devuelve (con, schema) usando cache. Si el cache vencio, relee la hoja.
    """
    ahora = time.time()
    if _cache["con"] is not None and (ahora - _cache["ts"]) < config.CACHE_TTL_SECONDS:
        return _cache["con"], _cache["schema"]

    # Cerrar conexion vieja si habia
    if _cache["con"] is not None:
        try:
            _cache["con"].close()
        except Exception:  # noqa: BLE001
            pass

    con, schema = _build_connection()
    _cache.update({"ts": ahora, "con": con, "schema": schema})
    return con, schema
