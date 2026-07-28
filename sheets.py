"""
Ingesta de los Sheets de DATOS -> DuckDB, por cliente, + lectura del CATALOGO.

Cada Sheet de datos de un cliente tiene:
  - Una o mas pestanias de DATOS (ventas, inventario, etc.) -> tablas consultables.
  - Una pestania obligatoria '_catalogo' con la metadata/governance.

Las pestanias que empiezan con '_' NO se cargan como tablas consultables
(son metadata, no datos). El catalogo se lee aparte y se usa para responder
preguntas de governance.

Estructura esperada de la pestania '_catalogo' (primera fila = encabezados):
  tabla | columna | descripcion | sistema_origen | frecuencia | dueño
Una fila con columna = '*' describe la tabla completa.
"""

import time
import logging
import re

import duckdb
import pandas as pd

import config
from gclient import abrir_libro

logger = logging.getLogger("fachavi.sheets")

CATALOGO_SHEET = "_catalogo"

# Cache por spreadsheet_id: {sid: {"ts", "con", "schema", "catalogo"}}
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


def _leer_catalogo(libro) -> str:
    """
    Lee la pestania '_catalogo' y arma un texto de metadata para el prompt.
    Lanza excepcion si no existe (catalogo obligatorio).
    """
    try:
        ws = libro.worksheet(CATALOGO_SHEET)
    except Exception:  # noqa: BLE001
        raise RuntimeError("SIN_CATALOGO")

    filas = ws.get_all_records()
    if not filas:
        raise RuntimeError("SIN_CATALOGO")

    lineas = []
    for f in filas:
        # Normaliza claves (quita espacios/tabs invisibles en encabezados)
        f = { str(k).strip().lower(): str(v).strip() for k, v in f.items() }
        tabla = f.get("tabla", "")
        columna = f.get("columna", "")
        desc = f.get("descripcion", "")
        sistema = f.get("sistema_origen", "")
        frec = f.get("frecuencia", "")
        dueno = f.get("dueño", "") or f.get("dueno", "")

        if columna == "*" or not columna:
            lineas.append(
                f"Tabla '{tabla}': {desc}. Sistema de origen: {sistema}. "
                f"Frecuencia de actualizacion: {frec}. Dueño del dato: {dueno}."
            )
        else:
            lineas.append(
                f"Columna '{columna}' (tabla '{tabla}'): {desc}. "
                f"Origen: {sistema}. Actualizacion: {frec}. Dueño: {dueno}."
            )
    return "\n".join(lineas)


def _build(spreadsheet_id: str):
    """Lee pestanias de datos -> DuckDB, y el catalogo aparte."""
    libro = abrir_libro(spreadsheet_id)
    con = duckdb.connect(database=":memory:")
    schema_parts = []

    for ws in libro.worksheets():
        # Saltar pestanias de metadata (empiezan con '_')
        if ws.title.startswith("_"):
            continue

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

    # Catalogo obligatorio
    catalogo = _leer_catalogo(libro)

    return con, "\n\n".join(schema_parts), catalogo


def get_data(spreadsheet_id: str):
    """
    Devuelve (con, schema, catalogo) para el Sheet de datos de un cliente,
    con cache TTL. 'catalogo' es el texto de metadata/governance.
    """
    ahora = time.time()
    entry = _cache.get(spreadsheet_id)
    if entry and (ahora - entry["ts"]) < config.DATA_CACHE_TTL:
        return entry["con"], entry["schema"], entry["catalogo"]

    if entry and entry.get("con") is not None:
        try:
            entry["con"].close()
        except Exception:  # noqa: BLE001
            pass

    con, schema, catalogo = _build(spreadsheet_id)
    _cache[spreadsheet_id] = {
        "ts": ahora, "con": con, "schema": schema, "catalogo": catalogo,
    }
    return con, schema, catalogo
