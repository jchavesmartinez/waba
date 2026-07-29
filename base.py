"""
Contrato base de una FUENTE de datos + helpers compartidos.

Una fuente es cualquier cosa capaz de:
  1. Cargar sus tablas dentro de una conexion DuckDB compartida.
  2. Describir esas tablas (schema) para el prompt de generacion de SQL.
  3. Aportar metadata/governance (catalogo) de esas tablas.

El resto del bot (nl2sql) NO sabe de donde salen los datos: solo ve una
conexion DuckDB, un texto de schema y un texto de catalogo. Por eso agregar
una fuente nueva = escribir una subclase de `Source`; no se toca el pipeline.
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import duckdb
import pandas as pd

logger = logging.getLogger("fachavi.sources")


# --------------------------------------------------------------------------
# Lo que una fuente devuelve tras cargarse
# --------------------------------------------------------------------------

@dataclass
class Fragmento:
    """Aporte de una fuente al pipeline, ya cargada en la DuckDB compartida."""
    schema: str = ""                       # descripcion de tablas para el prompt SQL
    catalogo: str = ""                     # metadata/governance para el prompt de governance
    tablas: List[str] = field(default_factory=list)  # nombres finales de las tablas cargadas


# --------------------------------------------------------------------------
# Contrato: toda fuente hereda de aca
# --------------------------------------------------------------------------

class Source(ABC):
    """Clase base de una fuente de datos consultable."""

    # Identificador del TIPO de fuente. Debe coincidir con la columna 'tipo'
    # del registro (p.ej. 'google_sheets', 'csv_url', 'postgres').
    tipo: str = "base"

    def __init__(self, fuente_id: str, config: dict):
        self.fuente_id = fuente_id      # id logico de ESTA fuente para el cliente
        self.config = config or {}

    @abstractmethod
    def cargar(self, con: duckdb.DuckDBPyConnection) -> Fragmento:
        """
        Carga las tablas de esta fuente DENTRO de `con` (compartida entre todas
        las fuentes activas del cliente) y devuelve el Fragmento con el schema
        y el catalogo que aporta.

        Convencion: usar `registrar_df(con, df, nombre, self.fuente_id)` para
        crear cada tabla; asi se resuelven choques de nombres entre fuentes.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} tipo={self.tipo} id={self.fuente_id}>"


# --------------------------------------------------------------------------
# Helpers compartidos por todas las fuentes
# --------------------------------------------------------------------------

def limpiar_nombre(name: str) -> str:
    """Normaliza un nombre de tabla/columna a un identificador SQL valido."""
    n = re.sub(r"[^0-9a-zA-Z_]", "_", str(name).strip().lower())
    if not n or n[0].isdigit():
        n = "t_" + n
    return n


def inferir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas de texto a numero o fecha cuando >=80% de los valores
    lo permiten. Quita separadores de miles y simbolos de moneda (₡, $).
    """
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


def _tablas_existentes(con: duckdb.DuckDBPyConnection) -> set:
    try:
        return {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    except Exception:  # noqa: BLE001
        return set()


def registrar_df(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    nombre_base: str,
    fuente_id: str,
) -> str:
    """
    Crea una tabla en `con` a partir de un DataFrame, resolviendo choques de
    nombres: si el nombre ya existe (otra fuente lo uso), se antepone el
    fuente_id -> `<fuente_id>_<nombre>`. Devuelve el nombre final.

    Nota: esto corre en NUESTRO codigo de carga (confiable), no en el SQL que
    genera el modelo; por eso puede usar CREATE/REGISTER sin pasar por el
    validador de solo-lectura de nl2sql.
    """
    nombre = limpiar_nombre(nombre_base)
    existentes = _tablas_existentes(con)
    if nombre in existentes:
        alterno = limpiar_nombre(f"{fuente_id}_{nombre_base}")
        logger.warning(
            "Choque de nombre de tabla '%s'; se renombra a '%s'", nombre, alterno
        )
        nombre = alterno

    tmp = f"_df_{nombre}"
    con.register(tmp, df)
    con.execute(f"CREATE TABLE {nombre} AS SELECT * FROM {tmp}")
    con.unregister(tmp)
    return nombre


def describir_tabla(con: duckdb.DuckDBPyConnection, tabla: str) -> str:
    """Arma el bloque de schema (columnas + 2 filas de ejemplo) de una tabla."""
    cols = con.execute(f"DESCRIBE {tabla}").fetchall()
    cols_txt = ", ".join(f"{c[0]} ({c[1]})" for c in cols)
    muestra = con.execute(f"SELECT * FROM {tabla} LIMIT 2").fetchall()
    return f"Tabla: {tabla}\nColumnas: {cols_txt}\nEjemplo de filas: {muestra}"


def construir_catalogo(filas: List[dict]) -> str:
    """
    Formatea filas de catalogo (misma estructura para toda fuente) a texto para
    el prompt de governance. Cada fila es un dict con las claves:
      tabla | columna | descripcion | sistema_origen | frecuencia | dueno
    Una fila con columna == '*' (o vacia) describe la tabla completa.
    """
    lineas = []
    for f in filas:
        f = {str(k).strip().lower(): str(v).strip() for k, v in f.items()}
        tabla = f.get("tabla", "")
        columna = f.get("columna", "")
        desc = f.get("descripcion", "")
        sistema = f.get("sistema_origen", "")
        frec = f.get("frecuencia", "")
        dueno = f.get("dueño", "") or f.get("dueno", "")

        if columna in ("*", ""):
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
