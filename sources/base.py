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
    catalogo: str = ""                     # metadata/governance como texto (para el prompt)
    tablas: List[str] = field(default_factory=list)  # nombres finales de las tablas cargadas
    # Filas estructuradas del catalogo, para persistirlas como TABLA en el
    # warehouse. Asi el bot consulta governance sin tocar Google Sheets.
    catalogo_filas: List[dict] = field(default_factory=list)
    # Filas del tab '_kpis' (capa semantica): metricas predefinidas que el bot
    # usa para responder de forma consistente. Se persisten en <esquema>._kpis.
    kpis_filas: List[dict] = field(default_factory=list)
    # Alertas de calidad que detecta la PROPIA fuente (paginacion truncada,
    # valores descartados al inferir tipos, tope de filas alcanzado...). sync.py
    # las suma a corrida.alertas y quedan en la bitacora. Antes estas cosas solo
    # se veian en el log, o sea: no se veian.
    alertas: List[str] = field(default_factory=list)


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


def normalizar_columnas(nombres) -> list:
    """
    Normaliza los encabezados de una hoja a identificadores SQL validos,
    DETERMINISTAMENTE (no dejamos que el motor decida).

    Resuelve dos casos que abundan en Sheets reales:
      - Encabezado vacio         -> columna_3 (por posicion)
      - Encabezado duplicado     -> monto, monto_2, monto_3...

    Importante: si no lo hacemos nosotros, DuckDB inventa 'C3'/'monto_1' y
    Postgres se comporta distinto -> la misma hoja daria nombres distintos
    segun el warehouse.
    """
    limpios = []
    vistos = {}
    usados = set()
    for i, n in enumerate(nombres):
        base = limpiar_nombre(n) if str(n).strip() else f"columna_{i + 1}"
        if base in vistos:
            # B-06: no basta con sumar el contador. Si la hoja YA trae una
            # columna 'monto_2' y ademas dos 'monto', el contador generaba dos
            # 'monto_2'. Se avanza hasta encontrar un nombre libre de verdad.
            candidato = base
            while candidato in usados:
                vistos[base] += 1
                candidato = f"{base}_{vistos[base]}"
            logger.warning("Encabezado duplicado; se renombra a '%s'", candidato)
            base = candidato
        else:
            vistos[base] = 1
        usados.add(base)
        limpios.append(base)
    return limpios


def _a_fecha(serie: pd.Series, dia_primero: bool = True) -> pd.Series:
    """
    Convierte a fecha tolerando los formatos que aparecen en la practica:
      - Google Sheets en es-CR : 07/01/2026        (dia primero)
      - ISO de APIs            : 2026-01-07T14:30:00Z
      - ISO con hora           : 2026-01-07 00:00:00

    'format=mixed' parsea cada valor por separado, que es lo unico que acierta
    en los cuatro casos.

    A-07: `dia_primero` desempata 07/01. True (default) = convencion tica, 7 de
    enero. Si el cliente exporta de un sistema en ingles es 1 de julio y nadie
    se entera: los dias del 13 en adelante se salvan solos, los del 1 al 12 no.
    Se controla por fuente con {"formato_fecha": "mes_primero"} en su config, o
    global con la variable FORMATO_FECHA.
    """
    try:
        return pd.to_datetime(serie, errors="coerce", format="mixed",
                              dayfirst=dia_primero)
    except (ValueError, TypeError):
        return pd.to_datetime(serie, errors="coerce", dayfirst=dia_primero)


# A-03: si al convertir una columna a numero/fecha se pierde mas de este
# porcentaje de valores NO vacios, se genera una alerta de calidad. El umbral de
# conversion sigue siendo 80%: entre 80% y 95% de aciertos, la columna se
# convierte igual PERO el 5-20% descartado deja de ser invisible.
UMBRAL_INFERENCIA = 0.8
UMBRAL_ALERTA_DESCARTE = 0.05


# --------------------------------------------------------------------------
# Separador de miles: el error mas caro que puede cometer este archivo.
#
# La version anterior hacia .str.replace(r"[,\s₡$]", "") y pasaba el resto a
# pd.to_numeric(). Eso quita la coma —separador de miles ANGLOSAJON— pero deja
# el punto, que es el separador de miles TICO. Resultado sobre datos reales de
# un cliente costarricense:
#
#     "₡5.000"      ->     5.0        (mil veces menos)
#     "₡1.250.000"  ->     NaN        (el valor se pierde entero)
#     "₡500"        ->   500.0        (correcto, por casualidad)
#
# Una columna entera de montos escrita "₡5.000" convierte al 100 %, o sea que
# pasa el umbral del 80 % sin descartar NADA: la alerta de A-03 no se dispara,
# porque desde el punto de vista del codigo no hubo ningun fallo. El bot
# responde "vendimos 47" cuando fueron 47.000, y el numero es plausible.
#
# La ambiguedad no se puede resolver valor por valor —"5.000" es 5000 en Costa
# Rica y 5.0 en Estados Unidos— asi que se resuelve por COLUMNA, con la
# convencion declarada de la fuente, igual que se hizo con las fechas (A-07).
# --------------------------------------------------------------------------

_MILES_PUNTO = re.compile(r"^\s*[₡$€]?\s*-?\d{1,3}(\.\d{3})+(,\d+)?\s*$")
_MILES_COMA  = re.compile(r"^\s*[₡$€]?\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*$")


def _limpiar_numero(serie: pd.Series, coma_decimal: bool = True) -> pd.Series:
    """
    Quita moneda y separadores de miles respetando la convencion de la fuente.

    coma_decimal=True  (default, convencion tica):  punto = miles, coma = decimal
    coma_decimal=False (convencion anglosajona):    coma  = miles, punto = decimal
    """
    t = serie.astype(str).str.strip()
    t = t.str.replace(r"[\s₡$€]", "", regex=True)      # moneda y espacios
    if coma_decimal:
        t = t.str.replace(".", "", regex=False)         # miles
        t = t.str.replace(",", ".", regex=False)        # decimal -> punto
    else:
        t = t.str.replace(",", "", regex=False)         # miles
    return t.str.replace(r"^$", "nan", regex=True)


def coma_decimal_de(config_fuente: dict) -> bool:
    """
    Resuelve la convencion numerica de una fuente: su propio
    {"formato_numero": "punto_decimal"} y, si no lo trae, la global.
    """
    import config as _cfg
    valor = str((config_fuente or {}).get("formato_numero", "")).strip().lower()
    if not valor:
        valor = getattr(_cfg, "FORMATO_NUMERO", "coma_decimal")
    return valor not in ("punto_decimal", "punto", "us", "en")


def detectar_convencion(serie: pd.Series):
    """
    Mira la columna y dice que convencion parece tener, o None si no hay señal.
    Solo cuenta los valores INEQUIVOCOS (los que tienen grupos de tres digitos
    repetidos, como 1.250.000 o 1,250,000). Sirve para avisar cuando la
    convencion declarada contradice a los datos.
    """
    t = serie.astype(str)
    punto = int(t.str.match(_MILES_PUNTO).sum())
    coma  = int(t.str.match(_MILES_COMA).sum())
    if punto > coma and punto > 0:
        return "coma_decimal"       # 1.250.000 -> punto es miles
    if coma > punto and coma > 0:
        return "punto_decimal"      # 1,250,000 -> coma es miles
    return None


def inferir_tipos(df: pd.DataFrame, alertas: list = None, contexto: str = "",
                  dia_primero: bool = True, coma_decimal: bool = True) -> pd.DataFrame:
    """
    Convierte columnas de texto a numero o fecha cuando >=80% de los valores
    lo permiten. Quita separadores de miles y simbolos de moneda (₡, $).

    B-07: trabaja sobre una COPIA. Antes modificaba el DataFrame que recibia, lo
    cual funcionaba de casualidad (nadie reusaba el original) y era una trampa
    puesta para el proximo cambio.

    A-03: `alertas` es una lista donde se anotan los valores que la conversion
    descarto. Sin esto, un codigo '001' perdia los ceros, un 'A45' dentro de una
    columna numerica se volvia vacio y una columna con '₡5.000' y '$10' producia
    5000 y 10 sumables entre si — todo en absoluto silencio.
    """
    df = df.copy()
    for col in df.columns:
        serie = df[col]
        no_vacios = serie.astype(str).str.strip().ne("").sum()
        umbral = max(1, int(UMBRAL_INFERENCIA * len(serie)))

        limpio = _limpiar_numero(serie, coma_decimal=coma_decimal)
        num = pd.to_numeric(limpio, errors="coerce")
        if num.notna().sum() >= umbral:
            detectada = detectar_convencion(serie)
            declarada = "coma_decimal" if coma_decimal else "punto_decimal"
            if detectada and detectada != declarada:
                msg = (f"[{contexto or 'fuente'}] la columna '{col}' parece usar la "
                       f"convencion numerica '{detectada}' pero la fuente declara "
                       f"'{declarada}'. Los montos pueden quedar mil veces mas "
                       "grandes o mas chicos. Revisa 'formato_numero' en la config "
                       "de la fuente.")
                logger.warning(msg)
                if alertas is not None:
                    alertas.append(msg)
            df[col] = num
            _avisar_descartes(alertas, contexto, col, "numero",
                              int(no_vacios - num.notna().sum()), int(no_vacios))
            continue

        fecha = _a_fecha(serie, dia_primero=dia_primero)
        if fecha.notna().sum() >= umbral:
            df[col] = fecha
            _avisar_descartes(alertas, contexto, col, "fecha",
                              int(no_vacios - fecha.notna().sum()), int(no_vacios))
            continue

        df[col] = serie.astype(str)
    return df


def _avisar_descartes(alertas, contexto: str, columna: str, tipo: str,
                      descartados: int, total: int) -> None:
    """Anota una alerta si la conversion perdio una fraccion no trivial (A-03)."""
    if descartados <= 0 or total <= 0:
        return
    fraccion = descartados / total
    if fraccion < UMBRAL_ALERTA_DESCARTE:
        return
    msg = (
        f"[{contexto or 'fuente'}] la columna '{columna}' se convirtio a {tipo} "
        f"y {descartados} de {total} valores ({fraccion:.0%}) quedaron VACIOS. "
        "Revisa si son datos con otro formato (codigos con ceros a la izquierda, "
        "monedas mezcladas, fechas anglosajonas)."
    )
    logger.warning(msg)
    if alertas is not None:
        alertas.append(msg)


def dia_primero_de(config_fuente: dict) -> bool:
    """
    Resuelve la convencion de fecha de una fuente (A-07):
    su propio {"formato_fecha": "..."} y, si no lo trae, la global.
    """
    import config as _cfg
    valor = str((config_fuente or {}).get("formato_fecha", "")).strip().lower()
    if not valor:
        valor = getattr(_cfg, "FORMATO_FECHA", "dia_primero")
    return valor not in ("mes_primero", "mdy", "us", "en")


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
      tabla | columna | descripcion | instruccion | sistema_origen | frecuencia | dueno
    Una fila con columna == '*' (o vacia) describe la tabla completa.

    'instruccion' es texto libre de gobernanza (p.ej. "esta tabla puede ser
    usada por el bot"). El bot lo usa para decidir que tablas puede consultar;
    aca solo se documenta.
    """
    lineas = []
    for f in filas:
        f = {str(k).strip().lower(): str(v).strip() for k, v in f.items()}
        tabla = f.get("tabla", "")
        columna = f.get("columna", "")
        desc = f.get("descripcion", "")
        instr = f.get("instruccion", "")
        sistema = f.get("sistema_origen", "")
        frec = f.get("frecuencia", "")
        dueno = f.get("dueño", "") or f.get("dueno", "")

        extra = f" Instruccion: {instr}." if instr else ""
        if columna in ("*", ""):
            lineas.append(
                f"Tabla '{tabla}': {desc}. Sistema de origen: {sistema}. "
                f"Frecuencia de actualizacion: {frec}. Dueño del dato: {dueno}.{extra}"
            )
        else:
            lineas.append(
                f"Columna '{columna}' (tabla '{tabla}'): {desc}. "
                f"Origen: {sistema}. Actualizacion: {frec}. Dueño: {dueno}.{extra}"
            )
    return "\n".join(lineas)
