"""
Fuente: Google Sheets.

Config esperada (columna 'config' del registro, como JSON):
  {"spreadsheet_id": "1AbC..."}

Comportamiento:
  - Cada pestania que NO empiece con '_' se carga como tabla consultable.
  - Cualquier pestania que empiece con '_' se excluye de las tablas (por si
    el Sheet trae una '_catalogo' o '_kpis' vieja; ya no se leen de aca).

El mismo service account (gclient.py) lee este Sheet. El cliente solo debe
compartir su hoja con el client_email del service account (permiso Lector).

CATALOGO Y KPIS: ya NO se leen de este Sheet. Antes cada fuente traia su
propio catalogo desde una pestania '_catalogo' interna; ahora el catalogo y
los KPIs viven en UN Sheet central por cliente (ver catalogo_cliente.py), que
documenta todas las fuentes de ese cliente juntas. sync.py lo lee una vez por
cliente y lo reparte — este conector ya no necesita saber nada de eso.
"""

import logging
import time

from gspread.exceptions import APIError
from gclient import abrir_libro
from .base import (
    Source,
    Fragmento,
    coma_decimal_de,
    dia_primero_de,
    inferir_tipos,
    registrar_df,
    describir_tabla,
    limpiar_nombre,
    normalizar_columnas,
)

import pandas as pd

logger = logging.getLogger("fachavi.sources.google_sheets")

_TAMANO_LOTE_LECTURA = 25
_REINTENTOS_LECTURA = 4


def _rango_hoja(titulo: str) -> str:
    """Devuelve el titulo escapado como rango A1 de hoja completa."""
    return "'" + titulo.replace("'", "''") + "'"


def _rellenar_filas(registros):
    """Iguala el ancho de las filas como hacia Worksheet.get_all_values()."""
    if not registros:
        return []
    ancho = max(len(fila) for fila in registros)
    return [list(fila) + [""] * (ancho - len(fila)) for fila in registros]


def _leer_hojas_en_lotes(libro, hojas):
    """Lee varias pestanias con values.batchGet, no una llamada por hoja."""
    resultado = {}
    for inicio in range(0, len(hojas), _TAMANO_LOTE_LECTURA):
        lote = hojas[inicio:inicio + _TAMANO_LOTE_LECTURA]
        rangos = [_rango_hoja(ws.title) for ws in lote]

        for intento in range(_REINTENTOS_LECTURA):
            try:
                respuesta = libro.values_batch_get(
                    rangos,
                    params={"majorDimension": "ROWS"},
                )
                break
            except APIError as exc:
                codigo = getattr(getattr(exc, "response", None), "status_code", None)
                if codigo != 429 or intento == _REINTENTOS_LECTURA - 1:
                    raise
                espera = 2 ** intento
                logger.warning(
                    "Cuota de Google Sheets agotada; reintento %d/%d en %ds",
                    intento + 1, _REINTENTOS_LECTURA, espera,
                )
                time.sleep(espera)

        valores = respuesta.get("valueRanges", [])
        if len(valores) != len(lote):
            raise RuntimeError(
                "Google Sheets devolvio una cantidad inesperada de rangos: "
                f"se pidieron {len(lote)} y llegaron {len(valores)}."
            )
        for ws, value_range in zip(lote, valores):
            resultado[ws.title] = _rellenar_filas(
                value_range.get("values", [])
            )

    return resultado

class GoogleSheetsSource(Source):
    tipo = "google_sheets"

    def cargar(self, con) -> Fragmento:
        spreadsheet_id = self.config.get("spreadsheet_id", "").strip()
        if not spreadsheet_id:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (google_sheets) sin 'spreadsheet_id' en config."
            )

        # Filtros opcionales de pestanias:
        #   "hojas":   ["ventas"]        -> SOLO estas (lista blanca)
        #   "excluir": ["notas","borra"] -> todas menos estas (lista negra)
        # Sirven para partir UN mismo Sheet en varias fuentes con frescuras
        # distintas, o para saltarse pestanias que no son tablas.
        solo = {str(h).strip().lower() for h in self.config.get("hojas", []) if str(h).strip()}
        excluir = {str(h).strip().lower() for h in self.config.get("excluir", []) if str(h).strip()}

        # Guarda: pedir una pestania que empieza con '_' no tiene efecto, porque
        # esas son metadata y se excluyen siempre. El catalogo NO se ingesta con
        # una fila propia en 'fuentes': viaja automaticamente con cada fuente que
        # lee ese Sheet y se guarda en <esquema>._catalogo.
        ocultas = {h for h in solo if h.startswith("_")}
        if ocultas:
            logger.error(
                "[%s] 'hojas' pide pestanias de metadata (%s). Esas se excluyen "
                "siempre y esta fuente cargaria CERO tablas. El catalogo ya se "
                "guarda solo; borra esta fila de la pestania 'fuentes'.",
                self.fuente_id, ", ".join(sorted(ocultas)),
            )

        # A-08: get_all_values() trae la hoja COMPLETA a memoria de un saque.
        # Una hoja de cientos de miles de filas puede agotar el contenedor de
        # Render y matar la corrida de TODOS los clientes. Con este tope, en vez
        # de morir se recorta y se avisa fuerte. 0 = sin tope.
        max_filas = int(self.config.get("max_filas", 0) or 0)
        dia_primero = dia_primero_de(self.config)
        coma_dec = coma_decimal_de(self.config)

        libro = abrir_libro(spreadsheet_id)
        schema_parts = []
        tablas = []
        vistas = []
        alertas = []

        hojas_seleccionadas = []
        for ws in libro.worksheets():
            titulo = ws.title
            vistas.append(titulo)

            if titulo.startswith("_"):
                continue  # metadata, no es tabla consultable
            if solo and titulo.strip().lower() not in solo:
                continue
            if titulo.strip().lower() in excluir:
                logger.info("[%s] pestania '%s' excluida por config", self.fuente_id, titulo)
                continue

            hojas_seleccionadas.append(ws)

        registros_por_hoja = _leer_hojas_en_lotes(libro, hojas_seleccionadas)

        for ws in hojas_seleccionadas:
            titulo = ws.title
            registros = registros_por_hoja[titulo]
            if not registros or len(registros) < 2:
                # B-12: esto era un logger.info y por lo tanto invisible. Si la
                # tabla YA existia con datos, saltarla significa que la guarda de
                # vaciado nunca llega a evaluarse (no se crea tabla para
                # comparar) y el dato viejo se queda ahi callado, envejeciendo.
                msg = (f"[{self.fuente_id}] la pestania '{titulo}' llego SIN FILAS "
                       "(solo encabezados o vacia); se salta y no se actualiza "
                       "nada de esa tabla.")
                logger.warning(msg)
                alertas.append(msg)
                continue

            headers = registros[0]
            filas_datos = registros[1:]
            if max_filas and len(filas_datos) > max_filas:
                msg = (f"[{self.fuente_id}] la pestania '{titulo}' tiene "
                       f"{len(filas_datos)} filas y el tope configurado es "
                       f"{max_filas}: se cargan las primeras {max_filas}. "
                       "HAY DATOS QUE NO ENTRARON.")
                logger.warning(msg)
                alertas.append(msg)
                filas_datos = filas_datos[:max_filas]

            df = pd.DataFrame(filas_datos, columns=headers)
            df.columns = normalizar_columnas(df.columns)
            df = inferir_tipos(df, alertas=alertas,
                               contexto=f"{self.fuente_id}/{titulo}",
                               dia_primero=dia_primero,
                           coma_decimal=coma_dec)

            tabla = registrar_df(con, df, titulo, self.fuente_id)
            tablas.append(tabla)
            schema_parts.append(describir_tabla(con, tabla))
            logger.info("[%s] tabla %s (%d filas)", self.fuente_id, tabla, len(df))

        if solo:
            faltan = solo - {v.strip().lower() for v in vistas}
            if faltan:
                logger.warning(
                    "[%s] pedidas en 'hojas' pero NO existen en el Sheet: %s",
                    self.fuente_id, ", ".join(sorted(faltan)),
                )

        # Catalogo y KPIs ya NO se leen de este Sheet: vienen del Sheet central
        # del cliente (catalogo_cliente.py), que sync.py lee una vez y reparte
        # a cada fuente despues de cargar(). Este conector solo entrega las
        # tablas y su schema.
        return Fragmento(
            schema="\n\n".join(schema_parts),
            tablas=tablas,
            alertas=alertas,
        )
