"""
Fuente: Google Sheets.

Config esperada (columna 'config' del registro, como JSON):
  {"spreadsheet_id": "1AbC..."}

Comportamiento:
  - Cada pestania que NO empiece con '_' se carga como tabla consultable.
  - La pestania '_catalogo' (opcional) aporta la metadata/governance.
  - Cualquier pestania que empiece con '_' se excluye de las tablas.

El mismo service account (gclient.py) lee este Sheet. El cliente solo debe
compartir su hoja con el client_email del service account (permiso Lector).
"""

import logging

from gclient import abrir_libro
from .base import (
    Source,
    Fragmento,
    inferir_tipos,
    registrar_df,
    describir_tabla,
    construir_catalogo,
    limpiar_nombre,
    normalizar_columnas,
)

import pandas as pd

logger = logging.getLogger("fachavi.sources.google_sheets")

CATALOGO_SHEET = "_catalogo"


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

        libro = abrir_libro(spreadsheet_id)
        schema_parts = []
        tablas = []
        vistas = []

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

            registros = ws.get_all_values()
            if not registros or len(registros) < 2:
                logger.info("[%s] pestania '%s' vacia o sin filas; se salta",
                            self.fuente_id, titulo)
                continue

            headers = registros[0]
            df = pd.DataFrame(registros[1:], columns=headers)
            df.columns = normalizar_columnas(df.columns)
            df = inferir_tipos(df)

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

        catalogo = self._leer_catalogo(libro)

        return Fragmento(
            schema="\n\n".join(schema_parts),
            catalogo=catalogo,
            tablas=tablas,
        )

    def _leer_catalogo(self, libro) -> str:
        """Lee '_catalogo' si existe. Si falta, devuelve '' (fuente sin documentar)."""
        try:
            ws = libro.worksheet(CATALOGO_SHEET)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Fuente '%s' sin pestania '%s' (sin catalogo/governance).",
                self.fuente_id, CATALOGO_SHEET,
            )
            return ""
        filas = ws.get_all_records()
        if not filas:
            return ""
        return construir_catalogo(filas)
