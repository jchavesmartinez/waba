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

        catalogo, catalogo_filas = self._leer_catalogo(libro)

        # El catalogo del Sheet documenta TODAS sus pestanias. Si esta fuente
        # solo cargo algunas (filtro 'hojas'), se queda unicamente con las filas
        # de esas tablas. Sin esto, dos fuentes sobre el mismo Sheet duplicarian
        # el catalogo entero y las consultas de governance darian filas repetidas.
        cargadas = {t.strip().lower() for t in tablas}
        if catalogo_filas and cargadas:
            antes = len(catalogo_filas)
            catalogo_filas = [
                f for f in catalogo_filas
                if str(f.get("tabla", "")).strip().lower() in cargadas
            ]
            if len(catalogo_filas) != antes:
                logger.info(
                    "[%s] catalogo filtrado a sus tablas: %d de %d filas",
                    self.fuente_id, len(catalogo_filas), antes,
                )

        return Fragmento(
            schema="\n\n".join(schema_parts),
            catalogo=catalogo,
            tablas=tablas,
            catalogo_filas=catalogo_filas,
        )

    def _leer_catalogo(self, libro):
        """Lee '_catalogo' si existe. Si falta, devuelve '' (fuente sin documentar)."""
        try:
            ws = libro.worksheet(CATALOGO_SHEET)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Fuente '%s' sin pestania '%s' (sin catalogo/governance).",
                self.fuente_id, CATALOGO_SHEET,
            )
            return "", []
        filas = ws.get_all_records()
        if not filas:
            return "", []
        # Normaliza claves para que la tabla del warehouse tenga columnas fijas.
        # 'instruccion' es gobernanza que consume el BOT (que tablas puede leer),
        # por eso ahora VIAJA hasta el warehouse en vez de quedarse en el Sheet.
        norm = []
        for f in filas:
            f = {str(k).strip().lower(): str(v).strip() for k, v in f.items()}
            norm.append({
                "tabla": f.get("tabla", ""),
                "columna": f.get("columna", ""),
                "descripcion": f.get("descripcion", ""),
                "instruccion": f.get("instruccion", ""),
                "sistema_origen": f.get("sistema_origen", ""),
                "frecuencia": f.get("frecuencia", ""),
                "dueno": f.get("dueño", "") or f.get("dueno", ""),
            })
        return construir_catalogo(filas), norm
