"""
Fuente: CSV por URL (o varios).

Sirve para clientes que publican datos como CSV (un endpoint, un export de su
ERP, un archivo en Drive/GCS publicado, etc.). Es tambien el ejemplo mas simple
de como se ve una fuente nueva.

Config esperada (JSON en la columna 'config' del registro):

  Una sola tabla:
    {"url": "https://.../ventas.csv", "tabla": "ventas"}

  Varias tablas:
    {"tablas": [
        {"url": "https://.../ventas.csv",    "tabla": "ventas"},
        {"url": "https://.../inventario.csv","tabla": "inventario"}
    ]}

  Catalogo opcional (mismo formato que la pestania _catalogo de Sheets):
    {"url": "...", "tabla": "ventas",
     "catalogo": [
        {"tabla":"ventas","columna":"*","descripcion":"Ventas diarias",
         "sistema_origen":"ERP","frecuencia":"Diaria","dueño":"Comercial"}
     ]}
"""

import logging

import pandas as pd

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

logger = logging.getLogger("fachavi.sources.csv_url")


class CSVURLSource(Source):
    tipo = "csv_url"

    def cargar(self, con) -> Fragmento:
        specs = self._normaliza_specs()
        if not specs:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (csv_url) sin 'url' ni 'tablas' en config."
            )

        schema_parts = []
        tablas = []
        for spec in specs:
            url = spec.get("url", "").strip()
            if not url:
                continue
            nombre_deseado = spec.get("tabla") or "datos"

            df = pd.read_csv(url)
            df.columns = normalizar_columnas(df.columns)
            df = inferir_tipos(df)

            tabla = registrar_df(con, df, nombre_deseado, self.fuente_id)
            tablas.append(tabla)
            schema_parts.append(describir_tabla(con, tabla))
            logger.info("[csv_url:%s] tabla %s (%d filas)", self.fuente_id, tabla, len(df))

        catalogo_filas = self.config.get("catalogo", [])
        catalogo = construir_catalogo(catalogo_filas) if catalogo_filas else ""

        return Fragmento(
            schema="\n\n".join(schema_parts),
            catalogo=catalogo,
            tablas=tablas,
        )

    def _normaliza_specs(self):
        """Acepta {url, tabla} o {tablas:[{url,tabla}, ...]} y devuelve una lista."""
        if self.config.get("tablas"):
            return list(self.config["tablas"])
        if self.config.get("url"):
            return [{"url": self.config["url"], "tabla": self.config.get("tabla", "datos")}]
        return []
