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

  Limites de descarga (opcionales; ver A-09):
    {"timeout_seg": 30, "max_mb": 50}
"""

import io
import logging

import httpx
import pandas as pd

from .base import (
    Source,
    Fragmento,
    coma_decimal_de,
    dia_primero_de,
    inferir_tipos,
    registrar_df,
    describir_tabla,
    construir_catalogo,
    limpiar_nombre,
    normalizar_columnas,
)

logger = logging.getLogger("fachavi.sources.csv_url")

# A-09: pandas.read_csv(url) descargaba directo, sin limite de tiempo ni de
# tamaño. Una direccion que responde muy lento colgaba la corrida ENTERA
# (todos los clientes) de forma indefinida, y un archivo gigante agotaba la
# memoria del contenedor. httpx ya es dependencia del proyecto.
TIMEOUT_DEFECTO_SEG = 30.0
MAX_MB_DEFECTO = 50


class CSVURLSource(Source):
    tipo = "csv_url"

    def cargar(self, con) -> Fragmento:
        specs = self._normaliza_specs()
        if not specs:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (csv_url) sin 'url' ni 'tablas' en config."
            )

        timeout = float(self.config.get("timeout_seg", TIMEOUT_DEFECTO_SEG))
        max_bytes = int(self.config.get("max_mb", MAX_MB_DEFECTO)) * 1024 * 1024
        dia_primero = dia_primero_de(self.config)
        coma_dec = coma_decimal_de(self.config)

        schema_parts = []
        tablas = []
        alertas = []
        for spec in specs:
            url = spec.get("url", "").strip()
            if not url:
                continue
            nombre_deseado = spec.get("tabla") or "datos"

            crudo = self._descargar(url, timeout, max_bytes)
            df = pd.read_csv(io.BytesIO(crudo))
            df.columns = normalizar_columnas(df.columns)
            df = inferir_tipos(df, alertas=alertas,
                               contexto=f"{self.fuente_id}/{nombre_deseado}",
                               dia_primero=dia_primero,
                           coma_decimal=coma_dec)

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
            alertas=alertas,
        )

    def _descargar(self, url: str, timeout: float, max_bytes: int) -> bytes:
        """
        Descarga el CSV con limite de tiempo y de tamaño (A-09). Se lee por
        pedazos para poder cortar ANTES de traerse un archivo enorme entero a
        memoria, no despues.
        """
        acumulado = bytearray()
        with httpx.Client(timeout=timeout, follow_redirects=True) as cli:
            with cli.stream("GET", url) as r:
                r.raise_for_status()
                for pedazo in r.iter_bytes():
                    acumulado += pedazo
                    if max_bytes and len(acumulado) > max_bytes:
                        raise RuntimeError(
                            f"Fuente '{self.fuente_id}': el CSV de {url[:80]} pasa "
                            f"de {max_bytes // (1024*1024)} MB. Subi 'max_mb' en la "
                            "config de la fuente si de verdad es asi de grande."
                        )
        return bytes(acumulado)

    def _normaliza_specs(self):
        """Acepta {url, tabla} o {tablas:[{url,tabla}, ...]} y devuelve una lista."""
        if self.config.get("tablas"):
            return list(self.config["tablas"])
        if self.config.get("url"):
            return [{"url": self.config["url"], "tabla": self.config.get("tabla", "datos")}]
        return []
