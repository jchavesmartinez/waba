"""
Fuente: API REST (JSON -> tabla).

La conversion API -> tabla ocurre ACA DENTRO, en cargar(). NO necesitas una
capa aparte: el connector ES esa capa. Trae el JSON, lo APLANA a filas/columnas
(pandas.json_normalize) y lo registra como tabla, igual que las demas fuentes.
El pipeline (nl2sql) ve una tabla normal y ni se entera de que vino de un API.

Config esperada (JSON en la columna 'config' del registro):
  {
    "url": "https://api.cliente.com/v1/ventas",
    "tabla": "ventas",
    "headers": {"Authorization": "Bearer XXX"},   # opcional (auth)
    "params":  {"desde": "2026-01-01"},            # opcional (querystring)
    "ruta_datos": "data.items",                    # opcional: donde esta la LISTA
                                                   #   de registros dentro del JSON
                                                   #   (ruta con puntos). Vacio = el
                                                   #   JSON ya es una lista.
    "paginacion": {"param": "page", "inicio": 1, "max_paginas": 20},  # opcional
    "catalogo": [ ... ]                            # opcional, formato _catalogo
  }
"""

import logging

import httpx
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

logger = logging.getLogger("fachavi.sources.api_rest")

_UA_DEFECTO = "FACHAVI-bot/1.0"


class ApiRestSource(Source):
    tipo = "api_rest"

    def cargar(self, con) -> Fragmento:
        url = self.config.get("url", "").strip()
        if not url:
            raise RuntimeError(f"Fuente '{self.fuente_id}' (api_rest) sin 'url'.")

        registros = self._traer_registros(url)
        if not registros:
            logger.warning("[api_rest:%s] el API no devolvio registros", self.fuente_id)

        # --- API -> tabla: aplanar el JSON a filas/columnas ---
        df = pd.json_normalize(registros)          # anidados -> col 'autor.nombre' etc.
        df.columns = normalizar_columnas(df.columns)
        df = inferir_tipos(df)

        nombre_deseado = self.config.get("tabla") or "datos"
        tabla = registrar_df(con, df, nombre_deseado, self.fuente_id)
        logger.info("[api_rest:%s] tabla %s (%d filas)", self.fuente_id, tabla, len(df))

        catalogo_filas = self.config.get("catalogo", [])
        catalogo = construir_catalogo(catalogo_filas) if catalogo_filas else ""

        return Fragmento(
            schema=describir_tabla(con, tabla),
            catalogo=catalogo,
            tablas=[tabla],
        )

    # -------- helpers --------

    def _traer_registros(self, url: str) -> list:
        """Hace el/los GET y devuelve la lista plana de registros del API."""
        headers = {"User-Agent": _UA_DEFECTO, **self.config.get("headers", {})}
        params = dict(self.config.get("params", {}))
        ruta = self.config.get("ruta_datos", "")
        pag = self.config.get("paginacion")

        with httpx.Client(timeout=30.0) as cli:
            if not pag:
                r = cli.get(url, headers=headers, params=params)
                r.raise_for_status()
                return _lista_en(r.json(), ruta)

            # Con paginacion: pedir paginas hasta que una venga vacia o llegue al tope.
            param = pag.get("param", "page")
            pagina = int(pag.get("inicio", 1))
            tope = int(pag.get("max_paginas", 20))
            acumulado = []
            for _ in range(tope):
                params[param] = pagina
                r = cli.get(url, headers=headers, params=params)
                r.raise_for_status()
                lote = _lista_en(r.json(), ruta)
                if not lote:
                    break
                acumulado += lote
                pagina += 1
            return acumulado


def _lista_en(payload, ruta: str) -> list:
    """
    Navega el JSON hasta la lista de registros. 'ruta' con puntos, p.ej.
    'data.items'. Si esta vacia, se asume que el payload ya es la lista
    (o un solo objeto, que se envuelve como lista de 1).
    """
    obj = payload
    for parte in [p for p in ruta.split(".") if p]:
        if not isinstance(obj, dict):
            raise RuntimeError(f"ruta_datos '{ruta}': '{parte}' no es un objeto.")
        obj = obj.get(parte)
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]  # un solo objeto -> tabla de una fila
