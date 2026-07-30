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
    "headers_env": "API_TOKEN_FERRETERIA_A",     # NOMBRE de la variable de
                                                 #   entorno con los headers de
                                                 #   autenticacion en JSON.
                                                 #   Ver nota de seguridad.
    "params":  {"desde": "2026-01-01"},          # opcional (querystring)
    "ruta_datos": "data.items",                  # opcional: donde esta la LISTA
                                                 #   de registros dentro del JSON
                                                 #   (ruta con puntos). Vacio = el
                                                 #   JSON ya es una lista.
    "paginacion": {"param": "page", "inicio": 1, "max_paginas": 20},  # opcional
    "catalogo": [ ... ]                          # opcional, formato _catalogo
  }

SEGURIDAD — C-04. El campo "headers" con el token escrito ADENTRO de la hoja de
calculo sigue funcionando por retrocompatibilidad, pero esta DESACONSEJADO y
deja una advertencia en el log:
  - cualquiera con acceso de lectura a la hoja maestra ve el token de todos los
    clientes, y una hoja se comparte con ligereza;
  - el historial de versiones de Google lo guarda para siempre, aunque despues
    se borre la celda.
Lo correcto es "headers_env": el nombre de la variable de entorno de Render que
guarda el JSON de headers. Es el MISMO mecanismo que ya se usa para el DSN del
warehouse (config.dsn_de_cliente), donde la regla se venia respetando.
"""

import json
import logging

import httpx
import pandas as pd

import config
from .base import (
    Source,
    Fragmento,
    dia_primero_de,
    inferir_tipos,
    registrar_df,
    describir_tabla,
    construir_catalogo,
    limpiar_nombre,
    normalizar_columnas,
)

logger = logging.getLogger("fachavi.sources.api_rest")

_UA_DEFECTO = "FACHAVI-bot/1.0"
MAX_PAGINAS_DEFECTO = 20


class ApiRestSource(Source):
    tipo = "api_rest"

    def cargar(self, con) -> Fragmento:
        url = self.config.get("url", "").strip()
        if not url:
            raise RuntimeError(f"Fuente '{self.fuente_id}' (api_rest) sin 'url'.")

        alertas = []
        registros = self._traer_registros(url, alertas)
        if not registros:
            # B-14: se crea una tabla vacia y la guarda de vaciado hace su
            # trabajo, pero esto tiene que gritar, no susurrar.
            msg = (f"[{self.fuente_id}] el API no devolvio NINGUN registro. Si la "
                   "tabla ya existia con datos, la guarda de vaciado va a impedir "
                   "que se pisen — pero el dato se queda viejo hasta que se arregle.")
            logger.warning(msg)
            alertas.append(msg)

        # --- API -> tabla: aplanar el JSON a filas/columnas ---
        df = pd.json_normalize(registros)          # anidados -> col 'autor.nombre' etc.
        df.columns = normalizar_columnas(df.columns)
        df = inferir_tipos(df, alertas=alertas, contexto=self.fuente_id,
                           dia_primero=dia_primero_de(self.config))

        nombre_deseado = self.config.get("tabla") or "datos"
        tabla = registrar_df(con, df, nombre_deseado, self.fuente_id)
        logger.info("[api_rest:%s] tabla %s (%d filas)", self.fuente_id, tabla, len(df))

        catalogo_filas = self.config.get("catalogo", [])
        catalogo = construir_catalogo(catalogo_filas) if catalogo_filas else ""

        return Fragmento(
            schema=describir_tabla(con, tabla),
            catalogo=catalogo,
            tablas=[tabla],
            alertas=alertas,
        )

    # -------- helpers --------

    def _headers(self) -> dict:
        """
        Resuelve los headers de autenticacion (C-04).

        Prioridad:
          1. "headers_env": nombre de la variable de entorno con el JSON. ✅
          2. "headers": el token literal en la hoja. ⚠️ desaconsejado.
        """
        nombre_var = str(self.config.get("headers_env", "")).strip()
        if nombre_var:
            crudo = config.secreto_de_env(
                nombre_var, para=f"La fuente '{self.fuente_id}' (api_rest)"
            )
            try:
                propios = json.loads(crudo)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"La variable '{nombre_var}' de la fuente '{self.fuente_id}' "
                    f"no tiene un JSON valido de headers: {e}"
                ) from e
            if not isinstance(propios, dict):
                raise RuntimeError(
                    f"La variable '{nombre_var}' debe contener un objeto JSON "
                    '(p.ej. {"Authorization": "Bearer XXX"}).'
                )
        else:
            propios = dict(self.config.get("headers", {}))
            if propios:
                logger.warning(
                    "[%s] los headers de autenticacion vienen ESCRITOS EN LA HOJA "
                    "de calculo. Cualquiera con acceso de lectura los ve, y el "
                    "historial de versiones de Google los guarda para siempre. "
                    "Movelos a una variable de entorno y usa \"headers_env\".",
                    self.fuente_id,
                )
        return {"User-Agent": _UA_DEFECTO, **propios}

    def _traer_registros(self, url: str, alertas: list) -> list:
        """Hace el/los GET y devuelve la lista plana de registros del API."""
        headers = self._headers()
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
            tope = int(pag.get("max_paginas", MAX_PAGINAS_DEFECTO))
            acumulado = []
            agoto_paginas = True
            for _ in range(tope):
                params[param] = pagina
                r = cli.get(url, headers=headers, params=params)
                r.raise_for_status()
                lote = _lista_en(r.json(), ruta)
                if not lote:
                    agoto_paginas = False   # se corto porque el API ya no da mas
                    break
                acumulado += lote
                pagina += 1

            # A-10: antes, si el API tenia 21 paginas simplemente faltaban datos
            # y NADIE se enteraba. Ahora queda en la bitacora como alerta.
            if agoto_paginas:
                msg = (f"[{self.fuente_id}] la paginacion llego al tope de {tope} "
                       f"paginas y el API todavia devolvia datos: FALTAN REGISTROS. "
                       "Subi 'max_paginas' en la config de la fuente.")
                logger.warning(msg)
                alertas.append(msg)
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
