"""
Destino: DuckDB (archivo local) y MotherDuck (DuckDB administrado en la nube).

Es EL MISMO codigo: solo cambia el DSN.
    Local (desarrollo):  WAREHOUSE_DSN=/data/fachavi.duckdb
    MotherDuck (prod):   WAREHOUSE_DSN=md:fachavi?motherduck_token=XXXX

Por eso arrancar local NO es trabajo tirado: cuando quieras pasar a MotherDuck
cambias una variable de entorno y listo. Es la ruta de menor friccion porque
todo el bot ya habla DuckDB.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import duckdb
import pandas as pd

from .base import Destino, Corrida, ESQUEMA_META, TABLA_CORRIDAS

logger = logging.getLogger("fachavi.warehouse.duckdb")


class DuckDBDestino(Destino):
    tipo = "duckdb"

    def __init__(self, dsn: str = ""):
        super().__init__(dsn or "fachavi.duckdb")
        self._con = None

    def conectar(self):
        if self._con is None:
            self._con = duckdb.connect(self.dsn)
            self._asegurar_meta()
        return self._con

    def cerrar(self):
        if self._con is not None:
            self._con.close()
            self._con = None

    # --- esquemas y tablas ---

    def asegurar_esquema(self, esquema: str):
        self.conectar().execute(f'CREATE SCHEMA IF NOT EXISTS "{esquema}"')

    def escribir_tabla(self, esquema: str, tabla: str, df: pd.DataFrame):
        con = self.conectar()
        self.asegurar_esquema(esquema)
        con.register("_df_entrante", df)
        # Full refresh atomico: se crea la nueva y se reemplaza.
        con.execute(
            f'CREATE OR REPLACE TABLE "{esquema}"."{tabla}" AS '
            f"SELECT * FROM _df_entrante"
        )
        con.unregister("_df_entrante")
        logger.info("escrito %s.%s (%d filas)", esquema, tabla, len(df))

    # --- metadata de corridas ---

    def _asegurar_meta(self):
        self._con.execute(f'CREATE SCHEMA IF NOT EXISTS "{ESQUEMA_META}"')
        self._con.execute(
            f"""CREATE TABLE IF NOT EXISTS "{ESQUEMA_META}"."{TABLA_CORRIDAS}" (
                    corrida_id   VARCHAR,
                    cliente_id   VARCHAR,
                    fuente_id    VARCHAR,
                    tipo         VARCHAR,
                    inicio       TIMESTAMPTZ,
                    fin          TIMESTAMPTZ,
                    estado       VARCHAR,
                    filas        BIGINT,
                    tablas       VARCHAR,
                    duracion_seg DOUBLE,
                    error        VARCHAR,
                    alertas      VARCHAR,
                    detalle      VARCHAR
                )"""
        )

    def registrar_corrida(self, corrida: Corrida):
        self.conectar().execute(
            f'INSERT INTO "{ESQUEMA_META}"."{TABLA_CORRIDAS}" VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            [
                corrida.corrida_id,
                corrida.cliente_id,
                corrida.fuente_id,
                corrida.tipo,
                corrida.inicio,
                corrida.fin,
                corrida.estado,
                corrida.filas,
                ",".join(corrida.tablas),
                corrida.duracion_seg,
                corrida.error[:500],
                " | ".join(corrida.alertas)[:500],
                json.dumps(corrida.detalle),
            ],
        )

    def ultimo_detalle(self, cliente_id: str, fuente_id: str) -> dict:
        fila = self.conectar().execute(
            f'SELECT detalle FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
            f"WHERE cliente_id=? AND fuente_id=? AND estado LIKE 'ok%' "
            "ORDER BY fin DESC LIMIT 1",
            [cliente_id, fuente_id],
        ).fetchone()
        if not fila or not fila[0]:
            return {}
        try:
            return json.loads(fila[0])
        except (ValueError, TypeError):
            return {}

    def ultima_corrida_ok(self, cliente_id: str, fuente_id: str) -> Optional[datetime]:
        fila = self.conectar().execute(
            f'SELECT MAX(fin) FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
            f"WHERE cliente_id=? AND fuente_id=? AND estado LIKE 'ok%'",
            [cliente_id, fuente_id],
        ).fetchone()
        return fila[0] if fila and fila[0] else None
