"""
Destino: DuckDB (archivo local) y MotherDuck (DuckDB administrado en la nube).

Es EL MISMO codigo: solo cambia el DSN.
    Local (desarrollo):  WAREHOUSE_DSN=/data/fachavi.duckdb
    MotherDuck (prod):   WAREHOUSE_DSN=md:fachavi?motherduck_token=XXXX

Por eso arrancar local NO es trabajo tirado: cuando quieras pasar a MotherDuck
cambias una variable de entorno y listo. Es la ruta de menor friccion porque
todo el bot ya habla DuckDB.

PARIDAD CON EL DESTINO POSTGRES (B-18) — que NO hace este destino:
  - escribir_kpis: no persiste <esquema>._kpis (usa el no-op de la clase base).
    Consecuencia practica: la capa semantica de KPIs no se puede probar en
    desarrollo local, solo contra Neon.
  - aplicar_comentarios: no escribe COMMENT ON nativos.
  - purgar_corridas: no purga la bitacora (A-04); en local no hace falta.
Todo lo demas —esquemas, tablas, catalogo, bitacora, frescura y la guarda de
vaciado— se comporta igual en los dos destinos. Si algun dia esto deja de
alcanzar, lo que falta esta listado aca arriba.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import duckdb
import pandas as pd

from .base import Destino, Corrida, ESQUEMA_META, TABLA_CORRIDAS

logger = logging.getLogger("fachavi.warehouse.duckdb")

# Igual que en el destino Postgres: cuantas corridas OK hacia atras se fusionan
# para armar el punto de comparacion de la guarda de vaciado (C-01).
_CORRIDAS_A_FUSIONAR = 20


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

    def escribir_catalogo(self, esquema: str, fuente_id: str, filas: list):
        con = self.conectar()
        self.asegurar_esquema(esquema)
        con.execute(
            f'CREATE TABLE IF NOT EXISTS "{esquema}"."_catalogo" ('
            "fuente_id VARCHAR, tabla VARCHAR, columna VARCHAR, descripcion VARCHAR,"
            "instruccion VARCHAR,"
            "sistema_origen VARCHAR, frecuencia VARCHAR, dueno VARCHAR)"
        )
        # Migracion suave para _catalogo creado por una version previa.
        con.execute(
            f'ALTER TABLE "{esquema}"."_catalogo" ADD COLUMN IF NOT EXISTS instruccion VARCHAR'
        )
        con.execute(f'DELETE FROM "{esquema}"."_catalogo" WHERE fuente_id=?', [fuente_id])
        for f in filas:
            con.execute(
                f'INSERT INTO "{esquema}"."_catalogo" '
                "(fuente_id,tabla,columna,descripcion,instruccion,"
                "sistema_origen,frecuencia,dueno) VALUES (?,?,?,?,?,?,?,?)",
                [fuente_id, f.get("tabla",""), f.get("columna",""), f.get("descripcion",""),
                 f.get("instruccion",""),
                 f.get("sistema_origen",""), f.get("frecuencia",""), f.get("dueno","")],
            )
        logger.info("catalogo de '%s': %d filas en %s._catalogo", fuente_id, len(filas), esquema)

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
        """
        Punto de comparacion de la guarda de vaciado. Misma semantica que el
        destino Postgres (C-01): se FUSIONAN las ultimas corridas OK en vez de
        tomar solo la ultima, para que un hueco en el historial —por ejemplo una
        corrida que bloqueo la escritura de una tabla— no deje a esa tabla sin
        con que compararse y desarme la guarda.
        """
        filas = self.conectar().execute(
            f'SELECT detalle FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
            f"WHERE cliente_id=? AND fuente_id=? AND estado LIKE 'ok%' "
            "ORDER BY fin DESC LIMIT ?",
            [cliente_id, fuente_id, _CORRIDAS_A_FUSIONAR],
        ).fetchall()

        fusionado: dict = {}
        for (crudo,) in reversed(filas):     # de la mas vieja a la mas nueva
            if not crudo:
                continue
            try:
                d = json.loads(crudo)
            except (ValueError, TypeError):
                continue
            if isinstance(d, dict):
                fusionado.update(d)
        return fusionado

    def ultima_corrida_ok(self, cliente_id: str, fuente_id: str) -> Optional[datetime]:
        fila = self.conectar().execute(
            f'SELECT MAX(fin) FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
            f"WHERE cliente_id=? AND fuente_id=? AND estado LIKE 'ok%'",
            [cliente_id, fuente_id],
        ).fetchone()
        return fila[0] if fila and fila[0] else None
