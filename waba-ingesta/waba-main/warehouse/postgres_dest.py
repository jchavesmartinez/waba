"""
Destino: PostgreSQL / Supabase / Neon.

Alternativa a DuckDB-MotherDuck. Conviene si preferis Postgres administrado
(Supabase, Neon) porque ya lo vas a necesitar para pgvector cuando metas RAG
en la Fase 8B, y asi operas una sola base en vez de dos.

    WAREHOUSE_TIPO=postgres
    WAREHOUSE_DSN=postgresql://usuario:clave@host:5432/basedatos

Requiere: sqlalchemy + psycopg2-binary (opcionales; si no estan instalados,
este destino simplemente no se registra).
"""

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from .base import Destino, Corrida, ESQUEMA_META, TABLA_CORRIDAS

logger = logging.getLogger("fachavi.warehouse.postgres")


class PostgresDestino(Destino):
    tipo = "postgres"

    def __init__(self, dsn: str = ""):
        super().__init__(dsn)
        self._eng = None

    def conectar(self):
        if self._eng is None:
            if not self.dsn:
                raise RuntimeError("Falta WAREHOUSE_DSN para el destino postgres.")
            self._eng = create_engine(self.dsn, pool_pre_ping=True)
            self._asegurar_meta()
        return self._eng

    def cerrar(self):
        if self._eng is not None:
            self._eng.dispose()
            self._eng = None

    # --- esquemas y tablas ---

    def asegurar_esquema(self, esquema: str):
        with self.conectar().begin() as cx:
            cx.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{esquema}"'))

    def escribir_tabla(self, esquema: str, tabla: str, df: pd.DataFrame):
        self.asegurar_esquema(esquema)
        # to_sql con replace: recrea la tabla en cada corrida (full refresh).
        df.to_sql(
            tabla,
            self.conectar(),
            schema=esquema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
        logger.info("escrito %s.%s (%d filas)", esquema, tabla, len(df))

    # --- metadata de corridas ---

    def _asegurar_meta(self):
        with self._eng.begin() as cx:
            cx.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{ESQUEMA_META}"'))
            cx.execute(text(
                f"""CREATE TABLE IF NOT EXISTS "{ESQUEMA_META}"."{TABLA_CORRIDAS}" (
                        corrida_id   TEXT,
                        cliente_id   TEXT,
                        fuente_id    TEXT,
                        tipo         TEXT,
                        inicio       TIMESTAMPTZ,
                        fin          TIMESTAMPTZ,
                        estado       TEXT,
                        filas        BIGINT,
                        tablas       TEXT,
                        duracion_seg DOUBLE PRECISION,
                        error        TEXT,
                        alertas      TEXT,
                        detalle      TEXT
                    )"""
            ))

    def registrar_corrida(self, corrida: Corrida):
        with self.conectar().begin() as cx:
            cx.execute(
                text(
                    f'INSERT INTO "{ESQUEMA_META}"."{TABLA_CORRIDAS}" VALUES '
                    "(:cid,:cli,:fid,:tipo,:ini,:fin,:est,:filas,:tablas,:dur,:err,:ale,:det)"
                ),
                {
                    "cid": corrida.corrida_id, "cli": corrida.cliente_id,
                    "fid": corrida.fuente_id, "tipo": corrida.tipo,
                    "ini": corrida.inicio, "fin": corrida.fin,
                    "est": corrida.estado, "filas": corrida.filas,
                    "tablas": ",".join(corrida.tablas),
                    "dur": corrida.duracion_seg, "err": corrida.error[:500],
                    "ale": " | ".join(corrida.alertas)[:500],
                    "det": json.dumps(corrida.detalle),
                },
            )

    def ultimo_detalle(self, cliente_id: str, fuente_id: str) -> dict:
        with self.conectar().begin() as cx:
            fila = cx.execute(
                text(
                    f'SELECT detalle FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
                    "WHERE cliente_id=:c AND fuente_id=:f AND estado LIKE 'ok%' "
                    "ORDER BY fin DESC LIMIT 1"
                ),
                {"c": cliente_id, "f": fuente_id},
            ).fetchone()
        if not fila or not fila[0]:
            return {}
        try:
            return json.loads(fila[0])
        except (ValueError, TypeError):
            return {}

    def ultima_corrida_ok(self, cliente_id: str, fuente_id: str) -> Optional[datetime]:
        with self.conectar().begin() as cx:
            fila = cx.execute(
                text(
                    f'SELECT MAX(fin) FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
                    "WHERE cliente_id=:c AND fuente_id=:f AND estado LIKE 'ok%'"
                ),
                {"c": cliente_id, "f": fuente_id},
            ).fetchone()
        return fila[0] if fila and fila[0] else None
