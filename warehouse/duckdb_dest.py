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
from datetime import datetime, timezone
from typing import Optional

import duckdb
import pandas as pd

from sources.google_calendar import (
    CAMPOS_EVENTO,
    fila_evento,
    fusionar_evento,
    hash_evento,
    limpiar_valor,
)
from .base import Destino, Corrida, ESQUEMA_META, TABLA_CORRIDAS

logger = logging.getLogger("fachavi.warehouse.duckdb")

# Igual que en el destino Postgres: cuantas corridas OK hacia atras se fusionan
# para armar el punto de comparacion de la guarda de vaciado (C-01).
_CORRIDAS_A_FUSIONAR = 20

_TIPOS_EVENTO = {
    "calendar_id": "VARCHAR",
    "recurso": "VARCHAR",
    "evento_id": "VARCHAR",
    "ical_uid": "VARCHAR",
    "titulo": "VARCHAR",
    "descripcion": "VARCHAR",
    "ubicacion": "VARCHAR",
    "inicio": "TIMESTAMPTZ",
    "fin": "TIMESTAMPTZ",
    "zona_horaria": "VARCHAR",
    "duracion_min": "BIGINT",
    "todo_el_dia": "BOOLEAN",
    "estado": "VARCHAR",
    "transparencia": "VARCHAR",
    "visibilidad": "VARCHAR",
    "tipo_evento": "VARCHAR",
    "evento_recurrente_id": "VARCHAR",
    "inicio_original": "TIMESTAMPTZ",
    "creado_en": "TIMESTAMPTZ",
    "actualizado_en": "TIMESTAMPTZ",
    "enlace_evento": "VARCHAR",
    "organizador": "VARCHAR",
    "invitados": "VARCHAR",
    "propiedades": "VARCHAR",
    "raw_evento": "VARCHAR",
}


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

    # --- Google Calendar: estado actual + historico SCD2 ---

    def _tabla_existe(self, esquema: str, tabla: str) -> bool:
        fila = self.conectar().execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema=? AND table_name=?",
            [esquema, tabla],
        ).fetchone()
        return bool(fila)

    def _columna_existe(self, esquema: str, tabla: str, columna: str) -> bool:
        fila = self.conectar().execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema=? AND table_name=? AND column_name=?",
            [esquema, tabla, columna],
        ).fetchone()
        return bool(fila)

    def _asegurar_tablas_calendar(
        self, esquema: str, tabla_actual: str, tabla_historial: str
    ) -> None:
        con = self.conectar()
        self.asegurar_esquema(esquema)

        # La version previa del conector hacia full refresh y no tenia
        # calendar_id. Se conserva completa bajo otro nombre.
        if (
            self._tabla_existe(esquema, tabla_actual)
            and not self._columna_existe(esquema, tabla_actual, "calendar_id")
        ):
            base = f"{tabla_actual}__snapshot_legacy"
            legado = base
            numero = 2
            while self._tabla_existe(esquema, legado):
                legado = f"{base}_{numero}"
                numero += 1
            con.execute(
                f'ALTER TABLE "{esquema}"."{tabla_actual}" RENAME TO "{legado}"'
            )
            logger.warning(
                "la tabla anterior %s.%s se preservo como %s antes de activar "
                "el historico de Calendar",
                esquema,
                tabla_actual,
                legado,
            )

        columnas = ", ".join(
            f'"{campo}" {_TIPOS_EVENTO[campo]}' for campo in CAMPOS_EVENTO
        )
        con.execute(
            f'CREATE TABLE IF NOT EXISTS "{esquema}"."{tabla_actual}" ('
            f"{columnas}, version_hash VARCHAR NOT NULL, "
            "_corrida_id VARCHAR, _fuente_id VARCHAR, "
            "_ingestado_en TIMESTAMPTZ, visto_por_ultima_vez TIMESTAMPTZ, "
            "PRIMARY KEY (calendar_id, evento_id))"
        )
        con.execute(
            f'CREATE TABLE IF NOT EXISTS "{esquema}"."{tabla_historial}" ('
            f"{columnas}, version_hash VARCHAR NOT NULL, "
            "vigente_desde TIMESTAMPTZ, vigente_hasta TIMESTAMPTZ, "
            "es_version_actual BOOLEAN, _corrida_id VARCHAR, "
            "_fuente_id VARCHAR, _ingestado_en TIMESTAMPTZ)"
        )

    def fusionar_eventos_calendar(
        self,
        esquema: str,
        tabla_actual: str,
        tabla_historial: str,
        df: pd.DataFrame,
        corrida: Corrida,
        calendarios_reiniciados: tuple | list = (),
    ) -> dict:
        con = self.conectar()
        self._asegurar_tablas_calendar(esquema, tabla_actual, tabla_historial)
        ahora = datetime.now(timezone.utc)
        cols_actual = list(CAMPOS_EVENTO) + [
            "version_hash",
            "_corrida_id",
            "_fuente_id",
            "_ingestado_en",
            "visto_por_ultima_vez",
        ]
        cols_hist = list(CAMPOS_EVENTO) + [
            "version_hash",
            "vigente_desde",
            "vigente_hasta",
            "es_version_actual",
            "_corrida_id",
            "_fuente_id",
            "_ingestado_en",
        ]
        seleccion = ", ".join(f'"{c}"' for c in list(CAMPOS_EVENTO) + ["version_hash"])
        existentes = {}
        for valores in con.execute(
            f'SELECT {seleccion} FROM "{esquema}"."{tabla_actual}"'
        ).fetchall():
            fila = dict(zip(list(CAMPOS_EVENTO) + ["version_hash"], valores))
            existentes[(fila["calendar_id"], fila["evento_id"])] = fila

        entrantes = {}
        for _, serie in df.iterrows():
            fila = fila_evento(serie)
            clave = (fila.get("calendar_id"), fila.get("evento_id"))
            if not all(clave):
                raise RuntimeError(
                    f"Google Calendar devolvio un evento con clave vacia: {clave}"
                )
            entrantes[clave] = fila

        stats = {
            "nuevos": 0,
            "cambiados": 0,
            "sin_cambios": 0,
            "retirados_actual": 0,
        }
        reiniciados = set(calendarios_reiniciados or ())
        marcas_actual = ", ".join("?" for _ in cols_actual)
        marcas_hist = ", ".join("?" for _ in cols_hist)
        nombres_actual = ", ".join(f'"{c}"' for c in cols_actual)
        nombres_hist = ", ".join(f'"{c}"' for c in cols_hist)

        con.execute("BEGIN TRANSACTION")
        try:
            for clave, entrante in entrantes.items():
                anterior = existentes.get(clave)
                unido = fusionar_evento(anterior, entrante)
                version = hash_evento(unido)

                if anterior and anterior.get("version_hash") == version:
                    con.execute(
                        f'UPDATE "{esquema}"."{tabla_actual}" SET '
                        "visto_por_ultima_vez=?, _corrida_id=?, _fuente_id=?, "
                        "_ingestado_en=? WHERE calendar_id=? AND evento_id=?",
                        [ahora, corrida.corrida_id, corrida.fuente_id, ahora, *clave],
                    )
                    stats["sin_cambios"] += 1
                    continue

                if anterior:
                    con.execute(
                        f'UPDATE "{esquema}"."{tabla_historial}" SET '
                        "vigente_hasta=?, es_version_actual=false "
                        "WHERE calendar_id=? AND evento_id=? "
                        "AND es_version_actual=true",
                        [ahora, *clave],
                    )
                    stats["cambiados"] += 1
                else:
                    stats["nuevos"] += 1

                con.execute(
                    f'DELETE FROM "{esquema}"."{tabla_actual}" '
                    "WHERE calendar_id=? AND evento_id=?",
                    list(clave),
                )
                valores_evento = [limpiar_valor(unido.get(c)) for c in CAMPOS_EVENTO]
                con.execute(
                    f'INSERT INTO "{esquema}"."{tabla_actual}" '
                    f"({nombres_actual}) VALUES ({marcas_actual})",
                    valores_evento
                    + [
                        version,
                        corrida.corrida_id,
                        corrida.fuente_id,
                        ahora,
                        ahora,
                    ],
                )
                con.execute(
                    f'INSERT INTO "{esquema}"."{tabla_historial}" '
                    f"({nombres_hist}) VALUES ({marcas_hist})",
                    valores_evento
                    + [
                        version,
                        ahora,
                        None,
                        True,
                        corrida.corrida_id,
                        corrida.fuente_id,
                        ahora,
                    ],
                )

            # Un 410 obliga a reconstruir el estado sincronizado de ese
            # calendario. Lo que ya no vino en la carga completa sale de la
            # tabla actual, pero su auditoria permanece en historial.
            for clave in existentes:
                if clave[0] in reiniciados and clave not in entrantes:
                    con.execute(
                        f'DELETE FROM "{esquema}"."{tabla_actual}" '
                        "WHERE calendar_id=? AND evento_id=?",
                        list(clave),
                    )
                    stats["retirados_actual"] += 1
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        stats["actual_total"] = con.execute(
            f'SELECT COUNT(*) FROM "{esquema}"."{tabla_actual}"'
        ).fetchone()[0]
        stats["historial_total"] = con.execute(
            f'SELECT COUNT(*) FROM "{esquema}"."{tabla_historial}"'
        ).fetchone()[0]
        logger.info(
            "Calendar %s.%s: %d nuevos, %d cambiados, %d sin cambios, "
            "%d retirados del estado actual",
            esquema,
            tabla_actual,
            stats["nuevos"],
            stats["cambiados"],
            stats["sin_cambios"],
            stats["retirados_actual"],
        )
        return stats

    def _asegurar_estado_calendar(self, esquema: str) -> None:
        self.asegurar_esquema(esquema)
        self.conectar().execute(
            f'CREATE TABLE IF NOT EXISTS "{esquema}"."_calendar_sync_state" ('
            "fuente_id VARCHAR, calendar_id VARCHAR, sync_token VARCHAR, "
            "actualizado_en TIMESTAMPTZ, PRIMARY KEY (fuente_id, calendar_id))"
        )

    def leer_estado_calendar(self, esquema: str, fuente_id: str) -> dict:
        if not self._tabla_existe(esquema, "_calendar_sync_state"):
            return {}
        filas = self.conectar().execute(
            f'SELECT calendar_id, sync_token FROM '
            f'"{esquema}"."_calendar_sync_state" WHERE fuente_id=?',
            [fuente_id],
        ).fetchall()
        return {calendar_id: token for calendar_id, token in filas}

    def guardar_estado_calendar(
        self, esquema: str, fuente_id: str, sync_tokens: dict
    ) -> None:
        self._asegurar_estado_calendar(esquema)
        con = self.conectar()
        ahora = datetime.now(timezone.utc)
        con.execute("BEGIN TRANSACTION")
        try:
            for calendar_id, token in sync_tokens.items():
                con.execute(
                    f'DELETE FROM "{esquema}"."_calendar_sync_state" '
                    "WHERE fuente_id=? AND calendar_id=?",
                    [fuente_id, calendar_id],
                )
                con.execute(
                    f'INSERT INTO "{esquema}"."_calendar_sync_state" '
                    "(fuente_id,calendar_id,sync_token,actualizado_en) "
                    "VALUES (?,?,?,?)",
                    [fuente_id, calendar_id, token, ahora],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

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
