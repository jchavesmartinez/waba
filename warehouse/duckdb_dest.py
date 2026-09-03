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
from sources.zoho_imap import (
    CAMPOS_CORREO,
    fila_correo,
    hash_correo,
    limpiar_valor_correo,
)
from sources.meta_ads import (
    CAMPOS_INSIGHT,
    TIPOS_INSIGHT,
    fila_insight,
    hash_insight,
    limpiar_valor_meta,
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

_TIPOS_CORREO = {
    "correo_id": "VARCHAR",
    "message_id": "VARCHAR",
    "buzon": "VARCHAR",
    "carpeta": "VARCHAR",
    "uid": "VARCHAR",
    "fecha": "TIMESTAMP",
    "remitente_nombre": "VARCHAR",
    "remitente_correo": "VARCHAR",
    "destinatarios": "VARCHAR",
    "cc": "VARCHAR",
    "asunto": "VARCHAR",
    "cuerpo": "VARCHAR",
    "n_adjuntos": "BIGINT",
    "adjuntos": "VARCHAR",
}

# Meta Ads son ~60 columnas: en vez de copiarlas a mano (y que se desincronicen
# con las de Postgres) se traduce el tipo LOGICO que declara el conector.
_SQL_META = {
    "entero": "BIGINT",
    "decimal": "DOUBLE",
    "texto": "VARCHAR",
    "fecha": "DATE",
    "fecha_hora": "TIMESTAMPTZ",
}

_TIPOS_INSIGHT = {
    columna: _SQL_META[TIPOS_INSIGHT[columna]] for columna in CAMPOS_INSIGHT
}

# Tipos logicos de la capa semantica (modelo/tipos.py) -> SQL de DuckDB.
_SQL_SEMANTICO = {
    "texto": "VARCHAR",
    "entero": "BIGINT",
    "decimal": "DOUBLE",
    "fecha_hora_es": "TIMESTAMP",
    "fecha_iso": "TIMESTAMP",
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

    # --- Google Calendar: una sola tabla acumulativa ---

    def _tabla_existe(self, esquema: str, tabla: str) -> bool:
        fila = self.conectar().execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema=? AND table_name=?",
            [esquema, tabla],
        ).fetchone()
        return bool(fila)

    def _asegurar_tabla_calendar(self, esquema: str, tabla: str) -> None:
        con = self.conectar()
        self.asegurar_esquema(esquema)
        definiciones = {
            **_TIPOS_EVENTO,
            "version_hash": "VARCHAR",
            "_corrida_id": "VARCHAR",
            "_fuente_id": "VARCHAR",
            "_ingestado_en": "TIMESTAMPTZ",
            "visto_por_ultima_vez": "TIMESTAMPTZ",
        }
        if not self._tabla_existe(esquema, tabla):
            columnas = ", ".join(
                f'"{nombre}" {tipo}' for nombre, tipo in definiciones.items()
            )
            con.execute(
                f'CREATE TABLE "{esquema}"."{tabla}" ({columnas})'
            )
            return

        # Migracion EN LA MISMA TABLA: si venia del conector snapshot, se
        # agregan las columnas nuevas sin renombrarla ni crear otra tabla.
        for nombre, tipo in definiciones.items():
            con.execute(
                f'ALTER TABLE "{esquema}"."{tabla}" '
                f'ADD COLUMN IF NOT EXISTS "{nombre}" {tipo}'
            )

    def actualizar_eventos_calendar(
        self,
        esquema: str,
        tabla: str,
        df: pd.DataFrame,
        corrida: Corrida,
    ) -> dict:
        con = self.conectar()
        self._asegurar_tabla_calendar(esquema, tabla)
        ahora = datetime.now(timezone.utc)
        columnas = list(CAMPOS_EVENTO) + [
            "version_hash",
            "_corrida_id",
            "_fuente_id",
            "_ingestado_en",
            "visto_por_ultima_vez",
        ]
        seleccion = ", ".join(f'"{c}"' for c in list(CAMPOS_EVENTO) + ["version_hash"])
        por_clave, por_recurso = {}, {}
        for valores in con.execute(
            f'SELECT {seleccion} FROM "{esquema}"."{tabla}"'
        ).fetchall():
            fila = dict(zip(list(CAMPOS_EVENTO) + ["version_hash"], valores))
            if fila.get("calendar_id") and fila.get("evento_id"):
                por_clave[(fila["calendar_id"], fila["evento_id"])] = fila
            if fila.get("recurso") and fila.get("evento_id"):
                por_recurso[(fila["recurso"], fila["evento_id"])] = fila

        entrantes = {}
        for _, serie in df.iterrows():
            fila = fila_evento(serie)
            clave = (fila.get("calendar_id"), fila.get("evento_id"))
            if not all(clave):
                raise RuntimeError(
                    f"Google Calendar devolvio un evento con clave vacia: {clave}"
                )
            entrantes[clave] = fila

        stats = {"nuevos": 0, "actualizados": 0, "sin_cambios": 0}
        marcas = ", ".join("?" for _ in columnas)
        nombres = ", ".join(f'"{c}"' for c in columnas)

        con.execute("BEGIN TRANSACTION")
        try:
            for clave, entrante in entrantes.items():
                recurso_clave = (entrante.get("recurso"), entrante.get("evento_id"))
                anterior = por_clave.get(clave) or por_recurso.get(recurso_clave)
                unido = fusionar_evento(anterior, entrante)
                version = hash_evento(unido)
                parametros_clave = [clave[0], clave[1], *recurso_clave]

                if anterior and anterior.get("version_hash") == version:
                    con.execute(
                        f'UPDATE "{esquema}"."{tabla}" SET '
                        "visto_por_ultima_vez=?, _corrida_id=?, _fuente_id=?, "
                        "_ingestado_en=? WHERE "
                        "(calendar_id=? AND evento_id=?) OR "
                        "(recurso=? AND evento_id=?)",
                        [ahora, corrida.corrida_id, corrida.fuente_id, ahora,
                         *parametros_clave],
                    )
                    stats["sin_cambios"] += 1
                    continue

                if anterior:
                    stats["actualizados"] += 1
                else:
                    stats["nuevos"] += 1

                con.execute(
                    f'DELETE FROM "{esquema}"."{tabla}" WHERE '
                    "(calendar_id=? AND evento_id=?) OR "
                    "(recurso=? AND evento_id=?)",
                    parametros_clave,
                )
                valores_evento = [limpiar_valor(unido.get(c)) for c in CAMPOS_EVENTO]
                con.execute(
                    f'INSERT INTO "{esquema}"."{tabla}" '
                    f"({nombres}) VALUES ({marcas})",
                    valores_evento
                    + [
                        version,
                        corrida.corrida_id,
                        corrida.fuente_id,
                        ahora,
                        ahora,
                    ],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        stats["actual_total"] = con.execute(
            f'SELECT COUNT(*) FROM "{esquema}"."{tabla}"'
        ).fetchone()[0]
        logger.info(
            "Calendar %s.%s: %d nuevos, %d actualizados, %d sin cambios; "
            "%d acumulados",
            esquema,
            tabla,
            stats["nuevos"],
            stats["actualizados"],
            stats["sin_cambios"],
            stats["actual_total"],
        )
        return stats

    # --- Zoho IMAP: create inicial + UPSERT acumulativo -----------------

    def _asegurar_tabla_correos(self, esquema: str, tabla: str) -> None:
        con = self.conectar()
        self.asegurar_esquema(esquema)
        definiciones = {
            **_TIPOS_CORREO,
            "version_hash": "VARCHAR",
            "_corrida_id": "VARCHAR",
            "_fuente_id": "VARCHAR",
            "_ingestado_en": "TIMESTAMPTZ",
            "visto_por_ultima_vez": "TIMESTAMPTZ",
        }
        if not self._tabla_existe(esquema, tabla):
            columnas = ", ".join(
                f'"{nombre}" {tipo}' for nombre, tipo in definiciones.items()
            )
            con.execute(f'CREATE TABLE "{esquema}"."{tabla}" ({columnas})')
            return

        # La primera corrida con esta version migra el snapshot EN SU MISMA
        # tabla y conserva todas las filas que ya existian.
        for nombre, tipo in definiciones.items():
            con.execute(
                f'ALTER TABLE "{esquema}"."{tabla}" '
                f'ADD COLUMN IF NOT EXISTS "{nombre}" {tipo}'
            )

    def actualizar_correos_zoho(
        self,
        esquema: str,
        tabla: str,
        df: pd.DataFrame,
        corrida: Corrida,
    ) -> dict:
        """Inserta/actualiza la ventana IMAP sin borrar correos mas antiguos."""
        con = self.conectar()
        self._asegurar_tabla_correos(esquema, tabla)
        ahora = datetime.now(timezone.utc)
        campos_lectura = list(CAMPOS_CORREO) + ["version_hash"]
        seleccion = ", ".join(f'"{c}"' for c in campos_lectura)
        columnas = list(CAMPOS_CORREO) + [
            "version_hash",
            "_corrida_id",
            "_fuente_id",
            "_ingestado_en",
            "visto_por_ultima_vez",
        ]

        por_id, por_uid, legado_por_uid = {}, {}, {}
        for valores in con.execute(
            f'SELECT {seleccion} FROM "{esquema}"."{tabla}"'
        ).fetchall():
            fila = dict(zip(campos_lectura, valores))
            if fila.get("correo_id"):
                por_id[fila["correo_id"]] = fila
            if fila.get("buzon") and fila.get("carpeta") and fila.get("uid"):
                por_uid[(fila["buzon"], fila["carpeta"], fila["uid"])] = fila
            elif fila.get("uid"):
                legado_por_uid[fila["uid"]] = fila

        entrantes = {}
        for _, serie in df.iterrows():
            fila = fila_correo(serie)
            clave = fila.get("correo_id")
            if not clave:
                raise RuntimeError("Zoho IMAP devolvio un correo sin correo_id")
            entrantes[clave] = fila

        stats = {"nuevos": 0, "actualizados": 0, "sin_cambios": 0}
        nombres = ", ".join(f'"{c}"' for c in columnas)
        marcas = ", ".join("?" for _ in columnas)
        donde = (
            "correo_id=? OR (buzon=? AND carpeta=? AND uid=?) OR "
            "(correo_id IS NULL AND uid=?)"
        )

        con.execute("BEGIN TRANSACTION")
        try:
            for clave, entrante in entrantes.items():
                clave_uid = (
                    entrante.get("buzon"),
                    entrante.get("carpeta"),
                    entrante.get("uid"),
                )
                anterior = (
                    por_id.get(clave)
                    or por_uid.get(clave_uid)
                    or legado_por_uid.get(entrante.get("uid"))
                )
                version = hash_correo(entrante)
                identidad = [
                    clave,
                    entrante.get("buzon"),
                    entrante.get("carpeta"),
                    entrante.get("uid"),
                    entrante.get("uid"),
                ]

                if anterior and anterior.get("version_hash") == version:
                    con.execute(
                        f'UPDATE "{esquema}"."{tabla}" SET '
                        "visto_por_ultima_vez=?, _corrida_id=?, _fuente_id=?, "
                        f'_ingestado_en=? WHERE {donde}',
                        [ahora, corrida.corrida_id, corrida.fuente_id, ahora,
                         *identidad],
                    )
                    stats["sin_cambios"] += 1
                    continue

                stats["actualizados" if anterior else "nuevos"] += 1
                if anterior:
                    con.execute(
                        f'DELETE FROM "{esquema}"."{tabla}" WHERE {donde}',
                        identidad,
                    )

                valores = [
                    limpiar_valor_correo(entrante.get(c)) for c in CAMPOS_CORREO
                ]
                con.execute(
                    f'INSERT INTO "{esquema}"."{tabla}" '
                    f"({nombres}) VALUES ({marcas})",
                    valores + [
                        version,
                        corrida.corrida_id,
                        corrida.fuente_id,
                        ahora,
                        ahora,
                    ],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        stats["actual_total"] = con.execute(
            f'SELECT COUNT(*) FROM "{esquema}"."{tabla}"'
        ).fetchone()[0]
        logger.info(
            "Zoho %s.%s: %d nuevos, %d actualizados, %d sin cambios; "
            "%d acumulados",
            esquema,
            tabla,
            stats["nuevos"],
            stats["actualizados"],
            stats["sin_cambios"],
            stats["actual_total"],
        )
        return stats

    # --- Meta Ads: create inicial + UPSERT acumulativo -------------------

    def _asegurar_tabla_insights(self, esquema: str, tabla: str) -> None:
        con = self.conectar()
        self.asegurar_esquema(esquema)
        definiciones = {
            **_TIPOS_INSIGHT,
            "version_hash": "VARCHAR",
            "_corrida_id": "VARCHAR",
            "_fuente_id": "VARCHAR",
            "_ingestado_en": "TIMESTAMPTZ",
            "visto_por_ultima_vez": "TIMESTAMPTZ",
        }
        if not self._tabla_existe(esquema, tabla):
            columnas = ", ".join(
                f'"{nombre}" {tipo}' for nombre, tipo in definiciones.items()
            )
            con.execute(f'CREATE TABLE "{esquema}"."{tabla}" ({columnas})')
            return

        # Meta agrega metricas nuevas seguido. ADD COLUMN IF NOT EXISTS deja
        # que la tabla crezca sin DROP y sin perder el historial acumulado.
        for nombre, tipo in definiciones.items():
            con.execute(
                f'ALTER TABLE "{esquema}"."{tabla}" '
                f'ADD COLUMN IF NOT EXISTS "{nombre}" {tipo}'
            )

    def actualizar_insights_meta(
        self,
        esquema: str,
        tabla: str,
        df: pd.DataFrame,
        corrida: Corrida,
    ) -> dict:
        """Corrige los dias de la ventana sin borrar los anteriores."""
        con = self.conectar()
        self._asegurar_tabla_insights(esquema, tabla)
        ahora = datetime.now(timezone.utc)
        campos_lectura = list(CAMPOS_INSIGHT) + ["version_hash"]
        seleccion = ", ".join(f'"{c}"' for c in campos_lectura)
        columnas = list(CAMPOS_INSIGHT) + [
            "version_hash",
            "_corrida_id",
            "_fuente_id",
            "_ingestado_en",
            "visto_por_ultima_vez",
        ]

        anteriores = {}
        for valores in con.execute(
            f'SELECT {seleccion} FROM "{esquema}"."{tabla}"'
        ).fetchall():
            fila = dict(zip(campos_lectura, valores))
            if fila.get("insight_id"):
                anteriores[fila["insight_id"]] = fila

        entrantes = {}
        for _, serie in df.iterrows():
            fila = fila_insight(serie)
            clave = fila.get("insight_id")
            if not clave:
                raise RuntimeError("Meta Ads devolvio una fila sin insight_id")
            # La misma clave dos veces en una corrida solo puede venir de un
            # breakdown mal declarado; gana la ultima y no se duplica la fila.
            entrantes[clave] = fila

        stats = {"nuevos": 0, "actualizados": 0, "sin_cambios": 0}
        nombres = ", ".join(f'"{c}"' for c in columnas)
        marcas = ", ".join("?" for _ in columnas)

        con.execute("BEGIN TRANSACTION")
        try:
            for clave, entrante in entrantes.items():
                anterior = anteriores.get(clave)
                version = hash_insight(entrante)

                if anterior and anterior.get("version_hash") == version:
                    con.execute(
                        f'UPDATE "{esquema}"."{tabla}" SET '
                        "visto_por_ultima_vez=?, _corrida_id=?, _fuente_id=?, "
                        '_ingestado_en=? WHERE insight_id=?',
                        [ahora, corrida.corrida_id, corrida.fuente_id, ahora,
                         clave],
                    )
                    stats["sin_cambios"] += 1
                    continue

                stats["actualizados" if anterior else "nuevos"] += 1
                if anterior:
                    con.execute(
                        f'DELETE FROM "{esquema}"."{tabla}" WHERE insight_id=?',
                        [clave],
                    )

                valores = [
                    limpiar_valor_meta(entrante.get(c)) for c in CAMPOS_INSIGHT
                ]
                con.execute(
                    f'INSERT INTO "{esquema}"."{tabla}" '
                    f"({nombres}) VALUES ({marcas})",
                    valores + [
                        version,
                        corrida.corrida_id,
                        corrida.fuente_id,
                        ahora,
                        ahora,
                    ],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        stats["actual_total"] = con.execute(
            f'SELECT COUNT(*) FROM "{esquema}"."{tabla}"'
        ).fetchone()[0]
        logger.info(
            "Meta Ads %s.%s: %d nuevos, %d corregidos, %d sin cambios; "
            "%d acumulados",
            esquema,
            tabla,
            stats["nuevos"],
            stats["actualizados"],
            stats["sin_cambios"],
            stats["actual_total"],
        )
        return stats

    # --- Capa semantica: reconstruccion completa -------------------------

    def reconstruir_tabla(self, esquema: str, tabla: str, columnas: list,
                          filas: list):
        con = self.conectar()
        self.asegurar_esquema(esquema)
        definicion = ", ".join(
            f'"{c}" {_SQL_SEMANTICO.get(t, "VARCHAR")}' for c, t in columnas)
        nombres = [c for c, _ in columnas]

        # DROP y CREATE en la MISMA transaccion que el INSERT: si algo falla a
        # mitad, la tabla anterior sigue en pie. Sin esto, un error al insertar
        # dejaria al cliente sin tabla hasta la proxima corrida.
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f'DROP TABLE IF EXISTS "{esquema}"."{tabla}"')
            con.execute(f'CREATE TABLE "{esquema}"."{tabla}" ({definicion})')
            if filas:
                marcas = ", ".join("?" for _ in nombres)
                con.executemany(
                    f'INSERT INTO "{esquema}"."{tabla}" VALUES ({marcas})',
                    [[limpiar_valor(f.get(c)) for c in nombres] for f in filas],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        logger.info("semantica %s.%s: %d filas", esquema, tabla, len(filas))

    def leer_filas(self, sql: str, params: dict = None) -> list:
        # Los :nombre se resuelven a marcadores posicionales para que el SQL
        # que escribe construir_modelos.py sea el mismo en los dos destinos.
        valores = []
        if params:
            for clave, valor in params.items():
                if f":{clave}" in sql:
                    sql = sql.replace(f":{clave}", "?")
                    valores.append(valor)
        con = self.conectar()
        res = con.execute(sql, valores) if valores else con.execute(sql)
        columnas = [d[0] for d in res.description]
        return [dict(zip(columnas, f)) for f in res.fetchall()]

    def escribir_catalogo(self, esquema: str, fuente_id: str, filas: list):
        con = self.conectar()
        self.asegurar_esquema(esquema)
        cols = (
            "fuente_id", "tabla", "columna", "descripcion", "instruccion",
            "sistema_origen", "frecuencia", "dueno", "editable",
            "acciones_permitidas", "origen_edicion", "clave_primaria",
            "origen_tipo", "origen_fuente_id", "hoja_origen", "anulacion_campo",
            "requerido", "editable_campo", "tipo_validacion",
            "valores_permitidos", "valor_por_defecto", "calculado_por_sistema",
            "generador", "etiqueta_usuario", "ejemplo",
        )
        con.execute(
            f'CREATE TABLE IF NOT EXISTS "{esquema}"."_catalogo" ('
            "fuente_id VARCHAR, tabla VARCHAR, columna VARCHAR, descripcion VARCHAR,"
            "instruccion VARCHAR,"
            "sistema_origen VARCHAR, frecuencia VARCHAR, dueno VARCHAR,"
            "editable VARCHAR, acciones_permitidas VARCHAR, origen_edicion VARCHAR,"
            "clave_primaria VARCHAR, origen_tipo VARCHAR, origen_fuente_id VARCHAR,"
            "hoja_origen VARCHAR, anulacion_campo VARCHAR, requerido VARCHAR,"
            "editable_campo VARCHAR, tipo_validacion VARCHAR, valores_permitidos VARCHAR,"
            "valor_por_defecto VARCHAR, calculado_por_sistema VARCHAR,"
            "generador VARCHAR, etiqueta_usuario VARCHAR, ejemplo VARCHAR)"
        )
        # Migracion suave para _catalogo creado por una version previa.
        for col in cols[4:]:
            con.execute(
                f'ALTER TABLE "{esquema}"."_catalogo" ADD COLUMN IF NOT EXISTS "{col}" VARCHAR'
            )
        con.execute(f'DELETE FROM "{esquema}"."_catalogo" WHERE fuente_id=?', [fuente_id])
        for f in filas:
            campos = ",".join(cols)
            marcas = ",".join("?" for _ in cols)
            valores = [fuente_id if c == "fuente_id" else f.get(c, "") for c in cols]
            con.execute(
                f'INSERT INTO "{esquema}"."_catalogo" ({campos}) VALUES ({marcas})', valores,
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

