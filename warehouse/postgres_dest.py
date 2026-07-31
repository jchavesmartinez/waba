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
import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from sources.google_calendar import (
    CAMPOS_EVENTO,
    fila_evento,
    fusionar_evento,
    hash_evento,
    limpiar_valor,
)
import config
from .base import Destino, Corrida, ESQUEMA_META, TABLA_CORRIDAS

logger = logging.getLogger("fachavi.warehouse.postgres")

# Cuantas corridas OK hacia atras se fusionan para armar el punto de
# comparacion de la guarda de vaciado (C-01). Ver ultimo_detalle().
_CORRIDAS_A_FUSIONAR = 20

_TIPOS_EVENTO = {
    "calendar_id": "TEXT",
    "recurso": "TEXT",
    "evento_id": "TEXT",
    "ical_uid": "TEXT",
    "titulo": "TEXT",
    "descripcion": "TEXT",
    "ubicacion": "TEXT",
    "inicio": "TIMESTAMPTZ",
    "fin": "TIMESTAMPTZ",
    "zona_horaria": "TEXT",
    "duracion_min": "BIGINT",
    "todo_el_dia": "BOOLEAN",
    "estado": "TEXT",
    "transparencia": "TEXT",
    "visibilidad": "TEXT",
    "tipo_evento": "TEXT",
    "evento_recurrente_id": "TEXT",
    "inicio_original": "TIMESTAMPTZ",
    "creado_en": "TIMESTAMPTZ",
    "actualizado_en": "TIMESTAMPTZ",
    "enlace_evento": "TEXT",
    "organizador": "TEXT",
    "invitados": "TEXT",
    "propiedades": "TEXT",
    "raw_evento": "TEXT",
}


def _ident(nombre: str) -> str:
    """
    Normaliza un identificador SQL (esquema, tabla, columna) que viene de una
    hoja de calculo del cliente. Es la MISMA regla que usa la ingesta al crear
    las tablas, asi que un nombre legitimo pasa sin cambios.

    C-03: sin esto, un nombre de columna con comillas dobles podia cerrar el
    identificador de un COMMENT ON y continuar con SQL propio.
    """
    limpio = re.sub(r"[^0-9a-zA-Z_]", "_", str(nombre).strip().lower())
    if not limpio or limpio[0].isdigit():
        limpio = "t_" + limpio
    return limpio


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
        """
        Full refresh ATOMICO (A-02, B-17).

        Antes: to_sql(if_exists="replace") hacia DROP + CREATE + INSERT. Entre
        el DROP y el final del INSERT la tabla NO EXISTIA: si el bot consultaba
        en ese instante, la consulta reventaba. Y el DROP se llevaba puestos los
        GRANT otorgados sobre la tabla (B-17), asi que un rol de solo lectura
        con permisos por tabla se rompia en cada corrida.

        Ahora: se escribe en una tabla temporal <tabla>__nueva y se hace el
        swap con DROP + RENAME dentro de UNA sola transaccion. Postgres hace
        DDL transaccional, asi que el cambio de nombre es instantaneo y atomico
        para cualquier lector.
        """
        self.asegurar_esquema(esquema)
        tabla = _ident(tabla)
        staging = f"{tabla}__nueva"[:63]     # Postgres trunca a 63 chars

        eng = self.conectar()
        with eng.begin() as cx:
            cx.execute(text(f'DROP TABLE IF EXISTS "{esquema}"."{staging}"'))

        df.to_sql(
            staging,
            eng,
            schema=esquema,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )

        with eng.begin() as cx:
            cx.execute(text(f'DROP TABLE IF EXISTS "{esquema}"."{tabla}"'))
            cx.execute(text(
                f'ALTER TABLE "{esquema}"."{staging}" RENAME TO "{tabla}"'
            ))
        logger.info("escrito %s.%s (%d filas, swap atomico)", esquema, tabla, len(df))

    # --- Google Calendar: estado actual + historico SCD2 ---

    def _asegurar_tablas_calendar(
        self, esquema: str, tabla_actual: str, tabla_historial: str
    ) -> tuple[str, str, str]:
        esquema = _ident(esquema)
        tabla_actual = _ident(tabla_actual)
        tabla_historial = _ident(tabla_historial)
        self.asegurar_esquema(esquema)

        with self.conectar().begin() as cx:
            existe = cx.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema=:e AND table_name=:t)"
                ),
                {"e": esquema, "t": tabla_actual},
            ).scalar()
            tiene_calendar_id = False
            if existe:
                tiene_calendar_id = cx.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                        "WHERE table_schema=:e AND table_name=:t "
                        "AND column_name='calendar_id')"
                    ),
                    {"e": esquema, "t": tabla_actual},
                ).scalar()

            if existe and not tiene_calendar_id:
                base = _ident(f"{tabla_actual}__snapshot_legacy")[:63]
                legado = base
                numero = 2
                while cx.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema=:e AND table_name=:t)"
                    ),
                    {"e": esquema, "t": legado},
                ).scalar():
                    sufijo = f"_{numero}"
                    legado = f"{base[:63-len(sufijo)]}{sufijo}"
                    numero += 1
                cx.execute(
                    text(
                        f'ALTER TABLE "{esquema}"."{tabla_actual}" '
                        f'RENAME TO "{legado}"'
                    )
                )
                logger.warning(
                    "la tabla anterior %s.%s se preservo como %s antes de "
                    "activar el historico de Calendar",
                    esquema,
                    tabla_actual,
                    legado,
                )

            columnas = ", ".join(
                f'"{campo}" {_TIPOS_EVENTO[campo]}' for campo in CAMPOS_EVENTO
            )
            cx.execute(
                text(
                    f'CREATE TABLE IF NOT EXISTS "{esquema}"."{tabla_actual}" ('
                    f"{columnas}, version_hash TEXT NOT NULL, "
                    "_corrida_id TEXT, _fuente_id TEXT, "
                    "_ingestado_en TIMESTAMPTZ, "
                    "visto_por_ultima_vez TIMESTAMPTZ, "
                    "PRIMARY KEY (calendar_id, evento_id))"
                )
            )
            cx.execute(
                text(
                    f'CREATE TABLE IF NOT EXISTS "{esquema}"."{tabla_historial}" ('
                    f"{columnas}, version_hash TEXT NOT NULL, "
                    "vigente_desde TIMESTAMPTZ, vigente_hasta TIMESTAMPTZ, "
                    "es_version_actual BOOLEAN, _corrida_id TEXT, "
                    "_fuente_id TEXT, _ingestado_en TIMESTAMPTZ)"
                )
            )
            indice = _ident(f"{tabla_historial}__evento_idx")[:63]
            cx.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "{indice}" ON '
                    f'"{esquema}"."{tabla_historial}" '
                    "(calendar_id, evento_id, vigente_desde)"
                )
            )
        return esquema, tabla_actual, tabla_historial

    def fusionar_eventos_calendar(
        self,
        esquema: str,
        tabla_actual: str,
        tabla_historial: str,
        df: pd.DataFrame,
        corrida: Corrida,
        calendarios_reiniciados: tuple | list = (),
    ) -> dict:
        esquema, tabla_actual, tabla_historial = self._asegurar_tablas_calendar(
            esquema, tabla_actual, tabla_historial
        )
        ahora = datetime.now(timezone.utc)
        campos_lectura = list(CAMPOS_EVENTO) + ["version_hash"]
        seleccion = ", ".join(f'"{c}"' for c in campos_lectura)
        columnas_actual = list(CAMPOS_EVENTO) + [
            "version_hash",
            "_corrida_id",
            "_fuente_id",
            "_ingestado_en",
            "visto_por_ultima_vez",
        ]
        columnas_hist = list(CAMPOS_EVENTO) + [
            "version_hash",
            "vigente_desde",
            "vigente_hasta",
            "es_version_actual",
            "_corrida_id",
            "_fuente_id",
            "_ingestado_en",
        ]

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
        eng = self.conectar()
        with eng.begin() as cx:
            existentes = {}
            for fila in cx.execute(
                text(f'SELECT {seleccion} FROM "{esquema}"."{tabla_actual}"')
            ).mappings():
                d = dict(fila)
                existentes[(d["calendar_id"], d["evento_id"])] = d

            nombres_actual = ", ".join(f'"{c}"' for c in columnas_actual)
            binds_actual = ", ".join(f":{c}" for c in columnas_actual)
            nombres_hist = ", ".join(f'"{c}"' for c in columnas_hist)
            binds_hist = ", ".join(f":{c}" for c in columnas_hist)

            for clave, entrante in entrantes.items():
                anterior = existentes.get(clave)
                unido = fusionar_evento(anterior, entrante)
                version = hash_evento(unido)
                clave_sql = {"calendar_id": clave[0], "evento_id": clave[1]}

                if anterior and anterior.get("version_hash") == version:
                    cx.execute(
                        text(
                            f'UPDATE "{esquema}"."{tabla_actual}" SET '
                            "visto_por_ultima_vez=:ahora, "
                            "_corrida_id=:corrida, _fuente_id=:fuente, "
                            "_ingestado_en=:ahora "
                            "WHERE calendar_id=:calendar_id "
                            "AND evento_id=:evento_id"
                        ),
                        {
                            **clave_sql,
                            "ahora": ahora,
                            "corrida": corrida.corrida_id,
                            "fuente": corrida.fuente_id,
                        },
                    )
                    stats["sin_cambios"] += 1
                    continue

                if anterior:
                    cx.execute(
                        text(
                            f'UPDATE "{esquema}"."{tabla_historial}" SET '
                            "vigente_hasta=:ahora, es_version_actual=false "
                            "WHERE calendar_id=:calendar_id "
                            "AND evento_id=:evento_id "
                            "AND es_version_actual=true"
                        ),
                        {**clave_sql, "ahora": ahora},
                    )
                    stats["cambiados"] += 1
                else:
                    stats["nuevos"] += 1

                cx.execute(
                    text(
                        f'DELETE FROM "{esquema}"."{tabla_actual}" '
                        "WHERE calendar_id=:calendar_id "
                        "AND evento_id=:evento_id"
                    ),
                    clave_sql,
                )
                base = {c: limpiar_valor(unido.get(c)) for c in CAMPOS_EVENTO}
                actual = {
                    **base,
                    "version_hash": version,
                    "_corrida_id": corrida.corrida_id,
                    "_fuente_id": corrida.fuente_id,
                    "_ingestado_en": ahora,
                    "visto_por_ultima_vez": ahora,
                }
                cx.execute(
                    text(
                        f'INSERT INTO "{esquema}"."{tabla_actual}" '
                        f"({nombres_actual}) VALUES ({binds_actual})"
                    ),
                    actual,
                )
                historica = {
                    **base,
                    "version_hash": version,
                    "vigente_desde": ahora,
                    "vigente_hasta": None,
                    "es_version_actual": True,
                    "_corrida_id": corrida.corrida_id,
                    "_fuente_id": corrida.fuente_id,
                    "_ingestado_en": ahora,
                }
                cx.execute(
                    text(
                        f'INSERT INTO "{esquema}"."{tabla_historial}" '
                        f"({nombres_hist}) VALUES ({binds_hist})"
                    ),
                    historica,
                )

            # Google exige reconstruir el store tras un 410. Solo se limpia la
            # tabla de estado actual; las versiones historicas se conservan.
            for clave in existentes:
                if clave[0] in reiniciados and clave not in entrantes:
                    cx.execute(
                        text(
                            f'DELETE FROM "{esquema}"."{tabla_actual}" '
                            "WHERE calendar_id=:calendar_id "
                            "AND evento_id=:evento_id"
                        ),
                        {"calendar_id": clave[0], "evento_id": clave[1]},
                    )
                    stats["retirados_actual"] += 1

            stats["actual_total"] = cx.execute(
                text(f'SELECT COUNT(*) FROM "{esquema}"."{tabla_actual}"')
            ).scalar()
            stats["historial_total"] = cx.execute(
                text(f'SELECT COUNT(*) FROM "{esquema}"."{tabla_historial}"')
            ).scalar()

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

    def _asegurar_estado_calendar(self, esquema: str) -> str:
        esquema = _ident(esquema)
        self.asegurar_esquema(esquema)
        with self.conectar().begin() as cx:
            cx.execute(
                text(
                    f'CREATE TABLE IF NOT EXISTS "{esquema}".'
                    '"_calendar_sync_state" ('
                    "fuente_id TEXT, calendar_id TEXT, sync_token TEXT, "
                    "actualizado_en TIMESTAMPTZ, "
                    "PRIMARY KEY (fuente_id, calendar_id))"
                )
            )
        return esquema

    def leer_estado_calendar(self, esquema: str, fuente_id: str) -> dict:
        esquema = _ident(esquema)
        with self.conectar().begin() as cx:
            existe = cx.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema=:e "
                    "AND table_name='_calendar_sync_state')"
                ),
                {"e": esquema},
            ).scalar()
            if not existe:
                return {}
            filas = cx.execute(
                text(
                    f'SELECT calendar_id, sync_token FROM "{esquema}".'
                    '"_calendar_sync_state" WHERE fuente_id=:f'
                ),
                {"f": fuente_id},
            ).fetchall()
        return {calendar_id: token for calendar_id, token in filas}

    def guardar_estado_calendar(
        self, esquema: str, fuente_id: str, sync_tokens: dict
    ) -> None:
        esquema = self._asegurar_estado_calendar(esquema)
        ahora = datetime.now(timezone.utc)
        with self.conectar().begin() as cx:
            for calendar_id, token in sync_tokens.items():
                cx.execute(
                    text(
                        f'INSERT INTO "{esquema}"."_calendar_sync_state" '
                        "(fuente_id,calendar_id,sync_token,actualizado_en) "
                        "VALUES (:f,:c,:t,:a) "
                        "ON CONFLICT (fuente_id,calendar_id) DO UPDATE SET "
                        "sync_token=EXCLUDED.sync_token, "
                        "actualizado_en=EXCLUDED.actualizado_en"
                    ),
                    {"f": fuente_id, "c": calendar_id, "t": token, "a": ahora},
                )

    def escribir_catalogo(self, esquema: str, fuente_id: str, filas: list):
        self.asegurar_esquema(esquema)
        with self.conectar().begin() as cx:
            cx.execute(text(
                f'CREATE TABLE IF NOT EXISTS "{esquema}"."_catalogo" ('
                "fuente_id TEXT, tabla TEXT, columna TEXT, descripcion TEXT,"
                "instruccion TEXT,"
                "sistema_origen TEXT, frecuencia TEXT, dueno TEXT)"
            ))
            # Migracion suave: _catalogo creado por una version vieja no tiene
            # 'instruccion'. La agregamos si falta, para no reventar el INSERT ni
            # obligar a recrear la tabla a mano en Neon.
            cx.execute(text(
                f'ALTER TABLE "{esquema}"."_catalogo" '
                "ADD COLUMN IF NOT EXISTS instruccion TEXT"
            ))
            cx.execute(text(f'DELETE FROM "{esquema}"."_catalogo" WHERE fuente_id=:f'),
                       {"f": fuente_id})
            for f in filas:
                cx.execute(text(
                    f'INSERT INTO "{esquema}"."_catalogo" '
                    "(fuente_id,tabla,columna,descripcion,instruccion,"
                    "sistema_origen,frecuencia,dueno) VALUES "
                    "(:fid,:tab,:col,:des,:ins,:sis,:fre,:due)"),
                    {"fid": fuente_id, "tab": f.get("tabla",""), "col": f.get("columna",""),
                     "des": f.get("descripcion",""), "ins": f.get("instruccion",""),
                     "sis": f.get("sistema_origen",""),
                     "fre": f.get("frecuencia",""), "due": f.get("dueno","")})
        logger.info("catalogo de '%s': %d filas en %s._catalogo", fuente_id, len(filas), esquema)

    def escribir_kpis(self, esquema: str, fuente_id: str, filas: list):
        """
        Persiste los KPIs (capa semantica) en <esquema>._kpis.

        A-05: antes se hacia REEMPLAZO TOTAL (DELETE sin WHERE). Con dos fuentes
        del mismo cliente que trajeran tab '_kpis', sobrevivian solo los de la
        ultima en correr — y cual corria ultima dependia de la frescura, asi que
        el resultado cambiaba entre corridas. Un no-determinismo silencioso.

        Ahora se borra e inserta SOLO lo de esta fuente, igual que el catalogo.
        Si dos fuentes definen el mismo kpi_id, el choque queda VISIBLE (dos
        filas) en vez de resolverse por orden de ejecucion, y se avisa fuerte.
        """
        self.asegurar_esquema(esquema)
        cols = ("kpi", "nombre", "descripcion", "preguntas_ejemplo", "formula_sql",
                "tabla", "dimensiones", "unidad", "supuestos", "minimo_datos",
                "instruccion")
        ids_nuevos = {str(f.get("kpi", "")).strip().lower() for f in filas if f.get("kpi")}
        with self.conectar().begin() as cx:
            cx.execute(text(
                f'CREATE TABLE IF NOT EXISTS "{esquema}"."_kpis" ('
                "fuente_id TEXT, " + ", ".join(f"{c} TEXT" for c in cols) + ")"
            ))
            # Solo esta fuente: no se pisan los KPIs de las otras.
            cx.execute(text(f'DELETE FROM "{esquema}"."_kpis" WHERE fuente_id=:f'),
                       {"f": fuente_id})

            # Aviso de colision entre fuentes: mismo kpi definido dos veces.
            if ids_nuevos:
                ajenos = cx.execute(text(
                    f'SELECT DISTINCT kpi, fuente_id FROM "{esquema}"."_kpis" '
                    "WHERE lower(trim(kpi)) = ANY(:ids)"
                ), {"ids": sorted(ids_nuevos)}).fetchall()
                for kpi_id, otra in ajenos:
                    logger.warning(
                        "KPI '%s' esta definido en dos fuentes ('%s' y '%s') del "
                        "mismo cliente. Dejalo en UN solo tab '_kpis'.",
                        kpi_id, otra, fuente_id,
                    )

            campos = ", ".join(["fuente_id"] + list(cols))
            binds = ", ".join([":fuente_id"] + [f":{c}" for c in cols])
            for f in filas:
                params = {"fuente_id": fuente_id}
                params.update({c: f.get(c, "") for c in cols})
                cx.execute(text(
                    f'INSERT INTO "{esquema}"."_kpis" ({campos}) VALUES ({binds})'
                ), params)
        logger.info("kpis de '%s': %d filas en %s._kpis", fuente_id, len(filas), esquema)

    def aplicar_comentarios(self, esquema: str, mapa_tablas: dict, filas: list):
        """
        BONUS de Postgres: ademas de la tabla, escribe la descripcion como
        COMMENT ON nativo. Asi Metabase, DBeaver, dbt y cualquier herramienta
        que lea information_schema muestran la documentacion sin saber nada
        de nuestro catalogo.

        C-03: COMMENT ON no acepta parametros ligados, asi que el SQL se arma
        pegando texto. Tres defensas:
          1. El nombre de columna pasa por _ident() (misma normalizacion con la
             que se creo la tabla). Una comilla doble deja de poder cerrar el
             identificador.
          2. Se valida contra information_schema que la columna EXISTA de
             verdad en esa tabla. Lista blanca, no lista negra.
          3. Los fallos se registran en WARNING, no en DEBUG: un intento de
             inyeccion tiene que dejar rastro visible en el log.
        """
        def esc(s):
            return str(s).replace("'", "''")

        with self.conectar().begin() as cx:
            # Lista blanca de columnas reales por tabla (una sola consulta).
            reales: dict = {}
            for t, c in cx.execute(text(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = :esq"
            ), {"esq": esquema}).fetchall():
                reales.setdefault(t, set()).add(c)

            for f in filas:
                tabla_real = mapa_tablas.get(str(f.get("tabla", "")).strip().lower())
                desc = f.get("descripcion", "")
                if not tabla_real or not desc:
                    continue
                if tabla_real not in reales:
                    continue
                crudo = str(f.get("columna", "")).strip()
                try:
                    if crudo in ("*", ""):
                        cx.execute(text(
                            f'COMMENT ON TABLE "{esquema}"."{tabla_real}" '
                            f"IS '{esc(desc)}'"))
                        continue

                    col = _ident(crudo)
                    if col not in reales[tabla_real]:
                        logger.warning(
                            "catalogo: la columna '%s' no existe en %s.%s; no se "
                            "comenta. Revisa la pestania _catalogo del cliente.",
                            crudo, esquema, tabla_real,
                        )
                        continue
                    cx.execute(text(
                        f'COMMENT ON COLUMN "{esquema}"."{tabla_real}"."{col}" '
                        f"IS '{esc(desc)}'"))
                except Exception as e:  # noqa: BLE001
                    logger.warning("comentario omitido (%s.%s): %s", tabla_real, crudo, e)

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
            # A-04: sin este indice, CADA revision de frescura y CADA busqueda
            # del detalle previo recorren la bitacora completa. Con 4 fuentes y
            # 96 corridas diarias son ~140.000 filas al año y la degradacion es
            # gradual, o sea dificil de atribuir.
            cx.execute(text(
                f'CREATE INDEX IF NOT EXISTS ix_corridas_cliente_fuente_fin '
                f'ON "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
                "(cliente_id, fuente_id, fin DESC)"
            ))

    def purgar_corridas(self, dias: int = 0) -> int:
        """
        A-04: borra corridas mas viejas que `dias` (0 = usa
        config.SYNC_RETENCION_DIAS; <=0 = no purga). Devuelve cuantas borro.
        La llama sync.py al final de cada corrida completa.
        """
        dias = int(dias or getattr(config, "SYNC_RETENCION_DIAS", 0) or 0)
        if dias <= 0:
            return 0
        with self.conectar().begin() as cx:
            res = cx.execute(text(
                f'DELETE FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
                "WHERE fin < now() - make_interval(days => :d)"
            ), {"d": dias})
        borradas = res.rowcount or 0
        if borradas:
            logger.info("bitacora: %d corridas de mas de %d dias purgadas",
                        borradas, dias)
        return borradas

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
        """
        Punto de comparacion de la guarda de vaciado.

        C-01: antes devolvia el detalle de la ULTIMA corrida OK. Si esa corrida
        no habia registrado una tabla —justamente lo que pasa cuando la guarda
        bloquea una escritura— la corrida siguiente se quedaba sin con que
        comparar y escribia las 0 filas encima. O sea: la guarda protegia 15
        minutos y despues se desarmaba sola.

        Ahora se FUSIONAN las ultimas N corridas OK, de la mas vieja a la mas
        nueva, de modo que el valor mas reciente de CADA tabla gane y ninguna
        desaparezca por un hueco en el historial.
        """
        with self.conectar().begin() as cx:
            filas = cx.execute(
                text(
                    f'SELECT detalle FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
                    "WHERE cliente_id=:c AND fuente_id=:f AND estado LIKE 'ok%' "
                    "ORDER BY fin DESC LIMIT :n"
                ),
                {"c": cliente_id, "f": fuente_id, "n": _CORRIDAS_A_FUSIONAR},
            ).fetchall()

        fusionado: dict = {}
        for (crudo,) in reversed(filas):   # de la mas vieja a la mas nueva
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
        with self.conectar().begin() as cx:
            fila = cx.execute(
                text(
                    f'SELECT MAX(fin) FROM "{ESQUEMA_META}"."{TABLA_CORRIDAS}" '
                    "WHERE cliente_id=:c AND fuente_id=:f AND estado LIKE 'ok%'"
                ),
                {"c": cliente_id, "f": fuente_id},
            ).fetchone()
        return fila[0] if fila and fila[0] else None
