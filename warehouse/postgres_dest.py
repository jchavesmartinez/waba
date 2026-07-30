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
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

import config
from .base import Destino, Corrida, ESQUEMA_META, TABLA_CORRIDAS

logger = logging.getLogger("fachavi.warehouse.postgres")

# Cuantas corridas OK hacia atras se fusionan para armar el punto de
# comparacion de la guarda de vaciado (C-01). Ver ultimo_detalle().
_CORRIDAS_A_FUSIONAR = 20


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
