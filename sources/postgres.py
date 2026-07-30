"""
Fuente: PostgreSQL (ejemplo listo para activar).

Trae tablas de un Postgres del cliente hacia la DuckDB compartida usando la
extension 'postgres' de DuckDB (no requiere dependencias Python extra; DuckDB
descarga la extension la primera vez).

Config esperada (JSON en la columna 'config'):
  {
    "dsn_env": "PG_FERRETERIA_A",              # NOMBRE de la variable de entorno
                                               #   que guarda la cadena de conexion
    "esquema": "public",
    "tablas": ["ventas", "clientes"],
    "where": {"ventas": "fecha >= current_date - 90"},   # opcional, ver A-11
    "catalogo": [ ... opcional, mismo formato que _catalogo ... ]
  }

SEGURIDAD — C-04. El campo "dsn" con la cadena completa escrita ADENTRO de la
hoja de calculo sigue funcionando por retrocompatibilidad, pero esta
DESACONSEJADO y deja una advertencia en el log. El README del propio proyecto
dice, en negrita, que el DSN nunca va en el Sheet porque llevaria la contraseña
en una hoja de calculo — y esa regla se venia aplicando solo al warehouse
propio, no a las bases de los clientes. "dsn_env" cierra el hueco reusando el
mecanismo que ya existe y esta probado en config.dsn_de_cliente().

Nota de seguridad (C-03): aca corremos ATTACH/INSTALL/LOAD directamente sobre la
conexion DuckDB desde NUESTRO codigo (confiable), pero la CADENA de conexion
viene de una celda del cliente, asi que se valida y se escapa antes de pegarla
en el SQL. El SQL que genera el modelo sigue pasando por el validador de
solo-lectura en nl2sql. Traemos una copia de cada tabla (CREATE TABLE AS
SELECT), asi la base del cliente nunca se toca durante las consultas.
"""

import logging
import re

import config
from .base import (
    Source,
    Fragmento,
    describir_tabla,
    limpiar_nombre,
)

logger = logging.getLogger("fachavi.sources.postgres")

# Un DSN legitimo no tiene comillas, punto y coma ni saltos de linea. Si los
# tiene, o alguien se equivoco al pegarlo o esta intentando cerrar el literal
# del ATTACH y seguir con SQL propio. En los dos casos: no corre.
_DSN_PROHIBIDO = re.compile(r"['\";\n\r]")


class PostgresSource(Source):
    tipo = "postgres"

    def cargar(self, con) -> Fragmento:
        dsn = self._dsn()
        tablas_pedidas = self.config.get("tablas", [])
        esquema = limpiar_nombre(self.config.get("esquema", "public") or "public")
        filtros = self.config.get("where", {}) or {}
        if not dsn or not tablas_pedidas:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (postgres) requiere 'dsn_env' (o 'dsn') "
                "y 'tablas'."
            )

        con.execute("INSTALL postgres; LOAD postgres;")
        alias = f"pg_{limpiar_nombre(self.fuente_id)}"
        con.execute(f"ATTACH '{dsn}' AS {alias} (TYPE postgres, READ_ONLY)")

        schema_parts = []
        tablas = []
        try:
            for t in tablas_pedidas:
                origen = f"{alias}.{esquema}.{limpiar_nombre(t)}"
                destino = limpiar_nombre(t)
                # A-11: sin filtro se copia la tabla ENTERA en cada corrida. Con
                # una tabla de millones de filas eso es caro y lento. El filtro
                # es SQL del administrador (no del modelo ni del usuario final),
                # se escribe una vez en la config de la fuente.
                cond = str(filtros.get(t, filtros.get(destino, ""))).strip()
                sql = f"CREATE TABLE {destino} AS SELECT * FROM {origen}"
                if cond:
                    sql += f" WHERE {cond}"
                    logger.info("[postgres:%s] %s con filtro: %s",
                                self.fuente_id, destino, cond)
                # copia de solo lectura hacia DuckDB (la base del cliente no se modifica)
                con.execute(sql)
                tablas.append(destino)
                schema_parts.append(describir_tabla(con, destino))
                logger.info("[postgres:%s] tabla %s copiada", self.fuente_id, destino)
        finally:
            con.execute(f"DETACH {alias}")

        catalogo_filas = self.config.get("catalogo", [])
        from .base import construir_catalogo
        catalogo = construir_catalogo(catalogo_filas) if catalogo_filas else ""

        return Fragmento(
            schema="\n\n".join(schema_parts),
            catalogo=catalogo,
            tablas=tablas,
        )

    def _dsn(self) -> str:
        """Resuelve la cadena de conexion: variable de entorno primero (C-04)."""
        nombre_var = str(self.config.get("dsn_env", "")).strip()
        if nombre_var:
            dsn = config.secreto_de_env(
                nombre_var, para=f"La fuente '{self.fuente_id}' (postgres)"
            )
        else:
            dsn = str(self.config.get("dsn", "")).strip()
            if dsn:
                logger.warning(
                    "[%s] la cadena de conexion (con la CONTRASEÑA del cliente) "
                    "esta escrita en la hoja de calculo. Cualquiera con acceso de "
                    "lectura la ve, y el historial de versiones de Google la "
                    "guarda para siempre. Movela a una variable de entorno y usa "
                    '"dsn_env".',
                    self.fuente_id,
                )

        if dsn and _DSN_PROHIBIDO.search(dsn):
            raise RuntimeError(
                f"La fuente '{self.fuente_id}' tiene una cadena de conexion con "
                "caracteres no permitidos (comillas, punto y coma o saltos de "
                "linea). Revisala."
            )
        return dsn
