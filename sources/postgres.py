"""
Fuente: PostgreSQL (ejemplo listo para activar).

Trae tablas de un Postgres del cliente hacia la DuckDB compartida usando la
extension 'postgres' de DuckDB (no requiere dependencias Python extra; DuckDB
descarga la extension la primera vez).

Config esperada (JSON en la columna 'config'):
  {
    "dsn": "postgresql://usuario:clave@host:5432/basedatos",
    "esquema": "public",
    "tablas": ["ventas", "clientes"],
    "catalogo": [ ... opcional, mismo formato que _catalogo ... ]
  }

Nota de seguridad: aca corremos ATTACH/INSTALL/LOAD directamente sobre la
conexion DuckDB desde NUESTRO codigo (confiable). El SQL que genera el modelo
sigue pasando por el validador de solo-lectura en nl2sql, que bloquea justamente
esas sentencias. Traemos una copia de cada tabla (CREATE TABLE AS SELECT), asi
la base del cliente nunca se toca durante las consultas.
"""

import logging

from .base import (
    Source,
    Fragmento,
    describir_tabla,
    limpiar_nombre,
)

logger = logging.getLogger("fachavi.sources.postgres")


class PostgresSource(Source):
    tipo = "postgres"

    def cargar(self, con) -> Fragmento:
        dsn = self.config.get("dsn", "").strip()
        tablas_pedidas = self.config.get("tablas", [])
        esquema = self.config.get("esquema", "public").strip() or "public"
        if not dsn or not tablas_pedidas:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (postgres) requiere 'dsn' y 'tablas'."
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
                # copia de solo lectura hacia DuckDB (la base del cliente no se modifica)
                con.execute(f"CREATE TABLE {destino} AS SELECT * FROM {origen}")
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
