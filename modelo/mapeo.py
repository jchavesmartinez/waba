"""Lectura segura del mapeo persistente de clasificaciones.

La tabla puede faltar antes de la primera clasificación. Una falla al leer una
tabla que sí existe nunca debe interpretarse como un mapeo vacío: publicar una
reconstrucción en ese estado devolvería gastos históricos a sin_clasificar.
"""


def leer_mapeo(destino, esquema: str, modelo_id: str | None = None) -> list[dict]:
    sql = f'SELECT * FROM "{esquema}"."_mapeo"'
    parametros = None
    if modelo_id is not None:
        sql += " WHERE modelo_id = :modelo_id"
        parametros = {"modelo_id": modelo_id}
    try:
        return destino.leer_filas(sql, parametros)
    except Exception as exc:
        # Solo la ausencia comprobada de la tabla equivale a primera corrida.
        # Una desconexión, falta de permisos o cambio de esquema debe detener
        # la publicación y conservar la tabla semántica anterior.
        try:
            existe = destino.leer_filas(
                "SELECT 1 AS existe FROM information_schema.tables "
                "WHERE table_schema = :esquema AND table_name = :tabla LIMIT 1",
                {"esquema": esquema, "tabla": "_mapeo"},
            )
        except Exception:
            raise RuntimeError(
                f"no se pudo verificar el mapeo de {esquema}; se conserva "
                "la clasificación publicada"
            ) from exc
        if not existe:
            return []
        raise RuntimeError(
            f"no se pudo leer {esquema}._mapeo; se conserva "
            "la clasificación publicada"
        ) from exc
