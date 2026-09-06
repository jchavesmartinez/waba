"""Puerta genérica y agnóstica de cliente para consultas de solo lectura."""

from bot import nl2sql, seguimiento


def validar(sql: str, pregunta: str, ctx, contrato: dict | None = None) -> tuple[bool, str]:
    """Valida seguridad, forma de salida y continuidad antes de ejecutar.

    No conoce tablas ni reglas de un cliente: ambas provienen de ``ctx`` y del
    contrato persistido. Así finanzas, inventario, reservas y ventas pasan por
    la misma puerta.
    """
    ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
    if ok:
        ok, motivo = nl2sql.validar_granularidad(pregunta, sql)
    if ok and contrato and contrato.get("relacion") in {"seguimiento", "modificacion"}:
        continuidad = dict(contrato)
        continuidad["agrupacion"] = continuidad.get("entidad", "")
        ok, motivo = seguimiento.validar_contrato_sql(sql, continuidad)
    return ok, motivo
