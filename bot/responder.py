"""
Orquestador del bot: junta registro, catalogo, text-to-SQL y ejecucion.

    responder(numero, pregunta) -> str  (texto listo para mandar por WhatsApp)

Es lo unico que necesita conocer app.py (el webhook). Toda la logica de
gobernanza (que tablas se pueden leer) esta en bot/catalogo.py; aca solo se
encadena el flujo y se manejan los caminos de error con mensajes claros.
"""

import logging

import registry
from bot import catalogo, nl2sql, warehouse_ro

logger = logging.getLogger("fachavi.bot.responder")

_NO_REGISTRADO = (
    "Tu número no está registrado para consultar datos. "
    "Contactá a la persona que administra este servicio."
)
_SIN_TABLAS = (
    "Todavía no hay tablas habilitadas para responder por este medio. "
    "El administrador debe marcarlas en el catálogo."
)
_NO_SEGURO = (
    "No pude armar esa consulta de forma segura. "
    "Probá preguntándolo de otra manera, por ejemplo: "
    "«¿cuánto vendimos ayer?» o «¿qué productos tienen bajo inventario?»."
)
_ERROR = "Tuve un problema consultando los datos. Intentá de nuevo en un momento."


def responder(numero: str, pregunta: str) -> str:
    cliente = registry.resolver(numero)
    if not cliente:
        logger.info("numero no registrado: %s", numero)
        return _NO_REGISTRADO

    cid = cliente["cliente_id"]

    ctx = catalogo.construir_contexto(cliente)
    if not ctx.tablas_reales:
        logger.info("[%s] sin tablas habilitadas por catalogo", cid)
        return _SIN_TABLAS

    # 1) Generar y validar el SQL. Un reintento pidiendo corregir si falla.
    sql = nl2sql.generar_sql(pregunta, ctx.schema_text)
    ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
    if not ok:
        logger.info("[%s] SQL rechazado (%s); reintento. sql=%s", cid, motivo, sql)
        sql = nl2sql.generar_sql(pregunta, ctx.schema_text,
                                 correccion=motivo, sql_previo=sql)
        ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
        if not ok:
            logger.warning("[%s] SQL invalido tras reintento (%s): %s", cid, motivo, sql)
            return _NO_SEGURO

    # 2) Ejecutar en solo-lectura.
    try:
        columnas, filas = warehouse_ro.ejecutar(cliente, sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error ejecutando SQL: %s", cid, e)
        return _ERROR

    # 3) Redactar la respuesta en lenguaje natural.
    try:
        return nl2sql.redactar_respuesta(pregunta, columnas, filas)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error redactando respuesta: %s", cid, e)
        # Fallback sin LLM: al menos devolver el dato crudo.
        if not filas:
            return "No encontré datos para eso."
        return nl2sql._tabla_texto(columnas, filas, tope=10)
