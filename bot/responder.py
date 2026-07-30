"""
Orquestador del bot: clasifica la intencion y encadena registro, memoria,
catalogo, text-to-SQL y ejecucion.

    responder(numero, pregunta) -> str  (texto listo para mandar por WhatsApp)

Flujo:
    1. Resolver numero -> cliente (registry).
    2. Comando 'olvidá' -> borra memoria.
    3. Cargar historial (memoria).
    4. Clasificar intencion (intencion): datos / meta / saludo.
         - saludo -> respuesta fija.
         - meta   -> se responde con el historial, SIN tocar la base.
         - datos  -> catalogo + text-to-SQL + ejecucion + redaccion (como siempre).
    5. Guardar el intercambio.

La gobernanza (que tablas se pueden leer) sigue en bot/catalogo.py y se
re-evalua en cada consulta de datos. El clasificador NO afecta la seguridad:
solo decide si hace falta ir a la base o no.
"""

import logging

import registry
from bot import catalogo, intencion, kpis, memoria, nl2sql, warehouse_ro

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
_OLVIDADO = "Listo, borré lo que veníamos hablando. Empezamos de cero. 🙂"
_SALUDO = (
    "¡Hola! 👋 Soy tu asistente de datos. Preguntame sobre tus ventas o "
    "inventario, por ejemplo: «¿cuál fue el producto más vendido?» o «¿cuánto "
    "vendimos ayer?»."
)

# Comandos para borrar la memoria del propio numero.
_CMD_OLVIDAR = {"olvidá", "olvida", "olvidate", "olvídate", "reset", "reiniciar",
                "borrá memoria", "borra memoria", "empezar de cero"}


def responder(numero: str, pregunta: str) -> str:
    cliente = registry.resolver(numero)
    if not cliente:
        logger.info("numero no registrado: %s", numero)
        return _NO_REGISTRADO

    cid = cliente["cliente_id"]

    # Comando explicito para olvidar el historial de este numero.
    if pregunta.strip().lower() in _CMD_OLVIDAR:
        memoria.olvidar(cliente, numero)
        return _OLVIDADO

    # La memoria es best-effort: si falla, seguimos sin historial.
    historial = memoria.cargar_historial(cliente, numero)

    # Ruteo por intencion.
    intent = intencion.clasificar(pregunta, historial)
    logger.info("[%s] intencion=%s", cid, intent)

    if intent == "saludo":
        respuesta = _SALUDO
    elif intent == "meta":
        # Pregunta sobre la conversacion: se responde con el historial, sin base.
        try:
            respuesta = intencion.responder_conversacional(pregunta, historial)
        except Exception as e:  # noqa: BLE001
            logger.exception("[%s] error respondiendo meta: %s", cid, e)
            respuesta = ("No pude procesar eso. Preguntame algo sobre tus datos de "
                         "ventas o inventario.")
    else:  # "datos"
        respuesta = _responder_datos(cliente, numero, pregunta, historial)

    # Guardar el intercambio para dar continuidad a los proximos mensajes.
    memoria.guardar_intercambio(cliente, numero, pregunta, respuesta)
    return respuesta


def _responder_datos(cliente: dict, numero: str, pregunta: str,
                     historial: list) -> str:
    cid = cliente["cliente_id"]

    ctx = catalogo.construir_contexto(cliente)
    if not ctx.tablas_reales:
        logger.info("[%s] sin tablas habilitadas por catalogo", cid)
        return _SIN_TABLAS

    # Capa semantica: ¿un KPI predefinido calza? ¿hay que pedir contexto o retar?
    kpis_def = kpis.cargar_kpis(cliente)
    plan = kpis.planificar(pregunta, kpis_def, ctx, historial=historial)
    logger.info("[%s] plan=%s kpi=%s", cid, plan["accion"], plan.get("kpi"))

    # Freno anti-interrogatorio: si el turno anterior el bot YA pregunto (su
    # ultimo mensaje termino en '?'), no volvemos a preguntar. El usuario ya
    # respondio algo; ejecutamos con lo que haya en vez de repreguntar.
    ya_pregunto = (
        bool(historial)
        and historial[-1].get("rol") == "assistant"
        and historial[-1].get("contenido", "").rstrip().endswith("?")
    )
    if plan["accion"] in ("pedir_contexto", "retar") and ya_pregunto:
        logger.info("[%s] ya se pregunto el turno previo; no repregunto, ejecuto", cid)
        plan = {"accion": "sql_libre", "kpi": "", "sql": "", "mensaje": ""}

    # El bot pregunta o advierte ANTES de responder: no improvisa un numero.
    if plan["accion"] in ("pedir_contexto", "retar") and plan.get("mensaje"):
        return plan["mensaje"]

    # 1) Conseguir el SQL: del KPI (definicion canonica) o del text-to-SQL libre.
    sql = ""
    if plan["accion"] == "usar_kpi" and plan.get("sql"):
        sql = plan["sql"]
        ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
        if not ok:
            logger.info("[%s] SQL de KPI '%s' invalido (%s); cae a sql_libre",
                        cid, plan.get("kpi"), motivo)
            sql = ""  # cae al camino libre abajo

    if not sql:
        sql = nl2sql.generar_sql(pregunta, ctx.schema_text, historial=historial)
        ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
        if not ok:
            logger.info("[%s] SQL rechazado (%s); reintento. sql=%s", cid, motivo, sql)
            sql = nl2sql.generar_sql(pregunta, ctx.schema_text,
                                     correccion=motivo, sql_previo=sql,
                                     historial=historial)
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

    # 3) Redactar la respuesta en lenguaje natural (con continuidad).
    try:
        return nl2sql.redactar_respuesta(pregunta, columnas, filas,
                                         historial=historial, sql=sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error redactando respuesta: %s", cid, e)
        # Fallback sin LLM: al menos devolver el dato crudo.
        if not filas:
            return "No encontré datos para eso."
        return nl2sql._tabla_texto(columnas, filas, tope=10)
