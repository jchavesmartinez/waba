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
import threading
from collections import defaultdict
from datetime import date

import config
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
# B-26: no es lo mismo "el catálogo dice que no hay nada habilitado" que "no
# pude leer el catálogo". El primero se arregla en el Sheet; el segundo es la
# base caída o un permiso. Antes los dos daban el mismo mensaje.
_SIN_CATALOGO = (
    "No pude leer la configuración de datos en este momento. "
    "Probá de nuevo en un ratito; si sigue igual, avisale al administrador."
)
_TOPE_DIARIO = (
    "Ya llegamos al tope de consultas de hoy para esta cuenta. "
    "Mañana se reinicia. Si necesitás más, hablá con el administrador."
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
#
# B-21: antes se exigia coincidencia EXACTA con el conjunto, asi que "olvidá
# todo" o "olvidate por favor" no funcionaban y se trataban como pregunta de
# datos (gastando llamadas al modelo para no hacer nada util). Ahora se acepta
# el comando como PREFIJO del mensaje, que es como la gente lo escribe.
_CMD_OLVIDAR = ("olvidá", "olvida", "olvidate", "olvídate", "olvidáte",
                "reset", "reiniciar", "borrá memoria", "borra memoria",
                "borrá la memoria", "borra la memoria", "empezar de cero")


def _es_comando_olvidar(texto: str) -> bool:
    t = (texto or "").strip().lower().rstrip("!.¡ ")
    if not t:
        return False
    # Se exige que el mensaje sea corto: "olvidá lo que te dije de las ventas de
    # marzo" es un comando; "olvidate de eso, mejor decime cuánto vendimos ayer"
    # no deberia borrar la memoria del cliente por accidente.
    if len(t.split()) > 5:
        return False
    return any(t == c or t.startswith(c + " ") for c in _CMD_OLVIDAR)


# --------------------------------------------------------------------------
# A-14: tope de consumo por cliente y por dia.
#
# Cada mensaje de datos gasta entre 3 y 5 llamadas al modelo (clasificacion +
# planificacion + generacion de SQL, hasta 2 con reintento + redaccion). No
# habia ningun tope: ni diario, ni por usuario, ni por cliente. Un usuario
# entusiasta —o un abuso del webhook— se traducia directo en factura.
#
# El contador vive en el proceso: con un solo worker es exacto, con varios cada
# uno lleva el suyo (o sea, el tope efectivo se multiplica por la cantidad de
# workers, nunca se vuelve mas estricto de lo configurado). Para un tope duro y
# compartido habria que moverlo a la tabla de memoria; esto ya evita el caso que
# importa, que es la factura sorpresa.
# --------------------------------------------------------------------------
_CONSUMO: dict = defaultdict(int)
_CONSUMO_DIA = {"fecha": date.today()}
_LOCK_CONSUMO = threading.Lock()


def _pasa_tope_diario(cliente_id: str) -> bool:
    tope = int(getattr(config, "BOT_MAX_MSJ_POR_DIA", 0) or 0)
    if tope <= 0:
        return True
    hoy = date.today()
    with _LOCK_CONSUMO:
        if _CONSUMO_DIA["fecha"] != hoy:
            _CONSUMO.clear()
            _CONSUMO_DIA["fecha"] = hoy
        if _CONSUMO[cliente_id] >= tope:
            return False
        _CONSUMO[cliente_id] += 1
    return True


def responder(numero: str, pregunta: str) -> str:
    cliente = registry.resolver(numero)
    if not cliente:
        logger.info("numero no registrado: %s", numero)
        return _NO_REGISTRADO

    cid = cliente["cliente_id"]

    # Comando explicito para olvidar el historial de este numero.
    if _es_comando_olvidar(pregunta):
        memoria.olvidar(cliente, numero)
        return _OLVIDADO

    if not _pasa_tope_diario(cid):
        logger.warning("[%s] tope diario de mensajes alcanzado (%s)",
                       cid, config.BOT_MAX_MSJ_POR_DIA)
        return _TOPE_DIARIO

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
    if ctx.error_lectura:
        logger.error("[%s] no se pudo leer el catalogo; no se responde con datos", cid)
        return _SIN_CATALOGO
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
        # A-19: la formula canonica del KPI se le manda al modelo para que la
        # "adapte" al periodo/dimension que pidio el usuario. El prompt dice
        # "respetala", pero nada lo garantiza: el validador comprueba que el SQL
        # sea SEGURO, no que sea CORRECTO. Un KPI mal adaptado devuelve un numero
        # plausible y equivocado, que es el peor tipo de error. Como minimo, que
        # quede el rastro para poder auditar despues cual formula produjo que
        # numero.
        logger.info("[%s] SQL derivado del KPI '%s': %s",
                    cid, plan.get("kpi"), " ".join(sql.split()))
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
        return nl2sql.tabla_texto(columnas, filas, tope=10)
