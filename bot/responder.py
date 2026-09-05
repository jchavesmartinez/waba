"""
Orquestador del bot: clasifica la intencion y encadena registro, memoria,
catalogo, text-to-SQL y ejecucion.

    responder(numero, pregunta) -> Respuesta(texto, adjuntos)

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

ADJUNTOS. Si el usuario pide grafico / Excel / PDF (bot/formato.py lo detecta
con reglas, sin gastar una llamada al modelo), el MISMO resultado del SELECT se
usa dos veces: para redactar el texto y para armar el archivo. No hay una
segunda consulta ni un camino de datos aparte — o sea, el archivo no puede
contener nada que el texto no pudiera contener, y toda la gobernanza que ya
existia lo cubre sin cambios.

El tipo de retorno cambio de str a Respuesta. Es lo unico que rompe hacia
atras: quien llame a responder() tiene que usar .texto y .adjuntos.
"""

import logging
import re
import threading
import time
import uuid
from collections import defaultdict

import config
import registry
from bot import (artefactos, catalogo, correo, dashboard, edicion, formato, intencion, kpis,
                 memoria, nl2sql, seguimiento, warehouse_ro)
from bot.salida import Respuesta
from bot.tiempo import fecha_local

logger = logging.getLogger("fachavi.bot.responder")

_NO_REGISTRADO = (
    "Tu número no está registrado para consultar datos. "
    "Contacte a la persona que administra este servicio."
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
    "Inténtelo nuevamente en unos minutos; si continúa igual, informe al administrador."
)
_TOPE_DIARIO = (
    "Ya llegamos al tope de consultas de hoy para esta cuenta. "
    "Mañana se reinicia. Si necesita más, contacte al administrador."
)
def _no_seguro(temas: str = "") -> str:
    mensaje = "No pude armar esa consulta de forma segura. Intente formularla de otra manera."
    return f"{mensaje} {_mensaje_capacidades(temas)}" if temas else mensaje
_ERROR = "Tuve un problema consultando los datos. Intentá de nuevo en un momento."
# El archivo se genera DESPUES de tener los datos. Si falla el armado (o la
# subida a Meta), el usuario igual se queda con la respuesta en texto: perder el
# grafico es un inconveniente, perder el dato es una consulta desperdiciada.
_SIN_GRAFICO = (
    "\n\n(No pude armar el gráfico con este resultado —necesito al menos una "
    "columna de texto y una numérica, con varias filas. Te dejo el dato arriba.)"
)
_ADJUNTO_FALLO = (
    "\n\n(Tuve un problema generando el archivo. El dato de arriba es correcto; "
    "solicítelo nuevamente.)"
)
_OLVIDADO = "Listo. Borré el historial de esta conversación. Podemos empezar de cero."


def _mensaje_capacidades(temas: str) -> str:
    if temas:
        return f"Puede consultarme sobre {temas}."
    return "Puede consultarme sobre los datos habilitados para su empresa."


def _saludo(temas: str) -> str:
    return f"Hola. Soy su asistente de datos. {_mensaje_capacidades(temas)}"

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


def _ultimo_asistente(historial: list) -> str:
    return next(
        (str(t.get("contenido", "")) for t in reversed(historial or [])
         if t.get("rol") == "assistant"),
        "",
    )


def _es_pedido_datos_usados(pregunta: str) -> bool:
    texto = nl2sql._normalizar_para_columnas(pregunta)
    return bool(re.search(
        r"\b(?:datos|cifras|valores)\b.*\b(?:usaste|utilizaste|ocupaste|analizaste)\b|"
        r"\b(?:datos|cifras|valores)\b.*\b(?:del|detras del|detrás del)\b.*\banalisis\b",
        texto,
    ))


def _confirmacion_de_detalle(pregunta: str, historial: list) -> tuple[str, bool]:
    """Convierte un "si" a la oferta de detalle en una orden no ambigua."""
    texto = (pregunta or "").strip().lower()
    palabras = texto.split()
    ultimo = _ultimo_asistente(historial).lower()
    ofrece_detalle = "detalle" in ultimo and "?" in ultimo
    refiere_detalle = (
        "detalle" in texto
        and any(x in texto for x in ("eso", "esos", "esas", "los "))
    )
    if len(palabras) <= 12 and (
            (formato.es_afirmacion(pregunta) and ofrece_detalle)
            or refiere_detalle):
        return (
            "Muestra las filas de detalle que componen el resultado anterior. "
            "Conserva exactamente su periodo, filtros y criterio. "
            f"Confirmacion del usuario: {pregunta}",
            True,
        )
    return pregunta, False


def _ultimo_sql_seguro(historial: list, tablas_reales) -> str:
    """Recupera la ultima consulta, solo si sigue pasando la lista blanca."""
    for turno in reversed(historial or []):
        if turno.get("rol") != "assistant":
            continue
        sql = str(turno.get("sql", "") or "").strip()
        if not sql:
            continue
        ok, motivo = nl2sql.validar_sql(sql, tablas_reales)
        if ok:
            return sql
        logger.info("SQL historico descartado para adjunto: %s", motivo)
    return ""


def _sql_una_linea(sql: str) -> str:
    limpio = " ".join(str(sql or "").split())
    tope = int(getattr(config, "BOT_LOG_SQL_MAX_CHARS", 20000) or 0)
    if tope > 0 and len(limpio) > tope:
        return limpio[:tope] + f" ... [recortado; {len(limpio)} caracteres]"
    return limpio


def _ejecutar_con_auditoria(cliente: dict, sql: str, limite, origen: str):
    """Ejecuta y deja una traza copiable en Render para cada SELECT de negocio."""
    cid = cliente.get("cliente_id", "")
    query_id = uuid.uuid4().hex[:10]
    auditar = bool(getattr(config, "BOT_LOG_SQL", True))
    if auditar:
        logger.info("[%s] SQL_AUDIT inicio id=%s origen=%s limite=%s sql=%s",
                    cid, query_id, origen, limite or config.BOT_MAX_FILAS,
                    _sql_una_linea(sql))
    inicio = time.perf_counter()
    try:
        columnas, filas = warehouse_ro.ejecutar(cliente, sql, limite=limite)
    except Exception:
        if auditar:
            logger.exception("[%s] SQL_AUDIT error id=%s origen=%s duracion_ms=%.1f",
                             cid, query_id, origen,
                             (time.perf_counter() - inicio) * 1000)
        raise
    if auditar:
        logger.info(
            "[%s] SQL_AUDIT fin id=%s origen=%s duracion_ms=%.1f filas=%d columnas=%d",
            cid, query_id, origen, (time.perf_counter() - inicio) * 1000,
            len(filas), len(columnas),
        )
    return columnas, filas, query_id


def _limitar_top_solicitado(kpi: str, pregunta: str, columnas, filas):
    """Acota rankings explicitos e implicitos a lo que pidio el usuario.

    La formula del KPI es la fuente canonica del calculo y normalmente ya
    ordena el resultado. Esta capa solo materializa la intencion de ranking:
    ``top 5`` y tambien expresiones naturales como ``la categoria con mayor
    gasto`` o ``donde mas gaste``. El LLM puede reconocer la intencion, pero
    esta defensa evita mostrar todas las filas si no la representa en el plan.
    """
    texto = str(pregunta or "").lower()
    # Limites explicitos: top 5, los 5, 5 comercios/conceptos.
    match = re.search(
        r"\b(?:top|los|las)\s+(\d{1,3})\b|\b(\d{1,3})\s+(?:comercios|conceptos|categorias|categorías)\b",
        texto,
    )
    tope = int(match.group(1) or match.group(2)) if match else None

    # Limite implicito: la forma singular y los superlativos piden una sola
    # fila, aunque no usen la palabra "top". No se activa para "categorias"
    # en plural sin un superlativo, que correctamente debe listar todas.
    if tope is None:
        singular = bool(re.search(
            r"\b(?:la|el|cual|qué|que)\s+(?:es\s+)?(?:la\s+)?(?:categor[ií]a|comercio|concepto)\b",
            texto,
        ))
        superlativo = bool(re.search(
            r"\b(?:mayor|máximo|máxima|principal|más\s+(?:alto|alta|grande|gast[eé]|gasto)|menos|mínimo|mínima)\b",
            texto,
        ))
        donde_mas = bool(re.search(
            r"\b(?:dónde|donde|en\s+qué|en\s+que)\s+más\s+(?:gast[eé]|gasto|dinero)\b",
            texto,
        ))
        if (singular and superlativo) or donde_mas:
            tope = 1

    if tope is None:
        return filas
    if tope < 1 or len(filas) <= tope:
        return filas
    logger.info("ranking %s limitado a top %d (de %d filas)", kpi, tope, len(filas))
    return filas[:tope]


def _reconciliar_presupuesto_fuente(cliente, ctx, columnas, filas):
    """Ata ``presupuesto`` al monto mensual raw sin sumar filas de un JOIN."""
    nombres = [str(c).strip().lower().replace(" ", "_") for c in columnas]
    i_linea = next((nombres.index(x) for x in
                    ("linea_id", "linea_presupuesto_id") if x in nombres), None)
    i_concepto = nombres.index("concepto") if "concepto" in nombres else None
    i_presupuesto = next((nombres.index(x) for x in
                          ("presupuesto_mensual", "monto_mensual", "presupuesto")
                          if x in nombres), None)
    if not filas or i_presupuesto is None or (i_linea is None and i_concepto is None):
        return list(filas), []
    tabla = next((p.tabla_real for p in ctx.permitidas
                  if str(p.tabla_logica).strip().lower() == "presupuesto"), "")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(tabla)):
        return list(filas), []
    lineas = sorted({str(f[i_linea]) for f in filas
                     if i_linea is not None and f[i_linea] not in (None, "")})
    conceptos = sorted({seguimiento.normalizar_clave(f[i_concepto]) for f in filas
                        if i_concepto is not None and f[i_concepto] not in (None, "")})
    condiciones = []
    params = {}
    if lineas:
        condiciones.append("linea_id = ANY(:lineas)")
        params["lineas"] = lineas
    if conceptos:
        condiciones.append(
            "TRANSLATE(LOWER(TRIM(concepto)), 'áéíóúüñ', 'aeiouun') = ANY(:conceptos)"
        )
        params["conceptos"] = conceptos
    if not condiciones:
        return list(filas), []
    sql_fuente = (
        f'SELECT linea_id, concepto, MIN(monto_mensual) AS minimo, '
        f'MAX(monto_mensual) AS maximo FROM "{tabla}" WHERE '
        + "(" + " OR ".join(condiciones) + ") "
        "AND LOWER(TRIM(tipo)) = 'gasto' GROUP BY linea_id, concepto"
    )
    auditar = bool(getattr(config, "BOT_LOG_SQL", True))
    audit_id = uuid.uuid4().hex[:10]
    inicio = time.perf_counter()
    if auditar:
        logger.info(
            "[%s] SQL_AUDIT inicio id=%s origen=validacion_presupuesto "
            "params=%s sql=%s",
            cliente.get("cliente_id"), audit_id, params, _sql_una_linea(sql_fuente),
        )
    try:
        fuente = warehouse_ro.leer_interno(cliente, sql_fuente, params)
    except Exception as e:  # noqa: BLE001
        if auditar:
            logger.exception(
                "[%s] SQL_AUDIT error id=%s origen=validacion_presupuesto "
                "duracion_ms=%.1f",
                cliente.get("cliente_id"), audit_id,
                (time.perf_counter() - inicio) * 1000,
            )
        logger.warning("[%s] no se pudo reconciliar presupuesto fuente: %s",
                       cliente.get("cliente_id"), e)
        return list(filas), []
    if auditar:
        logger.info(
            "[%s] SQL_AUDIT fin id=%s origen=validacion_presupuesto "
            "duracion_ms=%.1f filas=%d",
            cliente.get("cliente_id"), audit_id,
            (time.perf_counter() - inicio) * 1000, len(fuente),
        )
    presupuestos = {}
    for item in fuente:
        # Valores diferentes para una misma llave requieren corregir la fuente,
        # no elegir uno silenciosamente desde el bot.
        if item.get("minimo") != item.get("maximo"):
            continue
        valor = item.get("minimo")
        presupuestos[f"linea:{seguimiento.normalizar_clave(item.get('linea_id'))}"] = valor
        presupuestos[f"concepto:{seguimiento.normalizar_clave(item.get('concepto'))}"] = valor
    return seguimiento.reconciliar_presupuesto_fuente(
        columnas, filas, presupuestos,
    )


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
_CONSUMO_DIA = {"fecha": fecha_local()}
_LOCK_CONSUMO = threading.Lock()


def _pasa_tope_diario(cliente_id: str) -> bool:
    tope = int(getattr(config, "BOT_MAX_MSJ_POR_DIA", 0) or 0)
    if tope <= 0:
        return True
    hoy = fecha_local()
    with _LOCK_CONSUMO:
        if _CONSUMO_DIA["fecha"] != hoy:
            _CONSUMO.clear()
            _CONSUMO_DIA["fecha"] = hoy
        if _CONSUMO[cliente_id] >= tope:
            return False
        _CONSUMO[cliente_id] += 1
    return True


def responder(numero: str, pregunta: str) -> Respuesta:
    cliente = registry.resolver(numero)
    if not cliente:
        logger.info("numero no registrado: %s", numero)
        return Respuesta(_NO_REGISTRADO)

    cid = cliente["cliente_id"]

    # Comando explicito para olvidar el historial de este numero.
    if _es_comando_olvidar(pregunta):
        memoria.olvidar(cliente, numero)
        return Respuesta(_OLVIDADO)

    # El dashboard es una entrega, no una consulta conversacional. El numero ya
    # fue resuelto contra el registro y queda firmado dentro del enlace; no se
    # gasta una llamada al modelo ni consume el tope diario de preguntas.
    if dashboard.es_solicitud(pregunta):
        return Respuesta(dashboard.mensaje_enlace(cliente, numero, pregunta))

    if not _pasa_tope_diario(cid):
        logger.warning("[%s] tope diario de mensajes alcanzado (%s)",
                       cid, config.BOT_MAX_MSJ_POR_DIA)
        return Respuesta(_TOPE_DIARIO)

    # La memoria es best-effort: si falla, seguimos sin historial.
    historial = memoria.cargar_historial(cliente, numero)

    # Una edición iniciada desde el menú toma prioridad sobre el clasificador
    # conversacional. Así textos como "Monto: 12000" nunca terminan como SQL,
    # y la escritura sigue requiriendo la confirmación explícita del usuario.
    respuesta_edicion = edicion.procesar_mensaje(cliente, numero, pregunta, historial)
    if respuesta_edicion is not None:
        memoria.guardar_intercambio(
            cliente, numero, pregunta, respuesta_edicion.texto,
            sql=respuesta_edicion.sql, estado=respuesta_edicion.estado,
        )
        return respuesta_edicion

    # Un mismo turno puede pedir las dos acciones: "crea un PDF ... y envialo
    # a persona@empresa.com". En ese caso primero se consulta/genera el archivo
    # y solo despues se prepara el borrador. El envio sigue exigiendo un "si"
    # separado; nunca se salta la confirmacion por combinar las instrucciones.
    correo_compuesto = correo.es_generacion_y_correo(pregunta)
    pregunta_datos = (
        correo.pregunta_para_generar(pregunta)
        if correo_compuesto else pregunta
    )

    # El correo es una ACCION, no una consulta de datos. Se resuelve antes del
    # catalogo/LLM para que "envia el PDF anterior" no dispare text-to-SQL. La
    # confirmacion y el archivo temporal viven en Neon, asi que funcionan aun
    # con varios workers o tras un redeploy.
    respuesta_correo = (
        None if correo_compuesto
        else correo.procesar_mensaje(cliente, numero, pregunta)
    )
    if respuesta_correo is not None:
        memoria.guardar_intercambio(
            cliente, numero, pregunta, respuesta_correo.texto,
            sql=respuesta_correo.sql, estado=respuesta_correo.estado,
        )
        return respuesta_correo

    # Se carga una sola vez y se reutiliza para gobernanza, clasificación y
    # mensajes. Así el bot describe las tablas realmente habilitadas para este
    # cliente, sin una lista fija de ventas/inventario ni una segunda lectura.
    ctx = catalogo.construir_contexto(cliente)
    tablas_habilitadas = catalogo.nombres_habilitados(ctx)
    temas_habilitados = catalogo.resumir_habilitados(ctx)

    # Un pedido de archivo es una accion de datos aunque el texto sea tan corto
    # como "en PDF" o "si, ese". Resolverlo antes evita gastar una llamada en
    # clasificarlo como charla y perder el resultado al que hace referencia.
    # En un pedido compuesto, pregunta_datos ya no contiene "PDF/Excel": se
    # quitó esa envoltura para que el planificador vea una consulta limpia. El
    # formato se conserva leyéndolo del mensaje original.
    pregunta_formato = pregunta if correo_compuesto else pregunta_datos
    fmt_solicitado = formato.detectar_con_contexto(pregunta_formato, historial)
    intent = (
        "datos" if fmt_solicitado != formato.TEXTO
        else intencion.clasificar(
            pregunta_datos, historial, tablas_habilitadas=tablas_habilitadas,
        )
    )
    logger.info("[%s] intencion=%s", cid, intent)

    if intent == "saludo":
        respuesta = Respuesta(_saludo(temas_habilitados))
    elif intent == "meta":
        # Pregunta sobre la conversacion: se responde con el historial, sin base.
        try:
            respuesta = Respuesta(
                intencion.responder_conversacional(
                    pregunta_datos, historial,
                    temas_habilitados=temas_habilitados,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("[%s] error respondiendo meta: %s", cid, e)
            respuesta = Respuesta(
                "No pude procesar eso. "
                + _mensaje_capacidades(temas_habilitados)
            )
    else:  # "datos"
        respuesta = _responder_datos(
            cliente, numero, pregunta_datos, historial,
            fmt_solicitado=fmt_solicitado,
            ctx=ctx,
        )

    # Guardar el intercambio para dar continuidad a los proximos mensajes.
    #
    # En la memoria va SOLO el texto: guardar bytes de un PNG en la tabla de
    # turnos la haria crecer sin control y el modelo no puede hacer nada con
    # ellos. Pero si se mandaron adjuntos se deja una nota, para que el turno
    # siguiente ("mandame ese mismo en Excel") tenga contexto de que se envio.
    texto_memoria = respuesta.texto
    if respuesta.adjuntos:
        correo.guardar_artefactos(cliente, numero, respuesta.adjuntos)
        if correo_compuesto:
            previa_correo = correo.procesar_mensaje(cliente, numero, pregunta)
            if previa_correo is not None:
                respuesta.texto = f"{respuesta.texto}\n\n{previa_correo.texto}"
                texto_memoria = respuesta.texto
        nombres = ", ".join(a.nombre for a in respuesta.adjuntos)
        texto_memoria = f"{texto_memoria}\n[Se envió el archivo adjunto: {nombres}]"
    memoria.guardar_intercambio(
        cliente, numero, pregunta, texto_memoria, sql=respuesta.sql,
        estado=respuesta.estado)
    return respuesta


def _responder_datos(cliente: dict, numero: str, pregunta: str,
                     historial: list, fmt_solicitado: str | None = None,
                     ctx=None) -> Respuesta:
    cid = cliente["cliente_id"]

    ctx = ctx or catalogo.construir_contexto(cliente)
    if ctx.error_lectura:
        logger.error("[%s] no se pudo leer el catalogo; no se responde con datos", cid)
        return Respuesta(_SIN_CATALOGO)
    if not ctx.tablas_reales:
        logger.info("[%s] sin tablas habilitadas por catalogo", cid)
        return Respuesta(_SIN_TABLAS)

    # ¿Pidio un archivo? Reglas, no modelo (ver bot/formato.py). Se resuelve
    # ANTES de ejecutar porque cambia cuantas filas hay que traer: para responder
    # en texto alcanza con BOT_MAX_FILAS (200), pero un Excel de 200 filas cuando
    # el cliente pidio "el detalle completo" es un archivo mutilado y no se nota.
    fmt = fmt_solicitado or formato.detectar_con_contexto(pregunta, historial)
    if fmt != formato.TEXTO:
        logger.info("[%s] formato de salida pedido: %s", cid, fmt)

    # Un contrafactual sobre el ultimo agregado ("sin la transaccion de 195
    # mil") es aritmetica, no text-to-SQL. Se calcula sobre cifras persistidas
    # y verificadas; Gemini no vuelve a interpretar ni a sumar los montos.
    if fmt == formato.TEXTO:
        ajuste = seguimiento.resolver_ajuste(pregunta, historial)
        if ajuste is not None:
            texto_ajuste, estado_ajuste = ajuste
            return Respuesta(texto_ajuste, estado=estado_ajuste)

    # "Eso en PDF" no es una consulta nueva: reutiliza el ultimo SELECT
    # validado. Si el mensaje agrega tema o periodo ("ventas de agosto en
    # PDF"), sigue por el planificador normal.
    fmt_explicito = formato.detectar(pregunta)
    seguimiento_archivo = fmt != formato.TEXTO and (
        formato.es_pedido_solo_formato(pregunta)
        or fmt_explicito == formato.TEXTO
    )
    seguimiento_datos = _es_pedido_datos_usados(pregunta) and bool(historial)
    sql_reutilizado = (
        _ultimo_sql_seguro(historial, ctx.tablas_reales)
        if seguimiento_archivo or seguimiento_datos else ""
    )
    pregunta_efectiva, confirma_detalle = _confirmacion_de_detalle(
        pregunta, historial,
    )

    # Capa semantica: ¿un KPI predefinido calza? ¿hay que pedir contexto o retar?
    kpis_def = kpis.cargar_kpis(cliente)
    if sql_reutilizado:
        plan = {
            "accion": "reutilizar_sql", "kpi": "", "sql": sql_reutilizado,
            "mensaje": "",
        }
        logger.info("[%s] se reutiliza el ultimo SQL para entregar %s", cid, fmt)
    elif confirma_detalle:
        # Una afirmacion corta a una oferta explicita de detalle es el unico
        # seguimiento conversacional que resolvemos sin LLM. No hay ambiguedad:
        # hereda exactamente el estado ofrecido y fuerza filas, no otro resumen.
        previo = seguimiento.ultimo_estado(historial)
        plan = {
            "relacion": "seguimiento",
            "heredar_filtros": list((previo.get("filtros") or {}).keys()),
            "heredar_periodo": bool(previo.get("periodo")),
            "heredar_kpi": bool(previo.get("kpi")),
            "accion": "sql_libre", "kpi": "", "sql": "", "mensaje": "",
        }
    else:
        plan = kpis.planificar(
            pregunta_efectiva, kpis_def, ctx, historial=historial,
        )
        if seguimiento.es_consulta_composicion(pregunta_efectiva):
            # El planificador conserva la autoridad sobre la relacion y el
            # contexto; solo se impide usar un KPI de resumen para pedir filas.
            plan.update(accion="sql_libre", kpi="", sql="", mensaje="")
        if nl2sql.pide_atributos_registro(pregunta_efectiva):
            # Una pregunta de clasificacion necesita la fila identificada y sus
            # campos (cuenta contable/concepto), no un KPI agregado que pueda
            # devolver solo totales o repetir el ultimo resumen.
            plan.update(accion="sql_libre", kpi="", sql="", mensaje="")

    estado_previo = seguimiento.contexto_segun_plan(historial, plan)
    relacion_plan = plan.get("relacion", "nueva")
    tope_historial_sql = max(
        int(getattr(config, "BOT_PLAN_HISTORIAL_TURNOS", 6)), 0,
    )
    historial_sql = (
        list(historial)[-tope_historial_sql:]
        if relacion_plan in ("seguimiento", "modificacion")
        and tope_historial_sql else []
    )
    unidad_kpi = ""
    if plan.get("accion") == "usar_kpi":
        elegido = next(
            (k for k in kpis_def
             if str(k.get("kpi", "")).strip().lower()
             == str(plan.get("kpi", "")).strip().lower()),
            None,
        )
        if elegido:
            unidad_kpi = str(elegido.get("unidad", "")).strip()
    heredado = []
    if estado_previo.get("filtros"):
        heredado.extend(sorted(estado_previo["filtros"]))
    if estado_previo.get("periodo"):
        heredado.append("periodo")
    if estado_previo.get("kpi"):
        heredado.append("kpi")
    logger.info(
        "[%s] plan=%s kpi=%s relacion=%s contexto_heredado=%s",
        cid, plan["accion"], plan.get("kpi"), relacion_plan,
        heredado,
    )

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
        plan.update(accion="sql_libre", kpi="", sql="", mensaje="")

    # El bot pregunta o advierte ANTES de responder: no improvisa un numero.
    if plan["accion"] in ("pedir_contexto", "retar") and plan.get("mensaje"):
        return Respuesta(plan["mensaje"])

    # 1) Conseguir el SQL: del KPI (definicion canonica) o del text-to-SQL libre.
    sql = sql_reutilizado
    if plan["accion"] == "usar_kpi" and plan.get("sql"):
        sql = plan["sql"]
        filtros_kpi = dict(estado_previo.get("filtros") or {})
        # El LLM puede nombrar un filtro explícito de la pregunta actual, pero
        # jamás escribe SQL: el valor solo se aplica si la fórmula KPI expone
        # esa dimensión en su resultado canónico.
        filtros_kpi.update(plan.get("filtros_actuales") or {})
        periodo_actual = seguimiento.periodo_explicito(pregunta_efectiva)
        periodo_kpi = periodo_actual or estado_previo.get("periodo") or {}
        if not periodo_kpi and kpis.admite_periodo_parametrizado(sql):
            hoy = fecha_local()
            if hoy.month == 12:
                fin = f"{hoy.year + 1:04d}-01-01"
            else:
                fin = f"{hoy.year:04d}-{hoy.month + 1:02d}-01"
            periodo_kpi = {
                "inicio": f"{hoy.year:04d}-{hoy.month:02d}-01",
                "fin_exclusivo": fin,
                "granularidad": "mes",
            }
        if periodo_actual and not kpis.admite_periodo_parametrizado(sql):
            logger.info(
                "[%s] KPI '%s' no declara parámetros de período; cae a sql_libre",
                cid, plan.get("kpi"),
            )
            sql = ""
        else:
            try:
                sql, filtros_sql = kpis.parametrizar_sql(
                    sql, filtros_kpi, periodo_kpi,
                )
            except ValueError as e:
                logger.info(
                    "[%s] KPI '%s' no se pudo parametrizar (%s); cae a sql_libre",
                    cid, plan.get("kpi"), e,
                )
                sql = ""
                filtros_sql = {}
            if filtros_sql:
                logger.info("[%s] KPI parametrizado con contexto: %s",
                            cid, sorted(filtros_sql))
        if sql:
            # La formula ya viene materializada de forma deterministica desde la
            # metadata: el modelo eligio QUE KPI corresponde, pero no puede
            # reescribir sus joins, deduplicación ni sus rangos.
            logger.info("[%s] SQL derivado del KPI '%s': %s",
                        cid, plan.get("kpi"), " ".join(sql.split()))
            ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
            if ok:
                ok, motivo = nl2sql.validar_granularidad(pregunta_efectiva, sql)
            if not ok:
                logger.info("[%s] SQL de KPI '%s' invalido (%s); cae a sql_libre",
                            cid, plan.get("kpi"), motivo)
                sql = ""  # cae al camino libre abajo

    if not sql and plan.get("accion") == "usar_kpi":
        plan.update(accion="sql_libre", kpi="", sql="", mensaje="")

    if not sql:
        pregunta_sql = pregunta_efectiva
        if estado_previo:
            contexto = {
                "kpi": estado_previo.get("kpi", ""),
                "filtros": estado_previo.get("filtros", {}),
                "periodo": estado_previo.get("periodo", {}),
            }
            pregunta_sql = (
                f"{pregunta_efectiva}\n\n"
                "CONTEXTO ESTRUCTURADO OBLIGATORIO DEL RESULTADO ANTERIOR: "
                f"{contexto}. Conserva esos filtros salvo que el usuario los "
                "cambie explicitamente."
            )
        if seguimiento.es_consulta_composicion(pregunta_efectiva):
            pregunta_sql += (
                "\n\nREGLA DETERMINISTICA: esta es una consulta de composicion. "
                "Si el usuario no indico periodo, no agregues un filtro de mes; "
                "lista las filas coincidentes, incluye su fecha y compara texto "
                "sin tildes con TRANSLATE."
            )
        sql = nl2sql.generar_sql(
            pregunta_sql, ctx.schema_text, historial=historial_sql,
        )
        ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
        if ok:
            ok, motivo = nl2sql.validar_granularidad(pregunta_efectiva, sql)
        if not ok:
            logger.info("[%s] SQL rechazado (%s); reintento. sql=%s", cid, motivo, sql)
            sql = nl2sql.generar_sql(pregunta_sql, ctx.schema_text,
                                     correccion=motivo, sql_previo=sql,
                                     historial=historial_sql)
            ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
            if ok:
                ok, motivo = nl2sql.validar_granularidad(pregunta_efectiva, sql)
            if not ok:
                logger.warning("[%s] SQL invalido tras reintento (%s): %s", cid, motivo, sql)
                return Respuesta(_no_seguro(catalogo.resumir_habilitados(ctx)))

        # El modelo puede producir una consulta valida pero insuficiente: por
        # ejemplo, seleccionar comercio/fecha/monto cuando el usuario pidio la
        # cuenta contable y el concepto. Se corrige una sola vez antes de leer
        # la base, conservando la semantica y los filtros de la pregunta.
        faltan = nl2sql.faltan_campos_solicitados(
            pregunta_efectiva, sql, ctx.schema_text,
        )
        if faltan:
            correccion_campos = (
                "El SQL es valido pero no proyecta los campos solicitados: "
                + ", ".join(faltan)
                + ". Incluyelos explicitamente en SELECT (no devuelvas solo "
                "comercio, fecha y monto), manteniendo los filtros y joins "
                "originales."
            )
            sql_campos = nl2sql.generar_sql(
                pregunta_sql, ctx.schema_text, correccion=correccion_campos,
                sql_previo=sql, historial=historial_sql,
            )
            ok_campos, motivo_campos = nl2sql.validar_sql(
                sql_campos, ctx.tablas_reales,
            )
            if ok_campos:
                ok_campos, motivo_campos = nl2sql.validar_granularidad(
                    pregunta_efectiva, sql_campos,
                )
            if ok_campos:
                sql = sql_campos
            else:
                logger.warning(
                    "[%s] SQL sin campos solicitados y reintento invalido (%s)",
                    cid, motivo_campos,
                )

    # 2) Ejecutar en solo-lectura.
    #
    # Para un archivo se levanta el tope de filas. El limite de 200 existe para
    # proteger la memoria del proceso y el tamaño del prompt de redaccion (A-18);
    # ninguna de las dos cosas aplica al Excel, que no pasa por el modelo. El
    # freno real del warehouse sigue siendo el statement_timeout, que no se toca.
    limite = (int(config.BOT_ADJUNTO_MAX_FILAS)
              if fmt in (formato.EXCEL, formato.CSV, formato.PDF) else None)
    origen_sql = (
        "reutilizado" if sql_reutilizado
        else (f"kpi:{plan.get('kpi')}" if plan.get("accion") == "usar_kpi"
              else "sql_libre")
    )
    try:
        columnas, filas, query_id = _ejecutar_con_auditoria(
            cliente, sql, limite, origen_sql,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error ejecutando SQL: %s", cid, e)
        return Respuesta(_ERROR)

    # Una composicion sin periodo debe mirar las filas que forman el concepto,
    # no el mes calendario recien iniciado. Si el primer SQL devolvio cero, se
    # permite un unico reintento explicito sin filtro temporal y sin tildes.
    if (not filas and seguimiento.es_consulta_composicion(pregunta_efectiva)
            and not seguimiento.tiene_periodo_explicito(pregunta_efectiva)):
        correccion_vacia = (
            "La consulta devolvio cero filas. El usuario pregunto la composicion "
            "historica de un concepto sin indicar periodo: elimina cualquier "
            "filtro de mes/CURRENT_DATE, incluye la fecha y compara concepto sin "
            "tildes usando TRANSLATE en columna y literal."
        )
        sql_reintento = nl2sql.generar_sql(
            pregunta_efectiva, ctx.schema_text, correccion=correccion_vacia,
            sql_previo=sql, historial=historial_sql,
        )
        ok, motivo = nl2sql.validar_sql(sql_reintento, ctx.tablas_reales)
        if ok:
            ok, motivo = nl2sql.validar_granularidad(
                pregunta_efectiva, sql_reintento,
            )
        if ok:
            try:
                columnas_nuevas, filas_nuevas, query_id_nuevo = _ejecutar_con_auditoria(
                    cliente, sql_reintento, limite, "reintento_composicion",
                )
                if filas_nuevas:
                    sql, columnas, filas = sql_reintento, columnas_nuevas, filas_nuevas
                    query_id = query_id_nuevo
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] fallo reintento de composicion: %s", cid, e)

    # Una pequena errata en el nombre (por ejemplo, "Rogar" por "Roga") no
    # debe ocultar una fila cuando fecha/importe permiten identificarla. Se
    # relaja solo el texto, nunca el periodo ni el monto, y se reutiliza el
    # resultado unicamente si el segundo SELECT encuentra filas.
    if (not filas and nl2sql.puede_reintentar_nombre_aproximado(pregunta_efectiva)
            and not seguimiento.es_consulta_composicion(pregunta_efectiva)):
        correccion_nombre = (
            "La consulta devolvio cero filas. Puede haber una variacion "
            "ortografica en el comercio o descripcion. Conserva fecha, monto "
            "y moneda; relaja solo ese texto con coincidencias parciales por "
            "fragmentos distintivos usando ILIKE y devuelve el nombre real."
        )
        sql_nombre = nl2sql.generar_sql(
            pregunta_sql if 'pregunta_sql' in locals() else pregunta_efectiva,
            ctx.schema_text, correccion=correccion_nombre,
            sql_previo=sql, historial=historial_sql,
        )
        ok_nombre, motivo_nombre = nl2sql.validar_sql(
            sql_nombre, ctx.tablas_reales,
        )
        if ok_nombre:
            ok_nombre, motivo_nombre = nl2sql.validar_granularidad(
                pregunta_efectiva, sql_nombre,
            )
        if ok_nombre:
            try:
                columnas_nuevas, filas_nuevas, query_id_nuevo = _ejecutar_con_auditoria(
                    cliente, sql_nombre, limite, "reintento_nombre_aproximado",
                )
                if filas_nuevas:
                    sql, columnas, filas = sql_nombre, columnas_nuevas, filas_nuevas
                    query_id = query_id_nuevo
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] fallo reintento de nombre aproximado: %s", cid, e)
        else:
            logger.info("[%s] reintento tolerante rechazado (%s)", cid, motivo_nombre)

    # Reconciliacion independiente del denominador. Una consulta libre puede
    # duplicar monto_mensual al unir presupuesto con movimientos; antes el
    # cociente seguia siendo internamente consistente y pasaba el validador.
    # La tabla presupuesto es la fuente de verdad del monto mensual.
    filas, correcciones_presupuesto = _reconciliar_presupuesto_fuente(
        cliente, ctx, columnas, filas,
    )
    if correcciones_presupuesto:
        logger.warning("[%s] se corrigieron %d presupuesto(s) contra la fuente",
                       cid, len(correcciones_presupuesto))

    # Un KPI antiguo puede no proyectar la llave necesaria para filtrarlo en
    # SQL. Como segunda defensa se acota el resultado ya calculado usando solo
    # valores inequívocos del turno anterior.
    if estado_previo:
        filas_filtradas, filtros_aplicados = seguimiento.filtrar_filas_por_contexto(
            columnas, filas, estado_previo,
        )
        if filtros_aplicados:
            logger.info("[%s] resultado acotado por contexto: %s",
                        cid, sorted(filtros_aplicados))
            filas = filas_filtradas

    filas = _limitar_top_solicitado(plan.get("kpi", ""), pregunta_efectiva,
                                    columnas, filas)

    ok_resultado, motivo_resultado = seguimiento.validar_resultado(
        columnas, filas, contexto=estado_previo,
    )
    if not ok_resultado:
        logger.error("[%s] resultado no reconciliado: %s", cid, motivo_resultado)
        return Respuesta(
            "No pude reconciliar el resultado con sus filtros y cálculos "
            "anteriores, así que no voy a mostrar una cifra dudosa. Inténtelo "
            "nuevamente o indique el concepto y período completos."
        )

    # NO_RESPONDIBLE es una señal de control del planificador, no una fila de
    # negocio. Se debe detener aquí, antes de que PDF/Excel la conviertan en un
    # archivo válido y antes de que un pedido compuesto prepare un correo con
    # ese archivo vacío.
    if nl2sql.es_resultado_no_respondible(columnas, filas):
        texto = nl2sql.redactar_respuesta(
            pregunta, columnas, filas,
            temas_habilitados=catalogo.resumir_habilitados(ctx),
        )
        return Respuesta(texto, sql=sql)

    # 3) Redactar la respuesta en lenguaje natural (con continuidad).
    #
    # Al redactor NUNCA se le pasan las miles de filas del export: se le da la
    # muestra de siempre. Un prompt con 5.000 filas cuesta plata, tarda y no
    # mejora la frase "te mando el detalle en Excel".
    muestra = filas[:config.BOT_MAX_FILAS]
    # La respuesta de WhatsApp es una vista para personas: no mostrar llaves
    # técnicas como linea_id, pero conservar columnas/filas originales para
    # estado conversacional, auditoría y adjuntos.
    columnas_texto, muestra_texto = nl2sql.ocultar_columnas_tecnicas(
        columnas, muestra, pregunta,
    )
    estado_resultado = seguimiento.crear_estado(
        pregunta, sql, plan.get("kpi", ""), unidad_kpi,
        columnas, filas, previo=estado_previo,
    )
    estado_resultado["query_id"] = query_id
    try:
        if sql_reutilizado and fmt != formato.TEXTO:
            etiqueta = {
                formato.PDF: "PDF", formato.EXCEL: "Excel", formato.CSV: "CSV",
                formato.GRAFICO: "grafico",
            }.get(fmt, "archivo")
            sustantivo = "registro" if len(filas) == 1 else "registros"
            texto = (
                f"Listo. Le adjunto el {etiqueta} con "
                f"{len(filas)} {sustantivo}."
            )
        else:
            texto = nl2sql.redactar_respuesta(
                pregunta, columnas_texto, muestra_texto, historial=historial, sql=sql,
                unidad=unidad_kpi,
                temas_habilitados=catalogo.resumir_habilitados(ctx),
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error redactando respuesta: %s", cid, e)
        # Fallback sin LLM: al menos devolver el dato crudo.
        if not filas:
            texto = "No encontré datos para eso."
        else:
            texto = nl2sql.tabla_texto(columnas_texto, muestra_texto, tope=10)

    # El redactor a veces dibuja un grafico con caracteres (barras de |, ejes de
    # _). En el celular eso se desalinea y queda ilegible. Se saca SIEMPRE, pero
    # el intento es informacion util: significa que este resultado pide una
    # visual y el usuario simplemente no sabia que podia pedirla. En vez de
    # devolver el texto pelado, se le manda el grafico de verdad.
    texto, hubo_arte = nl2sql.limpiar_arte_ascii(texto)
    if hubo_arte and fmt == formato.TEXTO:
        logger.info("[%s] el redactor intentó un gráfico ASCII; se manda el "
                    "gráfico real en su lugar", cid)
        fmt = formato.GRAFICO

    pasa_critico, motivo_critico = seguimiento.criticar_respuesta(
        pregunta, texto, estado_resultado,
    )
    if not pasa_critico:
        logger.error("[%s] critico final rechazo la respuesta: %s",
                     cid, motivo_critico)
        return Respuesta(
            "El resultado pasó las validaciones matemáticas, pero no pude "
            "confirmar que la explicación conservara correctamente el contexto. "
            "Prefiero no enviarle una respuesta dudosa; reformule la consulta "
            "indicando concepto y período."
        )

    # 4) Si pidio archivo, armarlo con el MISMO resultado. Nunca se vuelve a
    #    consultar la base: el adjunto es otra presentacion de lo ya autorizado.
    if fmt == formato.TEXTO or not filas:
        return Respuesta(texto, sql=sql, estado=estado_resultado)
    # Una lista explicita de columnas tambien se respeta en el adjunto. La
    # respuesta textual aplica esta misma proyeccion dentro del redactor para
    # elegir el formato compacto sin reescribir el SQL ni cambiar valores.
    columnas, filas, _ = nl2sql.proyectar_columnas_solicitadas(
        pregunta, columnas, filas,
    )
    # Si el gráfico salió de nuestra propia inferencia (no lo pidió el usuario),
    # un aviso de "no pude armarlo" es ruido sobre algo que nunca prometimos.
    respuesta = _armar_adjunto(
        cid, fmt, pregunta, texto, columnas, filas, historial,
        avisar_si_falla=not hubo_arte)
    respuesta.sql = sql
    respuesta.estado = estado_resultado
    return respuesta


def _armar_adjunto(cid: str, fmt: str, pregunta: str, texto: str,
                   columnas, filas, historial=None,
                   avisar_si_falla: bool = True) -> Respuesta:
    """
    Genera el archivo pedido. Si algo falla, devuelve el TEXTO igual: la
    consulta ya se pago y el dato ya esta; quedarse sin nada seria peor.
    """
    titulo = _titulo(pregunta)
    if titulo == "Consulta" and historial:
        # "pasame ESO en Excel" no tiene contenido propio: el tema esta en el
        # turno anterior. Se busca ahi antes de resignarse a 'consulta.xlsx'.
        for turno in reversed(historial):
            if turno.get("rol") == "user":
                contenido = turno.get("contenido", "")
                if (formato.es_pedido_solo_formato(contenido)
                        or formato.es_afirmacion(contenido)):
                    continue
                previo = _titulo(contenido)
                if previo != "Consulta":
                    titulo = previo
                    break
    try:
        if fmt == formato.GRAFICO:
            adj = artefactos.grafico_png(columnas, filas, titulo=titulo)
            if adj is None:
                # Datos que no se pueden graficar (una sola celda, sin columna
                # numerica). Se avisa; no se manda una imagen vacia.
                logger.info("[%s] resultado no graficable (%d filas, %d cols)",
                            cid, len(filas), len(columnas))
                return Respuesta(texto + _SIN_GRAFICO if avisar_si_falla else texto)
        elif fmt == formato.EXCEL:
            adj = artefactos.excel_xlsx(columnas, filas, titulo=titulo)
        elif fmt == formato.CSV:
            adj = artefactos.csv_texto(columnas, filas, titulo=titulo)
        elif fmt == formato.PDF:
            resumen_pdf = (
                "" if texto.startswith("Listo. Le adjunto el PDF") else texto
            )
            adj = artefactos.pdf_reporte(columnas, filas, titulo=titulo,
                                         resumen=resumen_pdf)
        else:
            return Respuesta(texto)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error generando adjunto (%s): %s", cid, fmt, e)
        return Respuesta(texto + _ADJUNTO_FALLO if avisar_si_falla else texto)

    if adj.tamano_mb > float(config.BOT_ADJUNTO_MAX_MB):
        logger.warning("[%s] adjunto '%s' pesa %.1f MB; se responde solo texto",
                       cid, adj.nombre, adj.tamano_mb)
        return Respuesta(
            texto + "\n\n(El archivo es demasiado pesado para WhatsApp. Solicite "
                    "un período más corto o menos filas.)"
        )

    logger.info("[%s] adjunto listo: %s (%.0f KB, %d filas)",
                cid, adj.nombre, len(adj.contenido) / 1024, len(filas))
    return Respuesta(texto, adjuntos=[adj])


# Muletillas del pedido que no aportan al titulo del archivo. Sin esto, "pasame
# eso en Excel" produce un archivo llamado 'eso_en_excel_2026-08-03.xlsx', que
# en el celular del cliente no dice absolutamente nada tres semanas despues.
_RUIDO_TITULO = re.compile(
    r"\b(por favor|porfa|mandame|mandámelo|manda|pasame|pásame|pasa|"
    r"d[aá]me(?:lo|la|los|las)?|"
    r"envia(me)?|enviá(me)?|quiero|necesito|podes|podés|puedes|me das|"
    r"graficame|graficá|grafica|gráfica|graficar|un gr[aá]fico( de)?|"
    r"el gr[aá]fico( de)?|export[aá](r)?|descarga(r|me)?|gener[aá](r|me)?|"
    r"el reporte( de)?|un reporte( de)?|el informe( de)?|el archivo( de)?|"
    r"en excel|a excel|en pdf|a pdf|en csv|en un archivo|eso|esto|lo anterior|"
    r"lo mismo)\b",
    re.IGNORECASE,
)


def _titulo(pregunta: str) -> str:
    """Titulo del grafico/archivo a partir de la pregunta, sin gastar el modelo."""
    if formato.es_pedido_solo_formato(pregunta):
        return "Consulta"
    t = _RUIDO_TITULO.sub(" ", pregunta or "")
    t = " ".join(t.split()).strip(" ¿?¡!.,:;-")
    if len(t) < 4:                       # quedo vacio o casi ("de", "las")
        return "Consulta"
    return t[:60].capitalize()
