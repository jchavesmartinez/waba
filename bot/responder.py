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
from collections import defaultdict
from datetime import date, timedelta

import config
import registry
from bot import (artefactos, catalogo, correo, formato, intencion, kpis,
                 memoria, nl2sql, warehouse_ro)
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
_NO_SEGURO = (
    "No pude armar esa consulta de forma segura. "
    "Intente formularla de otra manera, por ejemplo: "
    "«¿cuánto vendimos ayer?» o «¿qué productos tienen bajo inventario?»."
)
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


def _saludo_texto() -> str:
    return (
        "Hola soy Dativa, su asistente empresarial. 📲\n\n"
        "Puedo ayudarle a revisar los principales datos de Inside Tours. 🌴"
    )


def _saludo_botones() -> list:
    from bot.salida import Boton
    return [
        Boton("A", "Correos sin leer"),
        Boton("B", "Reporte de ventas"),
        Boton("C", "Lista de reservas"),
    ]


def _saludo_menu_titulo() -> str:
    return (
        "Aquí un menú de las principales solicitudes de su empresa 📝\n\n"
        "Si quieres saber algo adicional a este menú, escriba su consulta en este chat"
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


def _obtener_total_ventas(cliente: dict, periodo: str) -> str | None:
    """
    Obtiene el total de ventas para un periodo especifico.

    Args:
        cliente: dict con datos del cliente
        periodo: "día", "semana" o "mes"

    Returns:
        String formateado como moneda (ej: "₡314.678.900") o None si hay error
    """
    try:
        hoy = fecha_local()

        if periodo == "día":
            fecha_inicio = hoy
            fecha_fin = hoy
        elif periodo == "semana":
            fecha_inicio = hoy - timedelta(days=6)
            fecha_fin = hoy
        elif periodo == "mes":
            fecha_inicio = hoy.replace(day=1)
            fecha_fin = hoy
        else:
            return None

        # Query para obtener el total de ventas del periodo
        sql = (
            "SELECT COALESCE(SUM(monto_original), 0) as total "
            "FROM pagos "
            f"WHERE fecha >= '{fecha_inicio}'::date "
            f"AND fecha <= '{fecha_fin}'::date"
        )

        columnas, filas = warehouse_ro.ejecutar(cliente, sql)
        if not filas:
            return "₡0"

        total = float(filas[0][0]) if filas[0][0] else 0
        # Formatear como moneda de Costa Rica
        return f"₡{total:,.0f}".replace(",", ".")
    except Exception as e:
        logger.exception("Error obteniendo total de ventas para %s: %s",
                        cliente.get("cliente_id"), e)
        return None


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

    if not _pasa_tope_diario(cid):
        logger.warning("[%s] tope diario de mensajes alcanzado (%s)",
                       cid, config.BOT_MAX_MSJ_POR_DIA)
        return Respuesta(_TOPE_DIARIO)

    # La memoria es best-effort: si falla, seguimos sin historial.
    historial = memoria.cargar_historial(cliente, numero)

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
            sql=respuesta_correo.sql,
        )
        return respuesta_correo

    # Detectar si el usuario presionó un botón del menú de saludo
    pregunta_limpia = (pregunta or "").strip().upper()
    es_boton_correos = pregunta_limpia in ("A", "CORREOS SIN LEER")
    es_boton_reportes = pregunta_limpia in ("B", "REPORTE DE VENTAS")
    es_boton_reservas = pregunta_limpia in ("C", "LISTA DE RESERVAS")

    if es_boton_correos:
        sin_leer = correo.contar_correos_sin_leer(cliente, numero)
        if sin_leer is None:
            texto_respuesta = (
                "Necesitas conectar tu correo Gmail primero. "
                "Escribe 'conectar correo' para autorizarme el acceso."
            )
        else:
            texto_respuesta = (
                f"Tienes *{sin_leer}* correos sin leer en tu bandeja de entrada. 📧"
                if sin_leer > 0
                else "¡Excelente! No tienes correos sin leer. ✅"
            )
        memoria.guardar_intercambio(cliente, numero, pregunta, texto_respuesta)
        return Respuesta(texto_respuesta)
    elif es_boton_reportes:
        # Detectar si ya está respondiendo al período o formato
        palabras_mensuales = ("mes", "mensual", "meses")
        palabras_semanales = ("semana", "semanal", "semanas")
        palabras_diarias = ("dia", "día", "diario", "dias", "días", "hoy", "today")
        palabras_pdf = ("pdf",)
        palabras_excel = ("excel", "xlsx", "xls")
        palabras_grafico = ("gráfico", "grafico", "gráfica", "grafica", "chart", "imagen", "image")

        es_mes = any(p in pregunta_limpia for p in palabras_mensuales)
        es_semana = any(p in pregunta_limpia for p in palabras_semanales)
        es_dia = any(p in pregunta_limpia for p in palabras_diarias)
        es_pdf = any(p in pregunta_limpia for p in palabras_pdf)
        es_excel = any(p in pregunta_limpia for p in palabras_excel)
        es_grafico = any(p in pregunta_limpia for p in palabras_grafico)

        # Determinar período si lo especificó
        periodo = None
        if es_mes:
            periodo = "mes"
        elif es_semana:
            periodo = "semana"
        elif es_dia:
            periodo = "día"

        if periodo:
            # Usuario respondió el período, mostrar total de ventas y preguntar formato
            total_ventas = _obtener_total_ventas(cliente, periodo)
            if total_ventas is None:
                total_ventas = "₡0"
            detalle = (
                f"📊 *Reporte de ventas - {periodo.upper()}*\n\n"
                f"Total ventas: {total_ventas}\n\n"
                f"¿En qué formato lo quieres?\n"
                f"• PDF\n"
                f"• Excel\n"
                f"• Gráfico"
            )
            memoria.guardar_intercambio(cliente, numero, pregunta, detalle)
            return Respuesta(detalle)
        elif es_pdf or es_excel or es_grafico:
            # Usuario eligió formato
            formato_txt = "PDF" if es_pdf else ("Excel" if es_excel else "Gráfico")
            texto_respuesta = f"Generando reporte en formato {formato_txt}... ⏳\n\n(Proximamente: envío del archivo)"
            memoria.guardar_intercambio(cliente, numero, pregunta, texto_respuesta)
            return Respuesta(texto_respuesta)
        else:
            # Primera vez: preguntar por el período
            texto_respuesta = (
                "¿Qué período deseas? Responde:\n"
                "• Mensual (este mes)\n"
                "• Semanal (esta semana)\n"
                "• Diario (hoy)"
            )
            memoria.guardar_intercambio(cliente, numero, pregunta, texto_respuesta)
            return Respuesta(texto_respuesta)
    elif es_boton_reservas:
        texto_respuesta = (
            "Te muestro las reservas futuras. "
            "¿Desde qué fecha quieres ver? (ej: mañana, próxima semana)"
        )
        memoria.guardar_intercambio(cliente, numero, pregunta, texto_respuesta)
        return Respuesta(texto_respuesta)
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
        respuesta = Respuesta(
            texto=f"{_saludo_texto()}\n\n{_saludo_menu_titulo()}",
            botones=_saludo_botones(),
        )
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
        cliente, numero, pregunta, texto_memoria, sql=respuesta.sql)
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
        # El KPI de resumen no sirve para "si, quiero el detalle". El camino
        # libre conserva los filtros anteriores y lista las filas subyacentes.
        plan = {"accion": "sql_libre", "kpi": "", "sql": "", "mensaje": ""}
    else:
        plan = kpis.planificar(
            pregunta_efectiva, kpis_def, ctx, historial=historial,
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
        return Respuesta(plan["mensaje"])

    # 1) Conseguir el SQL: del KPI (definicion canonica) o del text-to-SQL libre.
    sql = sql_reutilizado
    if plan["accion"] == "usar_kpi" and plan.get("sql"):
        sql = plan["sql"]
        # La formula ya viene materializada de forma deterministica desde la
        # metadata: el modelo eligio el KPI, pero no pudo reescribir su SQL.
        logger.info("[%s] SQL derivado del KPI '%s': %s",
                    cid, plan.get("kpi"), " ".join(sql.split()))
        ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
        if ok:
            ok, motivo = nl2sql.validar_granularidad(pregunta_efectiva, sql)
        if not ok:
            logger.info("[%s] SQL de KPI '%s' invalido (%s); cae a sql_libre",
                        cid, plan.get("kpi"), motivo)
            sql = ""  # cae al camino libre abajo

    if not sql:
        sql = nl2sql.generar_sql(
            pregunta_efectiva, ctx.schema_text, historial=historial,
        )
        ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
        if ok:
            ok, motivo = nl2sql.validar_granularidad(pregunta_efectiva, sql)
        if not ok:
            logger.info("[%s] SQL rechazado (%s); reintento. sql=%s", cid, motivo, sql)
            sql = nl2sql.generar_sql(pregunta_efectiva, ctx.schema_text,
                                     correccion=motivo, sql_previo=sql,
                                     historial=historial)
            ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
            if ok:
                ok, motivo = nl2sql.validar_granularidad(pregunta_efectiva, sql)
            if not ok:
                logger.warning("[%s] SQL invalido tras reintento (%s): %s", cid, motivo, sql)
                return Respuesta(_NO_SEGURO)

    # 2) Ejecutar en solo-lectura.
    #
    # Para un archivo se levanta el tope de filas. El limite de 200 existe para
    # proteger la memoria del proceso y el tamaño del prompt de redaccion (A-18);
    # ninguna de las dos cosas aplica al Excel, que no pasa por el modelo. El
    # freno real del warehouse sigue siendo el statement_timeout, que no se toca.
    limite = (int(config.BOT_ADJUNTO_MAX_FILAS)
              if fmt in (formato.EXCEL, formato.CSV, formato.PDF) else None)
    try:
        columnas, filas = warehouse_ro.ejecutar(cliente, sql, limite=limite)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error ejecutando SQL: %s", cid, e)
        return Respuesta(_ERROR)

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
                pregunta, columnas, muestra, historial=historial, sql=sql,
                unidad=unidad_kpi,
                temas_habilitados=catalogo.resumir_habilitados(ctx),
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error redactando respuesta: %s", cid, e)
        # Fallback sin LLM: al menos devolver el dato crudo.
        if not filas:
            texto = "No encontré datos para eso."
        else:
            texto = nl2sql.tabla_texto(columnas, muestra, tope=10)

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

    # 4) Si pidio archivo, armarlo con el MISMO resultado. Nunca se vuelve a
    #    consultar la base: el adjunto es otra presentacion de lo ya autorizado.
    if fmt == formato.TEXTO or not filas:
        return Respuesta(texto, sql=sql)
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
