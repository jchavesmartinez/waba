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
from datetime import date

import config
import registry
from bot import (artefactos, catalogo, formato, intencion, kpis, memoria,
                 nl2sql, warehouse_ro)
from bot.salida import Respuesta

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
# El archivo se genera DESPUES de tener los datos. Si falla el armado (o la
# subida a Meta), el usuario igual se queda con la respuesta en texto: perder el
# grafico es un inconveniente, perder el dato es una consulta desperdiciada.
_SIN_GRAFICO = (
    "\n\n(No pude armar el gráfico con este resultado —necesito al menos una "
    "columna de texto y una numérica, con varias filas. Te dejo el dato arriba.)"
)
_ADJUNTO_FALLO = (
    "\n\n(Tuve un problema generando el archivo. El dato de arriba es correcto; "
    "probá pidiéndomelo de nuevo.)"
)
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

    # Ruteo por intencion.
    intent = intencion.clasificar(pregunta, historial)
    logger.info("[%s] intencion=%s", cid, intent)

    if intent == "saludo":
        respuesta = Respuesta(_SALUDO)
    elif intent == "meta":
        # Pregunta sobre la conversacion: se responde con el historial, sin base.
        try:
            respuesta = Respuesta(
                intencion.responder_conversacional(pregunta, historial)
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("[%s] error respondiendo meta: %s", cid, e)
            respuesta = Respuesta("No pude procesar eso. Preguntame algo sobre "
                                  "tus datos de ventas o inventario.")
    else:  # "datos"
        respuesta = _responder_datos(cliente, numero, pregunta, historial)

    # Guardar el intercambio para dar continuidad a los proximos mensajes.
    #
    # En la memoria va SOLO el texto: guardar bytes de un PNG en la tabla de
    # turnos la haria crecer sin control y el modelo no puede hacer nada con
    # ellos. Pero si se mandaron adjuntos se deja una nota, para que el turno
    # siguiente ("mandame ese mismo en Excel") tenga contexto de que se envio.
    texto_memoria = respuesta.texto
    if respuesta.adjuntos:
        nombres = ", ".join(a.nombre for a in respuesta.adjuntos)
        texto_memoria = f"{texto_memoria}\n[Se envió el archivo adjunto: {nombres}]"
    memoria.guardar_intercambio(
        cliente, numero, pregunta, texto_memoria, sql=respuesta.sql)
    return respuesta


def _responder_datos(cliente: dict, numero: str, pregunta: str,
                     historial: list) -> Respuesta:
    cid = cliente["cliente_id"]

    ctx = catalogo.construir_contexto(cliente)
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
    fmt = formato.detectar(pregunta)
    if fmt != formato.TEXTO:
        logger.info("[%s] formato de salida pedido: %s", cid, fmt)

    # Capa semantica: ¿un KPI predefinido calza? ¿hay que pedir contexto o retar?
    kpis_def = kpis.cargar_kpis(cliente)
    plan = kpis.planificar(pregunta, kpis_def, ctx, historial=historial)
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
    sql = ""
    if plan["accion"] == "usar_kpi" and plan.get("sql"):
        sql = plan["sql"]
        # La formula ya viene materializada de forma deterministica desde la
        # metadata: el modelo eligio el KPI, pero no pudo reescribir su SQL.
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
                return Respuesta(_NO_SEGURO)

    # 2) Ejecutar en solo-lectura.
    #
    # Para un archivo se levanta el tope de filas. El limite de 200 existe para
    # proteger la memoria del proceso y el tamaño del prompt de redaccion (A-18);
    # ninguna de las dos cosas aplica al Excel, que no pasa por el modelo. El
    # freno real del warehouse sigue siendo el statement_timeout, que no se toca.
    limite = (int(config.BOT_ADJUNTO_MAX_FILAS)
              if fmt in (formato.EXCEL, formato.CSV) else None)
    try:
        columnas, filas = warehouse_ro.ejecutar(cliente, sql, limite=limite)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error ejecutando SQL: %s", cid, e)
        return Respuesta(_ERROR)

    # 3) Redactar la respuesta en lenguaje natural (con continuidad).
    #
    # Al redactor NUNCA se le pasan las miles de filas del export: se le da la
    # muestra de siempre. Un prompt con 5.000 filas cuesta plata, tarda y no
    # mejora la frase "te mando el detalle en Excel".
    muestra = filas[:config.BOT_MAX_FILAS]
    try:
        texto = nl2sql.redactar_respuesta(
            pregunta, columnas, muestra, historial=historial, sql=sql,
            unidad=unidad_kpi,
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
                previo = _titulo(turno.get("contenido", ""))
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
            adj = artefactos.pdf_reporte(columnas, filas, titulo=titulo,
                                         resumen=texto)
        else:
            return Respuesta(texto)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] error generando adjunto (%s): %s", cid, fmt, e)
        return Respuesta(texto + _ADJUNTO_FALLO if avisar_si_falla else texto)

    if adj.tamano_mb > float(config.BOT_ADJUNTO_MAX_MB):
        logger.warning("[%s] adjunto '%s' pesa %.1f MB; se responde solo texto",
                       cid, adj.nombre, adj.tamano_mb)
        return Respuesta(
            texto + "\n\n(El archivo salió muy pesado para WhatsApp. Pedime un "
                    "período más corto o menos filas y te lo mando.)"
        )

    logger.info("[%s] adjunto listo: %s (%.0f KB, %d filas)",
                cid, adj.nombre, len(adj.contenido) / 1024, len(filas))
    return Respuesta(texto, adjuntos=[adj])


# Muletillas del pedido que no aportan al titulo del archivo. Sin esto, "pasame
# eso en Excel" produce un archivo llamado 'eso_en_excel_2026-08-03.xlsx', que
# en el celular del cliente no dice absolutamente nada tres semanas despues.
_RUIDO_TITULO = re.compile(
    r"\b(por favor|porfa|mandame|mandámelo|manda|pasame|pásame|pasa|dame|"
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
    t = _RUIDO_TITULO.sub(" ", pregunta or "")
    t = " ".join(t.split()).strip(" ¿?¡!.,:;-")
    if len(t) < 4:                       # quedo vacio o casi ("de", "las")
        return "Consulta"
    return t[:60].capitalize()
