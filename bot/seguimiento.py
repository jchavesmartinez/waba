"""Estado estructurado, ajustes y reconciliacion entre turnos.

Gemini puede decidir la intencion, pero no es la memoria de calculo. Este
modulo conserva el resultado ejecutado en una forma pequeña y verificable y
resuelve localmente los seguimientos que modifican una cifra anterior.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from calendar import monthrange
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation

from bot.tiempo import fecha_local


_MAX_FILAS_ESTADO = 200
_GRUPOS_FILTRO = {
    "descripcion": ("descripcion", "comercio"),
    "linea_id": ("linea_id", "linea_presupuesto_id", "linea_presupuestaria_id"),
    "concepto": ("concepto",),
    "categoria": ("categoria",),
    "moneda": ("moneda",),
}
_PRESUPUESTO = ("presupuesto_mensual", "monto_mensual", "monto_presupuestado",
                "total_presupuesto", "presupuestado", "presupuesto")
_GASTADO = ("gastado", "gasto_real", "gasto_ejecutado", "ejecutado",
            "monto_ejecutado", "ejecucion", "total_gastado", "gasto_neto")
_DISPONIBLE = ("disponible", "saldo_disponible", "diferencia")
_EXCESO = ("exceso", "sobregiro")
_PORCENTAJE = ("porcentaje_consumido", "porcentaje_ejecutado", "pct_consumido",
               "pct_ejecutado")
_MONTOS_DETALLE = ("monto_crc", "monto_neto", "monto", "importe", "monto_total", "total")
_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
    "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
    "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def _normalizar(valor) -> str:
    texto = unicodedata.normalize("NFKD", str(valor or ""))
    return " ".join("".join(c for c in texto if not unicodedata.combining(c))
                    .strip().lower().split())


def _nombre(valor) -> str:
    return _normalizar(valor).replace(" ", "_")


def normalizar_clave(valor) -> str:
    """Normalizacion publica para cruzar dimensiones ejecutadas."""
    return _normalizar(valor)


def _json_valor(valor):
    if valor is None or isinstance(valor, (str, int, float, bool)):
        return valor
    if isinstance(valor, Decimal):
        return str(valor)
    if isinstance(valor, (date, datetime)):
        return valor.isoformat()
    return str(valor)


def _decimal(valor) -> Decimal | None:
    if valor is None or isinstance(valor, bool):
        return None
    try:
        return Decimal(str(valor).replace("\u00a0", "").strip())
    except (InvalidOperation, ValueError):
        return None


def _indice(columnas, candidatos) -> int | None:
    nombres = [_nombre(c) for c in columnas]
    return next((nombres.index(c) for c in candidatos if c in nombres), None)


def ultimo_estado(historial: list) -> dict:
    for turno in reversed(historial or []):
        estado = turno.get("estado") if turno.get("rol") == "assistant" else None
        if isinstance(estado, dict) and estado.get("columnas") is not None:
            return estado
    return {}


def contexto_segun_plan(historial: list, plan: dict) -> dict:
    """Materializa solo el contexto que el planificador pidio heredar.

    El LLM decide la relacion semantica, pero nunca entrega valores de negocio:
    las cifras y filtros se copian exclusivamente del ultimo estado verificado.
    Una clave inexistente se ignora, de modo que el planificador no puede
    inventar contexto ni ampliar el acceso a datos.
    """
    if str((plan or {}).get("relacion", "nueva")) not in (
        "seguimiento", "modificacion",
    ):
        return {}
    previo = ultimo_estado(historial)
    if not previo:
        return {}
    disponibles = previo.get("filtros") or {}
    solicitados = (plan or {}).get("heredar_filtros") or []
    filtros = {
        clave: disponibles[clave]
        for clave in solicitados
        if clave in disponibles and disponibles[clave] not in (None, "")
    }
    contexto = {
        "kpi": previo.get("kpi", "") if plan.get("heredar_kpi") else "",
        "filtros": filtros,
        "periodo": (
            dict(previo.get("periodo") or {})
            if plan.get("heredar_periodo") else {}
        ),
    }
    return contexto if any(contexto.values()) else {}


def es_consulta_composicion(pregunta: str) -> bool:
    """True para preguntas de detalle que no solicitan un agregado temporal."""
    t = _normalizar(pregunta)
    return bool(re.search(
        r"\b(?:que|cuales)\s+(?:gastos?|compras?|movimientos?|transacciones?)\b.*"
        r"\b(?:conforman?|componen?|forman?|incluye|hubo)\b|"
        r"\b(?:que|cuales)\b.*\b(?:conforman?|componen?|forman?)\b|"
        r"\b(?:gastos?|compras?|movimientos?|transacciones?)\b.*"
        r"\b(?:cuales|fueron|son)\b",
        t,
    ))


def tiene_periodo_explicito(pregunta: str) -> bool:
    t = _normalizar(pregunta)
    meses = "|".join(_MESES)
    return bool(re.search(
        rf"\b(?:hoy|ayer|anteayer|esta semana|semana pasada|este mes|"
        rf"mes pasado|ultimos?\s+\d+\s+dias?|{meses}|20\d{{2}})\b|"
        r"\b\d{1,2}[/.-]\d{1,2}",
        t,
    ))


def periodo_explicito(pregunta: str) -> dict:
    """Extrae un mes/año escrito por el usuario, sin inferirlo del historial.

    El resultado se puede aplicar de forma deterministica a una fórmula KPI.
    Si el usuario indica solo el mes, usa el año actual del negocio. Esto sigue
    la regla general de período actual y permite seguimientos naturales como
    "¿y en septiembre?" sin perder el comercio o concepto anterior.
    """
    t = _normalizar(pregunta)
    hoy = fecha_local()

    def rango(inicio: date, fin_exclusivo: date, granularidad: str = "rango"):
        return {
            "inicio": inicio.isoformat(),
            "fin_inclusivo": (fin_exclusivo - timedelta(days=1)).isoformat(),
            "fin_exclusivo": fin_exclusivo.isoformat(),
            "granularidad": granularidad,
        }

    if re.search(r"\banteayer\b", t):
        return rango(hoy - timedelta(days=2), hoy - timedelta(days=1), "dia")
    if re.search(r"\bayer\b", t):
        return rango(hoy - timedelta(days=1), hoy, "dia")
    if re.search(r"\bhoy\b", t):
        return rango(hoy, hoy + timedelta(days=1), "dia")
    if re.search(r"\besta semana\b", t):
        inicio = hoy - timedelta(days=hoy.weekday())
        return rango(inicio, hoy + timedelta(days=1), "semana")
    if re.search(r"\bsemana pasada\b", t):
        fin = hoy - timedelta(days=hoy.weekday())
        return rango(fin - timedelta(days=7), fin, "semana")
    m_dias = re.search(r"\bultimos?\s+(\d+)\s+d[ií]as?\b", t)
    if m_dias:
        cantidad = max(int(m_dias.group(1)), 1)
        return rango(hoy - timedelta(days=cantidad - 1), hoy + timedelta(days=1), "rango")
    if re.search(r"\beste mes\b", t):
        inicio = hoy.replace(day=1)
        fin = (inicio.replace(day=28) + timedelta(days=4)).replace(day=1)
        return rango(inicio, fin, "mes")
    if re.search(r"\bmes pasado\b", t):
        fin = hoy.replace(day=1)
        inicio = (fin - timedelta(days=1)).replace(day=1)
        return rango(inicio, fin, "mes")
    mes = next((numero for nombre, numero in _MESES.items()
                if re.search(rf"\b{nombre}\b", t)), None)
    anio_m = re.search(r"\b(20\d{2})\b", t)
    if not mes:
        return {}
    anio = int(anio_m.group(1)) if anio_m else hoy.year
    ultimo = monthrange(anio, mes)[1]
    if mes == 12:
        fin_exclusivo = f"{anio + 1:04d}-01-01"
    else:
        fin_exclusivo = f"{anio:04d}-{mes + 1:02d}-01"
    return {
        "inicio": f"{anio:04d}-{mes:02d}-01",
        "fin_inclusivo": f"{anio:04d}-{mes:02d}-{ultimo:02d}",
        "fin_exclusivo": fin_exclusivo,
        "granularidad": "mes",
    }


def modo_resultado(pregunta: str, columnas=None, filas=None) -> str:
    """Clasifica la forma del resultado para continuarla sin adivinar."""
    texto = _normalizar(pregunta)
    if es_consulta_composicion(pregunta):
        return "detalle"
    if re.search(r"\b(?:por|agrupad[oa]s?\s+por)\s+(?:categoria|concepto|comercio|moneda)\b", texto):
        return "desglose"
    if re.search(r"\b(?:presupuesto|disponible|porcentaje|exced|sobregir)\b", texto):
        return "desglose" if len(filas or []) > 1 else "resumen"
    return "resumen"


def operacion_resultado(pregunta: str, sql: str = "", columnas=None,
                        filas=None) -> str:
    """Describe la operacion ya ejecutada para poder continuarla.

    No intenta reconocer el KPI de negocio. Conserva una propiedad mucho mas
    pequeña y estable: si el usuario obtuvo un conteo, total, ranking, desglose
    o detalle. Esa forma es parte del contrato conversacional y no debe quedar
    a criterio del modelo en cada turno.
    """
    t = _normalizar(pregunta)
    sql_n = _normalizar(sql)
    if re.search(r"\b(?:cuant[oa]s|cantidad|numero de)\b", t) or re.search(
            r"\bcount\s*\(", sql_n):
        return "conteo"
    if re.search(r"\b(?:mayor|menor|top|mas alto|mas alta|mas caro|mas cara|menos)\b", t):
        return "ranking"
    if es_consulta_composicion(pregunta) or re.search(
            r"\b(?:detalle|lista|movimientos|transacciones)\b", t):
        return "detalle"
    if re.search(r"\b(?:exceso|exced|sobregir|presupuesto|disponible|porcentaje)\b", t):
        return "comparacion"
    if re.search(r"\b(?:cuanto|total|suman|suma|sumo)\b", t):
        return "total"
    nombres = {_nombre(c) for c in (columnas or [])}
    tiene_fecha = any(n == "fecha" or n.startswith("fecha_") for n in nombres)
    tiene_descripcion = bool(set(_GRUPOS_FILTRO["descripcion"]) & nombres)
    tiene_monto = any(n in nombres for n in _MONTOS_DETALLE)
    if tiene_fecha and tiene_descripcion and tiene_monto:
        return "detalle"
    if modo_resultado(pregunta, columnas, filas) == "desglose":
        return "desglose"
    return "resumen"


def agrupacion_resultado(pregunta: str, columnas=None) -> str:
    """Obtiene la dimension humana principal del resultado."""
    t = _normalizar(pregunta)
    patrones = (
        ("concepto", r"\bconceptos?\b"),
        ("categoria", r"\bcategorias?\b"),
        ("descripcion", r"\b(?:comercios?|descripciones?)\b"),
        ("moneda", r"\bmonedas?\b"),
    )
    for dimension, patron in patrones:
        if re.search(patron, t):
            return dimension
    nombres = {_nombre(c) for c in (columnas or [])}
    for dimension in ("concepto", "categoria", "descripcion", "moneda"):
        if any(alias in nombres for alias in _GRUPOS_FILTRO[dimension]):
            return dimension
    return ""


def metrica_resultado(columnas=None) -> str:
    """Reconoce la métrica desde columnas ejecutadas, no desde redacción."""
    nombres = [_nombre(c) for c in (columnas or [])]
    if any("exceso" in n or "sobregiro" in n for n in nombres):
        return "exceso"
    tiene_presupuesto = any(any(alias in n for alias in _PRESUPUESTO)
                            for n in nombres)
    tiene_gastado = any(any(alias in n for alias in _GASTADO)
                        for n in nombres)
    if tiene_presupuesto and tiene_gastado:
        return "comparacion"
    if tiene_presupuesto:
        return "presupuesto"
    if any(any(alias in n for alias in _DISPONIBLE) for n in nombres):
        return "disponible"
    if tiene_gastado or any(n in _MONTOS_DETALLE for n in nombres):
        return "gastado"
    if any("cantidad" in n or "conteo" in n for n in nombres):
        return "conteo"
    return ""


def _es_frase_seguimiento(pregunta: str) -> bool:
    t = _normalizar(pregunta)
    return bool(re.search(
        r"^(?:y\b|tambien\b)|"
        r"\b(?:ahi|ese|esa|esos|esas|lo|los|la|las|de esas|de esos|"
        r"dentro de esa|suman|sumo|las componen|los conforman|me refiero|"
        r"quiero decir|eso quiero|que cosas|hicieron que)\b",
        t,
    ))


def _parece_detalle(estado: dict) -> bool:
    nombres = {_nombre(c) for c in (estado.get("columnas") or [])}
    return (
        any(n == "fecha" or n.startswith("fecha_") for n in nombres)
        and bool(set(_GRUPOS_FILTRO["descripcion"]) & nombres)
        and any(n in nombres for n in _MONTOS_DETALLE)
    )


def aclaracion_necesaria(pregunta: str, historial: list):
    """Pide una dimensión cuando una referencia admite varias lecturas."""
    previo = ultimo_estado(historial)
    if not previo:
        return None
    t = _normalizar(pregunta)
    explicita = bool(re.search(
        r"\b(?:conceptos?|categorias?|comercios?|descripciones?|"
        r"movimientos?|transacciones?|compras?)\b", t,
    ))
    ranking_ambiguo = (
        bool(re.search(r"\b(?:que fue lo que mas|donde gaste mas|en que gaste mas)\b", t))
        and not explicita and not _parece_detalle(previo)
    )
    composicion_ambigua = (
        bool(re.search(r"\b(?:que cosas|que fue lo que).*(?:hicieron|componen|forman)\b", t))
        and not explicita
    )
    if not ranking_ambiguo and not composicion_ambigua:
        return None
    estado = dict(previo)
    estado["pendiente"] = {
        "operacion": "ranking" if ranking_ambiguo else "desglose",
        "metrica": "gastado" if ranking_ambiguo else "exceso",
    }
    mensaje = (
        "¿Quiere verlo por concepto presupuestario, por comercio o por "
        "transacción individual?"
    )
    return mensaje, estado


def _referencia_temporal(historial: list) -> dict:
    """Busca la última fila individual con fecha, sin inferir valores."""
    for turno in reversed(historial or []):
        estado = turno.get("estado") if turno.get("rol") == "assistant" else None
        if not isinstance(estado, dict) or int(estado.get("filas_totales", 0)) != 1:
            continue
        columnas = list(estado.get("columnas") or [])
        nombres = [_nombre(c) for c in columnas]
        i_fecha = next((i for i, n in enumerate(nombres)
                        if n == "fecha" or n.startswith("fecha_")), None)
        if i_fecha is None or not estado.get("filas"):
            continue
        valor = estado["filas"][0][i_fecha]
        if isinstance(valor, str) and re.match(r"20\d{2}-\d{2}-\d{2}", valor):
            return {"fecha": valor[:10]}
        if isinstance(valor, (date, datetime)):
            return {"fecha": valor.date().isoformat() if isinstance(valor, datetime)
                    else valor.isoformat()}
    return {}


def _valor_del_resultado_mencionado(estado: dict, dimension: str,
                                    texto_normalizado: str) -> str:
    """Encuentra una entidad visible que el usuario repite literalmente.

    Solo reutiliza valores devueltos por una consulta verificada; no intenta
    extraer entidades nuevas ni adivinar nombres. Sirve para frases naturales
    como "las compras de Comidas afuera" después de una lista de conceptos.
    """
    columnas = list((estado or {}).get("columnas") or [])
    aliases = _GRUPOS_FILTRO.get(dimension, ())
    indice = _indice(columnas, aliases)
    if indice is None:
        return ""
    valores = []
    for fila in (estado or {}).get("filas") or []:
        if indice >= len(fila) or fila[indice] in (None, ""):
            continue
        valor = str(fila[indice])
        normalizado = _normalizar(valor)
        if len(normalizado) >= 3 and normalizado in texto_normalizado:
            valores.append(valor)
    return valores[0] if len(set(valores)) == 1 else ""


def contrato_seguimiento(pregunta: str, historial: list,
                         plan: dict | None = None) -> dict:
    """Construye reglas verificables para un seguimiento real.

    El modelo puede reconocer una pregunta nueva, pero una frase referencial
    no puede borrar silenciosamente filtros, periodo u operacion. El contrato
    solo usa estado proveniente de una consulta ejecutada.
    """
    previo = ultimo_estado(historial)
    relacion = str((plan or {}).get("relacion", "nueva"))
    if not previo or not (_es_frase_seguimiento(pregunta) or relacion in (
            "seguimiento", "modificacion")):
        return {}

    t = _normalizar(pregunta)
    periodo_nuevo = periodo_explicito(pregunta)
    pendiente = previo.get("pendiente") if isinstance(previo.get("pendiente"), dict) else {}
    operacion_previa = str(previo.get("operacion") or "resumen")
    metrica_previa = str(previo.get("metrica") or metrica_resultado(
        previo.get("columnas") or [],
    ))
    agrupacion_previa = str(previo.get("agrupacion") or "")
    operacion = str(
        (plan or {}).get("operacion") or pendiente.get("operacion")
        or operacion_previa
    )
    agrupacion = str((plan or {}).get("entidad") or agrupacion_previa)

    if re.search(r"\b(?:cuant[oa]s|cantidad|numero de)\b", t):
        operacion = "conteo"
    elif re.search(r"\b(?:cuanto suman|cuanto suma|totalizan|suman|sumo)\b", t):
        operacion = "total"
    elif re.search(r"\b(?:mayor|menor|top|mas alto|mas alta|mas caro|mas cara|menos)\b", t):
        operacion = "ranking"
    elif es_consulta_composicion(pregunta):
        # Una composición pide movimientos individuales, no un desglose del
        # KPI anterior. Su métrica pasa a ser el monto de cada movimiento;
        # exigir ``exceso`` aquí impediría listar las compras que lo causan.
        operacion = "detalle"
    elif re.search(r"\b(?:muestrame|lista|detalle)\b", t):
        operacion = "desglose"
    elif re.search(r"\b(?:me pase|se paso|pasar del presupuesto)\b", t):
        operacion = "comparacion"

    if re.search(r"\bconceptos?\b", t):
        agrupacion = "concepto"
    elif re.search(r"\b(?:cosas|lineas)\s+del\s+presupuesto\b", t):
        agrupacion = "concepto"
    elif re.search(r"\bcategorias?\b", t):
        agrupacion = "categoria"
    elif re.search(r"\b(?:comercios?|descripciones?)\b", t):
        agrupacion = "descripcion"
    elif re.search(r"\bmonedas?\b", t):
        agrupacion = "moneda"

    metrica = str((plan or {}).get("metrica") or "")
    if re.search(r"\b(?:exceso|exced|sobregir)\b", t):
        metrica = "exceso"
    elif re.search(r"\bpresupuesto\b", t):
        metrica = "presupuesto"
    elif re.search(r"\b(?:monto|cuanto|gastado|gaste|suman|sumo|total)\b", t):
        metrica = "gastado"
    elif re.search(r"\b(?:me pase|se paso|pasar del presupuesto)\b", t):
        metrica = "exceso"
    if not metrica and pendiente.get("metrica"):
        metrica = str(pendiente["metrica"])
    if es_consulta_composicion(pregunta):
        metrica = "gastado"
    if not metrica:
        metrica = metrica_previa

    referencia_temporal = _referencia_temporal(historial)
    relacion_temporal = ""
    if re.search(r"\bantes\b", t):
        relacion_temporal = "antes"
    elif re.search(r"\bdespues\b", t):
        relacion_temporal = "despues"
    elif re.search(r"\b(?:contando|incluyendo)\b.*\b(?:esa|ese|tambien)\b", t):
        relacion_temporal = "hasta_inclusive"
    # "esas" después de un conteo o total temporal significa el mismo tramo
    # que se acaba de calcular (por ejemplo, "¿cuántas compras fueron
    # después?" -> "¿y si sumo esas?"). El rango no se vuelve a inferir con
    # el modelo: se conserva en el estado verificado.
    if (not relacion_temporal and re.search(r"\b(?:esas|esos)\b", t)
            and previo.get("relacion_temporal")):
        relacion_temporal = str(previo["relacion_temporal"])
        referencia_temporal = dict(previo.get("referencia_temporal") or referencia_temporal)
    if relacion_temporal and re.search(r"\bcuanto\b", t):
        operacion = "total"
    filtros = dict(previo.get("filtros") or {})
    filtros_actuales = dict((plan or {}).get("filtros_actuales") or {})
    categoria_actual = filtros_actuales.get("categoria")
    concepto_actual = filtros_actuales.get("concepto")
    descripcion_actual = filtros_actuales.get("descripcion")
    if not concepto_actual:
        concepto_actual = _valor_del_resultado_mencionado(previo, "concepto", t)
    if categoria_actual and (
            _normalizar(categoria_actual) != _normalizar(filtros.get("categoria"))
            or re.search(r"\b(?:toda|todo|completa|completo|entera|entero)\b", t)):
        for clave in ("linea_id", "concepto", "descripcion"):
            filtros.pop(clave, None)
        agrupacion = "categoria"
    if concepto_actual and _normalizar(concepto_actual) != _normalizar(
            filtros.get("concepto")):
        filtros.pop("linea_id", None)
        filtros.pop("descripcion", None)
        filtros["concepto"] = concepto_actual
    if descripcion_actual and _normalizar(descripcion_actual) != _normalizar(
            filtros.get("descripcion")):
        filtros.pop("descripcion", None)
    filtros.update({k: v for k, v in filtros_actuales.items()
                    if v not in (None, "")})
    if relacion_temporal or (
            re.search(r"\b(?:esas|esos)\b", t)
            and operacion != operacion_previa):
        filtros.pop("descripcion", None)
    if relacion_temporal and operacion in ("total", "conteo"):
        agrupacion = ""

    entidades = []
    if re.search(r"\b(?:esas|esos|las componen|los conforman|de esas|de esos)\b", t):
        columnas = list(previo.get("columnas") or [])
        nombres = [_nombre(c) for c in columnas]
        dimension = agrupacion_previa
        aliases = _GRUPOS_FILTRO.get(dimension, ())
        indice = next((nombres.index(a) for a in aliases if a in nombres), None)
        if indice is not None:
            for fila in previo.get("filas") or []:
                valor = fila[indice]
                if valor not in (None, "") and str(valor) not in entidades:
                    entidades.append(str(valor))

    return {
        "operacion_previa": operacion_previa,
        "operacion": operacion,
        "agrupacion_previa": agrupacion_previa,
        "agrupacion": agrupacion,
        "metrica": metrica,
        "metrica_previa": metrica_previa,
        "filtros": filtros,
        "periodo": periodo_nuevo or dict(previo.get("periodo") or {}),
        "periodo_cambiado": bool(periodo_nuevo),
        "entidades_previas": entidades,
        "referencia_temporal": referencia_temporal,
        "relacion_temporal": relacion_temporal,
        "sql_previo": str(previo.get("sql", "") or ""),
        "referencia_conjunto": bool(
            re.search(r"\b(?:esas|esos|esa|ese)\b", t)
            and operacion != operacion_previa
        ),
        "estado_previo": previo,
    }


def instruccion_contrato(contrato: dict) -> str:
    """Convierte el contrato en una orden compacta para text-to-SQL."""
    if not contrato:
        return ""
    datos = {
        "operacion_requerida": contrato.get("operacion", ""),
        "dimension_requerida": contrato.get("agrupacion", ""),
        "metrica_requerida": contrato.get("metrica", ""),
        "filtros_heredados": contrato.get("filtros", {}),
        "periodo": contrato.get("periodo", {}),
        "entidades_del_resultado_anterior": contrato.get("entidades_previas", []),
        "referencia_temporal": contrato.get("referencia_temporal", {}),
        "relacion_temporal": contrato.get("relacion_temporal", ""),
    }
    base = (
        "CONTRATO DETERMINISTICO DE SEGUIMIENTO: "
        + json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
        + ". La pregunta actual modifica solo lo que declara este contrato. "
        "Conserva el resto. Si hay entidades anteriores, limita el universo "
        "exactamente a ellas. Proyecta la dimension pedida y respeta la "
        "operacion: conteo=COUNT, total=SUM, ranking=agrega/ordena/limita, "
        "desglose=una fila por valor de la dimension. Antes/después usa la "
        "fecha de referencia como límite y no como filtro de descripción."
    )
    if (contrato.get("referencia_conjunto")
            and contrato.get("sql_previo")):
        base += (
            " Conserva exactamente el universo (FROM, JOIN y WHERE) de este "
            "SQL verificado anterior y cambia solamente la agregación solicitada: "
            + str(contrato["sql_previo"])
        )
    return base


def _indice_numerico(columnas, preferidos):
    nombres = [_nombre(c) for c in columnas]
    for candidato in preferidos:
        for i, nombre in enumerate(nombres):
            if candidato in nombre:
                return i
    return None


def resolver_referencia(pregunta: str, historial: list):
    """Resuelve referencias al resultado anterior sobre filas verificadas."""
    estado = ultimo_estado(historial)
    if not estado or not estado.get("filas") or not estado.get("columnas"):
        return None
    t = _normalizar(pregunta)
    # Una pregunta completa con fecha/mes siempre abre un contexto nuevo. Las
    # referencias de atributos ("su presupuesto") solo son seguras después de
    # que este resolver haya seleccionado una fila explícita.
    if tiene_periodo_explicito(pregunta):
        return None
    seleccion_explicita = bool(re.search(
        r"\b(?:primero|primera|ultimo|ultima|mayor|menor|mas\s+car[oa]|"
        r"que\s+mas|que\s+menos)\b",
        t,
    ))
    if not seleccion_explicita and not estado.get("seleccion"):
        return None
    if not re.search(
        r"\b(?:el|la|los|las)\s+(?:primero|primera|ultimo|ultima|mayor|menor|"
        r"mas\s+car[oa]|que\s+mas|que\s+menos)\b|"
        r"\b(?:su|ese|esa|esos|esas|de\s+esos|de\s+esas)\b",
        t,
    ) and not re.search(
        r"^[¿?¡!\s]*y\s+.*\b(?:presupuesto|gastado|disponible|porcentaje)\b",
        t,
    ):
        return None
    columnas = list(estado.get("columnas") or [])
    filas = [tuple(f) for f in estado.get("filas") or []]
    if not filas:
        return None
    nombres = [_nombre(c) for c in columnas]
    # "el concepto con mayor gasto dentro de esa categoría" no selecciona
    # una categoría de la tabla anterior: solicita una dimensión nueva. Igual
    # ocurre con "mayor exceso" cuando el resultado previo no calculó exceso.
    for dimension, patron in (
        ("concepto", r"\bconceptos?\b"),
        ("categoria", r"\bcategorias?\b"),
        ("descripcion", r"\b(?:comercios?|descripciones?)\b"),
    ):
        if re.search(patron, t) and not any(
                alias in nombres for alias in _GRUPOS_FILTRO[dimension]):
            return None
    if re.search(r"\b(?:exceso|exced|sobregir)\b", t) and _indice(
            columnas, ("exceso", "sobregiro")) is None:
        return None
    i_gastado = _indice_numerico(
        columnas, ("gastado", "gasto_neto", "gasto", "total", "monto"),
    )
    i_exceso = _indice_numerico(columnas, ("exceso", "sobregiro"))
    i_disponible = _indice_numerico(columnas, ("disponible", "saldo"))
    indice = 0
    criterio = "primero"
    if re.search(r"\b(?:ultimo|ultima)\b", t):
        indice, criterio = len(filas) - 1, "ultimo"
    elif re.search(r"\b(?:menor|menos)\b", t):
        i = i_disponible if "disponible" in t and i_disponible is not None else i_gastado
        valores = [(_decimal(fila[i]), n) for n, fila in enumerate(filas)] if i is not None else []
        valores = [(valor, n) for valor, n in valores if valor is not None]
        if valores:
            indice, criterio = min(valores)[1], "menor"
    elif re.search(r"\b(?:mayor|mas|más|caro|cara)\b", t):
        i = i_exceso if ("exced" in t or "sobregir" in t) and i_exceso is not None else i_gastado
        valores = [(_decimal(fila[i]), n) for n, fila in enumerate(filas)] if i is not None else []
        valores = [(valor, n) for valor, n in valores if valor is not None]
        if valores:
            indice, criterio = max(valores)[1], "mayor"
    seleccionada = [filas[indice]]
    estado_nuevo = dict(estado)
    estado_nuevo["filas"] = [[_json_valor(v) for v in seleccionada[0]]]
    estado_nuevo["filas_totales"] = 1
    estado_nuevo["seleccion"] = {"indice": indice, "criterio": criterio}
    estado_nuevo["modo"] = "resumen"
    return {
        "columnas": columnas,
        "filas": seleccionada,
        "estado": estado_nuevo,
        "sql": estado.get("sql", ""),
    }


def resolver_sobre_resultado(pregunta: str, historial: list):
    """Resuelve proyecciones y sumas que no necesitan consultar otra vez.

    Solo opera sobre el ultimo resultado verificado. Esto cubre preguntas como
    "¿cuál fue el monto?" después de identificar una única transacción y
    "¿cuánto suman?" después de ver su detalle. Nunca mezcla monedas.
    """
    estado = ultimo_estado(historial)
    columnas = list(estado.get("columnas") or [])
    filas = [tuple(f) for f in estado.get("filas") or []]
    if not estado or not estado.get("verificado") or not columnas or not filas:
        return None
    if tiene_periodo_explicito(pregunta):
        return None
    t = _normalizar(pregunta)
    nombres = [_nombre(c) for c in columnas]

    pide_suma = bool(re.search(
        r"\b(?:cuanto suman|cuanto suma|totalizan|suman|sumo|"
        r"todo eso cuanto da|cuanto da|cuanto es)\b", t,
    ))
    if pide_suma:
        if int(estado.get("filas_totales", len(filas))) != len(filas):
            return None
        i_monto = _indice_numerico(
            columnas, ("gasto_neto", "gastado", "monto_crc", "monto", "importe", "total"),
        )
        if i_monto is None:
            return None
        i_moneda = _indice(columnas, ("moneda", "monto_moneda", "currency", "codigo_moneda"))
        totales = {}
        for fila in filas:
            monto = _decimal(fila[i_monto])
            if monto is None:
                return None
            moneda = str(fila[i_moneda]).strip() if i_moneda is not None else ""
            totales[moneda] = totales.get(moneda, Decimal("0")) + monto
        columnas_nuevas = ["moneda", "gastado"] if i_moneda is not None else ["gastado"]
        filas_nuevas = (
            [(moneda, total) for moneda, total in totales.items()]
            if i_moneda is not None else [(next(iter(totales.values())),)]
        )
        nuevo = crear_estado(
            pregunta, estado.get("sql", ""), estado.get("kpi", ""),
            estado.get("unidad", ""), columnas_nuevas, filas_nuevas,
            previo=estado,
        )
        return {"columnas": columnas_nuevas, "filas": filas_nuevas,
                "estado": nuevo, "sql": estado.get("sql", "")}

    # Una proyeccion es segura únicamente cuando el resultado anterior ya
    # identificó exactamente una fila.
    if len(filas) != 1 or int(estado.get("filas_totales", len(filas))) != 1:
        return None
    if re.search(
            r"\b(?:mayor|menor|top|dentro|conforman|componen|exceso|exced|"
            r"antes|despues|contando|incluyendo)\b",
            t):
        return None

    if re.search(r"\b(?:mucho|poco|bien|mal|pase|pasado)\b", t):
        i_pre = _indice(columnas, _PRESUPUESTO)
        i_gas = _indice(columnas, _GASTADO)
        i_dis = _indice(columnas, _DISPONIBLE)
        i_pct = _indice(columnas, _PORCENTAJE)
        if i_pre is not None and i_gas is not None:
            presupuesto = _decimal(filas[0][i_pre])
            gastado = _decimal(filas[0][i_gas])
            disponible = _decimal(filas[0][i_dis]) if i_dis is not None else None
            porcentaje = _decimal(filas[0][i_pct]) if i_pct is not None else None
            if presupuesto is not None and gastado is not None:
                if disponible is None:
                    disponible = presupuesto - gastado
                if porcentaje is None and presupuesto:
                    porcentaje = gastado / presupuesto * Decimal("100")
                if gastado <= presupuesto:
                    texto = (
                        "Está dentro del presupuesto. Ha consumido "
                        f"{_formato_numero(porcentaje or Decimal('0'))}% y le "
                        f"quedan {_formato_numero(disponible)}."
                    )
                else:
                    texto = (
                        "Sí, excedió el presupuesto por "
                        f"{_formato_numero(abs(disponible))}; ha consumido "
                        f"{_formato_numero(porcentaje or Decimal('0'))}%."
                    )
                nuevo = crear_estado(
                    pregunta, estado.get("sql", ""), estado.get("kpi", ""),
                    estado.get("unidad", ""), columnas, filas, previo=estado,
                )
                return {"columnas": columnas, "filas": filas, "estado": nuevo,
                        "sql": estado.get("sql", ""), "texto": texto}
    pedidos = []
    campos = (
        ("categoria", ("categoria", "cuenta_contable"), r"\b(?:categoria|cuenta contable)\b"),
        ("concepto", ("concepto",), r"\bconcepto\b"),
        ("monto", ("gasto_neto", "gastado", "monto_neto", "monto_crc", "monto", "importe", "total"), r"\b(?:monto|gastado|gaste)\b"),
        ("disponible", ("disponible", "saldo_disponible", "diferencia"), r"\b(?:queda|quedan|disponible|saldo)\b"),
        ("moneda", ("moneda", "monto_moneda", "currency"), r"\bmoneda\b"),
        ("fecha", ("fecha", "fecha_transaccion"), r"\b(?:fecha|dia)\b"),
    )
    for _, aliases, patron in campos:
        if not re.search(patron, t):
            continue
        indice = next((nombres.index(a) for a in aliases if a in nombres), None)
        if indice is not None and indice not in pedidos:
            pedidos.append(indice)
    if not pedidos:
        return None
    columnas_nuevas = [columnas[i] for i in pedidos]
    filas_nuevas = [tuple(filas[0][i] for i in pedidos)]
    nuevo = crear_estado(
        pregunta, estado.get("sql", ""), estado.get("kpi", ""),
        estado.get("unidad", ""), columnas_nuevas, filas_nuevas,
        previo=estado,
    )
    return {"columnas": columnas_nuevas, "filas": filas_nuevas,
            "estado": nuevo, "sql": estado.get("sql", "")}


def sql_con_periodo_nuevo(pregunta: str, historial: list,
                          contrato: dict) -> str:
    """Reutiliza el SELECT anterior cambiando únicamente su rango temporal."""
    actual = periodo_explicito(pregunta)
    previo = (contrato or {}).get("estado_previo") or ultimo_estado(historial)
    anterior = dict(previo.get("periodo") or {})
    sql = str(previo.get("sql", "") or "")
    if not actual or not anterior or not sql:
        return ""
    if (contrato.get("operacion") != contrato.get("operacion_previa")
            or contrato.get("agrupacion") != contrato.get("agrupacion_previa")):
        return ""
    reemplazos = {
        anterior.get("inicio"): actual.get("inicio"),
        anterior.get("fin_exclusivo"): actual.get("fin_exclusivo"),
        anterior.get("fin_inclusivo"): actual.get("fin_inclusivo"),
    }
    validos = {str(viejo): str(nuevo) for viejo, nuevo in reemplazos.items()
               if viejo and nuevo and str(viejo) in sql}
    if not validos:
        return ""
    patron = "|".join(re.escape(v) for v in sorted(validos, key=len, reverse=True))
    salida = re.sub(patron, lambda m: validos[m.group(0)], sql)
    return salida


def sql_temporal_desde_estado(contrato: dict) -> str:
    """Deriva una suma o conteo temporal desde un detalle ya verificado.

    Evita que el modelo reescriba el universo de una conversación cuando el
    usuario dice "antes de esa" o "después". El SQL base ya contiene los JOIN,
    concepto, moneda y período correctos; aquí solo se agrega una comparación
    de fecha y una agregación sobre columnas que provinieron de ese SELECT.
    """
    if not contrato or contrato.get("operacion") not in {"total", "conteo"}:
        return ""
    relacion = str(contrato.get("relacion_temporal") or "")
    referencia = (contrato.get("referencia_temporal") or {}).get("fecha")
    previo = contrato.get("estado_previo") or {}
    base = str(previo.get("sql_detalle") or "").strip().rstrip(";")
    campos = dict(previo.get("campos_detalle") or {})
    fecha = str(campos.get("fecha") or "")
    monto = str(campos.get("monto") or "")
    moneda = str(campos.get("moneda") or "")
    if (relacion not in {"antes", "despues", "hasta_inclusive"}
            or not referencia or not base or not fecha):
        return ""
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", valor)
               for valor in (fecha, monto) if valor):
        return ""
    operador = {"antes": "<", "despues": ">", "hasta_inclusive": "<="}[relacion]
    where = (
        f"CAST(_seguimiento.{fecha} AS DATE) {operador} "
        f"CAST('{referencia}' AS DATE)"
    )
    if contrato.get("operacion") == "conteo":
        return f"SELECT COUNT(*) AS cantidad FROM ({base}) AS _seguimiento WHERE {where}"
    if not monto:
        return ""
    if moneda and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", moneda):
        return (
            f"SELECT _seguimiento.{moneda} AS moneda, "
            f"SUM(_seguimiento.{monto}) AS gastado FROM ({base}) AS _seguimiento "
            f"WHERE {where} GROUP BY _seguimiento.{moneda} ORDER BY moneda"
        )
    return f"SELECT SUM(_seguimiento.{monto}) AS gastado FROM ({base}) AS _seguimiento WHERE {where}"


def validar_contrato_sql(sql: str, contrato: dict) -> tuple[bool, str]:
    """Valida la forma mínima prometida por el seguimiento."""
    if not contrato:
        return True, ""
    try:
        import sqlglot
        from sqlglot import exp
        arbol = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return False, "no se pudo analizar el SQL del seguimiento"
    operacion = contrato.get("operacion")
    if operacion == "conteo" and not any(arbol.find_all(exp.Count)):
        return False, "el seguimiento debía conservar un conteo con COUNT"
    if operacion == "total" and not any(arbol.find_all(exp.Sum)):
        return False, "el seguimiento debía devolver un total con SUM"
    dimension = contrato.get("agrupacion")
    proyectadas = {
        _nombre(sel.alias_or_name or sel.sql())
        for sel in arbol.expressions
    }
    if contrato.get("metrica") == "exceso" and not (
            set(_EXCESO) & proyectadas):
        return False, "el seguimiento debía calcular y nombrar el exceso"
    if dimension and operacion in ("ranking", "desglose"):
        aliases = set(_GRUPOS_FILTRO.get(dimension, (dimension,)))
        if not (aliases & proyectadas):
            return False, f"el seguimiento debía proyectar {dimension}"
    return True, ""


def filtros_unicos(columnas, filas) -> dict:
    """Extrae dimensiones que tienen un solo valor en todo el resultado."""
    filtros = {}
    nombres = [_nombre(c) for c in columnas]
    for canonico, aliases in _GRUPOS_FILTRO.items():
        indice = next((nombres.index(a) for a in aliases if a in nombres), None)
        if indice is None:
            continue
        valores = [f[indice] for f in filas if f[indice] not in (None, "")]
        unicos = {str(v) for v in valores}
        if len(unicos) == 1:
            filtros[canonico] = _json_valor(valores[0])
    return filtros


def _periodo_resultado(pregunta: str, columnas, filas, previo: dict | None) -> dict:
    periodo = dict((previo or {}).get("periodo") or {})
    explicito = periodo_explicito(pregunta)
    if explicito:
        return explicito
    # Una fila seleccionada dentro de agosto no redefine el análisis como "un
    # solo día". Si el turno es seguimiento, el rango heredado sigue siendo el
    # universo de la conversación hasta que el usuario lo cambie expresamente.
    if periodo:
        return periodo

    nombres = [_nombre(c) for c in columnas]
    indices = [i for i, n in enumerate(nombres)
               if n == "mes" or n.startswith("fecha") or n.endswith("_fecha")]
    fechas = []
    for i in indices:
        for fila in filas:
            valor = fila[i]
            if isinstance(valor, datetime):
                fechas.append(valor.date())
            elif isinstance(valor, date):
                fechas.append(valor)
            elif isinstance(valor, str):
                try:
                    fechas.append(datetime.fromisoformat(valor[:19]).date())
                except ValueError:
                    pass
    if fechas:
        periodo = {
            "inicio": min(fechas).isoformat(),
            "fin_inclusivo": max(fechas).isoformat(),
            "granularidad": (
                "mes" if min(fechas).year == max(fechas).year
                and min(fechas).month == max(fechas).month else "rango"
            ),
        }
    return periodo


def crear_estado(pregunta: str, sql: str, kpi: str, unidad: str,
                 columnas, filas, previo: dict | None = None,
                 operacion: str | None = None,
                 agrupacion: str | None = None,
                 referencia_temporal: dict | None = None,
                 relacion_temporal: str | None = None) -> dict:
    """Crea el contrato persistible de una consulta ya ejecutada."""
    filas_json = [[_json_valor(v) for v in fila]
                  for fila in list(filas)[:_MAX_FILAS_ESTADO]]
    filtros = dict((previo or {}).get("filtros") or {})
    filtros.update(filtros_unicos(columnas, filas))
    nombres = [_nombre(c) for c in columnas]
    es_detalle = _parece_detalle({"columnas": columnas})
    campos_detalle = dict((previo or {}).get("campos_detalle") or {})
    sql_detalle = str((previo or {}).get("sql_detalle") or "")
    if es_detalle:
        fecha = next((str(columnas[i]) for i, n in enumerate(nombres)
                      if n == "fecha" or n.startswith("fecha_")), "")
        monto = next((str(columnas[i]) for i, n in enumerate(nombres)
                      if n in _MONTOS_DETALLE), "")
        moneda = next((str(columnas[i]) for i, n in enumerate(nombres)
                       if n in ("moneda", "monto_moneda", "currency", "codigo_moneda")), "")
        campos_detalle = {"fecha": fecha, "monto": monto, "moneda": moneda}
        sql_detalle = sql
    base = {
        "version": 1,
        "pregunta": pregunta,
        "sql": sql,
        "kpi": kpi or "",
        "unidad": unidad or "",
        "columnas": [str(c) for c in columnas],
        "filas": filas_json,
        "filtros": filtros,
        "periodo": _periodo_resultado(pregunta, columnas, filas, previo),
        "filas_totales": len(filas),
        "modo": modo_resultado(pregunta, columnas, filas),
        "operacion": operacion or operacion_resultado(
            pregunta, sql, columnas, filas,
        ),
        "agrupacion": agrupacion if agrupacion is not None else agrupacion_resultado(
            pregunta, columnas,
        ),
        "metrica": metrica_resultado(columnas),
        "dimensiones": [str(c) for c in columnas],
        "orden": "consulta",
    }
    if sql_detalle and campos_detalle.get("fecha") and campos_detalle.get("monto"):
        base["sql_detalle"] = sql_detalle
        base["campos_detalle"] = campos_detalle
    if referencia_temporal:
        base["referencia_temporal"] = dict(referencia_temporal)
    elif (previo or {}).get("referencia_temporal"):
        base["referencia_temporal"] = dict(previo["referencia_temporal"])
    if relacion_temporal:
        base["relacion_temporal"] = relacion_temporal
    elif (previo or {}).get("relacion_temporal"):
        base["relacion_temporal"] = str(previo["relacion_temporal"])
    canonico = json.dumps(base, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    base["resultado_hash"] = hashlib.sha256(canonico.encode("utf-8")).hexdigest()
    base["verificado"] = True
    return base


def criticar_respuesta(pregunta: str, texto: str, estado: dict) -> tuple[bool, str]:
    """Critico LLM opcional: puede bloquear, pero nunca corregir numeros."""
    import config
    import llm

    if not getattr(config, "BOT_CRITICO_RESPUESTAS", False):
        return True, ""
    esquema = {
        "type": "object",
        "properties": {
            "veredicto": {"type": "string", "enum": ["PASS", "FAIL"]},
            "motivo": {"type": "string"},
        },
        "required": ["veredicto", "motivo"],
        "additionalProperties": False,
    }
    contenido = json.dumps({
        "pregunta": pregunta,
        "respuesta": texto,
        "estado_verificado": {
            "kpi": estado.get("kpi", ""),
            "filtros": estado.get("filtros", {}),
            "periodo": estado.get("periodo", {}),
            "columnas": estado.get("columnas", []),
            "filas": estado.get("filas", [])[:20],
        },
    }, ensure_ascii=False)
    try:
        resp = llm.generar_texto(
            config.BOT_MODELO_RESPUESTA, contenido, max_tokens=180,
            thinking_level="low", response_schema=esquema,
            system=(
                "Audite coherencia, continuidad de filtros y correspondencia "
                "exacta de cifras. Devuelva PASS o FAIL. No calcule cifras "
                "nuevas, no reescriba la respuesta y no use conocimiento externo."
            ),
        )
        dato = json.loads(resp.texto)
        return dato.get("veredicto") == "PASS", str(dato.get("motivo", ""))
    except Exception:
        # El critico es defensa adicional; su indisponibilidad no invalida las
        # comprobaciones deterministicas que ya pasaron.
        return True, ""


def filtrar_filas_por_contexto(columnas, filas, estado: dict):
    """Aplica a un resultado nuevo los filtros inequívocos del turno anterior."""
    filtros = (estado or {}).get("filtros") or {}
    if not filtros or not filas:
        return filas, {}
    nombres = [_nombre(c) for c in columnas]
    salida = list(filas)
    aplicados = {}
    # linea_id es la llave mas estable; si existe no se agregan concepto/categoria.
    claves = ["linea_id"] if filtros.get("linea_id") else ["concepto", "categoria", "titular"]
    for clave in claves:
        esperado = filtros.get(clave)
        # El estado conversacional puede contener una clave antigua o inferida
        # por el planificador que no aplica al resultado actual. Nunca debe
        # convertir ese dato en un KeyError ni en un filtro fantasma.
        aliases = _GRUPOS_FILTRO.get(clave, ())
        if not aliases:
            continue
        indice = next((nombres.index(a) for a in aliases if a in nombres), None)
        if indice is None or esperado in (None, ""):
            continue
        candidatas = [f for f in salida if _normalizar(f[indice]) == _normalizar(esperado)]
        if candidatas:
            salida = candidatas
            aplicados[clave] = esperado
    return salida, aplicados


def validar_resultado(columnas, filas, contexto: dict | None = None) -> tuple[bool, str]:
    """Comprueba invariantes aritmeticas y de continuidad antes de responder."""
    if not filas:
        return True, ""
    contexto = contexto or {}
    # Contrato semántico: cuando el plan lo declara, la respuesta debe
    # proyectar la dimensión y métrica solicitadas. Se omite si no hay contrato
    # (compatibilidad con KPIs/metadatos antiguos).
    entidad = str(contexto.get("entidad") or "").strip().lower()
    entidad = {"comercio": "descripcion", "comercios": "descripcion",
               "registro": "transaccion", "registros": "transaccion"}.get(
                   entidad, entidad)
    operacion = str(contexto.get("operacion") or "").strip().lower()
    nombres = [_nombre(c) for c in columnas]
    if entidad and operacion in {"ranking", "desglose", "detalle", "comparacion"}:
        candidatos = {
            "categoria": ("categoria",),
            "concepto": ("concepto",),
            "descripcion": ("descripcion", "comercio"),
            "moneda": ("moneda", "currency", "codigo_moneda"),
            "transaccion": ("movimiento_id", "transaccion_id", "id"),
        }.get(entidad, ())
        if candidatos and not any(c in nombres for c in candidatos):
            return False, f"el resultado no proyecta la entidad solicitada {entidad}"
    metrica = str(contexto.get("metrica") or "").strip().lower()
    if metrica and operacion in {"total", "ranking", "desglose", "comparacion"}:
        metricas = {
            "gastado": _GASTADO + _MONTOS_DETALLE,
            "presupuesto": _PRESUPUESTO,
            "disponible": _DISPONIBLE,
            "exceso": _EXCESO,
            "conteo": ("conteo", "count", "total_movimientos", "cantidad"),
        }.get(metrica, ())
        if metricas and not any(_nombre(c) in nombres for c in metricas):
            return False, f"el resultado no proyecta la métrica solicitada {metrica}"
    i_pre = _indice(columnas, _PRESUPUESTO)
    i_gas = _indice(columnas, _GASTADO)
    i_dis = _indice(columnas, _DISPONIBLE)
    i_exceso = _indice(columnas, _EXCESO)
    i_pct = _indice(columnas, _PORCENTAJE)
    tolerancia = Decimal("0.02")
    if i_pre is not None and i_gas is not None:
        for fila in filas:
            pre, gas = _decimal(fila[i_pre]), _decimal(fila[i_gas])
            if pre is None or gas is None:
                continue
            if i_dis is not None:
                disponible = _decimal(fila[i_dis])
                if disponible is not None and abs(disponible - (pre - gas)) > tolerancia:
                    return False, "el disponible no coincide con presupuesto menos gastado"
            if i_exceso is not None:
                exceso = _decimal(fila[i_exceso])
                if exceso is not None and abs(exceso - (gas - pre)) > tolerancia:
                    return False, "el exceso no coincide con gastado menos presupuesto"
            if i_pct is not None and pre != 0:
                porcentaje = _decimal(fila[i_pct])
                calculado = gas / pre * Decimal("100")
                if porcentaje is not None and abs(porcentaje - calculado) > Decimal("0.11"):
                    return False, "el porcentaje no coincide con gastado dividido entre presupuesto"

    i_moneda = _indice(columnas, ("moneda", "currency", "codigo_moneda"))
    # Filas separadas por moneda NO son una suma entre monedas. La comparación
    # presupuestaria sí requiere una única moneda explícita.
    if i_moneda is not None and i_pre is not None and i_gas is not None:
        monedas = {_normalizar(f[i_moneda]) for f in filas if f[i_moneda] not in (None, "")}
        if len(monedas) > 1:
            return False, "el agregado mezcla monedas sin una conversion explicita"

    i_periodo_gasto = _indice(columnas, ("periodo_gasto", "mes_gasto"))
    i_periodo_pre = _indice(columnas, ("periodo_presupuesto", "mes_presupuesto"))
    if i_periodo_gasto is not None and i_periodo_pre is not None:
        if any(_normalizar(f[i_periodo_gasto]) != _normalizar(f[i_periodo_pre])
               for f in filas):
            return False, "gasto y presupuesto pertenecen a periodos distintos"

    # Si el resultado declara un total repetido, debe coincidir con el detalle.
    i_total = _indice(columnas, ("total_general",))
    i_monto = _indice(columnas, _MONTOS_DETALLE)
    if i_total is not None and i_monto is not None and i_total != i_monto:
        montos = [_decimal(f[i_monto]) for f in filas]
        totales = {_decimal(f[i_total]) for f in filas if _decimal(f[i_total]) is not None}
        if all(v is not None for v in montos) and len(totales) == 1:
            if abs(sum(montos, Decimal("0")) - next(iter(totales))) > tolerancia:
                return False, "la suma del detalle no coincide con el total declarado"

    filtros = (contexto or {}).get("filtros") or {}
    nombres = [_nombre(c) for c in columnas]
    for clave, esperado in filtros.items():
        aliases = _GRUPOS_FILTRO.get(clave, ())
        indice = next((nombres.index(a) for a in aliases if a in nombres), None)
        if indice is not None and any(
                (_normalizar(esperado) not in _normalizar(f[indice])
                 if clave == "descripcion" else
                 _normalizar(f[indice]) != _normalizar(esperado)) for f in filas):
            return False, f"el resultado mezclo valores fuera del filtro {clave}"
    return True, ""


def reconciliar_presupuesto_fuente(columnas, filas, presupuestos: dict):
    """Corrige un presupuesto agregado contra su valor mensual de origen.

    ``presupuestos`` contiene llaves ``linea:<id>`` y ``concepto:<nombre>``.
    Solo se toca el denominador cuando la fuente tiene un valor inequívoco; los
    gastos ejecutados siguen viniendo del resultado consultado.
    """
    i_linea = _indice(columnas, _GRUPOS_FILTRO["linea_id"])
    i_concepto = _indice(columnas, _GRUPOS_FILTRO["concepto"])
    i_pre = _indice(columnas, _PRESUPUESTO)
    if i_pre is None or (i_linea is None and i_concepto is None):
        return list(filas), []
    i_gas = _indice(columnas, _GASTADO)
    i_dis = _indice(columnas, _DISPONIBLE)
    i_exceso = _indice(columnas, _EXCESO)
    i_pct = _indice(columnas, _PORCENTAJE)
    salida, cambios = [], []
    for original in filas:
        fila = list(original)
        fuente = None
        if i_linea is not None:
            fuente = presupuestos.get(f"linea:{_normalizar(fila[i_linea])}")
        if fuente is None and i_concepto is not None:
            fuente = presupuestos.get(f"concepto:{_normalizar(fila[i_concepto])}")
        actual, correcto = _decimal(fila[i_pre]), _decimal(fuente)
        if correcto is not None and actual is not None and actual != correcto:
            fila[i_pre] = correcto
            gastado = _decimal(fila[i_gas]) if i_gas is not None else None
            if gastado is not None:
                if i_dis is not None:
                    fila[i_dis] = correcto - gastado
                if i_exceso is not None:
                    fila[i_exceso] = gastado - correcto
                if i_pct is not None and correcto != 0:
                    fila[i_pct] = gastado / correcto * Decimal("100")
            cambios.append({"anterior": actual, "correcto": correcto})
        salida.append(tuple(fila))
    return salida, cambios


def _monto_pedido(pregunta: str) -> Decimal | None:
    t = _normalizar(pregunta).replace("₡", " ")
    if not re.search(r"\b(?:quit|sin|exclu|sac)", t):
        return None
    m = re.search(r"(?<!\d)(\d[\d.,]*)\s*(mil|k)?\b", t)
    if not m:
        return None
    bruto, escala = m.group(1), m.group(2)
    if escala:
        bruto = bruto.replace(".", "").replace(",", ".")
        try:
            return Decimal(bruto) * Decimal("1000")
        except InvalidOperation:
            return None
    # En español, un unico separador seguido por tres cifras es millar.
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", bruto):
        bruto = bruto.replace(".", "").replace(",", "")
    elif "," in bruto and "." not in bruto:
        bruto = bruto.replace(",", ".")
    try:
        return Decimal(bruto)
    except InvalidOperation:
        return None


def _formato_numero(valor: Decimal) -> str:
    q = valor.quantize(Decimal("0.01"))
    entero, _, dec = f"{q:.2f}".partition(".")
    entero = f"{int(entero):,}".replace(",", ".")
    dec = dec.rstrip("0")
    return entero if not dec else f"{entero},{dec}"


def resolver_ajuste(pregunta: str, historial: list):
    """Resuelve un 'sin/quitar X' sobre el ultimo agregado verificado.

    Devuelve ``(texto, estado)`` o ``None`` si el mensaje no es un ajuste que
    podamos calcular sin volver a interpretar el negocio con un LLM.
    """
    monto = _monto_pedido(pregunta)
    if monto is None:
        return None
    estados = [t.get("estado") for t in historial or []
               if t.get("rol") == "assistant" and isinstance(t.get("estado"), dict)]
    agregado = next((e for e in reversed(estados)
                     if e.get("filas") and len(e["filas"]) == 1
                     and _indice(e.get("columnas", []), _PRESUPUESTO) is not None
                     and _indice(e.get("columnas", []), _GASTADO) is not None), None)
    if not agregado:
        return None

    # Cuando hay un detalle previo, verificamos que el monto identifique una
    # sola fila; nunca quitamos dos transacciones homonimas por accidente.
    coincidencias = []
    for estado in estados:
        columnas = estado.get("columnas", [])
        i_monto = _indice(columnas, _MONTOS_DETALLE)
        if i_monto is None or len(estado.get("filas", [])) <= 1:
            continue
        coincidencias.extend(
            fila for fila in estado["filas"]
            if _decimal(fila[i_monto]) == monto
        )
    if len(coincidencias) > 1:
        return (f"Encontré {len(coincidencias)} transacciones por "
                f"{_formato_numero(monto)}. Indique cuál desea excluir.", {})
    if estados and any(len(e.get("filas", [])) > 1 for e in estados) and not coincidencias:
        return ("No pude identificar una única transacción con ese monto en el "
                "detalle anterior. Indique el comercio o la fecha.", {})

    columnas = agregado["columnas"]
    fila = agregado["filas"][0]
    pre = _decimal(fila[_indice(columnas, _PRESUPUESTO)])
    gas = _decimal(fila[_indice(columnas, _GASTADO)])
    if pre is None or gas is None or pre == 0:
        return None
    ajustado = gas - monto
    disponible = pre - ajustado
    porcentaje = ajustado / pre * Decimal("100")
    concepto = (agregado.get("filtros") or {}).get("concepto", "Resultado")
    unidad = agregado.get("unidad", "")
    simbolo = "₡" if any(x in _normalizar(unidad) for x in ("colon", "crc")) else ""
    prefijo = lambda v: f"{simbolo}{_formato_numero(v)}"
    texto = (
        f"🧾 *{concepto} · ajuste verificado*\n\n"
        f"• Presupuesto: *{prefijo(pre)}*\n"
        f"• Gastado original: {prefijo(gas)}\n"
        f"• Transacción excluida: −{prefijo(monto)}\n"
        f"• Gastado ajustado: *{prefijo(ajustado)}*\n"
        f"• Disponible: *{prefijo(disponible)}*\n"
        f"• Ejecutado: *{_formato_numero(porcentaje)}%*"
    )
    nuevas_columnas = ["concepto", "presupuesto", "gastado_original",
                       "monto_excluido", "gastado", "disponible",
                       "porcentaje_consumido"]
    nueva_fila = [(concepto, pre, gas, monto, ajustado, disponible, porcentaje)]
    estado = crear_estado(pregunta, agregado.get("sql", ""), agregado.get("kpi", ""),
                          unidad, nuevas_columnas, nueva_fila, previo=agregado)
    return texto, estado
