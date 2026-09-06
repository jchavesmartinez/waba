"""Contrato semántico universal entre conversación, metadata y ejecución.

El LLM puede proponer una intención, pero el resto del sistema opera con este
objeto pequeño y serializable. No contiene números ni SQL: ambos siguen siendo
resultado exclusivo del ejecutor de datos.
"""

from __future__ import annotations

import re
from copy import deepcopy


OPERACIONES = {"", "total", "conteo", "ranking", "detalle", "desglose", "comparacion"}
METRICAS = {"", "gastado", "presupuesto", "disponible", "exceso", "conteo"}
ENTIDADES = {"", "categoria", "concepto", "descripcion", "moneda", "transaccion"}
FILTROS = {"linea_id", "concepto", "categoria", "moneda", "descripcion"}
RELACIONES = {"nueva", "seguimiento", "modificacion", "ambigua"}

_ALIASES_ENTIDAD = {
    "comercio": "descripcion", "comercios": "descripcion",
    "registro": "transaccion", "registros": "transaccion",
}

# Estas pistas describen el idioma de la interfaz, no un modelo de negocio ni
# un cliente. Sirven para decidir *qué parte* del contrato cambia; los valores
# de dimensiones, las fórmulas y los datos continúan viniendo de metadata y
# del resultado ya verificado.
_PISTAS_CONTINUIDAD = re.compile(
    r"^(?:y\b|tambien\b)|\b(?:ese|esa|esos|esas|lo|la|los|las|ahi|"
    r"dentro|antes|despues|contando|incluyendo|suman|sumo|vario|"
    r"cambio|quedan|queda|falta|resta)\b"
)
_PREGUNTA_INCOMPLETA = re.compile(
    r"^\s*(?:[¿?¡!]+\s*)?(?:cual|cuanto|cuantos|que)\b.*\b(?:mayor|menor|"
    r"mas|menos|primero|ultimo|detalle|comercio|concepto|categoria)\b"
)


def _normalizar_texto(valor) -> str:
    """Normaliza texto humano para comparar un filtro sin inventarlo."""
    texto = str(valor or "").strip().lower()
    # Los valores de metadata pueden llevar tildes y el usuario no; para
    # confirmar que el valor fue escrito explícitamente basta una comparación
    # conservadora sin esos signos.
    import unicodedata
    texto = unicodedata.normalize("NFKD", texto)
    return " ".join(
        "".join(c for c in texto if not unicodedata.combining(c)).split()
    )


def es_seguimiento(pregunta: str, previo: dict | None,
                   propuesta: dict | None = None) -> bool:
    """Decide si hay un contrato previo que deba conservarse.

    Una pregunta autosuficiente no hereda por accidente solo porque existe
    historial. La relación propuesta por el intérprete LLM es una señal
    adicional, nunca la fuente de los filtros heredados.
    """
    if not previo:
        return False
    relacion = _texto((propuesta or {}).get("relacion"))
    texto = _normalizar_texto(pregunta)
    return relacion in {"seguimiento", "modificacion"} or bool(
        _PISTAS_CONTINUIDAD.search(texto) or _PREGUNTA_INCOMPLETA.search(texto)
    )


def _cambio_explicito_de_metrica(texto: str, propuesta: dict) -> str:
    """Reconoce una métrica pedida expresamente por el usuario.

    La prioridad es deliberadamente la frase actual: un plan que llegue con
    ``gastado`` no puede ignorar «cuánto queda» y conservar la métrica previa.
    """
    if re.search(r"\b(?:disponible|queda[n]?|falta[n]?|resta[n]?|sobran|sobrante(?:s)?)\b", texto):
        return "disponible"
    if re.search(r"\b(?:exceso|exced|sobregir)\b", texto):
        return "exceso"
    # «gasto por categoría del presupuesto» no pide el presupuesto: el verbo
    # de gasto es más específico que el nombre de la tabla.
    if re.search(r"\b(?:gaste|gastado|gasto|suman|sumo|total)\b", texto):
        return "gastado"
    if re.search(r"\b(?:presupuesto|presupuestado|planeado)\b", texto):
        return "presupuesto"
    propuesta_metrica = _texto((propuesta or {}).get("metrica"))
    return propuesta_metrica if propuesta_metrica in METRICAS else ""


def _cambio_explicito_de_operacion(texto: str, propuesta: dict) -> str:
    if re.search(r"\b(?:cuant[oa]s|cantidad|numero)\b", texto):
        return "conteo"
    # «ese total» puede ser una referencia al resultado previo; si el usuario
    # pide las transacciones que lo forman, la intención es detalle y no otro
    # SUM. Las acciones de detalle tienen prioridad sobre esa palabra.
    if re.search(r"\b(?:detalle|transacciones?|movimientos?|compras?).*\b(?:forman|componen|conforman|lista|cuales)\b", texto):
        return "detalle"
    if re.search(r"\b(?:cuanto suman|cuanto suma|totalizan|suman|sumo|total)\b|\bcuanto\b.*\b(?:gaste|gastado|gasto)\b", texto):
        return "total"
    if re.search(r"\b(?:mayor|menor|top|mas alto|mas alta|mas caro|mas cara|menos)\b", texto):
        return "ranking"
    propuesta_operacion = _texto((propuesta or {}).get("operacion"))
    return propuesta_operacion if propuesta_operacion in OPERACIONES else ""


def _cambio_explicito_de_entidad(texto: str, propuesta: dict) -> str:
    for entidad, patron in (
        ("concepto", r"\bconceptos?\b|\b(?:lineas?|cosas?)\s+del\s+presupuesto\b"),
        ("categoria", r"\bcategorias?\b"),
        ("descripcion", r"\b(?:comercios?|descripciones?)\b"),
        ("moneda", r"\bmonedas?\b"),
        ("transaccion", r"\b(?:transacciones?|movimientos?|compras?)\b"),
    ):
        if re.search(patron, texto):
            return entidad
    entidad = _ALIASES_ENTIDAD.get(_texto((propuesta or {}).get("entidad")),
                                    _texto((propuesta or {}).get("entidad")))
    return entidad if entidad in ENTIDADES else ""


def aplicar_delta(pregunta: str, previo: dict | None,
                  propuesta: dict | None = None,
                  *, periodo: dict | None = None,
                  filtros_adicionales: dict | None = None) -> dict:
    """Aplica un delta seguro a un contrato semántico verificado.

    El retorno declara sólo los campos que cambiaron y entrega el contrato
    actualizado. Un filtro sólo se modifica si su nuevo valor aparece en el
    mensaje actual (o fue recuperado de filas verificadas y pasado como
    ``filtros_adicionales``). Así Gemini puede proponer una intención, pero no
    puede borrar ni sustituir filtros silenciosamente.
    """
    propuesta = propuesta or {}
    anterior = dict((previo or {}).get("contrato") or previo or {})
    seguimiento = es_seguimiento(pregunta, anterior, propuesta)
    if not seguimiento:
        texto = _normalizar_texto(pregunta)
        # Incluso en una pregunta nueva la intención explícita del usuario es
        # más confiable que una etiqueta vacía o incompatible del plan. Esto
        # no extrae valores de negocio: sólo normaliza la operación, métrica y
        # dimensión del contrato universal.
        operacion = _cambio_explicito_de_operacion(texto, propuesta)
        metrica = _cambio_explicito_de_metrica(texto, propuesta)
        entidad = _cambio_explicito_de_entidad(texto, propuesta)
        contrato = crear({
            "operacion": operacion or propuesta.get("operacion"),
            "metrica": metrica or propuesta.get("metrica"),
            "entidad": entidad or propuesta.get("entidad"),
            "filtros": propuesta.get("filtros_actuales") or {},
            "periodo": periodo or {}, "relacion": "nueva",
        })
        return {"es_seguimiento": False, "contrato": contrato, "cambios": {}}

    texto = _normalizar_texto(pregunta)
    base = crear({
        "operacion": anterior.get("operacion"),
        "metrica": anterior.get("metrica"),
        "entidad": anterior.get("entidad"),
        "filtros": anterior.get("filtros") or {},
        "periodo": anterior.get("periodo") or {}, "relacion": "seguimiento",
    }, previo=anterior)
    cambios = {}
    operacion = _cambio_explicito_de_operacion(texto, propuesta)
    metrica = _cambio_explicito_de_metrica(texto, propuesta)
    entidad = _cambio_explicito_de_entidad(texto, propuesta)
    if operacion and operacion != base["operacion"]:
        cambios["operacion"] = operacion
    if metrica and metrica != base["metrica"]:
        cambios["metrica"] = metrica
    if entidad and entidad != base["entidad"]:
        cambios["entidad"] = entidad
    # El detalle de movimientos explica un agregado anterior: usa el monto de
    # cada fila, no la métrica comparativa (exceso/disponible) del resumen.
    if operacion == "detalle" and base["metrica"] != "gastado":
        cambios["metrica"] = "gastado"
    if periodo:
        cambios["periodo"] = dict(periodo)

    candidatos = dict(propuesta.get("filtros_actuales") or {})
    candidatos.update(filtros_adicionales or {})
    filtros = dict(base["filtros"])
    ambiguedad = ""
    # «sumar X» cambia el universo, pero sin una dimensión reconocida no es
    # seguro decidir si X sustituye o acompaña al filtro anterior. En cambio,
    # «sumar esas» se resuelve sobre las filas ya verificadas y no entra aquí.
    if (re.search(r"\b(?:sumo|sumar)\b", texto)
            and not re.search(r"\b(?:eso|esa|ese|esas|esos)\b", texto)
            and filtros and not candidatos):
        ambiguedad = (
            "¿Quiere combinar el resultado anterior con otro filtro, o "
            "sustituirlo? Indique también la dimensión de ese valor."
        )
    for clave, valor in candidatos.items():
        if ambiguedad:
            break
        clave_n = _texto(clave)
        valor_s = str(valor or "").strip()
        if clave_n not in FILTROS or not valor_s:
            continue
        # El valor debe ser textual en este turno, salvo el que se obtuvo de
        # una fila mostrada y verificada. Esto bloquea filtros alucinados.
        recuperado = clave in (filtros_adicionales or {})
        if not recuperado and _normalizar_texto(valor_s) not in texto:
            continue
        anterior_valor = filtros.get(clave_n)
        if anterior_valor and _normalizar_texto(anterior_valor) != _normalizar_texto(valor_s):
            # Dos valores para la misma dimensión con una conjunción pueden
            # significar reemplazo o suma. El contrato canónico no elige por
            # el usuario: solicita una confirmación explícita.
            if re.search(r"\b(?:sumo|sumar|junto|incluy|ademas|tambien| y )\b", texto):
                ambiguedad = (
                    f"¿Quiere combinar {clave_n} «{anterior_valor}» con "
                    f"«{valor_s}», o sustituirlo?"
                )
                break
            filtros[clave_n] = valor_s
            cambios.setdefault("filtros", {})[clave_n] = valor_s
        elif not anterior_valor:
            filtros[clave_n] = valor_s
            cambios.setdefault("filtros", {})[clave_n] = valor_s

    datos = {
        "operacion": cambios.get("operacion", base["operacion"]),
        "metrica": cambios.get("metrica", base["metrica"]),
        "entidad": cambios.get("entidad", base["entidad"]),
        "filtros": filtros,
        "periodo": cambios.get("periodo", base["periodo"]),
        "relacion": "ambigua" if ambiguedad else "seguimiento",
    }
    contrato = crear(datos, previo=anterior)
    return {
        "es_seguimiento": True,
        "contrato": contrato,
        "cambios": cambios,
        "ambiguedad": ambiguedad,
    }


def _texto(valor) -> str:
    return str(valor or "").strip().lower()


def _periodo(valor) -> dict:
    """Normaliza periodo sin adivinar fechas ni aplicar defaults."""
    if isinstance(valor, dict):
        return {k: str(v) for k, v in valor.items() if v not in (None, "")}
    texto = str(valor or "").strip()
    if re.fullmatch(r"20\d{2}-\d{2}", texto):
        return {"mes": texto}
    return {}


def crear(datos: dict | None = None, previo: dict | None = None) -> dict:
    """Devuelve la representación canónica y segura de una intención.

    ``previo`` se usa exclusivamente si la relación es seguimiento o
    modificación. Para una consulta nueva nunca se filtran datos heredados.
    """
    datos = datos or {}
    relacion = _texto(datos.get("relacion") or "nueva")
    if relacion not in RELACIONES:
        relacion = "nueva"
    entidad = _ALIASES_ENTIDAD.get(_texto(datos.get("entidad")), _texto(datos.get("entidad")))
    operacion = _texto(datos.get("operacion"))
    metrica = _texto(datos.get("metrica"))
    filtros = {
        _texto(k): str(v).strip()
        for k, v in (datos.get("filtros") or datos.get("filtros_actuales") or {}).items()
        if _texto(k) in FILTROS and v not in (None, "")
    }
    anterior = (previo or {}).get("contrato") or previo or {}
    if relacion in {"seguimiento", "modificacion"}:
        heredados = dict(anterior.get("filtros") or {})
        heredados.update(filtros)
        filtros = heredados
        periodo = _periodo(datos.get("periodo")) or _periodo(anterior.get("periodo"))
    else:
        periodo = _periodo(datos.get("periodo"))
    return {
        "operacion": operacion if operacion in OPERACIONES else "",
        "metrica": metrica if metrica in METRICAS else "",
        "entidad": entidad if entidad in ENTIDADES else "",
        "filtros": filtros,
        "periodo": periodo,
        "relacion": relacion,
    }


def es_valido(contrato: dict | None) -> tuple[bool, str]:
    """Comprueba forma y evita contratos que parezcan completos sin serlo."""
    contrato = contrato or {}
    for clave in ("operacion", "metrica", "entidad", "filtros", "periodo", "relacion"):
        if clave not in contrato:
            return False, f"falta {clave} en el contrato"
    if not isinstance(contrato["filtros"], dict) or not isinstance(contrato["periodo"], dict):
        return False, "filtros y periodo deben ser objetos"
    if contrato["operacion"] in {"ranking", "desglose"} and not contrato["entidad"]:
        return False, "falta la entidad para una consulta agrupada"
    return True, ""


def copiar(contrato: dict | None) -> dict:
    """Copia defensiva para persistir estado sin referencias mutables."""
    return deepcopy(contrato or {})
