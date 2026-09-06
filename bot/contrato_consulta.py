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
