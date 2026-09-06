"""Validación metadata-driven de las capacidades que el bot puede ejecutar.

Una consulta no declarada no puede degradarse silenciosamente a un total de
gastos o presupuesto. Las capacidades viven en la metadata de KPI: operaciones,
periodicidades, aliases, dimensiones y métricas.
"""

from __future__ import annotations

import re
import unicodedata


_FORMA = {
    "como", "cual", "cuales", "cuanto", "cuantos", "cuantas", "dame",
    "decime", "dime", "donde", "este", "esta", "estos", "estas", "para",
    "quiero", "tengo", "total", "todos", "todas", "mostrar", "muestra",
    "necesito", "mes", "dato", "datos", "puedo", "puedes", "por", "desde",
    "hasta", "entre", "sobre", "cada", "otro", "otra", "actual", "anterior",
    "ultimo", "ultima", "primero", "primera", "mayor", "menor", "mas",
    "menos", "mejor", "peor", "real", "meses", "semana", "semanas", "ano",
    "anos", "dia", "dias",
    # Referencias de tiempo son parte de la gramática, no una dimensión de
    # negocio. Si no se excluyen, una pregunta como "flujo de caja en agosto"
    # puede coincidir accidentalmente con el mes que aparece en un ejemplo de
    # KPI y pasar la barrera de capacidades.
    "hoy", "ayer", "manana", "enero", "febrero", "marzo", "abril",
    "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
    "noviembre", "diciembre",
}
_FORMA_RAICES = set()


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto or ""))
    return texto.encode("ascii", "ignore").decode("ascii").lower()


def _raiz(palabra: str) -> str:
    """Reduce variantes sencillas: gasto/gasté, compras/comprar, etc."""
    for sufijo in (
        "amientos", "imiento", "aciones", "acion", "idades", "idad", "mente",
        "ados", "adas", "idos", "idas", "ando", "iendo", "aron", "ieron",
        "amos", "emos", "ado", "ido", "ar", "er", "ir", "es", "os", "as",
        "e", "o", "a",
    ):
        if len(palabra) - len(sufijo) >= 3 and palabra.endswith(sufijo):
            return palabra[:-len(sufijo)]
    return palabra


def _es_forma(token: str) -> bool:
    global _FORMA_RAICES
    if not _FORMA_RAICES:
        _FORMA_RAICES = {_raiz(palabra) for palabra in _FORMA}
    return token in _FORMA or token in _FORMA_RAICES


def tokens(texto: str) -> set[str]:
    salida = set()
    for palabra in re.findall(r"[a-z0-9_]+", _normalizar(texto)):
        if len(palabra) < 4:
            continue
        salida.add(palabra)
        salida.add(_raiz(palabra))
    return salida


def valores(kpi: dict, *campos: str) -> set[str]:
    salida = set()
    for campo in campos:
        for valor in re.split(r"[;,\|\n]", str((kpi or {}).get(campo, "") or "")):
            valor = _normalizar(valor).strip()
            if valor:
                salida.add(valor)
    return salida


def es_estructurada(kpi: dict | None) -> bool:
    """Un KPI nuevo opta por el contrato al declarar al menos un campo."""
    return bool(valores(
        kpi or {}, "operaciones", "operaciones_permitidas", "metricas",
        "periodicidades", "aliases",
    ))


def vocabulario(kpi: dict) -> set[str]:
    """Palabras de negocio permitidas, tomadas exclusivamente de metadata."""
    texto = " ".join(str(kpi.get(campo, "") or "") for campo in (
        "kpi", "nombre", "preguntas_ejemplo", "aliases", "metricas",
        "dimensiones", "tabla",
    ))
    return tokens(texto)


def periodicidad(pregunta: str) -> str:
    texto = _normalizar(pregunta)
    for nombre, patron in (
        ("anual", r"\b(?:anual|ano|anos)\b"),
        ("trimestral", r"\b(?:trimestre|trimestral)\b"),
        ("semanal", r"\b(?:semanal|semana)\b"),
        ("quincenal", r"\b(?:quincena|quincenal)\b"),
        ("diario", r"\b(?:diario|diariamente|por dia)\b"),
        ("mensual", r"\b(?:mensual|mes)\b"),
    ):
        if re.search(patron, texto):
            return nombre
    return ""


def operacion_especial(pregunta: str) -> str:
    """Operaciones transversales que sólo existen si la metadata las declara."""
    texto = _normalizar(pregunta)
    for nombre, patron in (
        (
            "proyeccion",
            r"\b(?:proyect|estim|pronostic|preve|futuro|voy\s+a\s+"
            r"(?:gast|paga|recib)|al\s+cierre\s+de|terminar(?:e|a)?\s+gast)",
        ),
        ("recomendacion", r"\b(?:recomiend|deberi[ao]|recortar|ahorrar)"),
        ("conversion", r"\b(?:convierte|convertir|conversion|tipo de cambio)"),
        ("anomalia", r"\b(?:anomal|inusual|atipic|fuera de lo normal)"),
        ("programacion", r"\b(?:proxim[oa]|vencimiento|agenda[rd]?)"),
        # "clasificación" también aparece como valor legítimo de la
        # dimensión estado (por ejemplo, "sin clasificar"). La validación de
        # dimensiones/KPI decide si una consulta de clasificación es posible;
        # tratar la palabra como operación transversal bloquearía consultas
        # válidas sobre ese valor.
        # No se puede inferir un flujo de caja a partir de movimientos de
        # gasto: requiere ingresos, saldos iniciales y reglas explícitas. Sólo
        # se habilita cuando la metadata declara esta operación.
        ("flujo_caja", r"\b(?:flujo\s+de\s+caja|cash\s+flow)\b"),
        # Un gasto de la categoría "Impuestos" no equivale a IVA declarado,
        # deducciones o una liquidación fiscal. Esas métricas requieren datos
        # tributarios que deben estar declarados por el cliente.
        ("impuesto_fiscal", r"\b(?:iva|deducibles?|declaracion\s+fiscal)\b"),
    ):
        if re.search(patron, texto):
            return nombre
    return ""


def dimension_derivada(pregunta: str) -> str:
    """Dimensiones calculadas que requieren una declaración explícita."""
    texto = _normalizar(pregunta)
    if re.search(r"\bdia\s+de\s+la\s+semana\b", texto):
        return "dia_semana"
    if re.search(r"\b(?:hora|franja horaria)\b", texto):
        return "hora"
    return ""


def validar(pregunta: str, plan: dict | None, kpis: list, ctx,
            *, es_seguimiento: bool = False) -> tuple[bool, str]:
    """Comprueba que una pregunta nueva tenga capacidad explícita.

    La migración es compatible: sólo se cierra el paso cuando el cliente ya
    declaró capacidades estructuradas. Un seguimiento conserva su contrato
    verificado y no se vuelve a interpretar aquí.
    """
    plan = plan or {}
    estructurados = [k for k in (kpis or []) if es_estructurada(k)]
    if not estructurados or es_seguimiento:
        return True, ""

    elegido = next((
        k for k in estructurados
        if _normalizar(k.get("kpi", "")) == _normalizar(plan.get("kpi", ""))
    ), None)
    # Un KPI heredado puede coexistir con el catálogo estructurado durante la
    # migración. No puede convertirse en una escapatoria: si el plan eligió
    # uno sin contrato propio, validamos contra las capacidades estructuradas
    # del mismo cliente. Para un cliente íntegramente legado se conserva la
    # salida temprana de arriba y no se cambia su comportamiento.
    candidatos = [elegido] if elegido and es_estructurada(elegido) else estructurados

    especial = operacion_especial(pregunta)
    if especial and not any(
        especial in valores(kpi, "operaciones", "operaciones_permitidas")
        for kpi in candidatos if kpi
    ):
        return False, "La operación «%s» no está declarada en los datos disponibles." % especial

    derivada = dimension_derivada(pregunta)
    if derivada and not any(
        derivada in valores(kpi, "dimensiones", "dimensiones_derivadas")
        for kpi in candidatos if kpi
    ):
        return False, "La dimensión «%s» no está declarada para esta consulta." % derivada

    periodo = periodicidad(pregunta)
    declaradas = set().union(*(
        valores(kpi, "periodicidades") for kpi in candidatos if kpi
    )) if candidatos else set()
    if periodo and declaradas and periodo not in declaradas:
        return False, "La periodicidad «%s» no está declarada para esta consulta." % periodo

    anclas = {
        token for token in tokens(pregunta)
        if not _es_forma(token) and not token.isdigit()
    }
    if not anclas:
        return True, ""
    permitidas_globales = set().union(*(vocabulario(kpi) for kpi in estructurados))
    permitidas_globales |= tokens(str(getattr(ctx, "schema_text", "") or ""))
    if elegido:
        permitidas = vocabulario(elegido)
        # Gemini puede escoger un KPI vecino (por ejemplo, gasto por categoría
        # para una pregunta de presupuesto). La metadata del cliente sigue
        # siendo la autoridad: si otro KPI estructurado declara los términos,
        # no convertimos esa elección imperfecta en un falso "no soportado".
        if not (anclas & permitidas):
            permitidas = permitidas_globales
    else:
        permitidas = permitidas_globales
    if anclas & permitidas:
        return True, ""
    return False, (
        "No hay una métrica o dimensión declarada para «%s» en los datos disponibles."
        % ", ".join(sorted(anclas)[:3])
    )
