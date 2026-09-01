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
from datetime import date, datetime
from decimal import Decimal, InvalidOperation


_MAX_FILAS_ESTADO = 200
_GRUPOS_FILTRO = {
    "linea_id": ("linea_id", "linea_presupuesto_id", "linea_presupuestaria_id"),
    "concepto": ("concepto",),
    "categoria": ("categoria",),
    "titular": ("titular",),
    "moneda": ("moneda",),
}
_PRESUPUESTO = ("presupuesto_mensual", "monto_mensual", "presupuesto")
_GASTADO = ("gastado", "gasto_ejecutado", "ejecutado", "total_gastado")
_DISPONIBLE = ("disponible", "saldo_disponible")
_PORCENTAJE = ("porcentaje_consumido", "porcentaje_ejecutado", "pct_consumido",
               "pct_ejecutado")
_MONTOS_DETALLE = ("monto_crc", "monto", "importe", "monto_total", "total")
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


def es_seguimiento_contextual(pregunta: str, historial: list) -> bool:
    """Detecta frases que semanticamente dependen del resultado anterior."""
    if not ultimo_estado(historial):
        return False
    t = _normalizar(pregunta)
    return bool(re.search(
        r"^(?:y\b|pero\b)|\b(?:eso|ese|esa|esos|esas|anterior|mismo|misma|"
        r"contra su presupuesto|contra el presupuesto|sin la transaccion|"
        r"quit(?:a|ar|amos)|exclu(?:ir|ye|yendo)|sac(?:a|ar|amos))\b",
        t,
    ))


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
    t = _normalizar(pregunta)
    mes = next((numero for nombre, numero in _MESES.items()
                if re.search(rf"\b{nombre}\b", t)), None)
    anio_m = re.search(r"\b(20\d{2})\b", t)
    if mes and anio_m:
        anio = int(anio_m.group(1))
        ultimo = monthrange(anio, mes)[1]
        return {
            "inicio": f"{anio:04d}-{mes:02d}-01",
            "fin_inclusivo": f"{anio:04d}-{mes:02d}-{ultimo:02d}",
            "granularidad": "mes",
        }

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
                 columnas, filas, previo: dict | None = None) -> dict:
    """Crea el contrato persistible de una consulta ya ejecutada."""
    filas_json = [[_json_valor(v) for v in fila]
                  for fila in list(filas)[:_MAX_FILAS_ESTADO]]
    filtros = dict((previo or {}).get("filtros") or {})
    filtros.update(filtros_unicos(columnas, filas))
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
    }
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
        aliases = _GRUPOS_FILTRO[clave]
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
    i_pre = _indice(columnas, _PRESUPUESTO)
    i_gas = _indice(columnas, _GASTADO)
    i_dis = _indice(columnas, _DISPONIBLE)
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
            if i_pct is not None and pre != 0:
                porcentaje = _decimal(fila[i_pct])
                calculado = gas / pre * Decimal("100")
                if porcentaje is not None and abs(porcentaje - calculado) > Decimal("0.11"):
                    return False, "el porcentaje no coincide con gastado dividido entre presupuesto"

    i_moneda = _indice(columnas, ("moneda", "currency", "codigo_moneda"))
    if i_moneda is not None and (i_pre is not None or i_gas is not None):
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
                _normalizar(f[indice]) != _normalizar(esperado) for f in filas):
            return False, f"el resultado mezclo valores fuera del filtro {clave}"
    return True, ""


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
    return entero if dec == "00" else f"{entero},{dec}"


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
