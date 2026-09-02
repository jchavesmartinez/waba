"""Dashboard financiero multi-cliente servido mediante enlaces temporales.

La plantilla visual es unica. Los datos se obtienen exclusivamente de los KPI
habilitados en ``_kpis`` y se materializan con el mismo resolvedor/validador SQL
que usa el bot. El HTML recibe un snapshot embebido: una vez cargada la pagina,
el navegador no consulta Neon ni expone credenciales.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import threading
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import config
import registry
from bot import catalogo, kpis, nl2sql, seguimiento, warehouse_ro
from bot.tiempo import fecha_local

logger = logging.getLogger("fachavi.bot.dashboard")

ASSETS_DIR = Path(__file__).resolve().parent.parent / "dashboard"
_PLANTILLA = ASSETS_DIR / "index.html"
_SOLICITUD = re.compile(
    r"\b(?:dashboard|tablero|panel\s+(?:financiero|de\s+control)|"
    r"mis\s+indicadores|ver\s+(?:mis\s+)?kpis?)\b",
    re.IGNORECASE,
)
_CACHE: dict[tuple[str, str, str], tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()


class EnlaceInvalido(ValueError):
    """El enlace no es autentico, expiro o ya no corresponde al usuario."""


def habilitado() -> bool:
    return bool(
        config.BOT_DASHBOARD
        and config.APP_PUBLIC_URL
        and config.DASHBOARD_SECRET
    )


def es_solicitud(pregunta: str) -> bool:
    return bool(_SOLICITUD.search(str(pregunta or "")))


def _b64(datos: bytes) -> str:
    return base64.urlsafe_b64encode(datos).rstrip(b"=").decode("ascii")


def _desb64(texto: str) -> bytes:
    return base64.urlsafe_b64decode(texto + "=" * (-len(texto) % 4))


def _firma(payload: str) -> str:
    digest = hmac.new(
        config.DASHBOARD_SECRET.encode("utf-8"),
        payload.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64(digest)


def _periodo_actual() -> dict:
    hoy = fecha_local()
    fin = (
        date(hoy.year + 1, 1, 1)
        if hoy.month == 12
        else date(hoy.year, hoy.month + 1, 1)
    )
    return {
        "inicio": date(hoy.year, hoy.month, 1).isoformat(),
        "fin_exclusivo": fin.isoformat(),
        "granularidad": "mes",
    }


def periodo_solicitado(pregunta: str) -> dict:
    return seguimiento.periodo_explicito(pregunta) or _periodo_actual()


def _etiqueta_periodo(periodo: dict) -> str:
    meses = (
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    )
    try:
        inicio = date.fromisoformat(str(periodo["inicio"]))
        return f"{meses[inicio.month - 1]} {inicio.year}"
    except (KeyError, TypeError, ValueError):
        return "período seleccionado"


def crear_enlace(cliente: dict, numero: str, pregunta: str = "") -> tuple[str, str]:
    if not habilitado():
        raise RuntimeError("dashboard no configurado")
    periodo = periodo_solicitado(pregunta)
    ahora = int(time.time())
    payload = {
        "v": 1,
        "cid": str(cliente.get("cliente_id", "")),
        "num": "".join(c for c in str(numero) if c.isdigit()),
        "inicio": periodo["inicio"],
        "fin": periodo["fin_exclusivo"],
        "iat": ahora,
        "exp": ahora + int(config.DASHBOARD_TOKEN_TTL_MINUTOS) * 60,
        "nonce": secrets.token_urlsafe(10),
    }
    cuerpo = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    token = f"{cuerpo}.{_firma(cuerpo)}"
    return (
        f"{config.APP_PUBLIC_URL}/dashboard/{token}",
        _etiqueta_periodo(periodo),
    )


def validar_enlace(token: str, ahora: int | None = None) -> tuple[dict, dict]:
    try:
        cuerpo, firma = str(token).split(".", 1)
        if not hmac.compare_digest(_firma(cuerpo), firma):
            raise EnlaceInvalido("firma invalida")
        payload = json.loads(_desb64(cuerpo).decode("utf-8"))
    except EnlaceInvalido:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EnlaceInvalido("token malformado") from exc

    instante = int(time.time()) if ahora is None else int(ahora)
    if int(payload.get("exp", 0)) <= instante:
        raise EnlaceInvalido("enlace vencido")
    if payload.get("v") != 1 or not payload.get("cid") or not payload.get("num"):
        raise EnlaceInvalido("payload incompleto")

    cliente = registry.resolver(str(payload["num"]))
    if not cliente or str(cliente.get("cliente_id")) != str(payload["cid"]):
        raise EnlaceInvalido("usuario revocado o cliente distinto")
    return payload, cliente


def _serializable(valor):
    if isinstance(valor, Decimal):
        return float(valor)
    if isinstance(valor, (date, datetime)):
        return valor.isoformat()
    if isinstance(valor, bytes):
        return valor.decode("utf-8", errors="replace")
    return valor


def _orden_kpi(kpi: dict) -> tuple[int, str]:
    preferidos = (
        "presupuesto_disponible", "categorias_sobregiradas",
        "ejecucion_presupuesto_mes", "gasto_por_categoria",
        "gasto_por_comercio", "gasto_total",
    )
    nombre = str(kpi.get("kpi", "")).lower()
    try:
        return preferidos.index(nombre), nombre
    except ValueError:
        return len(preferidos), nombre


def _ejecutar_kpis(cliente: dict, periodo: dict) -> list[dict]:
    ctx = catalogo.construir_contexto(cliente)
    if ctx.error_lectura or not ctx.tablas_reales:
        raise RuntimeError("no se pudo cargar el catálogo del cliente")

    definiciones = sorted(kpis.cargar_kpis(cliente), key=_orden_kpi)
    resultados = []
    for definicion in definiciones[: int(config.DASHBOARD_MAX_KPIS)]:
        nombre = str(definicion.get("kpi", "")).strip()
        try:
            sql = kpis.sql_canonico(definicion, ctx)
            sql, _ = kpis.parametrizar_sql(sql, {}, periodo)
            if "{{" in sql or "}}" in sql:
                raise ValueError("el KPI requiere parámetros adicionales")
            ok, motivo = nl2sql.validar_sql(sql, ctx.tablas_reales)
            if not ok:
                raise ValueError(motivo)
            columnas, filas = warehouse_ro.ejecutar(
                cliente,
                sql,
                limite=int(config.DASHBOARD_MAX_FILAS_POR_KPI),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] KPI '%s' omitido del dashboard: %s",
                cliente.get("cliente_id"), nombre, exc,
            )
            continue
        resultados.append({
            "kpi": nombre,
            "nombre": definicion.get("nombre") or nombre.replace("_", " ").title(),
            "descripcion": definicion.get("descripcion", ""),
            "unidad": definicion.get("unidad", ""),
            "columnas": [str(c) for c in columnas],
            "filas": [[_serializable(v) for v in fila] for fila in filas],
        })
    return resultados


def generar_snapshot(cliente: dict, periodo: dict) -> dict:
    clave = (
        str(cliente.get("cliente_id", "")),
        str(periodo.get("inicio", "")),
        str(periodo.get("fin_exclusivo", "")),
    )
    ahora = time.time()
    ttl = max(0, int(config.DASHBOARD_CACHE_MINUTOS)) * 60
    with _CACHE_LOCK:
        guardado = _CACHE.get(clave)
        if guardado and ahora - guardado[0] <= ttl:
            return guardado[1]

    snapshot = {
        "cliente": {
            "id": str(cliente.get("cliente_id", "")),
            "nombre": str(cliente.get("nombre", "")) or "Dashboard financiero",
        },
        "periodo": {
            **periodo,
            "etiqueta": _etiqueta_periodo(periodo),
        },
        "actualizado_en": datetime.now(
            ZoneInfo(config.BOT_TIMEZONE)
        ).isoformat(timespec="minutes"),
        "kpis": _ejecutar_kpis(cliente, periodo),
    }
    with _CACHE_LOCK:
        _CACHE[clave] = (ahora, snapshot)
        if len(_CACHE) > 100:
            mas_antigua = min(_CACHE, key=lambda k: _CACHE[k][0])
            _CACHE.pop(mas_antigua, None)
    return snapshot


def renderizar(token: str) -> str:
    payload, cliente = validar_enlace(token)
    periodo = {
        "inicio": payload["inicio"],
        "fin_exclusivo": payload["fin"],
        "granularidad": "mes",
    }
    snapshot = generar_snapshot(cliente, periodo)
    plantilla = _PLANTILLA.read_text(encoding="utf-8")
    datos = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    # Evita cerrar el elemento script si un valor de negocio contiene </script>.
    datos = datos.replace("</", "<\\/")
    return plantilla.replace("__DASHBOARD_DATA__", datos)


def mensaje_enlace(cliente: dict, numero: str, pregunta: str = "") -> str:
    if not habilitado():
        return (
            "El dashboard todavía no está disponible. "
            "El administrador debe completar su configuración."
        )
    try:
        # Se materializa al pedirlo, antes de enviar el enlace. En el servicio
        # actual (una instancia) la apertura reutiliza este snapshot y no vuelve
        # a ejecutar los KPI durante la ventana de cache.
        generar_snapshot(cliente, periodo_solicitado(pregunta))
        url, periodo = crear_enlace(cliente, numero, pregunta)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[%s] dashboard no disponible al solicitar enlace: %s",
            cliente.get("cliente_id"), exc,
        )
        return (
            "No pude preparar el dashboard en este momento. "
            "Inténtelo nuevamente en unos minutos."
        )
    return (
        f"Aquí tiene su dashboard financiero de {periodo}:\n{url}\n\n"
        f"El enlace es personal y vence en "
        f"{config.DASHBOARD_TOKEN_TTL_MINUTOS} minutos."
    )
