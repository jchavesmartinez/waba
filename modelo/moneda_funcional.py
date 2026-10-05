"""Conversión histórica de importes a la moneda funcional del cliente.

La fuente original nunca se altera. La conversión se calcula mientras se
reconstruye el modelo semántico y queda acompañada por el importe/moneda de
origen, la tasa y su fecha. Así un reporte puede sumar una sola moneda sin
perder el dato auditable que llegó del banco o de la hoja.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

import httpx


_ISO_MONEDA = re.compile(r"[A-Z]{3}\Z")
_URL = "https://api.frankfurter.dev/v2/rates"
# El contrato de los movimientos canónicos y las operaciones de cuenta admite
# hasta seis decimales. Las tasas históricas pueden traer muchos más; dejar
# que esa precisión se propague convierte una compra válida en un error al
# proyectar saldos. Se conserva la tasa completa para auditoría y se redondea
# únicamente el importe monetario ya convertido.
_PRECISION_MONETARIA = Decimal("0.000001")


class ErrorTipoCambio(RuntimeError):
    """No fue posible obtener una tasa verificable para un movimiento."""


class ConversorMoneda:
    """Convierte a una moneda objetivo y memoriza cada par/día durante el build.

    Frankfurter publica tasas diarias de bancos centrales y permite consultar
    fechas históricas sin credencial. El cache evita una llamada por cada fila:
    un lote con cientos de compras USD del mismo día hace una sola consulta.
    """

    def __init__(self, moneda_funcional: str, cliente_http=None,
                 equivalencias: list[dict] | None = None):
        self.moneda_funcional = _validar_moneda(moneda_funcional)
        self._http = cliente_http or httpx.Client(timeout=10.0)
        self._tasas: dict[tuple[str, str], tuple[Decimal, str]] = {}
        self._equivalencias = _equivalencias(equivalencias or [])

    def convertir(self, monto, moneda_origen: str, fecha) -> dict:
        origen = _validar_moneda(moneda_origen)
        fecha_iso = _fecha_iso(fecha)
        importe = _decimal(monto)
        moneda_mercado, factor = self._equivalencias.get(
            origen, (origen, Decimal("1")))
        importe_mercado = importe * factor
        if moneda_mercado == self.moneda_funcional:
            return {
                "monto": _monto_monetario(importe_mercado),
                "moneda": self.moneda_funcional,
                "tasa": factor,
                "fecha_tasa": fecha_iso,
                "proveedor": "origen" if origen == moneda_mercado else "metadata",
            }
        tasa, fecha_tasa = self._tasa(moneda_mercado, fecha_iso)
        return {
            "monto": _monto_monetario(importe_mercado * tasa),
            "moneda": self.moneda_funcional,
            # Tasa efectiva por unidad de la moneda que venía en la fuente.
            "tasa": factor * tasa,
            "fecha_tasa": fecha_tasa,
            "proveedor": "frankfurter",
        }

    def _tasa(self, origen: str, fecha_iso: str) -> tuple[Decimal, str]:
        clave = (origen, fecha_iso)
        if clave in self._tasas:
            return self._tasas[clave]
        try:
            respuesta = self._http.get(
                _URL,
                params={"base": origen, "quotes": self.moneda_funcional, "date": fecha_iso},
            )
            respuesta.raise_for_status()
            datos = respuesta.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ErrorTipoCambio(
                f"no se pudo obtener el tipo de cambio {origen}/{self.moneda_funcional} del {fecha_iso}"
            ) from exc
        if not isinstance(datos, list) or not datos or not isinstance(datos[0], dict):
            raise ErrorTipoCambio(
                f"no hay tipo de cambio {origen}/{self.moneda_funcional} para {fecha_iso}"
            )
        fila = datos[0]
        try:
            tasa = _decimal(fila.get("rate"))
        except ErrorTipoCambio:
            raise ErrorTipoCambio(
                f"la tasa {origen}/{self.moneda_funcional} de {fecha_iso} no es válida"
            ) from None
        if tasa <= 0:
            raise ErrorTipoCambio(
                f"la tasa {origen}/{self.moneda_funcional} de {fecha_iso} no es positiva"
            )
        fecha_tasa = str(fila.get("date") or fecha_iso)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha_tasa):
            raise ErrorTipoCambio("el proveedor devolvió una fecha de tasa inválida")
        self._tasas[clave] = (tasa, fecha_tasa)
        return self._tasas[clave]


def _validar_moneda(valor: str) -> str:
    moneda = str(valor or "").strip().upper()
    if not _ISO_MONEDA.fullmatch(moneda):
        raise ErrorTipoCambio(f"moneda ISO inválida: '{valor}'")
    return moneda


def _fecha_iso(valor) -> str:
    if isinstance(valor, datetime):
        valor = valor.date()
    if isinstance(valor, date):
        return valor.isoformat()
    texto = str(valor or "").strip()[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", texto):
        raise ErrorTipoCambio(f"fecha inválida para tipo de cambio: '{valor}'")
    return texto


def _decimal(valor) -> Decimal:
    try:
        decimal = Decimal(str(valor))
    except (InvalidOperation, ValueError) as exc:
        raise ErrorTipoCambio(f"importe inválido para conversión: '{valor}'") from exc
    if not decimal.is_finite():
        raise ErrorTipoCambio(f"importe no finito para conversión: '{valor}'")
    return decimal


def _monto_monetario(valor: Decimal) -> Decimal:
    """Fija el importe canónico a la precisión que admite el libro mayor."""
    return valor.quantize(_PRECISION_MONETARIA, rounding=ROUND_HALF_UP)


def _equivalencias(filas: list[dict]) -> dict[str, tuple[str, Decimal]]:
    salida: dict[str, tuple[str, Decimal]] = {}
    for fila in filas:
        origen = _validar_moneda(fila.get("codigo_origen", ""))
        mercado = _validar_moneda(fila.get("codigo_mercado", ""))
        factor = _decimal(fila.get("factor_unidades", "1"))
        if factor <= 0:
            raise ErrorTipoCambio(
                f"factor_unidades debe ser positivo para equivalencia {origen}/{mercado}"
            )
        if origen in salida:
            raise ErrorTipoCambio(f"equivalencia de moneda duplicada: {origen}")
        salida[origen] = (mercado, factor)
    return salida
