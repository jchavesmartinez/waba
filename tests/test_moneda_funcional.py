from datetime import date
from decimal import Decimal

import pytest

from modelo.moneda_funcional import ConversorMoneda, ErrorTipoCambio
from modelo.movimientos_canonicos import _proyectar


class _Respuesta:
    def raise_for_status(self):
        return None

    def json(self):
        return [{"date": "2026-09-05", "base": "USD", "quote": "CRC", "rate": 451.64}]


class _ClienteHttp:
    def __init__(self):
        self.llamadas = []

    def get(self, url, params):
        self.llamadas.append((url, params))
        return _Respuesta()


def test_convierte_historico_y_reutiliza_tasa_del_mismo_dia():
    cliente = _ClienteHttp()
    conversor = ConversorMoneda("CRC", cliente)

    primero = conversor.convertir(Decimal("10"), "USD", date(2026, 9, 5))
    segundo = conversor.convertir(Decimal("2"), "USD", date(2026, 9, 5))

    assert primero == {
        "monto": Decimal("4516.40"), "moneda": "CRC", "tasa": Decimal("451.64"),
        "fecha_tasa": "2026-09-05", "proveedor": "frankfurter",
    }
    assert segundo["monto"] == Decimal("903.28")
    assert len(cliente.llamadas) == 1
    assert cliente.llamadas[0][1] == {
        "base": "USD", "quotes": "CRC", "date": "2026-09-05",
    }


def test_moneda_que_ya_es_funcional_no_consulta_el_proveedor():
    cliente = _ClienteHttp()
    salida = ConversorMoneda("CRC", cliente).convertir(Decimal("1200"), "CRC", "2026-09-05")

    assert salida == {
        "monto": Decimal("1200"), "moneda": "CRC", "tasa": Decimal("1"),
        "fecha_tasa": "2026-09-05", "proveedor": "origen",
    }
    assert cliente.llamadas == []


def test_movimiento_canonico_conserva_origen_y_normaliza_el_contrato():
    class ConversorFalso:
        moneda_funcional = "CRC"

        def convertir(self, monto, moneda, fecha):
            # El extractor normaliza el tipo de entrada antes de llegar al
            # conversor; lo esencial es que no se pierdan importe, moneda ni
            # día al construir el contrato canónico.
            assert (str(monto), moneda, fecha.date().isoformat()) == ("10.0", "USD", "2026-09-05")
            return {
                "monto": Decimal("4516.40"), "moneda": "CRC", "tasa": Decimal("451.64"),
                "fecha_tasa": "2026-09-05", "proveedor": "frankfurter",
            }

    fila, motivo = _proyectar(
        {"fecha": "2026-09-05", "descripcion": "Servicio USD", "moneda": "USD", "monto": "10", "id": "x-1"},
        {"fuente": "banco", "fecha": "fecha", "descripcion": "descripcion", "moneda": "moneda", "monto": "monto", "clave": "id"},
        "movimientos", {}, ConversorFalso(),
    )

    assert motivo == ""
    assert fila["monto_original"] == Decimal("10")
    assert fila["moneda_original"] == "USD"
    assert fila["monto"] == Decimal("4516.40")
    assert fila["monto_neto"] == Decimal("4516.40")
    assert fila["moneda"] == fila["moneda_funcional"] == "CRC"
    assert fila["tipo_cambio"] == Decimal("451.64")


def test_error_de_tasa_detiene_el_build_en_vez_de_omitir_el_gasto_extranjero():
    class ConversorQueFalla:
        moneda_funcional = "CRC"

        def convertir(self, monto, moneda, fecha):
            raise ErrorTipoCambio("no hay tasa verificable")

    with pytest.raises(ErrorTipoCambio, match="no hay tasa verificable"):
        _proyectar(
            {"fecha": "2026-09-05", "descripcion": "Servicio USD", "moneda": "USD", "monto": "10", "id": "x-1"},
            {"fuente": "banco", "fecha": "fecha", "descripcion": "descripcion", "moneda": "moneda", "monto": "monto", "clave": "id"},
            "movimientos", {}, ConversorQueFalla(),
        )
