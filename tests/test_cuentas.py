"""Contrato de saldos: corte, cargos, ingresos y movimientos de cuenta."""

from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bot import cuentas
from bot import app as app_mod


def _cuenta(cuenta_id, tipo, saldo_crc, *, ultimos4="", saldo_usd="0", corte_en=None):
    return {
        "cuenta_id": cuenta_id, "nombre": cuenta_id, "tipo": tipo,
        "ultimos4": ultimos4, "moneda": "CRC,USD" if tipo == "credito" else "CRC",
        "saldo_inicial_crc": Decimal(str(saldo_crc)),
        "saldo_inicial_usd": Decimal(str(saldo_usd)),
        "fecha_corte": date(2026, 9, 29), "cuenta_pago_default": "bac_salario",
        "corte_en": corte_en,
    }


CUENTAS = [
    _cuenta("bac_salario", "banco", "517754.95", ultimos4="8774"),
    _cuenta("amex_8715", "credito", "161681.30", ultimos4="8715", saldo_usd="284.50"),
    _cuenta("mismart", "ahorro", "6798159.75"),
]


def _movimiento(clave, medio, monto, moneda="CRC", original=None, tipo="GASTO",
                fecha="2026-09-30"):
    return {
        "_clave": clave, "fecha": fecha, "descripcion": clave,
        "medio_pago": medio, "monto_neto": Decimal(str(monto)),
        "monto_original": Decimal(str(original if original is not None else monto)),
        "moneda_original": moneda, "tipo_movimiento": tipo,
    }


def _operacion(identificador, tipo, origen, destino, monto_origen, monto_destino,
               moneda_destino="CRC", regla_id=""):
    return {
        "operacion_id": identificador, "tipo": tipo, "fecha": date(2026, 9, 30),
        "descripcion": identificador, "cuenta_origen": origen, "cuenta_destino": destino,
        "monto_origen": Decimal(str(monto_origen)), "moneda_origen": "CRC",
        "monto_destino": Decimal(str(monto_destino)),
        "moneda_destino": moneda_destino, "regla_id": regla_id,
    }


def _por_id(resultado):
    return {c["cuenta_id"]: c for c in resultado["cuentas"]}


def test_cargos_usan_debito_y_deuda_en_moneda_original():
    cargos = [
        _movimiento("super", "************8774", "2500"),
        _movimiento("compra-usd", "AMEX ****-8715", "5200", "USD", "10"),
    ]
    resultado = _por_id(cuentas.proyectar_saldos(CUENTAS, cargos, [], date(2026, 9, 30)))
    assert resultado["bac_salario"]["saldo_crc"] == "515254.95"
    assert resultado["amex_8715"]["saldo_crc"] == "161681.30"
    assert resultado["amex_8715"]["saldo_usd"] == "294.50"
    assert resultado["mismart"]["saldo_crc"] == "6798159.75"


def test_resolver_medio_de_cuenta_acepta_id_y_ultimos4(monkeypatch):
    monkeypatch.setattr(cuentas, "_leer_config", lambda _: (CUENTAS, [], []))
    cliente = {"cliente_id": "cliente_a"}

    assert cuentas.resolver_medio_de_cuenta(cliente, "cuenta:bac_salario") == "cuenta:bac_salario"
    assert cuentas.resolver_medio_de_cuenta(cliente, "8774") == "cuenta:bac_salario"
    assert cuentas.resolver_medio_de_cuenta(cliente, "cuenta:ajena") is None


def test_lee_cuentas_desde_hoja_tarjetas_sin_requerir_numero_para_ahorro(monkeypatch):
    class Hoja:
        def get_all_records(self):
            return [
                {"cuenta_id": "bac_salario", "nombre_cuenta": "CR - Salario Walmart",
                 "tipo_cuenta": "banco", "ultimos4": 8774,
                 "saldo_corte_crc": 517754.95, "fecha_corte": "2026-09-29"},
                {"cuenta_id": "mismart", "nombre_cuenta": "Mi Smart",
                 "tipo_cuenta": "ahorro", "ultimos4": "",
                 "saldo_corte_crc": 6798159.75, "fecha_corte": "2026-09-29"},
            ]

    class Libro:
        def worksheet(self, nombre):
            assert nombre == "tarjetas"
            return Hoja()

    monkeypatch.setattr(cuentas, "abrir_libro", lambda _: Libro())
    monkeypatch.setattr(cuentas, "fecha_local", lambda: date(2026, 9, 29))
    cliente = {"fuentes": [{"tipo": "google_sheets", "config": {
        "spreadsheet_id": "finanzas", "hojas": ["tarjetas"]}}]}
    semillas = cuentas._semillas_hoja(cliente)
    assert [(s["cuenta_id"], s["ultimos4"]) for s in semillas] == [
        ("bac_salario", "8774"), ("mismart", ""),
    ]


def test_canonicos_usa_columnas_fisicas_no_catalogo_publico(monkeypatch):
    cliente = {"cliente_id": "a"}
    tabla = SimpleNamespace(tabla_real="movimientos", columnas_config={"fecha": {}})
    monkeypatch.setattr(cuentas.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(cuentas.dashboard, "tabla_movimientos_canonicos", lambda *_: tabla)
    monkeypatch.setattr(cuentas.warehouse_ro, "listar_columnas", lambda *_: {
        "movimientos": [(n, "text") for n in (
            "_clave", "fecha", "descripcion", "medio_pago", "monto_neto",
            "monto_original", "moneda_original", "tipo_movimiento")],
    })
    consultas = []
    monkeypatch.setattr(cuentas.warehouse_ro, "leer_interno", lambda _, sql, params: (
        consultas.append((sql, params)) or []))

    assert cuentas._canonicos(cliente, date(2026, 9, 29), date(2026, 9, 30)) == []
    assert "monto_original, moneda_original" in consultas[0][0]
    assert consultas[0][1]["corte"] == date(2026, 9, 29)


def test_canonicos_anteriores_sin_moneda_original_degradan_con_advertencia(monkeypatch):
    cliente = {"cliente_id": "a"}
    tabla = SimpleNamespace(tabla_real="movimientos", columnas_config={})
    monkeypatch.setattr(cuentas.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(cuentas.dashboard, "tabla_movimientos_canonicos", lambda *_: tabla)
    monkeypatch.setattr(cuentas.warehouse_ro, "listar_columnas", lambda *_: {
        "movimientos": [(n, "text") for n in (
            "_clave", "fecha", "medio_pago", "monto_neto", "moneda")],
    })
    consultas = []
    monkeypatch.setattr(cuentas.warehouse_ro, "leer_interno", lambda _, sql, params: (
        consultas.append(sql) or [_movimiento("cargo", "8715", "5000") | {
            "monto_original": None, "moneda_original": "CRC", "moneda_estimada": True}]))

    filas = cuentas._canonicos(cliente, date(2026, 9, 29), date(2026, 9, 30))
    resultado = cuentas.proyectar_saldos(CUENTAS, filas, [], date(2026, 9, 30))
    assert "NULL::numeric AS monto_original" in consultas[0]
    assert resultado["advertencias"]


def test_deuda_usd_sin_monto_original_no_inventa_importe():
    cargo = _movimiento("usd", "8715", "5200", "USD", "10")
    cargo["monto_original"] = None
    resultado = cuentas.proyectar_saldos(CUENTAS, [cargo], [], date(2026, 9, 30))
    assert _por_id(resultado)["amex_8715"]["saldo_usd"] == "284.50"
    assert len(resultado["sin_conversion"]) == 1


def test_pago_tarjeta_y_transferencia_no_duplican_egreso():
    operaciones = [
        _operacion("pago", "pago_tarjeta", "bac_salario", "amex_8715", "5000", "10", "USD"),
        _operacion("ahorro", "transferencia", "bac_salario", "mismart", "2000", "2000"),
    ]
    resultado = _por_id(cuentas.proyectar_saldos(CUENTAS, [], operaciones, date(2026, 9, 30)))
    assert resultado["bac_salario"]["saldo_crc"] == "510754.95"
    assert resultado["amex_8715"]["saldo_usd"] == "274.50"
    assert resultado["mismart"]["saldo_crc"] == "6800159.75"


def test_pago_tarjeta_importado_y_registrado_no_rebaja_dos_veces():
    pago = _operacion("pago", "pago_tarjeta", "bac_salario", "amex_8715", "5000", "5000")
    cargo_banco = _movimiento("PAGO TARJETA AMEX", "8774", "5000")
    resultado = _por_id(cuentas.proyectar_saldos(
        CUENTAS, [cargo_banco], [pago], date(2026, 9, 30)))
    assert resultado["bac_salario"]["saldo_crc"] == "512754.95"
    assert resultado["amex_8715"]["saldo_crc"] == "156681.30"


def test_ingreso_importado_concilia_registro_recurrente():
    ingreso = _movimiento("salario", "8774", "100000", tipo="INGRESO")
    recurrencia = _operacion("recurrente:uno:2026-09-30", "ingreso", "",
                            "bac_salario", "0", "100000", regla_id="uno")
    resultado = _por_id(cuentas.proyectar_saldos(
        CUENTAS, [ingreso], [recurrencia], date(2026, 9, 30)))
    assert resultado["bac_salario"]["saldo_crc"] == "617754.95"
    assert [m["id"] for m in resultado["bac_salario"]["movimientos"]] == ["salario"]


def test_corte_excluye_gastos_previos_y_efecto_posterior_de_edicion():
    cargos = [
        _movimiento("antes", "8774", "9000", fecha="2026-09-29"),
        _movimiento("despues", "8774", "0"),
    ]
    resultado = _por_id(cuentas.proyectar_saldos(CUENTAS, cargos, [], date(2026, 9, 30)))
    assert resultado["bac_salario"]["saldo_crc"] == "517754.95"
    assert [m["id"] for m in resultado["bac_salario"]["movimientos"]] == ["despues"]


def test_corte_con_hora_incluye_solo_los_gastos_posteriores_del_mismo_dia():
    cuentas_con_hora = [
        _cuenta("bac_salario", "banco", "517754.95", ultimos4="8774",
                 corte_en=datetime(2026, 9, 29, 18, 40)),
    ]
    cargos = [
        _movimiento("antes", "************8774", "1000", fecha="2026-09-29T18:39:00"),
        _movimiento("despues", "************8774", "83270", fecha="2026-09-29T19:08:00"),
        _movimiento("despues-2", "************8774", "31240", fecha="2026-09-29T20:46:00"),
    ]
    resultado = _por_id(cuentas.proyectar_saldos(
        cuentas_con_hora, cargos, [], date(2026, 9, 29)))
    assert resultado["bac_salario"]["saldo_crc"] == "403244.95"
    assert [m["id"] for m in resultado["bac_salario"]["movimientos"]] == [
        "despues-2", "despues",
    ]
    assert resultado["bac_salario"]["corte_en"] == "2026-09-29T18:40"


def test_obtener_filtra_el_detalle_al_mes_solicitado(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    monkeypatch.setattr(cuentas, "_leer_config", lambda _: (CUENTAS, [], []))
    monkeypatch.setattr(cuentas, "_materializar_ingresos", lambda *_: None)
    monkeypatch.setattr(cuentas, "fecha_local", lambda: date(2026, 10, 31))
    monkeypatch.setattr(cuentas, "_aplicar_pendientes", lambda _, movimientos: movimientos)
    monkeypatch.setattr(cuentas, "_canonicos", lambda *_: [
        _movimiento("septiembre", "8774", "100", fecha="2026-09-30"),
        _movimiento("octubre", "8774", "200", fecha="2026-10-01"),
    ])

    resultado = cuentas.obtener(cliente, date(2026, 10, 31), desde=date(2026, 10, 1))

    assert [m["id"] for m in _por_id(resultado)["bac_salario"]["movimientos"]] == ["octubre"]


def test_edicion_pendiente_actualiza_monto_y_metodo_sin_duplicar(monkeypatch):
    movimiento = _movimiento("usd", "8715", "5200", "USD", "10")
    monkeypatch.setattr(cuentas.dashboard_edicion, "proyecciones_pendientes", lambda _: [{
        "tipo": "editar", "movimiento_clave": "usd", "monto": "10400",
        "medio_pago": "8774",
    }])
    actualizados = cuentas._aplicar_pendientes({"cliente_id": "a"}, [movimiento])
    assert len(actualizados) == 1
    assert actualizados[0]["monto_original"] == Decimal("20")
    resultado = _por_id(cuentas.proyectar_saldos(CUENTAS, actualizados, [], date(2026, 9, 30)))
    assert resultado["bac_salario"]["saldo_crc"] == "507354.95"
    assert resultado["amex_8715"]["saldo_usd"] == "284.50"


def test_recurrencias_mensuales_se_ajustan_al_ultimo_dia_y_semanales():
    mensual = {"desde": date(2026, 1, 30), "hasta": None, "frecuencia": "mensual",
               "dias_mes": [14, 31]}
    assert list(cuentas._fechas_regla(mensual, date(2026, 3, 31))) == [
        date(2026, 1, 31), date(2026, 2, 14), date(2026, 2, 28),
        date(2026, 3, 14), date(2026, 3, 31),
    ]
    semanal = {"desde": date(2026, 9, 29), "hasta": None,
               "frecuencia": "semanal", "dia_semana": 1}
    assert list(cuentas._fechas_regla(semanal, date(2026, 10, 13))) == [
        date(2026, 9, 29), date(2026, 10, 6), date(2026, 10, 13),
    ]


@pytest.mark.parametrize("valor", ["0", "-1", "NaN", "Infinity", "texto"])
def test_operaciones_rechazan_montos_no_positivos(valor):
    with pytest.raises(cuentas.ErrorCuentas):
        cuentas._decimal(valor, positivo=True)


def test_ingreso_manual_acepta_ajuste_negativo_y_rechaza_cero():
    assert cuentas._monto_destino_operacion("ingreso", "-1250.50") == Decimal("-1250.50")
    with pytest.raises(cuentas.ErrorCuentas, match="no puede ser cero"):
        cuentas._monto_destino_operacion("ingreso", "0")
    with pytest.raises(cuentas.ErrorCuentas):
        cuentas._monto_destino_operacion("transferencia", "-1250.50")


def test_ajuste_negativo_reduce_el_saldo_de_la_cuenta_destino():
    ajuste = _operacion("ajuste", "ingreso", "", "bac_salario", "0", "-1250.50")
    resultado = _por_id(cuentas.proyectar_saldos(CUENTAS, [], [ajuste], date(2026, 9, 30)))
    cuenta = resultado["bac_salario"]
    assert cuenta["saldo_crc"] == "516504.45"
    assert cuenta["movimientos"][0]["monto"] == "-1250.50"


def test_endpoint_cuentas_requiere_sesion_y_devuelve_resultado(monkeypatch):
    app = TestClient(app_mod.app)
    sin_sesion = app.get("/api/dashboard/cuentas")
    assert sin_sesion.status_code == 401
    monkeypatch.setattr(app_mod, "_sesion_dashboard", lambda request: ({}, {"cliente_id": "a"}))
    monkeypatch.setattr(cuentas, "obtener", lambda cliente: {
        "ok": True, "configurado": True, "cuentas": [], "reglas": [],
    })
    respuesta = app.get("/api/dashboard/cuentas")
    assert respuesta.status_code == 200
    assert respuesta.json()["configurado"] is True
    assert respuesta.headers["cache-control"] == "no-store"


def test_endpoint_cuentas_acepta_el_mes_del_dashboard(monkeypatch):
    app = TestClient(app_mod.app)
    recibido = {}
    monkeypatch.setattr(app_mod, "_sesion_dashboard", lambda request: ({}, {"cliente_id": "a"}))
    monkeypatch.setattr(cuentas, "obtener", lambda cliente, hasta, desde=None: recibido.update(hasta=hasta, desde=desde) or {
        "ok": True, "configurado": True, "cuentas": [], "reglas": [],
    })

    respuesta = app.get("/api/dashboard/cuentas?inicio=2026-09-01")

    assert respuesta.status_code == 200
    assert recibido["hasta"] == date(2026, 9, 30)
    assert recibido["desde"] == date(2026, 9, 1)


def test_dashboard_descarga_reporte_pdf_del_mes(monkeypatch):
    app = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "_sesion_dashboard", lambda request: ({}, {"cliente_id": "a"}))
    monkeypatch.setattr(app_mod.dashboard, "obtener_snapshot", lambda *_: ({
        "cliente": {"nombre": "Cliente A"},
        "periodo": {"inicio": "2026-09-01", "etiqueta": "septiembre 2026"},
        "kpis": [{
            "kpi": "presupuesto_disponible", "columnas": ["presupuesto", "gastado"],
            "filas": [[100000, 24000]],
        }],
        "movimientos": [{
            "fecha": "2026-09-12", "categoria": "Hogar", "concepto": "Internet",
            "descripcion": "Proveedor", "monto": 24000, "moneda": "CRC",
        }],
    }, False))
    monkeypatch.setattr(cuentas, "obtener", lambda *_: {
        "ok": True, "fecha": "2026-09-30", "cuentas": [{
            "nombre": "Banco", "tipo": "banco", "moneda": "CRC",
            "saldo_crc": "76000", "saldo_usd": "0", "fecha_corte": "2026-09-01",
            "movimientos": [],
        }],
    })

    respuesta = app.get("/api/dashboard/reporte.pdf?inicio=2026-09-01")

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"] == "application/pdf"
    assert "reporte-financiero-2026-09.pdf" in respuesta.headers["content-disposition"]
    assert respuesta.content.startswith(b"%PDF-")
