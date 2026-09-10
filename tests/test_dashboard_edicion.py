import pytest

from bot import dashboard_edicion


def test_reclasificar_guarda_override_reconstruye_e_invalida(monkeypatch):
    cliente = {"cliente_id": "cliente_a", "catalogo_spreadsheet_id": "sheet"}
    guardado = {}
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda _cliente, _clave: (
            {"_modelo_id": "movimientos", "fuente": "banco", "clave_origen": "correo-1"},
            object(),
        ),
    )
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_comedera", "categoria": "Alimentacion", "concepto": "Comedera"},
    )
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {"modelos": [], "movimientos_canonicos": []})
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("semantic", {"modelo_id": "transacciones"}))
    monkeypatch.setattr(
        dashboard_edicion, "_guardar_override",
        lambda *args: guardado.update(modelo=args[1], clave=args[2], linea=args[3], nota=args[4]),
    )
    monkeypatch.setattr(dashboard_edicion, "_reconstruir", lambda _: guardado.update(reconstruido=True))
    monkeypatch.setattr(dashboard_edicion.dashboard, "invalidar_cache", lambda cid: guardado.update(cache=cid))

    resultado = dashboard_edicion.reclasificar("token", "bac:movimiento-1", "gas_comedera")

    assert resultado == {
        "ok": True, "linea_id": "gas_comedera",
        "categoria": "Alimentacion", "concepto": "Comedera",
    }
    assert guardado["modelo"] == "transacciones"
    assert guardado["clave"] == "correo-1"
    assert guardado["linea"] == "gas_comedera"
    assert guardado["reconstruido"] is True
    assert guardado["cache"] == "cliente_a"


def test_reclasificar_rechaza_identificador_no_seguro(monkeypatch):
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, {"cliente_id": "cliente_a"}))
    with pytest.raises(dashboard_edicion.ErrorReclasificacion, match="no es válida"):
        dashboard_edicion.reclasificar("token", "x'; DROP TABLE", "gas_comedera")


def test_modelo_origen_resuelve_fuente_manual():
    datos = {
        "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "manual", "capa_origen": "raw",
            "tabla_origen": "gastos_manuales",
        }],
        "modelos": [],
    }
    capa, configuracion = dashboard_edicion._modelo_origen(
        datos, {"_modelo_id": "movimientos", "fuente": "manual"},
    )
    assert capa == "raw"
    assert configuracion["tabla_origen"] == "gastos_manuales"


def test_reclasificar_manual_actualiza_origen_y_luego_reconstruye(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    llamadas = []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({"_modelo_id": "movimientos", "fuente": "manual", "clave_origen": "MAN-1"}, object()),
    )
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_comedera", "categoria": "Alimentacion", "concepto": "Comedera"},
    )
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {})
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("raw", {"tabla_origen": "gastos_manuales"}))
    monkeypatch.setattr(dashboard_edicion, "_actualizar_movimiento_manual", lambda *args: llamadas.append(("origen", args[2], args[3])))
    monkeypatch.setattr(dashboard_edicion, "_sincronizar_fuente_manual", lambda *_: llamadas.append(("sync",)))
    monkeypatch.setattr(dashboard_edicion, "_reconstruir", lambda *_: llamadas.append(("reconstruir",)))
    monkeypatch.setattr(dashboard_edicion.dashboard, "invalidar_cache", lambda *_: llamadas.append(("cache",)))

    dashboard_edicion.reclasificar("token", "manual-1", "gas_comedera")

    assert llamadas == [
        ("origen", "MAN-1", "gas_comedera"), ("sync",), ("reconstruir",), ("cache",),
    ]
