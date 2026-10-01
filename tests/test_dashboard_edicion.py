import pytest
from dataclasses import replace
from decimal import Decimal

from bot import dashboard_edicion
from bot.edicion import CampoEdicion, PoliticaEdicion


def _politica_creacion():
    return PoliticaEdicion(
        tabla="gastos_manuales", origen="Google Sheets", clave_primaria="movimiento_id",
        anulacion_campo="activo", origen_tipo="google_sheets",
        origen_fuente_id="finanzas", hoja_origen="gastos_manuales", acciones=("crear",),
        campos={
            "movimiento_id": CampoEdicion("movimiento_id", "Identificador", calculado=True,
                                             generador="id_aleatorio_fecha"),
            "fecha": CampoEdicion("fecha", "Fecha", requerido=True, tipo="fecha_iso"),
            "descripcion": CampoEdicion("descripcion", "Descripción", requerido=True),
            "categoria": CampoEdicion("categoria", "Categoría", tipo="lista",
                                        valores=("Alimentacion",)),
            "linea_presupuesto_id": CampoEdicion(
                "linea_presupuesto_id", "Concepto presupuestario", requerido=True,
                generador="concepto_a_linea_id"),
            "monto": CampoEdicion("monto", "Monto", requerido=True, tipo="monto_positivo"),
            "moneda": CampoEdicion("moneda", "Moneda", requerido=True,
                                    tipo="moneda_iso", defecto="CRC"),
        },
    )


def test_reclasificar_guarda_override_y_encola_materializacion(monkeypatch):
    cliente = {"cliente_id": "cliente_a", "catalogo_spreadsheet_id": "sheet"}
    guardado, overrides = {}, []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda _cliente, _clave: (
            {"_modelo_id": "movimientos", "fuente": "banco", "clave_origen": "correo-1", "medio_pago": "Tarjeta anterior"},
            object(),
        ),
    )
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_comedera", "categoria": "Alimentacion", "concepto": "Comedera"},
    )
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {
        "modelos": [], "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "banco", "medio_pago": "tarjeta",
        }],
    })
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("semantic", {"modelo_id": "transacciones"}))
    monkeypatch.setattr(
        dashboard_edicion, "_guardar_override",
        lambda *args: overrides.append(args[1:]),
    )
    monkeypatch.setattr(
        dashboard_edicion, "_encolar_reconstruccion",
        lambda *args: guardado.update(cola=args[1:], version=8) or 8,
    )

    resultado = dashboard_edicion.reclasificar(
        "token", "bac:movimiento-1", "gas_comedera", "SINPE",
    )

    assert resultado == {
        "ok": True, "linea_id": "gas_comedera",
        "categoria": "Alimentacion", "concepto": "Comedera", "medio_pago": "SINPE",
        "estado": "pendiente", "version": 8,
    }
    assert overrides == [
        ("transacciones", "correo-1", "linea_presupuesto_id", "gas_comedera", "Reclasificado desde dashboard: Alimentacion > Comedera"),
        ("transacciones", "correo-1", "tarjeta", "SINPE", "Método de pago actualizado desde dashboard"),
    ]
    assert guardado["cola"][:2] == ("bac:movimiento-1", "")
    assert guardado["cola"][2] == {
        "tipo": "editar", "movimiento_clave": "bac:movimiento-1",
        "linea_id": "gas_comedera", "categoria": "Alimentacion",
        "concepto": "Comedera", "medio_pago": "SINPE",
    }


def test_reclasificar_monto_guarda_override_en_campo_declarado_por_metadata(monkeypatch):
    """Un importe bancario se corrige en la fuente semántica, no en Neon."""
    cliente = {"cliente_id": "cliente_a", "catalogo_spreadsheet_id": "sheet"}
    overrides = []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({
            "_modelo_id": "movimientos", "fuente": "banco", "clave_origen": "correo-1",
            "medio_pago": "Tarjeta", "monto": 100, "monto_original": 100,
            "moneda": "CRC", "moneda_original": "CRC", "linea_presupuesto_id": "gas_comedera",
        }, object()),
    )
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_comedera", "categoria": "Alimentacion", "concepto": "Comedera"},
    )
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {
        "modelos": [], "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "banco", "medio_pago": "tarjeta",
            "monto": "importe_capturado",
        }],
    })
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("semantic", {"modelo_id": "transacciones"}))
    monkeypatch.setattr(dashboard_edicion, "_guardar_override", lambda *args: overrides.append(args[1:]))
    monkeypatch.setattr(dashboard_edicion, "_encolar_reconstruccion", lambda *_: 5)

    resultado = dashboard_edicion.reclasificar(
        "token", "banco:1", "gas_comedera", monto="1,250.50",
    )

    assert resultado["monto"] == "1250.50"
    assert resultado["moneda"] == "CRC"
    assert overrides == [
        ("transacciones", "correo-1", "importe_capturado", "1250.50", "Monto visible actualizado desde dashboard: CRC 1250.50"),
    ]


def test_reclasificar_monto_visible_en_crc_guarda_equivalente_de_origen(monkeypatch):
    """Editar ₡21.000 nunca debe convertirse en USD 21.000."""
    cliente = {"cliente_id": "cliente_a", "catalogo_spreadsheet_id": "sheet"}
    overrides = []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({
            "_modelo_id": "movimientos", "fuente": "banco", "clave_origen": "correo-1",
            "linea_presupuesto_id": "gas_ropa", "medio_pago": "Tarjeta",
            "monto_original": 43.29, "moneda_original": "USD",
            "monto": 19649.331, "moneda": "CRC", "tipo_cambio": 453.9,
        }, object()),
    )
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {"linea_id": "gas_ropa", "categoria": "Ropa", "concepto": "Ropa"})
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {
        "modelos": [], "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "banco", "medio_pago": "tarjeta", "monto": "importe",
        }],
    })
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("semantic", {"modelo_id": "transacciones"}))
    monkeypatch.setattr(dashboard_edicion, "_guardar_override", lambda *args: overrides.append(args[1:]))
    monkeypatch.setattr(dashboard_edicion, "_encolar_reconstruccion", lambda *_: 1)

    resultado = dashboard_edicion.reclasificar("token", "banco:1", "gas_ropa", monto="21000")

    monto_fuente = Decimal(overrides[0][3])
    assert monto_fuente * Decimal("453.9") == Decimal("21000")
    assert resultado["monto"] == "21000"
    assert resultado["moneda"] == "CRC"


@pytest.mark.parametrize("monto", ["", "-1", "NaN", "infinito"])
def test_reclasificar_rechaza_monto_invalido(monkeypatch, monto):
    cliente = {"cliente_id": "cliente_a"}
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({"_modelo_id": "movimientos", "fuente": "manual", "clave_origen": "MAN-1", "medio_pago": "Efectivo"}, object()),
    )
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {"linea_id": "gas", "categoria": "A", "concepto": "B"})

    with pytest.raises(dashboard_edicion.ErrorReclasificacion, match="monto no es válido"):
        dashboard_edicion.reclasificar("token", "manual-1", "gas", monto=monto)


def test_reclasificar_permite_monto_cero_para_conservar_movimiento(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    guardado = []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({
            "_modelo_id": "movimientos", "fuente": "manual", "clave_origen": "MAN-1",
            "medio_pago": "Efectivo", "monto": 100, "moneda": "CRC",
        }, object()),
    )
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {"linea_id": "gas", "categoria": "A", "concepto": "B"})
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {
        "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "manual", "medio_pago": "medio_pago", "monto": "monto",
        }],
    })
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("raw", {"tabla_origen": "gastos_manuales"}))
    monkeypatch.setattr(dashboard_edicion, "_actualizar_movimiento_manual", lambda *args: guardado.append(args[5]) or "finanzas")
    monkeypatch.setattr(dashboard_edicion, "_encolar_reconstruccion", lambda *_: 1)

    resultado = dashboard_edicion.reclasificar("token", "manual-1", "gas", monto="0")

    assert guardado == ["0"]
    assert resultado["monto"] == "0"


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


def test_reclasificar_manual_actualiza_origen_y_encola_sincronizacion(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    llamadas = []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({"_modelo_id": "movimientos", "fuente": "manual", "clave_origen": "MAN-1", "medio_pago": "Efectivo"}, object()),
    )
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_comedera", "categoria": "Alimentacion", "concepto": "Comedera"},
    )
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {
        "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "manual", "medio_pago": "medio_pago",
        }],
    })
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("raw", {"tabla_origen": "gastos_manuales"}))
    monkeypatch.setattr(dashboard_edicion, "_actualizar_movimiento_manual", lambda *args: llamadas.append(("origen", args[2], args[3], args[4])) or "googledrive_db")
    monkeypatch.setattr(dashboard_edicion, "_encolar_reconstruccion", lambda *args: llamadas.append(("cola", args[1], args[2])) or 1)

    dashboard_edicion.reclasificar("token", "manual-1", "gas_comedera")

    assert llamadas == [
        ("origen", "MAN-1", "gas_comedera", "Efectivo"), ("cola", "manual-1", "googledrive_db"),
    ]


def test_reclasificar_manual_actualiza_monto_en_misma_fila(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    llamadas = []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({
            "_modelo_id": "movimientos", "fuente": "manual", "clave_origen": "MAN-1",
            "medio_pago": "Efectivo", "monto": 100, "moneda": "CRC",
        }, object()),
    )
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {"linea_id": "gas", "categoria": "A", "concepto": "B"})
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {
        "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "manual", "medio_pago": "medio_pago", "monto": "monto",
        }],
    })
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("raw", {"tabla_origen": "gastos_manuales"}))
    monkeypatch.setattr(
        dashboard_edicion, "_actualizar_movimiento_manual",
        lambda *args: llamadas.append((args[2], args[3], args[4], args[5])) or "finanzas",
    )
    monkeypatch.setattr(dashboard_edicion, "_encolar_reconstruccion", lambda *args: llamadas.append(("cola", args[1], args[2])) or 2)

    resultado = dashboard_edicion.reclasificar("token", "manual-1", "gas", monto="2500")

    assert llamadas == [("MAN-1", "gas", "Efectivo", "2500"), ("cola", "manual-1", "finanzas")]
    assert resultado["monto"] == "2500"


def test_reclasificar_regla_guarda_metadata_por_comercio(monkeypatch):
    cliente = {"cliente_id": "cliente_a", "catalogo_spreadsheet_id": "sheet"}
    guardado, overrides, reglas = {}, [], []
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({
            "_modelo_id": "movimientos", "fuente": "banco", "clave_origen": "correo-1",
            "medio_pago": "Tarjeta anterior", "descripcion": "Walmart Heredia",
            "concepto": "Comedera", "linea_presupuesto_id": "gas_viejo",
        }, object()),
    )
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_comedera", "categoria": "Alimentacion", "concepto": "Comedera"},
    )
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {
        "modelos": [], "overrides": [], "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "banco", "medio_pago": "tarjeta",
        }],
    })
    monkeypatch.setattr(dashboard_edicion, "_modelo_origen", lambda *_: ("semantic", {"modelo_id": "transacciones"}))
    monkeypatch.setattr(
        dashboard_edicion, "_regla_por_comercio",
        lambda *_: ("transacciones", "comercio_concepto", "MXM SAN FRANCISCO"),
    )
    monkeypatch.setattr(dashboard_edicion, "_guardar_override", lambda *args: overrides.append(args[1:]))
    monkeypatch.setattr(dashboard_edicion, "_guardar_regla_clasificacion", lambda *args: reglas.append(args[1:]))
    monkeypatch.setattr(dashboard_edicion, "_encolar_reconstruccion", lambda *args: guardado.update(version=3) or 3)

    resultado = dashboard_edicion.reclasificar(
        "token", "bac:movimiento-1", "gas_comedera", "SINPE", "regla",
    )

    assert resultado["alcance"] == "regla"
    assert resultado["comercio"] == "MXM SAN FRANCISCO"
    assert overrides == [
        ("transacciones", "correo-1", "tarjeta", "SINPE", "Método de pago actualizado desde dashboard"),
    ]
    assert reglas == [
        ("transacciones", "comercio_concepto", "MXM SAN FRANCISCO", "gas_comedera"),
    ]


def test_guardar_regla_clasificacion_agrega_fila_exacta_en_metadata(monkeypatch):
    class Hoja:
        encabezados = ["modelo_id", "columnas", "patron", "valor", "prioridad", "clasifica_en"]

        def __init__(self):
            self.agregadas = []

        def row_values(self, _):
            return self.encabezados

        def get_all_values(self):
            return [self.encabezados]

        def append_row(self, valores, **_):
            self.agregadas.append(valores)

    class Libro:
        def __init__(self, hoja):
            self.hoja = hoja

        def worksheet(self, nombre):
            assert nombre == "_clasificacion"
            return self.hoja

    hoja = Hoja()
    monkeypatch.setattr(dashboard_edicion, "abrir_libro_escritura", lambda _: Libro(hoja))

    dashboard_edicion._guardar_regla_clasificacion(
        {"catalogo_spreadsheet_id": "sheet"}, "transacciones_bac",
        "comercio_concepto", "MXM SAN FRANCISCO", "gas_comedera",
    )

    assert hoja.agregadas == [[
        "transacciones_bac", "comercio_concepto", "MXM SAN FRANCISCO",
        "gas_comedera", "0", "linea_presupuesto_id",
    ]]


def test_regla_por_comercio_usa_modelo_semantico_no_catalogo_publico(monkeypatch):
    class ModeloFalso:
        modelo_id = "transacciones_bac"
        campos = [{"columna": "comercio_concepto", "clasifica_en": "linea_presupuesto_id"}]

        def __init__(self, *_):
            pass

        def columnas(self):
            return [("_clave", "texto"), ("comercio_concepto", "texto")]

    llamadas = []
    monkeypatch.setattr(dashboard_edicion, "Modelo", ModeloFalso)
    monkeypatch.setattr(
        dashboard_edicion.warehouse_ro, "leer_interno",
        lambda *args: llamadas.append(args[1:]) or [{"comercio": "MXM PASEO D LAS FLORES"}],
    )
    datos = {
        "movimientos_canonicos": [{
            "modelo_id": "movimientos", "fuente": "bac", "clave": "_clave",
        }],
    }
    movimiento = {"_modelo_id": "movimientos", "fuente": "bac", "clave_origen": "correo-1"}

    assert dashboard_edicion._regla_por_comercio(
        {"cliente_id": "cliente_a"}, datos, movimiento,
        {"modelo_id": "transacciones_bac", "tabla_destino": "finanzas__transacciones"},
    ) == ("transacciones_bac", "comercio_concepto", "MXM PASEO D LAS FLORES")
    assert 'FROM "finanzas__transacciones"' in llamadas[0][0]
    assert llamadas[0][1] == {"clave": "correo-1"}


def test_reclasificar_rechaza_medio_pago_vacio(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(
        dashboard_edicion, "_movimiento",
        lambda *_: ({"_modelo_id": "movimientos", "fuente": "manual", "clave_origen": "MAN-1", "medio_pago": "Efectivo"}, object()),
    )
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {"linea_id": "gas", "categoria": "A", "concepto": "B"})
    monkeypatch.setattr(dashboard_edicion.metadata, "leer", lambda _: {"movimientos_canonicos": []})

    with pytest.raises(dashboard_edicion.ErrorReclasificacion, match="indique un método"):
        dashboard_edicion.reclasificar("token", "manual-1", "gas", "")


def test_procesar_reconstrucciones_agrupa_lote_e_invalida_version_vigente(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    trabajos = iter([
        {"movimiento_clave": "manual-1", "version": 3, "fuente_id": "googledrive_db"},
        {"movimiento_clave": "bac-2", "version": 4, "fuente_id": ""},
        None,
    ])
    llamadas = []
    monkeypatch.setattr(dashboard_edicion, "_tomar_reconstruccion", lambda _: next(trabajos))
    monkeypatch.setattr(dashboard_edicion, "_sincronizar_fuentes_manuales", lambda _c, fuentes: llamadas.append(("sync", fuentes)))
    monkeypatch.setattr(dashboard_edicion, "_reconstruir", lambda _: llamadas.append(("reconstruir",)))
    # La primera terminó después de una edición más nueva y no puede invalidar
    # el snapshot; la segunda sí corresponde a la última versión.
    vigentes = iter([False, True])
    monkeypatch.setattr(dashboard_edicion, "_terminar_reconstruccion", lambda *_: next(vigentes))
    monkeypatch.setattr(dashboard_edicion.dashboard, "invalidar_cache", lambda cid: llamadas.append(("cache", cid)))

    assert dashboard_edicion.procesar_reconstrucciones(cliente, agrupar=False) == 2
    assert llamadas == [
        ("sync", {"googledrive_db"}), ("reconstruir",), ("cache", "cliente_a"),
    ]


def test_crear_manual_guarda_en_origen_encola_e_impone_linea(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    llamadas, recibido = [], {}
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: ({}, cliente))
    monkeypatch.setattr(dashboard_edicion.edicion, "politica_para", lambda *_: _politica_creacion())
    monkeypatch.setattr(dashboard_edicion.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_comedera", "categoria": "Alimentacion", "concepto": "Comedera"},
    )
    monkeypatch.setattr(
        dashboard_edicion.escritura_google_sheets, "aplicar_confirmado",
        lambda _cliente, _politica, accion, valores: recibido.update(accion=accion, valores=valores) or {"clave": "MAN-1"},
    )
    monkeypatch.setattr(dashboard_edicion, "_encolar_reconstruccion", lambda *args: llamadas.append(args[1:]) or 4)

    resultado = dashboard_edicion.crear_movimiento("token", {
        "fecha": "2026-09-05", "descripcion": "Compra manual", "monto": "25.5",
        "moneda": "usd", "linea_presupuesto_id": "gas_comedera", "categoria": "Otra",
    })

    assert resultado["ok"] is True
    assert resultado["movimiento_id"] == "MAN-1"
    assert resultado["estado"] == "pendiente"
    assert resultado["movimiento_clave"] == "manual:MAN-1"
    assert resultado["movimiento"]["monto"] == "25.5"
    assert resultado["movimiento"]["moneda"] == "USD"
    assert recibido["accion"] == "crear"
    assert recibido["valores"]["linea_presupuesto_id"] == "gas_comedera"
    assert recibido["valores"]["categoria"] == "Alimentacion"
    assert recibido["valores"]["moneda"] == "USD"
    assert llamadas[0][:2] == ("manual:MAN-1", "finanzas")
    assert llamadas[0][2]["tipo"] == "crear"
    assert llamadas[0][2]["movimiento"] == resultado["movimiento"]


def test_lote_verifica_todas_las_fuentes_con_una_sincronizacion(monkeypatch):
    capturado = {}
    monkeypatch.setattr(
        dashboard_edicion.sync, "sincronizar_todo",
        lambda **kwargs: capturado.update(kwargs) or {"fuentes": [
            {"fuente_id": "manuales", "estado": "ok"},
            {"fuente_id": "tarjetas", "estado": "ok"},
        ]},
    )

    dashboard_edicion._sincronizar_fuentes_manuales(
        {"cliente_id": "cliente_a"}, {"manuales", "tarjetas"},
    )

    assert capturado == {"cliente_filtro": "cliente_a", "forzar": True}


def test_registrar_pago_reutiliza_creacion_manual_con_linea_fecha_y_monto(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    recibido = {}
    monkeypatch.setattr(
        dashboard_edicion.dashboard, "validar_enlace",
        lambda _: ({"inicio": "2026-09-01", "fin": "2026-10-01"}, cliente),
    )
    monkeypatch.setattr(dashboard_edicion.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_cuota", "categoria": "Vivienda", "concepto": "Cuota condominal", "pagable": True},
    )
    monkeypatch.setattr(dashboard_edicion.edicion, "politica_para", lambda *_: _politica_creacion())
    monkeypatch.setattr(
        dashboard_edicion, "crear_movimiento",
        lambda token, valores, **kwargs: recibido.update(token=token, valores=valores, periodo=kwargs.get("periodo")) or {"ok": True, "movimiento_id": "MAN-1"},
    )

    resultado = dashboard_edicion.registrar_pago("token", "gas_cuota", "80917", "2026-09-15")

    assert resultado == {"ok": True, "movimiento_id": "MAN-1"}
    assert recibido == {
        "token": "token",
        "valores": {
            "linea_presupuesto_id": "gas_cuota", "fecha": "2026-09-15", "monto": "80917",
            "descripcion": "Pago - Cuota condominal",
        },
        "periodo": {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
    }


def test_registrar_pago_envia_cuenta_elegida_al_movimiento_manual(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    recibido = {}
    politica = _politica_creacion()
    politica = replace(politica, campos={
        **politica.campos, "medio_pago": CampoEdicion("medio_pago", "Método de pago"),
    })
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: (
        {"inicio": "2026-09-01", "fin": "2026-10-01"}, cliente))
    monkeypatch.setattr(dashboard_edicion.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {
        "linea_id": "gas_cuota", "categoria": "Vivienda", "concepto": "Cuota", "pagable": True,
    })
    monkeypatch.setattr(dashboard_edicion.edicion, "politica_para", lambda *_: politica)
    monkeypatch.setattr(dashboard_edicion, "crear_movimiento", lambda _, valores, **kw: (
        recibido.update(valores) or {"ok": True}))

    dashboard_edicion.registrar_pago("token", "gas_cuota", "75000", "2026-09-30",
                                      "cuenta:mismart")
    assert recibido["medio_pago"] == "cuenta:mismart"
    assert recibido["monto"] == "75000"


def test_registrar_pago_usa_debito_cuenta_si_la_hoja_restringe_el_medio(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    recibido = {}
    politica = _politica_creacion()
    politica = replace(politica, campos={
        **politica.campos,
        "medio_pago": CampoEdicion(
            "medio_pago", "Método de pago", tipo="lista",
            valores=("sinpe", "transferencia", "efectivo", "debito_cuenta", "otro"),
        ),
    })
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: (
        {"inicio": "2026-09-01", "fin": "2026-10-01"}, cliente))
    monkeypatch.setattr(dashboard_edicion.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {
        "linea_id": "gas_cuota", "categoria": "Vivienda", "concepto": "Cuota", "pagable": True,
    })
    monkeypatch.setattr(dashboard_edicion.edicion, "politica_para", lambda *_: politica)
    from bot import cuentas
    monkeypatch.setattr(cuentas, "resolver_medio_de_cuenta", lambda *_: None)
    monkeypatch.setattr(dashboard_edicion, "crear_movimiento", lambda _, valores, **kw: (
        recibido.update(valores) or {"ok": True}))

    dashboard_edicion.registrar_pago(
        "token", "gas_cuota", "75000", "2026-09-30", "cuenta:mismart", "CR - Salario Walmart")

    assert recibido["medio_pago"] == "debito_cuenta"
    assert recibido["descripcion"] == "Pago - Cuota · CR - Salario Walmart"


def test_registrar_pago_vincula_la_cuenta_elegida_aun_con_lista_de_medios(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    recibido = {}
    politica = _politica_creacion()
    politica = replace(politica, campos={
        **politica.campos,
        "medio_pago": CampoEdicion("medio_pago", "Método de pago", tipo="lista", valores=("debito_cuenta",)),
    })
    monkeypatch.setattr(dashboard_edicion.dashboard, "validar_enlace", lambda _: (
        {"inicio": "2026-09-01", "fin": "2026-10-01"}, cliente))
    monkeypatch.setattr(dashboard_edicion.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(dashboard_edicion, "_validar_linea", lambda *_: {
        "linea_id": "gas_cuota", "categoria": "Vivienda", "concepto": "Cuota", "pagable": True,
    })
    monkeypatch.setattr(dashboard_edicion.edicion, "politica_para", lambda *_: politica)
    from bot import cuentas
    monkeypatch.setattr(cuentas, "resolver_medio_de_cuenta", lambda *_: "cuenta:bac_salario")
    monkeypatch.setattr(dashboard_edicion, "crear_movimiento", lambda _, valores, **kw: (
        recibido.update(valores=valores, **kw) or {"ok": True}))

    dashboard_edicion.registrar_pago(
        "token", "gas_cuota", "75000", "2026-09-30", "cuenta:bac_salario", "CR - Salario Walmart")

    assert recibido["valores"]["medio_pago"] == "cuenta:bac_salario"
    assert recibido["permitir_medio_cuenta"] is True


def test_registrar_pago_rechaza_linea_no_pagable_y_fecha_fuera_del_mes(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    monkeypatch.setattr(
        dashboard_edicion.dashboard, "validar_enlace",
        lambda _: ({"inicio": "2026-09-01", "fin": "2026-10-01"}, cliente),
    )
    with pytest.raises(dashboard_edicion.ErrorReclasificacion, match="dentro del período"):
        dashboard_edicion.registrar_pago("token", "gas_cuota", "100", "2026-10-01")

    monkeypatch.setattr(dashboard_edicion.catalogo, "construir_contexto", lambda _: object())
    monkeypatch.setattr(
        dashboard_edicion, "_validar_linea",
        lambda *_: {"linea_id": "gas_variable", "categoria": "Otros", "concepto": "Variable", "pagable": False},
    )
    with pytest.raises(dashboard_edicion.ErrorReclasificacion, match="no está habilitado"):
        dashboard_edicion.registrar_pago("token", "gas_variable", "100", "2026-09-15")


def test_sincronizacion_manual_ignora_error_de_otra_fuente(monkeypatch):
    monkeypatch.setattr(
        dashboard_edicion.sync, "sincronizar_todo",
        lambda **_: {
            "error": 1,
            "fuentes": [
                {"fuente_id": "googledrive_db", "estado": "ok"},
                {"fuente_id": "correo_zoho", "estado": "error", "error": "credencial"},
            ],
        },
    )

    dashboard_edicion._sincronizar_fuente_manual(
        {"cliente_id": "cliente_a"}, "googledrive_db", "el movimiento",
    )


def test_sincronizacion_manual_rechaza_fallo_de_su_propia_fuente(monkeypatch):
    monkeypatch.setattr(
        dashboard_edicion.sync, "sincronizar_todo",
        lambda **_: {"fuentes": [{"fuente_id": "googledrive_db", "estado": "error"}]},
    )

    with pytest.raises(dashboard_edicion.ErrorReclasificacion, match="guardé el movimiento"):
        dashboard_edicion._sincronizar_fuente_manual(
            {"cliente_id": "cliente_a"}, "googledrive_db", "el movimiento",
        )
