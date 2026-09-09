import time

import pytest

import config
from bot import catalogo, dashboard


@pytest.fixture
def dashboard_configurado(monkeypatch):
    monkeypatch.setattr(config, "BOT_DASHBOARD", True)
    monkeypatch.setattr(config, "APP_PUBLIC_URL", "https://app.example.com")
    monkeypatch.setattr(config, "DASHBOARD_SECRET", "s" * 48)
    monkeypatch.setattr(config, "DASHBOARD_TOKEN_TTL_MINUTOS", 30)
    return {"cliente_id": "cliente_a", "nombre": "Cliente A"}


def test_detecta_solicitudes_de_dashboard():
    assert dashboard.es_solicitud("Mándame mi dashboard")
    assert dashboard.es_solicitud("Quiero ver mis KPIs")
    assert dashboard.es_solicitud("Abre el panel financiero")
    assert not dashboard.es_solicitud("¿Cuánto gasté este mes?")


def test_enlace_firmado_conserva_cliente_numero_y_periodo(
    monkeypatch, dashboard_configurado,
):
    cliente = dashboard_configurado
    monkeypatch.setattr(dashboard.registry, "resolver", lambda numero: cliente)

    url, etiqueta = dashboard.crear_enlace(
        cliente, "+506 8888-9999", "dashboard de agosto 2026",
    )
    token = url.rsplit("/", 1)[-1]
    payload, resuelto = dashboard.validar_enlace(token)

    assert etiqueta == "agosto 2026"
    assert payload["cid"] == "cliente_a"
    assert payload["num"] == "50688889999"
    assert payload["inicio"] == "2026-08-01"
    assert payload["fin"] == "2026-09-01"
    assert resuelto == cliente


def test_enlace_alterado_o_vencido_se_rechaza(monkeypatch, dashboard_configurado):
    cliente = dashboard_configurado
    monkeypatch.setattr(dashboard.registry, "resolver", lambda numero: cliente)
    url, _ = dashboard.crear_enlace(cliente, "50688889999", "dashboard")
    token = url.rsplit("/", 1)[-1]

    with pytest.raises(dashboard.EnlaceInvalido):
        dashboard.validar_enlace(token + "x")
    with pytest.raises(dashboard.EnlaceInvalido):
        dashboard.validar_enlace(token, ahora=int(time.time()) + 3600)


def test_render_incrusta_snapshot_sin_llamadas_del_frontend(
    monkeypatch, dashboard_configurado,
):
    cliente = dashboard_configurado
    monkeypatch.setattr(dashboard.registry, "resolver", lambda numero: cliente)
    monkeypatch.setattr(
        dashboard,
        "generar_snapshot",
        lambda cliente, periodo: {
            "cliente": {"id": "cliente_a", "nombre": "Cliente A"},
            "periodo": {**periodo, "etiqueta": "agosto 2026"},
            "actualizado_en": "2026-09-02T02:00-06:00",
            "kpis": [],
        },
    )
    url, _ = dashboard.crear_enlace(
        cliente, "50688889999", "dashboard de agosto 2026",
    )
    html = dashboard.renderizar(url.rsplit("/", 1)[-1])

    assert "__DASHBOARD_DATA__" not in html
    assert '"nombre":"Cliente A"' in html
    assert "/dashboard-assets/app.js" in html


def test_jerarquia_prefiere_movimientos_canonicos_para_detalle(monkeypatch):
    """El árbol no debe perder cargos bancarios al existir la tabla canónica."""
    presupuesto = catalogo.TablaPermitida(
        tabla_logica="presupuesto", tabla_real="presupuesto", fuente_id="sheet",
        columnas_config={"linea_id": {}, "categoria": {}, "concepto": {}},
    )
    canonicos = catalogo.TablaPermitida(
        tabla_logica="movimientos", tabla_real="finanzas__movimientos", fuente_id="modelo",
        columnas_config={
            "linea_presupuesto_id": {}, "fecha": {}, "descripcion": {},
            "moneda": {}, "monto_neto": {}, "tipo_movimiento": {},
        },
    )
    manuales = catalogo.TablaPermitida(
        tabla_logica="gastos_manuales", tabla_real="gastos_manuales", fuente_id="sheet",
        columnas_config={
            "linea_presupuesto_id": {}, "fecha": {}, "descripcion": {},
            "monto": {}, "moneda": {}, "tipo_movimiento": {}, "activo": {},
            "incluir_en_gasto": {},
        },
    )
    ctx = catalogo.Contexto(
        schema_text="", tablas_reales={"presupuesto", "finanzas__movimientos", "gastos_manuales"},
        permitidas=[presupuesto, canonicos, manuales],
    )
    capturado = {}

    def ejecutar(_cliente, sql, limite):
        capturado["sql"] = sql
        capturado["limite"] = limite
        return (
            ["linea_id", "categoria", "concepto", "fecha", "descripcion", "moneda", "monto"],
            [("gas_comedera", "Alimentacion", "Comedera", "2026-09-05",
              "WALMART", "CRC", 23148)],
        )

    monkeypatch.setattr(dashboard.nl2sql, "validar_sql", lambda *_: (True, ""))
    monkeypatch.setattr(dashboard.warehouse_ro, "ejecutar", ejecutar)

    filas = dashboard._movimientos_jerarquia(
        {"cliente_id": "cliente_a"}, ctx,
        {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
    )

    assert 'FROM "finanzas__movimientos" m' in capturado["sql"]
    assert 'FROM "gastos_manuales" g' not in capturado["sql"]
    assert filas == [{
        "linea_id": "gas_comedera", "categoria": "Alimentacion",
        "concepto": "Comedera", "fecha": "2026-09-05",
        "descripcion": "WALMART", "moneda": "CRC", "monto": 23148,
    }]


def test_jerarquia_no_oculta_movimientos_sin_linea_de_presupuesto(monkeypatch):
    """Una línea no mapeada aparece explícitamente como sin clasificar."""
    presupuesto = catalogo.TablaPermitida(
        tabla_logica="presupuesto", tabla_real="presupuesto", fuente_id="sheet",
        columnas_config={"linea_id": {}, "categoria": {}, "concepto": {}},
    )
    canonicos = catalogo.TablaPermitida(
        tabla_logica="movimientos", tabla_real="movimientos", fuente_id="modelo",
        columnas_config={
            "linea_presupuesto_id": {}, "fecha": {}, "descripcion": {},
            "moneda": {}, "monto_neto": {},
        },
    )
    ctx = catalogo.Contexto(
        schema_text="", tablas_reales={"presupuesto", "movimientos"},
        permitidas=[presupuesto, canonicos],
    )

    def ejecutar(_cliente, _sql, limite):
        assert limite > 0
        return (
            ["linea_id", "categoria", "concepto", "fecha", "descripcion", "moneda", "monto"],
            [("sin_clasificar", "Sin clasificar", "Gastos sin identificar",
              "2026-09-05", "Comercio pendiente", "CRC", 1000)],
        )

    monkeypatch.setattr(dashboard.nl2sql, "validar_sql", lambda *_: (True, ""))
    monkeypatch.setattr(dashboard.warehouse_ro, "ejecutar", ejecutar)
    filas = dashboard._movimientos_jerarquia(
        {"cliente_id": "cliente_a"}, ctx,
        {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
    )
    assert filas[0]["categoria"] == "Sin clasificar"
    assert filas[0]["concepto"] == "Gastos sin identificar"
