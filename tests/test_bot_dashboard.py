import time

import pytest

import config
from bot import dashboard


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
