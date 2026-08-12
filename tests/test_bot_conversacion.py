"""Regresiones de continuidad conversacional y SQL libre."""

from bot import intencion, nl2sql


def test_una_respuesta_literal_inventada_no_cuenta_como_consulta():
    ok, motivo = nl2sql.validar_sql(
        "SELECT 'No tengo contexto de la pregunta anterior' AS respuesta",
        {"finanzas__transacciones"},
    )
    assert not ok
    assert "leer una tabla" in motivo


def test_el_sentinel_no_respondible_sigue_permitido():
    ok, motivo = nl2sql.validar_sql(
        "SELECT 'NO_RESPONDIBLE' AS nota;", {"finanzas__transacciones"})
    assert ok, motivo


def test_un_select_real_sigue_permitido():
    ok, motivo = nl2sql.validar_sql(
        "SELECT comercio FROM finanzas__transacciones",
        {"finanzas__transacciones"},
    )
    assert ok, motivo


def test_en_total_responde_la_pregunta_anterior_sobre_datos():
    historial = [{
        "rol": "assistant",
        "contenido": "¿En qué periodo: este mes o los últimos 30 días?",
    }]
    assert intencion.clasificar("en total", historial) == "datos"


def test_esas_es_referencia_al_ultimo_resultado_de_datos():
    historial = [{
        "rol": "assistant",
        "contenido": "Este mes gastaste ₡53.000 en 9 transacciones.",
    }]
    assert intencion.clasificar("esas", historial) == "datos"
