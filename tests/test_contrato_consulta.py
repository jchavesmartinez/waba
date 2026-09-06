from types import SimpleNamespace

from bot import contrato_consulta, ejecutor_consultas


def test_contrato_nuevo_no_hereda_filtros_ni_periodo():
    previo = {
        "contrato": {
            "operacion": "total", "metrica": "presupuesto",
            "entidad": "concepto", "filtros": {"categoria": "Alimentacion"},
            "periodo": {"mes": "2026-08"}, "relacion": "seguimiento",
        }
    }
    contrato = contrato_consulta.crear(
        {"operacion": "detalle", "metrica": "gastado", "relacion": "nueva"},
        previo=previo,
    )
    assert contrato["filtros"] == {}
    assert contrato["periodo"] == {}


def test_contrato_seguimiento_hereda_y_actualiza_solo_filtros_explicitos():
    previo = {
        "operacion": "total", "metrica": "gastado", "entidad": "concepto",
        "filtros": {"categoria": "Alimentacion"},
        "periodo": {"mes": "2026-08"}, "relacion": "nueva",
    }
    contrato = contrato_consulta.crear(
        {"operacion": "detalle", "relacion": "seguimiento",
         "filtros": {"concepto": "Comidas afuera"}},
        previo=previo,
    )
    assert contrato["filtros"] == {
        "categoria": "Alimentacion", "concepto": "Comidas afuera",
    }
    assert contrato["periodo"] == {"mes": "2026-08"}


def test_contrato_agrupado_sin_entidad_no_es_valido():
    ok, motivo = contrato_consulta.es_valido({
        "operacion": "ranking", "metrica": "gastado", "entidad": "",
        "filtros": {}, "periodo": {}, "relacion": "nueva",
    })
    assert not ok
    assert "entidad" in motivo


def test_ejecutor_rechaza_sql_fuera_de_la_lista_blanca():
    ok, motivo = ejecutor_consultas.validar(
        "SELECT * FROM secreta", "dame el detalle",
        SimpleNamespace(tablas_reales={"ventas"}),
        contrato_consulta.crear({"operacion": "detalle", "relacion": "nueva"}),
    )
    assert not ok
    assert "no permitida" in motivo
