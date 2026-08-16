"""Regresiones: el texto y el Excel deben compartir una sola verdad numerica."""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from bot import intencion, kpis, nl2sql
from sources.base import _limpiar_numero


def test_presupuesto_se_redacta_con_totales_que_vienen_del_sql():
    columnas = [
        "categoria", "concepto", "mensual", "total_categoria", "total_general",
    ]
    filas = [
        ("Servicios", "Internet", 29000, 50000, 5355326),
        ("Servicios", "Celular Jose", 11000, 50000, 5355326),
        ("Servicios", "Celular Aline", 10000, 50000, 5355326),
    ]

    with patch.object(nl2sql, "_anthropic", side_effect=AssertionError("no LLM")):
        texto = nl2sql.redactar_respuesta(
            "presupuesto por categoria y concepto", columnas, filas,
            unidad="colones",
        )

    assert "₡5.355.326" in texto
    assert "*Servicios — ₡50.000*" in texto
    assert "Internet: ₡29.000" in texto
    assert "Utilidades" not in texto


def test_resultado_generico_no_inventa_un_total_que_sql_no_trajo():
    texto = nl2sql.redactar_respuesta(
        "ventas por producto", ["producto", "monto"],
        [("A", 100), ("B", 200)], unidad="colones",
    )
    assert "₡100" in texto and "₡200" in texto
    assert "₡300" not in texto


def test_una_sola_celda_se_devuelve_directa_sin_llm():
    with patch.object(nl2sql, "_anthropic", side_effect=AssertionError("no LLM")):
        texto = nl2sql.redactar_respuesta(
            "total", ["presupuesto_total"], [(5355326,)], unidad="colones",
        )
    assert texto == "*Presupuesto total:* ₡5.355.326"


def test_objecion_a_una_categoria_vuelve_a_consultar_datos():
    historial = [{"rol": "assistant", "contenido": "Utilidades: 751.000"}]
    assert intencion.clasificar(
        "¿Por qué agregaste Utilidades como categoría?", historial,
    ) == "datos"


def test_kpis_del_final_se_seleccionan_por_relevancia_y_no_por_posicion():
    relleno = [
        {"kpi": f"ventas_{i}", "nombre": "Ventas", "preguntas_ejemplo": "ventas"}
        for i in range(30)
    ]
    objetivo = {
        "kpi": "plan_presupuesto_concepto",
        "nombre": "Presupuesto por categoria y concepto",
        "preguntas_ejemplo": "presupuesto total mensual por categoria y concepto",
    }
    seleccion = kpis._seleccionar_kpis(
        relleno + [objetivo], "presupuesto por categoria y concepto",
    )
    assert objetivo in seleccion


def test_planificador_elige_kpi_pero_no_puede_reescribir_su_sql():
    definicion = {
        "kpi": "plan_presupuesto",
        "nombre": "Presupuesto planeado",
        "preguntas_ejemplo": "cuanto tengo presupuestado",
        "formula_sql": (
            "SELECT categoria, SUM(monto_mensual) AS mensual "
            "FROM presupuesto WHERE tipo='gasto' GROUP BY categoria"
        ),
    }
    capturado = {}

    class _Mensajes:
        def create(self, **kwargs):
            capturado.update(kwargs)
            bloque = SimpleNamespace(
                type="text",
                text=(
                    '{"accion":"usar_kpi","kpi":"plan_presupuesto",'
                    '"sql":"SELECT 999 AS total","mensaje":""}'
                ),
            )
            return SimpleNamespace(content=[bloque])

    cliente = SimpleNamespace(messages=_Mensajes())
    ctx = SimpleNamespace(
        schema_text="sharepoint_db__presupuesto(categoria, monto_mensual, tipo)",
        tablas_reales={"sharepoint_db__presupuesto"},
        permitidas=[SimpleNamespace(
            tabla_logica="presupuesto",
            tabla_real="sharepoint_db__presupuesto",
        )],
    )
    with patch.object(kpis, "_anthropic", return_value=cliente):
        plan = kpis.planificar("cuanto tengo presupuestado", [definicion], ctx)

    assert plan["accion"] == "usar_kpi"
    assert "SELECT 999" not in plan["sql"]
    assert "sharepoint_db__presupuesto" in plan["sql"]
    assert "deja 'sql' VACIO" in capturado["system"]


def test_decimal_de_excel_no_se_multiplica_por_diez_en_locale_tico():
    serie = pd.Series(["390037.5", "5.000", "1.250.000", "3,25"])
    valores = pd.to_numeric(_limpiar_numero(serie, coma_decimal=True)).tolist()
    assert valores == [390037.5, 5000.0, 1250000.0, 3.25]

