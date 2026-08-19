"""Regresiones para seguimientos cortos y exportacion del ultimo resultado."""

from types import SimpleNamespace
from unittest.mock import patch

from bot import formato
from bot import responder as R
from bot.salida import Adjunto


CLIENTE = {"cliente_id": "cliente_a"}
CTX = SimpleNamespace(
    error_lectura=False,
    tablas_reales={"transacciones"},
    schema_text=(
        "transacciones(fecha_transaccion, comercio, monto, cuenta_contable)"
    ),
    permitidas=[],
)


def test_pedir_pdf_reutiliza_ultimo_sql_sin_volver_a_planificar():
    sql_detalle = (
        "SELECT fecha_transaccion, comercio, monto FROM transacciones "
        "WHERE cuenta_contable = 'sin_clasificar'"
    )
    historial = [
        {"rol": "user", "contenido": "Quiero el detalle de los gastos sin clasificar"},
        {
            "rol": "assistant",
            "contenido": "Resultado exacto (5 registros)",
            "sql": sql_detalle,
        },
    ]
    ejecutado = {}
    pdf = {}

    def _ejecutar(cliente, sql, limite=None):
        ejecutado.update(sql=sql, limite=limite)
        return ["comercio", "monto"], [(f"Comercio {i}", i) for i in range(5)]

    def _pdf(columnas, filas, titulo="", resumen="", **kwargs):
        pdf.update(columnas=columnas, filas=filas, titulo=titulo, resumen=resumen)
        return Adjunto(
            tipo="document", contenido=b"%PDF-prueba", nombre="detalle.pdf",
            mime="application/pdf",
        )

    with (
        patch.object(R.catalogo, "construir_contexto", return_value=CTX),
        patch.object(R.kpis, "cargar_kpis", return_value=[]),
        patch.object(
            R.kpis, "planificar",
            side_effect=AssertionError("no debe llamar al planificador"),
        ),
        patch.object(R.warehouse_ro, "ejecutar", side_effect=_ejecutar),
        patch.object(R.artefactos, "pdf_reporte", side_effect=_pdf),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50600000000", "Dámelo en un PDF por favor", historial,
        )

    assert ejecutado["sql"] == sql_detalle
    assert len(respuesta.adjuntos) == 1
    assert respuesta.adjuntos[0].nombre == "detalle.pdf"
    assert respuesta.texto == "Listo. Le adjunto el PDF con 5 registros."
    assert "Resultado exacto" not in respuesta.texto
    assert len(pdf["filas"]) == 5
    assert pdf["resumen"] == ""
    assert pdf["titulo"].lower() != "en un pdf"


def test_pedir_excel_con_damelo_reutiliza_el_ultimo_sql():
    sql_detalle = (
        "SELECT comercio, monto FROM transacciones "
        "ORDER BY comercio"
    )
    historial = [
        {"rol": "user", "contenido": "Muéstrame todo el presupuesto"},
        {
            "rol": "assistant",
            "contenido": "Resultado exacto (44 registros)",
            "sql": sql_detalle,
        },
    ]
    ejecutado = {}

    def _ejecutar(cliente, sql, limite=None):
        ejecutado.update(sql=sql, limite=limite)
        return ["concepto", "disponible"], [(f"Concepto {i}", i) for i in range(44)]

    def _excel(columnas, filas, titulo="", **kwargs):
        return Adjunto(
            tipo="document", contenido=b"PK-prueba", nombre="presupuesto.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with (
        patch.object(R.catalogo, "construir_contexto", return_value=CTX),
        patch.object(R.kpis, "cargar_kpis", return_value=[]),
        patch.object(
            R.kpis, "planificar",
            side_effect=AssertionError("no debe llamar al planificador"),
        ),
        patch.object(R.warehouse_ro, "ejecutar", side_effect=_ejecutar),
        patch.object(R.artefactos, "excel_xlsx", side_effect=_excel),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50600000000", "Dámelo en Excel sí", historial,
        )

    assert ejecutado["sql"] == sql_detalle
    assert len(respuesta.adjuntos) == 1
    assert respuesta.adjuntos[0].nombre == "presupuesto.xlsx"
    assert respuesta.texto == "Listo. Le adjunto el Excel con 44 registros."


def test_pedir_datos_usados_repite_exactamente_la_consulta_del_analisis():
    sql_analisis = (
        "SELECT DATE_TRUNC('month', fecha_transaccion) AS mes, SUM(monto) AS total "
        "FROM transacciones GROUP BY 1 ORDER BY 1"
    )
    historial = [
        {"rol": "user", "contenido": "Las ventas suben o bajan mes a mes?"},
        {
            "rol": "assistant",
            "contenido": "Las ventas fluctúan mes a mes.",
            "sql": sql_analisis,
        },
    ]
    ejecutado = {}

    def _ejecutar(_cliente, sql, limite=None):
        ejecutado.update(sql=sql, limite=limite)
        return ["mes", "total"], [("2026-01", 100), ("2026-02", 140)]

    with (
        patch.object(R.catalogo, "construir_contexto", return_value=CTX),
        patch.object(R.kpis, "cargar_kpis", return_value=[]),
        patch.object(
            R.kpis, "planificar",
            side_effect=AssertionError("debe reutilizar el SQL anterior"),
        ),
        patch.object(R.warehouse_ro, "ejecutar", side_effect=_ejecutar),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50600000000", "Me das los datos que usaste por favor", historial,
            fmt_solicitado=formato.TEXTO,
        )

    assert ejecutado["sql"] == sql_analisis
    assert "100" in respuesta.texto and "140" in respuesta.texto


def test_si_a_oferta_de_detalle_fuerza_desglose_y_no_repite_resumen():
    historial = [
        {"rol": "user", "contenido": "Cuantos gastos sin clasificar hay?"},
        {
            "rol": "assistant",
            "contenido": (
                "Pendientes: 5. ¿Quieres ver el detalle de esos 5 gastos "
                "sin clasificar?"
            ),
            "sql": (
                "SELECT COUNT(*) AS pendientes FROM transacciones "
                "WHERE cuenta_contable = 'sin_clasificar'"
            ),
        },
    ]
    generado = {}
    sql_detalle = (
        "SELECT comercio, monto FROM transacciones "
        "WHERE cuenta_contable = 'sin_clasificar'"
    )

    def _generar(pregunta, schema_text, **kwargs):
        generado["pregunta"] = pregunta
        return sql_detalle

    with (
        patch.object(R.catalogo, "construir_contexto", return_value=CTX),
        patch.object(R.kpis, "cargar_kpis", return_value=[]),
        patch.object(
            R.kpis, "planificar",
            side_effect=AssertionError("no debe repetir el KPI de resumen"),
        ),
        patch.object(R.nl2sql, "generar_sql", side_effect=_generar),
        patch.object(
            R.warehouse_ro, "ejecutar",
            return_value=(["comercio", "monto"], [("Tienda", 1000)]),
        ),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50600000000", "Si, por favor", historial,
            fmt_solicitado=formato.TEXTO,
        )

    assert "filas de detalle" in generado["pregunta"]
    assert respuesta.sql == sql_detalle
    assert "Tienda" in respuesta.texto
