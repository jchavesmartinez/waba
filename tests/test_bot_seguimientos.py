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
            CLIENTE, "50600000000", "En un PDF por favor", historial,
        )

    assert ejecutado["sql"] == sql_detalle
    assert len(respuesta.adjuntos) == 1
    assert respuesta.adjuntos[0].nombre == "detalle.pdf"
    assert respuesta.texto == "Listo, te adjunto el PDF con 5 registros."
    assert "Resultado exacto" not in respuesta.texto
    assert len(pdf["filas"]) == 5


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
