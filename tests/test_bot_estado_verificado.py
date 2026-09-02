from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from bot import formato, kpis, seguimiento
from bot import responder as R


CLIENTE = {"cliente_id": "cliente_prueba"}
CTX = SimpleNamespace(
    error_lectura=False,
    tablas_reales={"finanzas__transacciones", "finanzas__presupuesto"},
    schema_text=(
        "finanzas__transacciones(linea_presupuesto_id, concepto, monto_crc); "
        "finanzas__presupuesto(linea_id, concepto, monto_mensual)"
    ),
    permitidas=[
        SimpleNamespace(tabla_logica="transacciones", tabla_real="finanzas__transacciones"),
        SimpleNamespace(tabla_logica="presupuesto", tabla_real="finanzas__presupuesto"),
    ],
)


def test_estado_conserva_filtro_unico_y_hash():
    estado = seguimiento.crear_estado(
        "qué conforma imprevistos",
        "SELECT * FROM finanzas__transacciones",
        "detalle_gastos", "crc",
        ["linea_presupuesto_id", "concepto", "monto_crc"],
        [("gas_imprevistos_jose", "Imprevistos Jose", 26900),
         ("gas_imprevistos_jose", "Imprevistos Jose", 195000)],
    )
    assert estado["filtros"] == {
        "linea_id": "gas_imprevistos_jose",
        "concepto": "Imprevistos Jose",
    }
    assert len(estado["resultado_hash"]) == 64


def test_kpi_se_parametriza_con_linea_verificada():
    sql, aplicados = kpis.parametrizar_sql(
        "SELECT linea_id, concepto, SUM(presupuesto) AS presupuesto "
        "FROM finanzas__presupuesto GROUP BY 1, 2",
        {"linea_id": "gas_imprevistos_jose", "concepto": "Imprevistos Jose"},
    )
    assert aplicados == {"linea_id": "gas_imprevistos_jose"}
    assert "AS _kpi" in sql
    assert "_kpi.linea_id" in sql
    assert "'gas_imprevistos_jose'" in sql


def test_kpi_de_seguimiento_conserva_el_mes_aunque_cambie_current_date():
    sql, aplicados = kpis.parametrizar_sql(
        "SELECT linea_id, SUM(gastado) AS gastado FROM finanzas__presupuesto "
        "WHERE fecha >= DATE_TRUNC('month', CURRENT_DATE) GROUP BY 1",
        {"linea_id": "gas_imprevistos_jose"},
        {"inicio": "2026-08-01", "fin_inclusivo": "2026-08-31",
         "granularidad": "mes"},
    )
    assert "CURRENT_DATE" not in sql
    assert "CAST('2026-08-01' AS DATE)" in sql
    assert aplicados["periodo"]["inicio"] == "2026-08-01"


def test_reconciliador_rechaza_porcentaje_con_denominador_anual():
    ok, motivo = seguimiento.validar_resultado(
        ["concepto", "presupuesto", "gastado", "disponible", "porcentaje_consumido"],
        [("Imprevistos Jose", 1560000, 127625, 1432375, 98.17)],
    )
    assert ok is False
    assert "porcentaje" in motivo


def test_presupuesto_duplicado_se_corrige_con_la_fuente_mensual():
    columnas = ["concepto", "presupuesto_mensual", "gasto_real", "diferencia",
                "porcentaje_ejecutado"]
    filas, cambios = seguimiento.reconciliar_presupuesto_fuente(
        columnas,
        [("Imprevistos Jose", 260000, 322625, -62625, 124.09)],
        {"concepto:imprevistos jose": Decimal("130000")},
    )
    assert len(cambios) == 1
    assert filas[0][1] == Decimal("130000")
    assert filas[0][3] == Decimal("-192625")
    assert filas[0][4].quantize(Decimal("0.01")) == Decimal("248.17")


def test_consulta_de_composicion_no_implica_mes_actual():
    assert seguimiento.es_consulta_composicion(
        "Dime, sobre el concepto de imprevistos José, ¿qué gastos lo conforman?"
    )


def test_composicion_vacia_reintenta_sin_mes_actual_y_sin_tildes():
    sql_mes = (
        "SELECT fecha, concepto, monto_crc FROM finanzas__transacciones "
        "WHERE fecha >= DATE_TRUNC('month', CURRENT_DATE)"
    )
    sql_historia = (
        "SELECT fecha, concepto, monto_crc FROM finanzas__transacciones "
        "WHERE TRANSLATE(LOWER(concepto), 'áéíóúüñ', 'aeiouun') "
        "= 'imprevistos jose'"
    )
    consultas = []

    def ejecutar(_cliente, sql, limite=None):
        consultas.append(sql)
        if sql == sql_mes:
            return ["fecha", "concepto", "monto_crc"], []
        return ["fecha", "concepto", "monto_crc"], [
            ("2026-08-31", "Imprevistos Jose", 26900),
        ]

    with (
        patch.object(R.kpis, "cargar_kpis", return_value=[]),
        patch.object(R.nl2sql, "generar_sql", side_effect=[sql_mes, sql_historia]),
        patch.object(R.warehouse_ro, "ejecutar", side_effect=ejecutar),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50600000000",
            "Dime, sobre imprevistos José, qué gastos lo conforman?", [],
            fmt_solicitado=formato.TEXTO, ctx=CTX,
        )
    assert consultas == [sql_mes, sql_historia]
    assert "Imprevistos Jose" in respuesta.texto


def test_ajuste_de_195mil_usa_presupuesto_mensual_y_no_gemini():
    detalle = seguimiento.crear_estado(
        "qué gastos lo conforman", "SELECT detalle", "", "crc",
        ["linea_presupuesto_id", "concepto", "descripcion", "monto_crc"],
        [
            ("gas_imprevistos_jose", "Imprevistos Jose", "STEREN", 26900),
            ("gas_imprevistos_jose", "Imprevistos Jose", "EXTREME TECH", 195000),
            ("gas_imprevistos_jose", "Imprevistos Jose", "OTROS", 100725),
        ],
    )
    agregado = seguimiento.crear_estado(
        "cómo está contra su presupuesto", "SELECT kpi",
        "ejecucion_presupuesto_concepto", "crc",
        ["linea_id", "concepto", "presupuesto", "gastado", "disponible",
         "porcentaje_consumido"],
        [("gas_imprevistos_jose", "Imprevistos Jose", 130000, 322625,
          -192625, Decimal("248.1730769"))],
        previo=detalle,
    )
    historial = [
        {"rol": "assistant", "contenido": "detalle", "estado": detalle},
        {"rol": "assistant", "contenido": "presupuesto", "estado": agregado},
    ]
    texto, estado = seguimiento.resolver_ajuste(
        "Cuánto daría si quitamos la transacción de 195mil?", historial,
    )
    assert "₡127.625" in texto
    assert "₡2.375" in texto
    assert "98,17%" in texto
    assert estado["filas"][0][4] == "127625"


def test_seguimiento_de_presupuesto_devuelve_solo_el_concepto_anterior():
    estado = seguimiento.crear_estado(
        "qué conforma imprevistos", "SELECT detalle", "", "crc",
        ["linea_presupuesto_id", "concepto", "monto_crc"],
        [("gas_imprevistos_jose", "Imprevistos Jose", 1000)],
    )
    historial = [
        {"rol": "user", "contenido": "qué conforma imprevistos"},
        {"rol": "assistant", "contenido": "detalle", "sql": "SELECT detalle",
         "estado": estado},
    ]
    definicion = {
        "kpi": "ejecucion_presupuesto_concepto",
        "unidad": "crc",
        "formula_sql": (
            "SELECT linea_id, concepto, presupuesto, gastado, disponible, "
            "porcentaje_consumido FROM finanzas__presupuesto"
        ),
    }
    ejecutado = {}

    def ejecutar(_cliente, sql, limite=None):
        ejecutado["sql"] = sql
        return (
            ["linea_id", "concepto", "presupuesto", "gastado", "disponible",
             "porcentaje_consumido"],
            [("gas_imprevistos_jose", "Imprevistos Jose", 130000, 120000,
              10000, 92.3076923)],
        )

    with (
        patch.object(R.kpis, "cargar_kpis", return_value=[definicion]),
        patch.object(R.kpis, "planificar", return_value={
            "accion": "usar_kpi", "kpi": "ejecucion_presupuesto_concepto",
            "sql": definicion["formula_sql"], "mensaje": "",
        }),
        patch.object(R.warehouse_ro, "ejecutar", side_effect=ejecutar),
        patch.object(R.warehouse_ro, "leer_interno", return_value=[{
            "linea_id": "gas_imprevistos_jose", "concepto": "Imprevistos Jose",
            "minimo": 130000, "maximo": 130000,
        }]),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50600000000", "¿Y cómo está contra su presupuesto?",
            historial, fmt_solicitado=formato.TEXTO, ctx=CTX,
        )

    assert "_kpi.linea_id" in ejecutado["sql"]
    assert "Imprevistos Jose" in respuesta.texto
    assert respuesta.estado["filtros"]["linea_id"] == "gas_imprevistos_jose"


def test_seguimiento_contra_presupuesto_fuerza_kpi_sin_sortearlo_con_gemini():
    estado = seguimiento.crear_estado(
        "gastos de agosto", "SELECT detalle", "", "crc",
        ["fecha", "linea_presupuesto_id", "concepto", "monto_crc"],
        [("2026-08-31", "gas_imprevistos_jose", "Imprevistos Jose", 322625)],
    )
    historial = [{"rol": "assistant", "contenido": "detalle", "estado": estado,
                  "sql": "SELECT detalle"}]
    definicion = {
        "kpi": "ejecucion_presupuesto_concepto", "unidad": "crc",
        "formula_sql": (
            "SELECT linea_id, concepto, monto_mensual AS presupuesto, "
            "0 AS gastado, monto_mensual AS disponible, 0 AS porcentaje_consumido "
            "FROM presupuesto"
        ),
    }
    with (
        patch.object(R.kpis, "cargar_kpis", return_value=[definicion]),
        patch.object(R.kpis, "planificar",
                     side_effect=AssertionError("no debe llamar Gemini")),
        patch.object(R.warehouse_ro, "ejecutar", return_value=(
            ["linea_id", "concepto", "presupuesto", "gastado", "disponible",
             "porcentaje_consumido"],
            [("gas_imprevistos_jose", "Imprevistos Jose", 130000, 0, 130000, 0)],
        )),
        patch.object(R.warehouse_ro, "leer_interno", return_value=[{
            "linea_id": "gas_imprevistos_jose", "concepto": "Imprevistos Jose",
            "minimo": 130000, "maximo": 130000,
        }]),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50600000000", "¿Y cómo está contra su presupuesto?",
            historial, fmt_solicitado=formato.TEXTO, ctx=CTX,
        )
    assert respuesta.estado["kpi"] == "ejecucion_presupuesto_concepto"
