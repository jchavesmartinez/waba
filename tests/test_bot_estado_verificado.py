from decimal import Decimal
import logging
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


def test_ejecucion_deja_sql_copiable_y_metricas_en_logs(caplog):
    sql = "SELECT concepto,\n       SUM(monto_crc) AS total\nFROM finanzas__transacciones"
    with (
        patch.object(R.config, "BOT_LOG_SQL", True),
        patch.object(R.config, "BOT_LOG_SQL_MAX_CHARS", 20000),
        patch.object(
            R.warehouse_ro,
            "ejecutar",
            return_value=(["concepto", "total"], [("Comida", 25000)]),
        ) as ejecutar,
        caplog.at_level(logging.INFO, logger="fachavi.bot.responder"),
    ):
        columnas, filas, query_id = R._ejecutar_con_auditoria(
            CLIENTE, sql, 100, "sql_libre",
        )

    ejecutar.assert_called_once_with(CLIENTE, sql, limite=100)
    assert columnas == ["concepto", "total"]
    assert filas == [("Comida", 25000)]
    assert len(query_id) == 10
    assert f"SQL_AUDIT inicio id={query_id}" in caplog.text
    assert "origen=sql_libre limite=100" in caplog.text
    assert (
        "sql=SELECT concepto, SUM(monto_crc) AS total "
        "FROM finanzas__transacciones"
    ) in caplog.text
    assert f"SQL_AUDIT fin id={query_id}" in caplog.text
    assert "filas=1 columnas=2" in caplog.text


def test_auditoria_sql_se_puede_desactivar(caplog):
    with (
        patch.object(R.config, "BOT_LOG_SQL", False),
        patch.object(R.warehouse_ro, "ejecutar", return_value=(["x"], [(1,)])),
        caplog.at_level(logging.INFO, logger="fachavi.bot.responder"),
    ):
        R._ejecutar_con_auditoria(CLIENTE, "SELECT 1 AS x", None, "kpi:prueba")

    assert "SQL_AUDIT" not in caplog.text


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


def test_plan_nuevo_no_hereda_estado_anterior():
    previo = seguimiento.crear_estado(
        "imprevistos jose", "SELECT detalle", "", "crc",
        ["concepto", "titular", "monto_crc"],
        [("Imprevistos Jose", "Jose", 1000)],
    )
    historial = [{"rol": "assistant", "contenido": "detalle", "estado": previo}]
    contexto = seguimiento.contexto_segun_plan(historial, {
        "relacion": "nueva",
        "heredar_filtros": ["concepto", "titular"],
        "heredar_periodo": True,
        "heredar_kpi": True,
    })
    assert contexto == {}


def test_modificacion_ignora_filtros_no_habilitados_para_kpis():
    previo = seguimiento.crear_estado(
        "imprevistos jose", "SELECT detalle", "detalle_gastos", "crc",
        ["concepto", "titular", "monto_crc"],
        [("Imprevistos Jose", "Jose", 1000)],
    )
    historial = [{"rol": "assistant", "contenido": "detalle", "estado": previo}]
    contexto = seguimiento.contexto_segun_plan(historial, {
        "relacion": "modificacion",
        "heredar_filtros": ["titular", "categoria"],
        "heredar_periodo": False,
        "heredar_kpi": True,
    })
    assert contexto == {
        "kpi": "detalle_gastos",
        "filtros": {},
        "periodo": {},
    }


def test_pregunta_completa_nueva_no_arrastra_titular_anterior():
    previo = seguimiento.crear_estado(
        "imprevistos jose en agosto 2026", "SELECT detalle", "", "crc",
        ["fecha", "concepto", "titular", "monto_crc"],
        [("2026-08-31", "Imprevistos Jose", "Jose", 1000)],
    )
    historial = [
        {"rol": "user", "contenido": "imprevistos jose en agosto 2026"},
        {"rol": "assistant", "contenido": "detalle", "sql": "SELECT detalle",
         "estado": previo},
    ]
    pregunta_sql = {}
    sql_alimentacion = (
        "SELECT concepto, SUM(monto_crc) AS gastado "
        "FROM finanzas__transacciones "
        "WHERE LOWER(concepto) = 'alimentacion' "
        "AND fecha >= DATE '2026-08-01' AND fecha < DATE '2026-09-01' "
        "GROUP BY concepto"
    )

    def generar(pregunta, _schema, **kwargs):
        pregunta_sql["texto"] = pregunta
        pregunta_sql["historial"] = kwargs.get("historial")
        return sql_alimentacion

    with (
        patch.object(R.kpis, "cargar_kpis", return_value=[{
            "kpi": "ejecucion_presupuesto_concepto", "unidad": "crc",
        }]),
        patch.object(R.kpis, "planificar", return_value={
            "relacion": "nueva",
            "heredar_filtros": [],
            "heredar_periodo": False,
            "heredar_kpi": False,
            "accion": "sql_libre", "kpi": "", "sql": "", "mensaje": "",
        }),
        patch.object(R.nl2sql, "generar_sql", side_effect=generar),
        patch.object(R.warehouse_ro, "ejecutar", return_value=(
            ["concepto", "gastado"], [("Alimentacion", 250000)],
        )),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50683919244",
            "¿Cuánto gasté en alimentación en agosto 2026 y cómo se compara "
            "contra su presupuesto?",
            historial, fmt_solicitado=formato.TEXTO, ctx=CTX,
        )

    assert "CONTEXTO ESTRUCTURADO OBLIGATORIO" not in pregunta_sql["texto"]
    assert pregunta_sql["historial"] == []
    assert "jose" not in respuesta.sql.lower()
    assert "alimentacion" in respuesta.sql.lower()


def test_periodo_explicito_mantiene_kpi_consolidado_y_no_degrada_a_sql_libre():
    ctx = SimpleNamespace(
        error_lectura=False,
        tablas_reales={"finanzas__transacciones", "googledrive_db__gastos_manuales"},
        schema_text=(
            "finanzas__transacciones(fecha_transaccion, comercio, cuenta_contable, monto); "
            "googledrive_db__gastos_manuales(fecha, descripcion, categoria, monto)"
        ),
        permitidas=[],
    )
    formula = (
        "WITH movimientos AS ("
        "SELECT fecha_transaccion AS fecha, comercio, cuenta_contable AS categoria, monto "
        "FROM finanzas__transacciones "
        "UNION ALL "
        "SELECT fecha, descripcion AS comercio, categoria, monto "
        "FROM googledrive_db__gastos_manuales) "
        "SELECT categoria, comercio, SUM(monto) AS gastado "
        "FROM movimientos "
        "WHERE fecha >= DATE '{{periodo_inicio}}' "
        "AND fecha < DATE '{{periodo_fin}}' "
        "GROUP BY categoria, comercio"
    )
    definicion = {
        "kpi": "gasto_por_comercio", "unidad": "colones",
        "formula_sql": formula,
    }
    ejecutado = {}

    def ejecutar(_cliente, sql, limite=None):
        ejecutado["sql"] = sql
        return ["categoria", "comercio", "gastado"], [
            ("Alimentacion", "Pinchos", 22500),
        ]

    with (
        patch.object(R.kpis, "cargar_kpis", return_value=[definicion]),
        patch.object(R.kpis, "planificar", return_value={
            "relacion": "nueva", "heredar_filtros": [],
            "filtros_actuales": {"categoria": "Alimentacion"},
            "heredar_periodo": False, "heredar_kpi": False,
            "accion": "usar_kpi", "kpi": "gasto_por_comercio",
            "sql": formula, "mensaje": "",
        }),
        patch.object(R.warehouse_ro, "ejecutar", side_effect=ejecutar),
    ):
        respuesta = R._responder_datos(
            CLIENTE, "50683919244",
            "Podrías agregar por comercio los gastos de alimentación para agosto 2026",
            [], fmt_solicitado=formato.TEXTO, ctx=ctx,
        )

    assert "finanzas__transacciones" in ejecutado["sql"]
    assert "googledrive_db__gastos_manuales" in ejecutado["sql"]
    assert "CAST('2026-08-01' AS DATE)" in ejecutado["sql"]
    assert "CAST('2026-09-01' AS DATE)" in ejecutado["sql"]
    assert "_kpi.categoria" in ejecutado["sql"]
    assert "Pinchos" in respuesta.texto


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


def test_kpi_parametrizado_reemplaza_rango_y_filtro_actual_sin_reescribir_joins():
    sql, aplicados = kpis.parametrizar_sql(
        "SELECT categoria, comercio, SUM(monto) AS gastado "
        "FROM finanzas__movimientos "
        "WHERE fecha >= DATE '{{periodo_inicio}}' "
        "AND fecha < DATE '{{periodo_fin}}' "
        "GROUP BY categoria, comercio",
        {"categoria": "Alimentacion"},
        {"inicio": "2026-08-01", "fin_exclusivo": "2026-09-01", "granularidad": "mes"},
    )
    assert "{{periodo_" not in sql
    assert "CAST('2026-08-01' AS DATE)" in sql
    assert "CAST('2026-09-01' AS DATE)" in sql
    assert "_kpi.categoria" in sql
    assert aplicados["periodo"]["inicio"] == "2026-08-01"
    assert aplicados["categoria"] == "Alimentacion"


def test_periodo_explicito_extrae_rango_mensual():
    assert seguimiento.periodo_explicito("gastos de agosto 2026") == {
        "inicio": "2026-08-01",
        "fin_inclusivo": "2026-08-31",
        "fin_exclusivo": "2026-09-01",
        "granularidad": "mes",
    }


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
            "relacion": "seguimiento",
            "heredar_filtros": ["linea_id", "concepto"],
            "heredar_periodo": False,
            "heredar_kpi": False,
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


def test_seguimiento_contra_presupuesto_lo_decide_el_planificador():
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
        patch.object(R.kpis, "planificar", return_value={
            "relacion": "seguimiento",
            "heredar_filtros": ["linea_id", "concepto"],
            "heredar_periodo": True,
            "heredar_kpi": False,
            "accion": "usar_kpi", "kpi": "ejecucion_presupuesto_concepto",
            "sql": definicion["formula_sql"].replace(
                "FROM presupuesto", "FROM finanzas__presupuesto",
            ),
            "mensaje": "",
        }) as planificar,
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
    planificar.assert_called_once()
    assert respuesta.estado["kpi"] == "ejecucion_presupuesto_concepto"
