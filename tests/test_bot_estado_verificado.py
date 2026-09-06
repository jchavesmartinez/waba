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


def test_kpi_con_linea_conserva_filtro_independiente_de_moneda():
    sql, aplicados = kpis.parametrizar_sql(
        "SELECT linea_id, concepto, moneda, SUM(gastado) AS gastado "
        "FROM movimientos GROUP BY 1, 2, 3",
        {"linea_id": "gas_comedera", "concepto": "Comedera", "moneda": "CRC"},
    )
    assert aplicados == {"linea_id": "gas_comedera", "moneda": "CRC"}
    assert "_kpi.linea_id" in sql
    assert "_kpi.moneda" in sql


def test_kpi_filtra_descripcion_parcial_y_no_pierde_comercio():
    sql, aplicados = kpis.parametrizar_sql(
        "SELECT descripcion, moneda, SUM(monto_neto) AS gasto_neto "
        "FROM movimientos GROUP BY 1, 2",
        {"descripcion": "Walmart", "moneda": "CRC"},
    )
    assert aplicados == {"moneda": "CRC", "descripcion": "Walmart"}
    assert "ILIKE '%walmart%'" in sql


def test_kpi_no_ignora_un_filtro_que_su_salida_no_admite():
    import pytest
    with pytest.raises(ValueError, match="descripcion"):
        kpis.parametrizar_sql(
            "SELECT moneda, SUM(monto_neto) AS gasto_neto "
            "FROM movimientos GROUP BY 1",
            {"descripcion": "Walmart"},
        )


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


def test_periodos_relativos_se_resuelven_con_fecha_del_negocio(monkeypatch):
    from datetime import date
    monkeypatch.setattr(seguimiento, "fecha_local", lambda: date(2026, 9, 5))
    assert seguimiento.periodo_explicito("gastos de ayer") == {
        "inicio": "2026-09-04", "fin_inclusivo": "2026-09-04",
        "fin_exclusivo": "2026-09-05", "granularidad": "dia",
    }
    assert seguimiento.periodo_explicito("gastos esta semana")["fin_exclusivo"] == "2026-09-06"
    assert seguimiento.periodo_explicito("y en octubre?") == {
        "inicio": "2026-10-01", "fin_inclusivo": "2026-10-31",
        "fin_exclusivo": "2026-11-01", "granularidad": "mes",
    }


def test_estado_conserva_operacion_y_agrupacion_del_resultado():
    estado = seguimiento.crear_estado(
        "¿Cuántos movimientos hubo por concepto en agosto 2026?",
        "SELECT concepto, COUNT(*) AS cantidad FROM movimientos GROUP BY concepto",
        "", "", ["concepto", "cantidad"], [("Comedera", 17)],
    )
    assert estado["operacion"] == "conteo"
    assert estado["agrupacion"] == "concepto"


def test_contrato_cambia_de_categoria_a_ranking_por_concepto():
    estado = seguimiento.crear_estado(
        "¿Cuánto gasté en alimentación en agosto 2026?",
        "SELECT categoria, moneda, SUM(monto) AS gastado FROM movimientos "
        "WHERE fecha >= DATE '2026-08-01' AND fecha < DATE '2026-09-01' "
        "GROUP BY categoria, moneda",
        "", "CRC", ["categoria", "moneda", "gastado"],
        [("Alimentacion", "CRC", 793457.09)],
    )
    contrato = seguimiento.contrato_seguimiento(
        "¿Y cuál fue el concepto con mayor gasto dentro de esa categoría?",
        [{"rol": "assistant", "contenido": "resultado", "estado": estado}],
        {"relacion": "nueva"},
    )
    assert contrato["operacion"] == "ranking"
    assert contrato["agrupacion"] == "concepto"
    assert contrato["filtros"]["categoria"] == "Alimentacion"
    assert contrato["periodo"]["inicio"] == "2026-08-01"


def test_contrato_limita_pronombre_a_entidades_del_resultado_anterior():
    estado = seguimiento.crear_estado(
        "Tres categorías con mayor gasto en agosto 2026",
        "SELECT categoria, SUM(monto) AS gastado FROM movimientos GROUP BY categoria",
        "", "CRC", ["categoria", "gastado"],
        [("Vivienda", 1700), ("Alimentacion", 790), ("Otros", 500)],
    )
    contrato = seguimiento.contrato_seguimiento(
        "¿Cuál de esas tuvo mayor exceso?",
        [{"rol": "assistant", "contenido": "resultado", "estado": estado}],
    )
    assert contrato["metrica"] == "exceso"
    assert contrato["entidades_previas"] == ["Vivienda", "Alimentacion", "Otros"]


def test_suma_de_detalle_anterior_se_resuelve_localmente_por_moneda():
    estado = seguimiento.crear_estado(
        "Qué movimientos conforman comidas afuera en agosto 2026",
        "SELECT moneda, monto FROM movimientos",
        "", "", ["descripcion", "moneda", "monto"],
        [("A", "CRC", 100), ("B", "CRC", 250), ("C", "USD", 4)],
    )
    resultado = seguimiento.resolver_sobre_resultado(
        "¿Y cuánto suman?",
        [{"rol": "assistant", "contenido": "detalle", "estado": estado}],
    )
    assert resultado["columnas"] == ["moneda", "gastado"]
    assert ("CRC", Decimal("350")) in resultado["filas"]
    assert ("USD", Decimal("4")) in resultado["filas"]


def test_monto_de_fila_identificada_se_proyecta_sin_nueva_consulta():
    estado = seguimiento.crear_estado(
        "Clasificación de Roga del 4 de septiembre de 2026",
        "SELECT fecha, descripcion, categoria, concepto, moneda, monto FROM movimientos",
        "", "", ["fecha", "descripcion", "categoria", "concepto", "moneda", "monto"],
        [("2026-09-04", "ROGA", "Otros", "Salud imprevistos", "CRC", 68000)],
    )
    resultado = seguimiento.resolver_sobre_resultado(
        "¿Cuál fue el monto?",
        [{"rol": "assistant", "contenido": "detalle", "estado": estado}],
    )
    assert resultado["columnas"] == ["monto"]
    assert resultado["filas"] == [(68000,)]


def test_cambio_de_periodo_reutiliza_sql_y_conserva_conteo():
    sql = (
        "SELECT COUNT(*) AS cantidad FROM movimientos "
        "WHERE concepto = 'Comedera' AND fecha >= DATE '2026-08-01' "
        "AND fecha < DATE '2026-09-01'"
    )
    estado = seguimiento.crear_estado(
        "¿Cuántos gastos de Comedera hubo en agosto 2026?", sql, "", "",
        ["cantidad"], [(17,)],
    )
    historial = [{"rol": "assistant", "contenido": "17", "estado": estado}]
    contrato = seguimiento.contrato_seguimiento(
        "¿Y en septiembre de 2026?", historial,
    )
    nuevo = seguimiento.sql_con_periodo_nuevo(
        "¿Y en septiembre de 2026?", historial, contrato,
    )
    assert "COUNT(*)" in nuevo
    assert "2026-09-01" in nuevo
    assert "2026-10-01" in nuevo
    assert "2026-08-01" not in nuevo


def test_ranking_sin_dimension_pide_aclaracion_y_guarda_intencion():
    estado = seguimiento.crear_estado(
        "Cuanto gaste en alimentacion en agosto 2026", "SELECT 1", "", "CRC",
        ["categoria", "moneda", "gastado"], [("Alimentacion", "CRC", 763007)],
    )
    resultado = seguimiento.aclaracion_necesaria(
        "Y de eso, que fue lo que mas gaste?",
        [{"rol": "assistant", "contenido": "total", "estado": estado}],
    )
    assert "concepto" in resultado[0].lower()
    assert resultado[1]["pendiente"]["operacion"] == "ranking"
    assert resultado[1]["filtros"]["categoria"] == "Alimentacion"


def test_aclaracion_posterior_recupera_ranking_y_dimension_concepto():
    estado = seguimiento.crear_estado(
        "Cuanto gaste en alimentacion en agosto 2026", "SELECT 1", "", "CRC",
        ["categoria", "moneda", "gastado"], [("Alimentacion", "CRC", 763007)],
    )
    estado["pendiente"] = {"operacion": "ranking", "metrica": "gastado"}
    contrato = seguimiento.contrato_seguimiento(
        "Me refiero a las cosas del presupuesto, como Comedera",
        [{"rol": "assistant", "contenido": "aclare", "estado": estado}],
        {"relacion": "nueva"},
    )
    assert contrato["operacion"] == "ranking"
    assert contrato["agrupacion"] == "concepto"
    assert contrato["periodo"]["inicio"] == "2026-08-01"


def test_todo_eso_cuanto_da_suma_el_detalle_anterior():
    estado = seguimiento.crear_estado(
        "Que compras forman eso", "SELECT detalle", "", "CRC",
        ["fecha", "descripcion", "moneda", "monto_neto"],
        [("2026-09-01", "A", "CRC", 4050),
         ("2026-09-02", "B", "CRC", 20000)],
    )
    resultado = seguimiento.resolver_sobre_resultado(
        "Todo eso cuanto da?",
        [{"rol": "assistant", "contenido": "detalle", "estado": estado}],
    )
    assert resultado["filas"] == [("CRC", Decimal("24050"))]


def test_solo_cuanto_me_queda_proyecta_disponible():
    estado = seguimiento.crear_estado(
        "Cuanto llevo", "SELECT presupuesto", "ejecucion", "colones",
        ["concepto", "presupuesto", "gastado", "disponible", "porcentaje_consumido"],
        [("Comidas afuera", 220000, 42350, 177650, Decimal("19.3"))],
    )
    resultado = seguimiento.resolver_sobre_resultado(
        "No me diga porcentajes, solo cuanto me queda",
        [{"rol": "assistant", "contenido": "resultado", "estado": estado}],
    )
    assert resultado["columnas"] == ["disponible"]
    assert resultado["filas"] == [(177650,)]


def test_eso_es_mucho_responde_contra_presupuesto_sin_nueva_consulta():
    estado = seguimiento.crear_estado(
        "Cuanto llevo", "SELECT presupuesto", "ejecucion", "colones",
        ["presupuesto", "gastado", "disponible", "porcentaje_consumido"],
        [(220000, 42350, 177650, Decimal("19.3"))],
    )
    resultado = seguimiento.resolver_sobre_resultado(
        "Eso es mucho o no?",
        [{"rol": "assistant", "contenido": "resultado", "estado": estado}],
    )
    assert "dentro del presupuesto" in resultado["texto"]
    assert "19,3%" in resultado["texto"]


def test_antes_de_esa_usa_fecha_como_limite_y_no_como_descripcion():
    estado = seguimiento.crear_estado(
        "La mas cara", "SELECT detalle", "", "CRC",
        ["fecha", "descripcion", "concepto", "moneda", "monto_neto"],
        [("2026-09-02", "Pinchos", "Comidas afuera", "CRC", 20000)],
    )
    historial = [{"rol": "assistant", "contenido": "fila", "estado": estado}]
    contrato = seguimiento.contrato_seguimiento(
        "Y antes de esa cuanto habia gastado?", historial,
        {"relacion": "seguimiento"},
    )
    assert contrato["operacion"] == "total"
    assert contrato["relacion_temporal"] == "antes"
    assert contrato["referencia_temporal"] == {"fecha": "2026-09-02"}
    assert "descripcion" not in contrato["filtros"]


def test_seleccionar_fila_no_reduce_el_periodo_heredado_a_un_dia():
    previo = seguimiento.crear_estado(
        "Detalle de septiembre 2026", "SELECT detalle", "", "CRC",
        ["fecha", "descripcion", "monto_neto"],
        [("2026-09-01", "A", 100), ("2026-09-02", "B", 200)],
    )
    seleccionado = seguimiento.crear_estado(
        "La mas cara", "SELECT detalle ORDER BY monto DESC LIMIT 1", "", "CRC",
        ["fecha", "descripcion", "monto_neto"],
        [("2026-09-02", "B", 200)], previo=previo,
    )
    assert seleccionado["periodo"] == previo["periodo"]


def test_sumar_esas_conserva_where_del_conteo_anterior_en_contrato():
    sql = (
        "SELECT COUNT(*) AS cantidad FROM movimientos "
        "WHERE concepto = 'Comidas afuera' AND fecha > DATE '2026-09-02'"
    )
    estado = seguimiento.crear_estado(
        "Cuantas compras fueron despues", sql, "", "CRC",
        ["cantidad"], [(2,)],
    )
    contrato = seguimiento.contrato_seguimiento(
        "Y si sumo esas, cuanto es?",
        [{"rol": "assistant", "contenido": "2", "estado": estado}],
        {"relacion": "seguimiento"},
    )
    assert contrato["operacion"] == "total"
    assert contrato["referencia_conjunto"] is True
    assert sql in seguimiento.instruccion_contrato(contrato)


def test_toda_categoria_elimina_filtro_de_concepto_anterior():
    estado = seguimiento.crear_estado(
        "Presupuesto de Comedera", "SELECT 1", "", "CRC",
        ["categoria", "concepto", "presupuesto"],
        [("Alimentacion", "Comedera", 400000)],
    )
    contrato = seguimiento.contrato_seguimiento(
        "No, perdon, para toda Alimentacion",
        [{"rol": "assistant", "contenido": "400000", "estado": estado}],
        {"relacion": "modificacion", "filtros_actuales": {
            "categoria": "Alimentacion",
        }},
    )
    assert contrato["filtros"]["categoria"] == "Alimentacion"
    assert "concepto" not in contrato["filtros"]


def test_concepto_nuevo_reemplaza_linea_y_concepto_anteriores():
    estado = seguimiento.crear_estado(
        "Conceptos excedidos", "SELECT 1", "", "CRC",
        ["linea_id", "categoria", "concepto", "gastado"],
        [("gas_comedera", "Alimentacion", "Comedera", 457737)],
    )
    contrato = seguimiento.contrato_seguimiento(
        "Y las compras de Comidas afuera cuales fueron?",
        [{"rol": "assistant", "contenido": "Comedera", "estado": estado}],
        {"relacion": "seguimiento", "filtros_actuales": {
            "concepto": "Comidas afuera",
        }},
    )
    assert contrato["filtros"]["concepto"] == "Comidas afuera"
    assert "linea_id" not in contrato["filtros"]


def test_referencia_selecciona_mayor_y_conserva_fila_para_seguimiento():
    estado = seguimiento.crear_estado(
        "gastos por concepto", "SELECT 1", "ejecucion", "CRC",
        ["concepto", "gastado", "presupuesto"],
        [("A", 100, 200), ("B", 900, 1000)],
    )
    resultado = seguimiento.resolver_referencia(
        "¿Cuál fue el que más gasté?",
        [{"rol": "assistant", "contenido": "resultado", "estado": estado}],
    )
    assert resultado["filas"] == [("B", 900, 1000)]
    assert resultado["estado"]["seleccion"]["criterio"] == "mayor"


def test_total_simple_consolida_movimientos_por_moneda():
    columnas, filas = R._consolidar_total_si_corresponde(
        "¿Cuánto gasté en alimentación?",
        ["comercio", "moneda", "monto"],
        [("A", "CRC", 100), ("B", "CRC", 250), ("C", "USD", 4)],
    )
    assert columnas == ["moneda", "gastado"]
    assert ("CRC", 350) in filas
    assert ("USD", 4) in filas


def test_agregados_separados_por_moneda_no_se_rechazan_como_mezcla():
    ok, motivo = seguimiento.validar_resultado(
        ["moneda", "gasto_neto"], [("CRC", 1000), ("USD", 5)],
    )
    assert ok is True
    assert motivo == ""


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


def test_presupuesto_multiplicado_con_alias_se_corrige_junto_con_exceso():
    columnas = ["concepto", "monto_presupuestado", "gasto_real", "exceso"]
    filas, cambios = seguimiento.reconciliar_presupuesto_fuente(
        columnas,
        [("Comedera", 13600000, 457737, -13142263)],
        {"concepto:comedera": Decimal("400000")},
    )
    assert len(cambios) == 1
    assert filas == [("Comedera", Decimal("400000"), 457737, Decimal("57737"))]


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


def test_compras_que_forman_eso_conserva_el_modo_detalle():
    assert seguimiento.es_consulta_composicion("¿Y qué compras forman eso?")
    assert seguimiento.operacion_resultado(
        "¿Y qué compras forman eso?",
        columnas=["fecha", "descripcion", "monto_neto"],
    ) == "detalle"


def test_la_mas_cara_selecciona_el_mayor_del_detalle_anterior():
    estado = seguimiento.crear_estado(
        "compras de comidas afuera", "SELECT detalle", "", "CRC",
        ["fecha", "descripcion", "monto_neto", "moneda"],
        [
            ("2026-09-01", "NINA CAFE", 4050, "CRC"),
            ("2026-09-02", "Pinchos el Pelon", 20000, "CRC"),
        ],
    )
    salida = seguimiento.resolver_referencia(
        "¿La más cara cuál fue?",
        [{"rol": "assistant", "contenido": "detalle", "estado": estado}],
    )
    assert salida is not None
    assert salida["filas"] == [("2026-09-02", "Pinchos el Pelon", 20000, "CRC")]


def test_agregacion_temporal_reutiliza_el_detalle_verificado():
    detalle = seguimiento.crear_estado(
        "compras", "SELECT fecha, descripcion, monto_neto, moneda FROM movimientos",
        "detalle", "CRC",
        ["fecha", "descripcion", "monto_neto", "moneda"],
        [("2026-09-02", "Pinchos", 20000, "CRC")],
    )
    contrato = {
        "operacion": "total",
        "relacion_temporal": "antes",
        "referencia_temporal": {"fecha": "2026-09-02"},
        "estado_previo": detalle,
    }
    sql = seguimiento.sql_temporal_desde_estado(contrato)
    assert "SUM(_seguimiento.monto_neto) AS gastado" in sql
    assert "CAST(_seguimiento.fecha AS DATE) < CAST('2026-09-02' AS DATE)" in sql


def test_proyeccion_de_una_fila_conserva_la_base_de_detalle():
    detalle = seguimiento.crear_estado(
        "compras", "SELECT fecha, descripcion, monto_neto, moneda FROM movimientos",
        "detalle", "CRC",
        ["fecha", "descripcion", "monto_neto", "moneda"],
        [("2026-09-02", "Pinchos", 20000, "CRC")],
    )
    proyectado = seguimiento.crear_estado(
        "¿qué día fue?", detalle["sql"], "detalle", "CRC",
        ["fecha"], [("2026-09-02",)], previo=detalle,
    )
    assert proyectado["sql_detalle"] == detalle["sql"]
    assert proyectado["campos_detalle"]["monto"] == "monto_neto"


def test_cuantas_compras_es_un_conteo_y_no_un_total():
    assert seguimiento.operacion_resultado("¿Cuántas compras fueron después?") == "conteo"


def test_composicion_desde_exceso_pide_detalle_y_monto():
    previo = seguimiento.crear_estado(
        "conceptos que excedieron", "SELECT concepto, exceso", "", "CRC",
        ["categoria", "concepto", "presupuesto", "gastado", "exceso"],
        [("Alimentacion", "Comidas afuera", 220000, 305270, 85270)],
    )
    contrato = seguimiento.contrato_seguimiento(
        "¿Y las compras de Comidas afuera cuáles fueron?",
        [{"rol": "assistant", "contenido": "exceso", "estado": previo}],
        {"relacion": "seguimiento", "filtros_actuales": {"concepto": "Comidas afuera"}},
    )
    assert contrato["operacion"] == "detalle"
    assert contrato["metrica"] == "gastado"


def test_contrato_usa_operacion_metrica_y_entidad_semanticas_del_plan():
    previo = seguimiento.crear_estado(
        "alimentacion", "SELECT categoria, gastado FROM movimientos", "", "CRC",
        ["categoria", "gastado"], [("Alimentacion", 1000)],
    )
    contrato = seguimiento.contrato_seguimiento(
        "y por concepto",
        [{"rol": "assistant", "contenido": "resultado", "estado": previo}],
        {
            "relacion": "seguimiento", "operacion": "desglose",
            "metrica": "gastado", "entidad": "concepto",
            "filtros_actuales": {},
        },
    )
    assert contrato["operacion"] == "desglose"
    assert contrato["metrica"] == "gastado"
    assert contrato["agrupacion"] == "concepto"


def test_planificador_exige_y_normaliza_el_contrato_semantico():
    plan = kpis._parsear(
        '{"operacion":"total","metrica":"gastado","entidad":"categoria",'
        '"relacion":"nueva","heredar_filtros":[],"filtros_actuales":{},'
        '"heredar_periodo":false,"heredar_kpi":false,"accion":"sql_libre",'
        '"kpi":"","sql":"","mensaje":""}'
    )
    assert plan["operacion"] == "total"
    assert plan["metrica"] == "gastado"
    assert plan["entidad"] == "categoria"


def test_mencionar_un_concepto_de_la_lista_lo_convierte_en_filtro():
    previo = seguimiento.crear_estado(
        "excesos", "SELECT concepto", "", "CRC",
        ["categoria", "concepto", "monto_presupuestado", "monto_ejecutado"],
        [
            ("Alimentacion", "Comedera", 400000, 457737),
            ("Alimentacion", "Comidas afuera", 220000, 305270),
        ],
    )
    contrato = seguimiento.contrato_seguimiento(
        "las compras de comidas afuera cuales fueron",
        [{"rol": "assistant", "contenido": "excesos", "estado": previo}],
        {"relacion": "seguimiento", "filtros_actuales": {}},
    )
    assert contrato["filtros"]["concepto"] == "Comidas afuera"


def test_reconciliacion_actualiza_exceso_con_monto_ejecutado():
    columnas = ["concepto", "monto_presupuestado", "monto_ejecutado", "exceso"]
    filas, _ = seguimiento.reconciliar_presupuesto_fuente(
        columnas, [("Comidas afuera", 7700000, 305270, -7394730)],
        {"concepto:comidas afuera": 220000},
    )
    assert filas == [("Comidas afuera", 220000, 305270, 85270)]
