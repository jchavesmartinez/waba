"""Regresiones: el texto y el Excel deben compartir una sola verdad numerica."""

from types import SimpleNamespace
from datetime import datetime
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


def test_movimientos_se_agrupan_para_leerse_bien_en_whatsapp():
    columnas = ["fecha", "descripcion", "monto_usd", "titular", "categoria"]
    filas = [
        (datetime(2026, 8, 17, 15, 25), "GOOGLE SERVICES", 10, "Aline", "Suscripciones"),
        (datetime(2026, 8, 15, 15, 46), "DISNEY PLUS", 13.99, "Aline", "Suscripciones"),
    ]

    texto = nl2sql.redactar_respuesta(
        "Que conforma el total de suscripciones?", columnas, filas, unidad="usd",
    )

    assert texto.startswith("💳 *Suscripciones · Aline*\n2 movimientos\n")
    assert "• *GOOGLE SERVICES* — USD 10 · 17 ago" in texto
    assert "• *DISNEY PLUS* — USD 13,99 · 15 ago" in texto
    assert "Categoria:" not in texto
    assert " | " not in texto
    assert "USD 23,99" not in texto


def test_movimientos_muestran_total_solo_si_viene_del_sql():
    columnas = [
        "fecha", "descripcion", "monto_usd", "titular", "categoria", "total_general",
    ]
    filas = [
        (datetime(2026, 8, 17), "GOOGLE SERVICES", 10, "Aline", "Suscripciones", 23.99),
        (datetime(2026, 8, 15), "DISNEY PLUS", 13.99, "Aline", "Suscripciones", 23.99),
    ]

    texto = nl2sql.redactar_respuesta("detalle", columnas, filas, unidad="usd")

    assert "2 movimientos · *USD 23,99*" in texto


def test_ventas_usan_lista_movil_y_no_barras_verticales():
    columnas = [
        "fecha", "producto", "categoria", "cantidad", "precio_unitario",
        "monto_total", "vendedor", "sucursal", "metodo_pago",
    ]
    filas = [
        (datetime(2026, 7, 20), "Teclado Logitech K380", "Accesorios", 6,
         28000, 151200, "Maria Jimenez", "San Jose", "Efectivo"),
        (datetime(2026, 7, 20), "Memoria RAM 16GB", "Componentes", 2,
         65000, 130000, "Luis Mora", "Heredia Centro", "SINPE"),
    ]

    texto = nl2sql.redactar_respuesta("ultimas ventas", columnas, filas)

    assert texto.startswith("📋 *Producto*\n2 registros\n")
    assert "• *Teclado Logitech K380*" in texto
    assert "20 jul" in texto
    assert "Monto total: 151.200" in texto
    assert " | " not in texto
    assert "Solicite el Excel" not in texto


def test_pregunta_de_tendencia_recibe_analisis_y_no_datos_crudos(monkeypatch):
    capturado = {}

    def generar(_modelo, contenido, **kwargs):
        capturado["contenido"] = contenido
        capturado.update(kwargs)
        return SimpleNamespace(
            texto=(
                "📈 *Las ventas suben en el período, pero con alta variación.*\n\n"
                "• El último mes supera al primero en 25%.\n"
                "• Marzo tuvo la mayor caída mensual: 12%."
            ),
            truncada=False,
        )

    monkeypatch.setattr(nl2sql.llm, "generar_texto", generar)
    texto = nl2sql.redactar_respuesta(
        "Mes a mes, las ventas aumentan o bajan?",
        ["mes", "total", "crecimiento_pct"],
        [("2026-01", 100, None), ("2026-02", 140, 40), ("2026-03", 125, -10.7)],
        sql="SELECT mes, total, crecimiento_pct FROM tendencia",
    )

    assert texto.startswith("📈 *Las ventas suben")
    assert "Registro 1" not in texto
    assert "2026-02\t140\t40%" in capturado["contenido"]
    assert "no sugiera pedir Excel" in capturado["system"]


def test_si_falla_el_analista_se_da_una_conclusion_numerica_local(monkeypatch):
    def fallar(*_args, **_kwargs):
        raise RuntimeError("modelo no disponible")

    monkeypatch.setattr(nl2sql.llm, "generar_texto", fallar)
    texto = nl2sql.redactar_respuesta(
        "Analiza la tendencia de ventas",
        ["mes", "total"],
        [("2026-01", 100), ("2026-02", 140)],
    )

    assert "tendencia ascendente" in texto
    assert "+40%" in texto
    assert "Primer valor: 100; último valor: 140" in texto


def test_mes_a_mes_rechaza_sql_semanal():
    ok, motivo = nl2sql.validar_granularidad(
        "Dime, mes a mes, las ventas aumentan o bajan?",
        "SELECT DATE_TRUNC('week', fecha) AS sem, SUM(total) FROM ventas GROUP BY 1",
    )

    assert ok is False
    assert "mes a mes" in motivo


def test_mes_a_mes_acepta_sql_mensual():
    ok, motivo = nl2sql.validar_granularidad(
        "Dime, mes a mes, las ventas aumentan o bajan?",
        "SELECT DATE_TRUNC('month', fecha) AS mes, SUM(total) FROM ventas GROUP BY 1",
    )

    assert (ok, motivo) == (True, "")


def test_una_sola_celda_se_devuelve_directa_sin_llm():
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

    def generar(modelo, contenido, **kwargs):
        capturado.update(kwargs)
        capturado["contenido"] = contenido
        return SimpleNamespace(
            texto=(
                '{"accion":"usar_kpi","kpi":"plan_presupuesto",'
                '"sql":"SELECT 999 AS total","mensaje":""}'
            ),
            truncada=False,
        )
    ctx = SimpleNamespace(
        schema_text="sharepoint_db__presupuesto(categoria, monto_mensual, tipo)",
        tablas_reales={"sharepoint_db__presupuesto"},
        permitidas=[SimpleNamespace(
            tabla_logica="presupuesto",
            tabla_real="sharepoint_db__presupuesto",
        )],
    )
    original = kpis.llm.generar_texto
    kpis.llm.generar_texto = generar
    try:
        plan = kpis.planificar("cuanto tengo presupuestado", [definicion], ctx)
    finally:
        kpis.llm.generar_texto = original

    assert plan["accion"] == "usar_kpi"
    assert "SELECT 999" not in plan["sql"]
    assert "sharepoint_db__presupuesto" in plan["sql"]
    assert "deja 'sql' VACIO" in capturado["system"]


def test_ventas_de_hoy_no_usan_un_kpi_historico_estatico(monkeypatch):
    definicion = {
        "kpi": "ventas_totales",
        "nombre": "Ventas totales",
        "preguntas_ejemplo": "ventas totales",
        "formula_sql": "SELECT SUM(total) AS ventas FROM ventas",
    }
    ctx = SimpleNamespace(schema_text="ventas(fecha, total)")

    def no_debe_llamarse(*_args, **_kwargs):
        raise AssertionError("No debe elegirse un KPI estático para una fecha")

    monkeypatch.setattr(kpis.llm, "generar_texto", no_debe_llamarse)
    plan = kpis.planificar("ventas totales de hoy", [definicion], ctx)

    assert plan == {
        "accion": "sql_libre", "kpi": "", "sql": "", "mensaje": "",
    }


def test_decimal_de_excel_no_se_multiplica_por_diez_en_locale_tico():
    serie = pd.Series(["390037.5", "5.000", "1.250.000", "3,25"])
    valores = pd.to_numeric(_limpiar_numero(serie, coma_decimal=True)).tolist()
    assert valores == [390037.5, 5000.0, 1250000.0, 3.25]


def test_pedido_explicito_proyecta_solo_las_columnas_solicitadas():
    columnas = [
        "linea_id", "categoria", "concepto", "titular", "presupuesto",
        "gastado", "disponible", "porcentaje_consumido",
    ]
    filas = [
        ("gas_internet", "Servicios", "Internet", "", 29000, 29895.4, -895.4, 103.1),
    ]

    proyectadas, valores, cambio = nl2sql.proyectar_columnas_solicitadas(
        "No pongas la linea id, solo categoria, concepto y porcentaje, nada mas",
        columnas,
        filas,
    )

    assert cambio is True
    assert proyectadas == ["categoria", "concepto", "porcentaje_consumido"]
    assert valores == [("Servicios", "Internet", 103.1)]


def test_respuesta_compacta_respeta_categoria_concepto_y_porcentaje():
    columnas = [
        "linea_id", "categoria", "concepto", "titular", "presupuesto",
        "gastado", "disponible", "porcentaje_consumido",
    ]
    filas = [
        ("gas_internet", "Servicios", "Internet", "", 29000, 29895.4, -895.4, 103.1),
        ("gas_celular_jose", "Servicios", "Celular Jose", "", 11000, 21120.06, -10120.06, 192),
    ]

    texto = nl2sql.redactar_respuesta(
        "La respuesta es categoria, concepto y porcentaje, nada mas",
        columnas,
        filas,
        unidad="colones y %",
    )

    assert "*Categoria | Concepto | Porcentaje consumido*" in texto
    assert "• Servicios | Internet | 103,1%" in texto
    assert "Linea id" not in texto
    assert "Presupuesto" not in texto
    assert "Gastado" not in texto


def test_ajuste_de_columnas_conserva_contexto_para_seleccionar_kpi():
    relleno = [
        {
            "kpi": f"ventas_{i}",
            "nombre": "Ventas por fecha y saldo",
            "preguntas_ejemplo": "fecha saldo",
            "formula_sql": "SELECT 1 AS dato FROM ventas",
        }
        for i in range(30)
    ]
    objetivo = {
        "kpi": "flujo_caja_semanal",
        "nombre": "Flujo de caja semanal",
        "preguntas_ejemplo": "flujo de caja por semana",
        "formula_sql": "SELECT 1 AS dato FROM ventas",
    }
    capturado = {}

    def generar(modelo, contenido, **kwargs):
        capturado.update(kwargs)
        capturado["contenido"] = contenido
        return SimpleNamespace(
            texto='{"accion":"sql_libre","kpi":"","sql":"","mensaje":""}',
            truncada=False,
        )

    ctx = SimpleNamespace(
        schema_text="ventas(dato)",
        tablas_reales={"ventas"},
        permitidas=[SimpleNamespace(tabla_logica="ventas", tabla_real="ventas")],
    )
    historial = [
        {"rol": "user", "contenido": "mostrame el flujo de caja semanal"},
        {"rol": "assistant", "contenido": "Resultado", "sql": "SELECT 1 FROM ventas"},
    ]
    original = kpis.llm.generar_texto
    kpis.llm.generar_texto = generar
    try:
        kpis.planificar(
            "No pongas otras columnas; la respuesta es fecha y saldo, nada mas",
            relleno + [objetivo],
            ctx,
            historial=historial,
        )
    finally:
        kpis.llm.generar_texto = original

    assert "kpi: flujo_caja_semanal" in capturado["contenido"]
