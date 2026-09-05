from types import SimpleNamespace

from bot import nl2sql


def test_generar_sql_exige_json_y_razonamiento_minimo(monkeypatch):
    llamadas = []

    def generar(_modelo, _contenido, **kwargs):
        llamadas.append(kwargs)
        return SimpleNamespace(
            texto='{"sql":"SELECT concepto FROM finanzas__transacciones;"}',
            truncada=False,
            finish_reason="STOP",
        )

    monkeypatch.setattr(nl2sql.llm, "generar_texto", generar)

    sql = nl2sql.generar_sql(
        "¿Qué gastos conforman imprevistos José?",
        "finanzas__transacciones(concepto)",
    )

    assert sql == "SELECT concepto FROM finanzas__transacciones"
    assert llamadas[0]["thinking_level"] == "minimal"
    assert llamadas[0]["response_schema"] == nl2sql._SQL_SCHEMA
    assert llamadas[0]["max_tokens"] == nl2sql._MAX_TOKENS_SQL


def test_generar_sql_no_entrega_fragmento_truncado(monkeypatch):
    monkeypatch.setattr(
        nl2sql.llm,
        "generar_texto",
        lambda *_a, **_k: SimpleNamespace(
            texto='{"sql":"SELECT concepto FRO',
            truncada=True,
            finish_reason="MAX_TOKENS",
        ),
    )

    assert nl2sql.generar_sql(
        "consulta", "finanzas__transacciones(concepto)",
    ) == ""


def test_reintento_sql_recibe_mas_tokens(monkeypatch):
    llamada = {}

    def generar(_modelo, _contenido, **kwargs):
        llamada.update(kwargs)
        return SimpleNamespace(
            texto='{"sql":"SELECT 1 AS valor"}',
            truncada=False,
            finish_reason="STOP",
        )

    monkeypatch.setattr(nl2sql.llm, "generar_texto", generar)

    nl2sql.generar_sql(
        "consulta", "tabla(valor)", correccion="SQL vacio",
        sql_previo="",
    )

    assert llamada["max_tokens"] == max(nl2sql._MAX_TOKENS_SQL * 2, 2400)


def test_pregunta_de_clasificacion_exige_campos_del_registro():
    pregunta = "¿Cuál es la cuenta contable y categoría de ese gasto?"
    assert nl2sql.pide_atributos_registro(pregunta)
    assert not nl2sql.pide_atributos_registro("¿Cuánto gasté por concepto presupuestario?")


def test_detecta_campos_de_clasificacion_omitidos_del_select():
    pregunta = "¿Cuál es la cuenta contable y concepto de ese gasto?"
    sql = "SELECT comercio, fecha_transaccion, monto FROM transacciones"
    esquema = "transacciones(comercio, fecha_transaccion, monto, cuenta_contable, concepto)"
    assert nl2sql.faltan_campos_solicitados(pregunta, sql, esquema) == [
        "cuenta_contable", "concepto",
    ]
    assert nl2sql.faltan_campos_solicitados(
        pregunta, "SELECT * FROM transacciones", esquema,
    ) == []


def test_nombre_aproximado_reintenta_solo_con_contexto_identificable():
    assert nl2sql.puede_reintentar_nombre_aproximado(
        "El gasto de Rogar Cent Med y Dental del 4 de septiembre de 2026"
    )
    assert not nl2sql.puede_reintentar_nombre_aproximado("¿Dónde gasté más?")
