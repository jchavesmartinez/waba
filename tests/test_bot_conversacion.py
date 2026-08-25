"""Regresiones de continuidad conversacional y SQL libre."""

from types import SimpleNamespace

from bot import catalogo, intencion, nl2sql
from bot import responder as R


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


def test_pedir_los_datos_usados_no_se_clasifica_como_conversacion():
    historial = [{
        "rol": "assistant",
        "contenido": "Las ventas fluctúan mes a mes.",
        "sql": "SELECT mes, total FROM ventas_mensuales",
    }]
    assert intencion.clasificar(
        "Me das los datos que usaste por favor", historial,
    ) == "datos"


def _historial_con_ventas_sql():
    return [
        {"rol": "user", "contenido": "Cuáles fueron las ventas totales de hoy"},
        {
            "rol": "assistant",
            "contenido": "Ventas: ₡314.678.900",
            "sql": "SELECT SUM(total) FROM ventas",
        },
    ]


def test_fecha_corta_sigue_la_consulta_anterior_y_no_va_a_meta():
    historial = _historial_con_ventas_sql()
    assert intencion.clasificar("El día de hoy?", historial) == "datos"
    assert intencion.clasificar(
        "de hoy 17 de agosto del 2026?", historial,
    ) == "datos"


def test_correccion_numerica_corta_vuelve_a_los_datos():
    historial = _historial_con_ventas_sql()
    assert intencion.clasificar("3?", historial) == "datos"
    assert intencion.clasificar(
        "Esa cifra no es la total más bien?", historial,
    ) == "datos"


def test_pregunta_exclusiva_sobre_fecha_actual_sigue_siendo_meta():
    assert intencion.clasificar("¿Qué día es hoy?", []) == "meta"


def test_interpretar_venta_promedio_no_vuelve_a_consultar_las_filas():
    historial = _historial_con_ventas_sql()
    assert intencion.clasificar(
        "Y la venta promedio por producto, deberia ser cercana al precio de venta?",
        historial,
    ) == "meta"
    assert intencion.clasificar(
        "Que formula usaste para la venta promedio?", historial,
    ) == "meta"


def test_pedir_valores_de_venta_promedio_sigue_siendo_consulta_de_datos():
    assert intencion.clasificar(
        "Dame la venta promedio de cada producto", [],
    ) == "datos"


def test_temas_del_bot_salen_de_las_tablas_habilitadas_del_catalogo():
    ctx = SimpleNamespace(permitidas=[
        SimpleNamespace(tabla_logica="excursiones"),
        SimpleNamespace(tabla_logica="asignaciones_transporte"),
        SimpleNamespace(tabla_logica="pagos"),
    ])

    assert catalogo.nombres_habilitados(ctx) == [
        "excursiones", "asignaciones transporte", "pagos",
    ]
    assert catalogo.resumir_habilitados(ctx) == (
        "excursiones, asignaciones transporte y pagos"
    )
    assert R._saludo_texto() == (
        "Hola soy Dativa, su asistente empresarial. 📲\n\n"
        "Puedo ayudarle a revisar los principales datos de Inside Tours. 🌴"
    )
    botones = R._saludo_botones()
    assert len(botones) == 3
    assert botones[0].id == "A"
    assert botones[1].id == "B"
    assert botones[2].id == "C"


def test_una_tabla_habilitada_clasifica_el_tema_sin_vocabulario_fijo():
    assert intencion.clasificar(
        "¿De qué mes son las excursiones?", [],
        tablas_habilitadas=["excursiones", "reservas"],
    ) == "datos"


def test_no_respondible_menciona_los_temas_del_catalogo():
    texto = nl2sql.redactar_respuesta(
        "¿Cuál fue la temperatura?",
        ["nota"],
        [("NO_RESPONDIBLE",)],
        temas_habilitados="excursiones, reservas y pagos",
    )

    assert "excursiones, reservas y pagos" in texto
    assert "ventas o inventario" not in texto


def test_respuesta_meta_es_profesional_y_reintenta_si_se_corta(monkeypatch):
    llamadas = []
    respuestas = iter([
        SimpleNamespace(
            texto="La respuesta se cortó a mitad de ",
            truncada=True,
            finish_reason="MAX_TOKENS",
        ),
        SimpleNamespace(
            texto="La respuesta completa es esta.",
            truncada=False,
            finish_reason="STOP",
        ),
    ])

    def generar(_modelo, _contenido, **kwargs):
        llamadas.append(kwargs)
        return next(respuestas)

    monkeypatch.setattr(intencion.llm, "generar_texto", generar)
    texto = intencion.responder_conversacional(
        "Repita la explicación", [],
        temas_habilitados="excursiones, reservas y pagos",
    )

    assert texto == "La respuesta completa es esta."
    assert len(llamadas) == 2
    assert llamadas[0]["thinking_level"] == "minimal"
    assert llamadas[1]["max_tokens"] > llamadas[0]["max_tokens"]
    assert "Trate al usuario de usted" in llamadas[0]["system"]
    assert "No use jerga" in llamadas[0]["system"]
    assert "No repita la lista completa" in llamadas[0]["system"]
    assert "TEMAS HABILITADOS: excursiones, reservas y pagos" in llamadas[0]["system"]


def test_no_envia_fragmento_si_gemini_corta_tambien_el_reintento(monkeypatch):
    def generar(_modelo, _contenido, **_kwargs):
        return SimpleNamespace(
            texto="fragmento",
            truncada=True,
            finish_reason="MAX_TOKENS",
        )

    monkeypatch.setattr(intencion.llm, "generar_texto", generar)
    texto = intencion.responder_conversacional("Repita", [])

    assert texto == (
        "No pude completar la respuesta. Por favor, formule nuevamente "
        "la consulta."
    )
