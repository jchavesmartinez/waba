"""
Pruebas del job de clasificacion.

El LLM se simula a proposito: lo que hay que probar no es que Claude clasifique
bien —eso no se puede afirmar con una prueba— sino que el job se defienda de lo
que un modelo de lenguaje hace de verdad: inventar categorias, saltarse
elementos de un lote, devolver valores que nadie pidio y cortar la respuesta a
la mitad.
"""

import json

import pytest

from modelo import clasificar as C
from modelo.motor import SIN_CLASIFICAR, Modelo
from warehouse.duckdb_dest import DuckDBDestino


CATEGORIAS = [
    {"modelo_id": "bac", "categoria": "Supermercado",
     "descripcion": "Compras de abarrotes", "ejemplos": "Automercado, Walmart"},
    {"modelo_id": "bac", "categoria": "Alimentacion",
     "descripcion": "Restaurantes y panaderias", "ejemplos": "Panaderia"},
    {"modelo_id": "bac", "categoria": "Mascotas", "descripcion": "", "ejemplos": ""},
]


def _metadata(categorias=None):
    return {"campos": [], "clasificacion": [], "overrides": [],
            "categorias": CATEGORIAS if categorias is None else categorias}


def _modelo(metadata):
    return Modelo(
        {"modelo_id": "bac", "tabla_origen": "correos",
         "tabla_destino": "transacciones", "extractor": "bac_transacciones",
         "columna_texto": "cuerpo"},
        metadata,
    )


@pytest.fixture
def destino():
    d = DuckDBDestino(":memory:")
    d.asegurar_esquema("semantic_cliente_a")
    # Tabla derivada como la dejaria construir_modelos.py
    d.reconstruir_tabla(
        "semantic_cliente_a", "transacciones",
        [("comercio", "texto"), ("cuenta_contable", "texto")],
        [{"comercio": "AUTOMERCADO ESCAZU", "cuenta_contable": SIN_CLASIFICAR},
         {"comercio": "AUTOMERCADO ESCAZU", "cuenta_contable": SIN_CLASIFICAR},
         {"comercio": "PANADERIA PANES", "cuenta_contable": SIN_CLASIFICAR},
         {"comercio": "AM PM", "cuenta_contable": "Supermercado"}],
    )
    yield d
    d.cerrar()


def _simular(monkeypatch, respuesta):
    """Reemplaza la llamada al LLM por una respuesta fija."""
    llamadas = []

    def falso(valores, categorias):
        llamadas.append(list(valores))
        datos = respuesta(valores) if callable(respuesta) else respuesta
        return {str(i["valor"]).strip(): i for i in datos if i.get("valor")}

    monkeypatch.setattr(C, "_preguntar", falso)
    return llamadas


# --- que se le manda al modelo -------------------------------------------

def test_solo_se_mandan_valores_distintos_y_sin_clasificar(destino, monkeypatch):
    """
    Si 300 transacciones son del mismo comercio, al modelo se le manda ese
    comercio UNA vez: es la diferencia entre 300 llamadas y una, y ademas
    garantiza que las 300 filas reciban la misma categoria.
    """
    llamadas = _simular(monkeypatch, [])
    C.clasificar_modelo(destino, "semantic_cliente_a",
                        _modelo(_metadata()), _metadata())
    enviados = sorted(llamadas[0])
    assert enviados == ["AUTOMERCADO ESCAZU", "PANADERIA PANES"]
    assert "AM PM" not in enviados        # ya estaba clasificado


def test_el_llm_recibe_las_columnas_de_contexto_elegidas(monkeypatch):
    d = DuckDBDestino(":memory:")
    d.asegurar_esquema("semantic_cliente_a")
    d.reconstruir_tabla(
        "semantic_cliente_a", "transacciones_contexto",
        [("comercio", "texto"), ("asunto", "texto"),
         ("tipo_transaccion", "texto"), ("cuenta_contable", "texto")],
        [{"comercio": "AM PM", "asunto": "Compra aprobada",
          "tipo_transaccion": "COMPRA", "cuenta_contable": SIN_CLASIFICAR}],
    )
    meta = _metadata()
    meta["campos"] = [{
        "modelo_id": "bac", "columna": "comercio", "tipo": "texto",
        "patron": "Comercio", "clasifica_en": "cuenta_contable",
        "clasifica_con": "asunto,tipo_transaccion",
    }]
    modelo = Modelo(
        {"modelo_id": "bac", "tabla_origen": "correos",
         "tabla_destino": "transacciones_contexto",
         "extractor": "bac_transacciones", "columna_texto": "cuerpo"},
        meta,
    )
    llamadas = _simular(monkeypatch, [])
    C.clasificar_modelo(d, "semantic_cliente_a", modelo, meta)
    assert llamadas[0] == [
        "comercio: AM PM | asunto: Compra aprobada | tipo_transaccion: COMPRA"
    ]
    d.cerrar()


def test_no_se_reclasifica_lo_que_ya_esta_en_el_mapeo(destino, monkeypatch):
    """El costo tiende a cero: la segunda corrida no vuelve a preguntar."""
    _simular(monkeypatch, [
        {"valor": "AUTOMERCADO ESCAZU", "categoria": "Supermercado",
         "confianza": "alta"},
        {"valor": "PANADERIA PANES", "categoria": "Alimentacion",
         "confianza": "alta"},
    ])
    meta, modelo = _metadata(), None
    modelo = _modelo(meta)
    primera = C.clasificar_modelo(destino, "semantic_cliente_a", modelo, meta)
    segunda = C.clasificar_modelo(destino, "semantic_cliente_a", modelo, meta)

    assert primera["clasificados"] == 2 and primera["llamadas"] == 1
    assert segunda["pendientes"] == 0 and segunda["llamadas"] == 0


def test_los_lotes_son_chicos(destino, monkeypatch):
    """Los modelos pierden precision en listas largas y apuran el final."""
    lotes = list(C._lotes(list(range(60)), 25))
    assert [len(x) for x in lotes] == [25, 25, 10]


# --- defensas contra lo que hace un LLM ----------------------------------

def test_una_categoria_inventada_se_descarta(destino, monkeypatch):
    """
    Sin conjunto cerrado, el cliente termina con quince cuentas contables que
    nadie definio, todas plausibles y ninguna comparable entre meses.
    """
    _simular(monkeypatch, [
        {"valor": "AUTOMERCADO ESCAZU", "categoria": "Gastos varios"},
        {"valor": "PANADERIA PANES", "categoria": "Alimentacion"},
    ])
    meta = _metadata()
    r = C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)
    assert r["clasificados"] == 1        # solo la valida
    assert r["sin_resolver"] == 1

    guardado = destino.leer_filas(
        'SELECT valor_asignado FROM "semantic_cliente_a"."_mapeo"')
    assert [f["valor_asignado"] for f in guardado] == ["Alimentacion"]


def test_un_valor_que_el_modelo_se_salta_queda_pendiente(destino, monkeypatch):
    """No se inventa nada por completar el lote; se reintenta la proxima."""
    _simular(monkeypatch, [
        {"valor": "AUTOMERCADO ESCAZU", "categoria": "Supermercado"},
    ])
    meta = _metadata()
    r = C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)
    assert r["clasificados"] == 1 and r["sin_resolver"] == 1


def test_un_valor_que_nadie_pidio_se_ignora(destino, monkeypatch):
    """Aceptarlo escribiria en el mapeo una entrada que nadie solicito."""
    _simular(monkeypatch, [
        {"valor": "AUTOMERCADO ESCAZU", "categoria": "Supermercado"},
        {"valor": "PANADERIA PANES", "categoria": "Alimentacion"},
        {"valor": "TIENDA QUE NO EXISTE", "categoria": "Supermercado"},
    ])
    meta = _metadata()
    C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)
    guardado = destino.leer_filas(
        'SELECT valor_original FROM "semantic_cliente_a"."_mapeo"')
    originales = {f["valor_original"] for f in guardado}
    assert "TIENDA QUE NO EXISTE" not in originales


def test_sin_clasificar_es_una_respuesta_valida(destino, monkeypatch):
    """
    Un modelo obligado a elegir siempre elige. Adivinar con seguridad es peor
    que no responder: el error no se detecta nunca.
    """
    _simular(monkeypatch, [
        {"valor": "AUTOMERCADO ESCAZU", "categoria": SIN_CLASIFICAR},
        {"valor": "PANADERIA PANES", "categoria": "Alimentacion"},
    ])
    meta = _metadata()
    r = C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)
    assert r["clasificados"] == 1 and r["sin_resolver"] == 1


def test_un_lote_que_falla_no_tumba_a_los_demas(destino, monkeypatch):
    """Idempotencia: lo que fallo se reintenta en la proxima corrida."""
    def explota(valores, categorias):
        raise RuntimeError("timeout de la API")

    monkeypatch.setattr(C, "_preguntar", explota)
    meta = _metadata()
    r = C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)
    assert r["clasificados"] == 0
    assert len(r["alertas"]) == 1 and "timeout" in r["alertas"][0]


def test_la_respuesta_se_indexa_por_valor_no_por_posicion(destino, monkeypatch):
    """
    Si se mapeara por posicion y el modelo se saltara un elemento, todo lo
    demas quedaria corrido y se asignarian categorias equivocadas a valores que
    nadie reviso.
    """
    _simular(monkeypatch, [                       # orden invertido a proposito
        {"valor": "PANADERIA PANES", "categoria": "Alimentacion"},
        {"valor": "AUTOMERCADO ESCAZU", "categoria": "Supermercado"},
    ])
    meta = _metadata()
    C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)
    guardado = {f["valor_original"]: f["valor_asignado"] for f in
                destino.leer_filas(
                    'SELECT valor_original, valor_asignado '
                    'FROM "semantic_cliente_a"."_mapeo"')}
    assert guardado["AUTOMERCADO ESCAZU"] == "Supermercado"
    assert guardado["PANADERIA PANES"] == "Alimentacion"


# --- configuracion --------------------------------------------------------

def test_sin_categorias_declaradas_no_se_clasifica_nada(destino, monkeypatch):
    """Sin conjunto cerrado no hay contra que validar; mejor no correr."""
    llamadas = _simular(monkeypatch, [])
    meta = _metadata(categorias=[])
    r = C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)
    assert llamadas == [] and r["clasificados"] == 0
    assert "_categorias" in r["alertas"][0]


def test_el_modo_probar_no_llama_al_llm_ni_escribe(destino, monkeypatch):
    llamadas = _simular(monkeypatch, [
        {"valor": "AUTOMERCADO ESCAZU", "categoria": "Supermercado"}])
    meta = _metadata()
    r = C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta,
                            probar=True)
    assert llamadas == [] and r["clasificados"] == 0
    with pytest.raises(Exception):
        destino.leer_filas('SELECT * FROM "semantic_cliente_a"."_mapeo"')


def test_el_mapeo_conserva_las_entradas_de_otros_modelos(destino, monkeypatch):
    """La tabla se reescribe entera: perder los otros modelos seria facil."""
    destino.reconstruir_tabla(
        "semantic_cliente_a", "_mapeo", C.COLUMNAS_MAPEO,
        [{"modelo_id": "otro", "valor_normalizado": "X", "valor_original": "X",
          "valor_asignado": "Cat", "confianza": "alta", "modelo_llm": "m",
          "clasificado_en": "2026-01-01"}])
    _simular(monkeypatch, [
        {"valor": "AUTOMERCADO ESCAZU", "categoria": "Supermercado"}])
    meta = _metadata()
    C.clasificar_modelo(destino, "semantic_cliente_a", _modelo(meta), meta)

    modelos = {f["modelo_id"] for f in destino.leer_filas(
        'SELECT modelo_id FROM "semantic_cliente_a"."_mapeo"')}
    assert modelos == {"otro", "bac"}


# --- parseo de la respuesta cruda ----------------------------------------

class _Resp:
    def __init__(self, texto, stop_reason="end_turn"):
        self.texto = texto
        self.truncada = stop_reason == "max_tokens"
        self.finish_reason = stop_reason


def _cliente_falso(monkeypatch, resp):
    monkeypatch.setattr(C.llm, "generar_texto", lambda *args, **kwargs: resp)
    monkeypatch.setattr(C.config, "BOT_MODELO_KPIS", "modelo-de-prueba")


def test_se_tolera_que_el_modelo_envuelva_el_json_en_backticks(monkeypatch):
    """Pedirle 'solo JSON' no garantiza que no agregue el bloque de codigo."""
    _cliente_falso(monkeypatch, _Resp(
        '```json\n[{"valor": "AM PM", "categoria": "Supermercado"}]\n```'))
    r = C._preguntar(["AM PM"], CATEGORIAS)
    assert r["AM PM"]["categoria"] == "Supermercado"


def test_una_respuesta_cortada_falla_en_vez_de_parsear_la_mitad(monkeypatch):
    """
    Con max_tokens el JSON queda incompleto. Parsear lo que llego daria un lote
    a medias sin ningun aviso, y los valores faltantes se darian por
    clasificados.
    """
    _cliente_falso(monkeypatch, _Resp(
        '[{"valor": "AM PM", "categoria": "Super', stop_reason="max_tokens"))
    with pytest.raises(RuntimeError, match="corto por longitud"):
        C._preguntar(["AM PM"], CATEGORIAS)


def test_las_categorias_llegan_al_prompt_con_sus_ejemplos(monkeypatch):
    """
    'Servicios' no le dice nada a un modelo; 'Servicios: agua, luz, internet'
    si. Los ejemplos son la parte que mas mueve la precision.
    """
    capturado = {}

    def generar(modelo, contenido, **kwargs):
        capturado["prompt"] = contenido
        return _Resp("[]")

    monkeypatch.setattr(C.llm, "generar_texto", generar)
    monkeypatch.setattr(C.config, "BOT_MODELO_KPIS", "modelo-de-prueba")
    C._preguntar(["X"], CATEGORIAS)

    assert "Automercado, Walmart" in capturado["prompt"]
    assert "Compras de abarrotes" in capturado["prompt"]
    assert SIN_CLASIFICAR in capturado["prompt"]
