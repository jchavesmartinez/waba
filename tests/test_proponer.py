"""
Pruebas del generador de metadata.

Lo que importa no es que el LLM proponga bien —eso no se puede afirmar en una
prueba— sino que la propuesta pase por un filtro antes de llegar a manos de una
persona: tipos que no existen, extractores inventados, identificadores
guardados como numero, y sobre todo que la propuesta se PRUEBE contra los
ejemplos en vez de mostrarse a ciegas.
"""

import json

import pytest

from modelo import proponer as P


CUERPO_BAC = """Comercio: AM PM VEROLIZ
Ciudad y país: HEREDIA, Costa Rica
Fecha: Ago 9, 2026 , 16:22
MASTER: ************8774
Autorización: 328882
Tipo de Transacción: COMPRA
Monto: CRC 3,320.00
Enviado por baccredomatic.cr"""

FACTURA = """Proveedor: DISTRIBUIDORA LA UNION
Numero de factura: FE-00123
Fecha de emision: Jul 3, 2026 , 09:15
Subtotal: CRC 120,000.00
Total: CRC 135,600.00"""


class _Bloque:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, texto, stop_reason="end_turn"):
        self.content = [_Bloque(texto)]
        self.stop_reason = stop_reason


def _llm(monkeypatch, payload, capturar=None):
    class _Msgs:
        def create(self, **kw):
            if capturar is not None:
                capturar["prompt"] = kw["messages"][0]["content"]
            texto = payload if isinstance(payload, str) else json.dumps(payload)
            return _Resp(texto)

    class _Cli:
        messages = _Msgs()

    monkeypatch.setattr(P, "_anthropic", lambda: _Cli())
    monkeypatch.setattr(P.config, "BOT_MODELO_KPIS", "modelo-de-prueba")


_PROPUESTA_FACTURA = {
    "filtro": "cuerpo ILIKE '%Numero de factura%'",
    "extractor": "etiqueta_valor",
    "tabla_destino": "compras__facturas",
    "campos": [
        {"columna": "proveedor", "tipo": "texto", "patron": "Proveedor",
         "requerido": "si", "clasifica_en": "cuenta_contable"},
        {"columna": "numero_factura", "tipo": "texto",
         "patron": "Numero de factura", "requerido": "si"},
        {"columna": "fecha_emision", "tipo": "fecha_hora_es",
         "patron": "Fecha de emision", "requerido": "si"},
        {"columna": "total", "tipo": "monto_con_moneda", "patron": "Total",
         "requerido": "si"},
    ],
}


# --- reutilizar lo que ya existe -----------------------------------------

def test_si_ya_hay_un_extractor_para_esto_no_se_inventan_campos(monkeypatch):
    """
    Un preset ya viene probado, con sus tipos y sus casos raros resueltos.
    Proponer campos nuevos para un documento que el sistema ya sabe leer es el
    caso caro: se pierde el trabajo hecho y se introduce una segunda forma de
    leer lo mismo.
    """
    llamo = {"si": False}

    def _no_deberia(*a, **k):
        llamo["si"] = True
        raise AssertionError("no habia que llamar al LLM")

    monkeypatch.setattr(P, "_preguntar", _no_deberia)
    r = P.proponer([{"cuerpo": CUERPO_BAC}] * 3, "correos")

    assert not llamo["si"]
    assert r["preset"] == "bac_transacciones"
    assert r["propuesta"]["extractor"] == "bac_transacciones"
    assert r["propuesta"]["campos"] == []          # el preset ya los trae
    assert r["prueba"]["filas"] and not r["prueba"]["rechazos"]


def test_un_documento_nuevo_si_va_al_llm(monkeypatch):
    _llm(monkeypatch, _PROPUESTA_FACTURA)
    r = P.proponer([{"cuerpo": FACTURA}], "correos", modelo_id="facturas")
    assert r["preset"] == ""
    assert len(r["propuesta"]["campos"]) == 4


# --- validacion de la propuesta -------------------------------------------

def test_un_tipo_inexistente_se_corrige_y_se_avisa(monkeypatch):
    mala = dict(_PROPUESTA_FACTURA)
    mala["campos"] = [{"columna": "total", "tipo": "moneda_rara",
                       "patron": "Total", "requerido": "si"}]
    _llm(monkeypatch, mala)
    r = P.proponer([{"cuerpo": FACTURA}], "correos")
    assert r["propuesta"]["campos"][0]["tipo"] == "texto"
    assert any("no existe" in a for a in r["avisos"])


def test_un_extractor_inventado_se_corrige_y_se_avisa(monkeypatch):
    mala = dict(_PROPUESTA_FACTURA)
    mala["extractor"] = "leer_pdf_magico"
    _llm(monkeypatch, mala)
    r = P.proponer([{"cuerpo": FACTURA}], "correos")
    assert r["propuesta"]["extractor"] == "etiqueta_valor"
    assert any("no existe" in a for a in r["avisos"])


def test_un_identificador_como_entero_se_pasa_a_texto(monkeypatch):
    """
    '000382' guardado como numero pierde los ceros de la izquierda y
    'AMWD1020' ni siquiera es numerico. Es un error silencioso: la columna se
    crea, el pipeline corre y el dato queda mal.
    """
    mala = dict(_PROPUESTA_FACTURA)
    mala["campos"] = [{"columna": "numero_autorizacion", "tipo": "entero",
                       "patron": "Autorizacion"}]
    _llm(monkeypatch, mala)
    r = P.proponer([{"cuerpo": FACTURA}], "correos")
    assert r["propuesta"]["campos"][0]["tipo"] == "texto"
    assert any("ceros de la izquierda" in a for a in r["avisos"])


def test_una_propuesta_sin_campos_utiles_falla(monkeypatch):
    _llm(monkeypatch, {"filtro": "", "extractor": "etiqueta_valor",
                       "campos": []})
    with pytest.raises(RuntimeError, match="ningun campo"):
        P.proponer([{"cuerpo": FACTURA}], "correos")


def test_una_respuesta_cortada_falla_en_vez_de_parsear_la_mitad(monkeypatch):
    class _Msgs:
        def create(self, **kw):
            return _Resp('{"filtro": "x", "campos": [{"col', "max_tokens")

    class _Cli:
        messages = _Msgs()

    monkeypatch.setattr(P, "_anthropic", lambda: _Cli())
    monkeypatch.setattr(P.config, "BOT_MODELO_KPIS", "m")
    with pytest.raises(RuntimeError, match="corto por longitud"):
        P.proponer([{"cuerpo": FACTURA}], "correos")


# --- la propuesta se prueba antes de mostrarse ----------------------------

def test_la_propuesta_se_corre_contra_los_ejemplos(monkeypatch):
    """
    Es la parte que hace util a todo esto: revisar un resultado en vez de
    revisar texto que se ve razonable.
    """
    _llm(monkeypatch, _PROPUESTA_FACTURA)
    r = P.proponer([{"cuerpo": FACTURA}], "correos", modelo_id="facturas")
    fila = r["prueba"]["filas"][0]
    assert fila["proveedor"] == "DISTRIBUIDORA LA UNION"
    assert fila["total"] == 135600.0 and fila["total_moneda"] == "CRC"
    assert fila["fecha_emision"].month == 7


def test_una_etiqueta_equivocada_se_delata_como_columna_vacia(monkeypatch):
    """
    El error mas comun y el mas facil de pasar por alto: la tabla se crea
    igual, el pipeline corre y la columna simplemente queda en blanco para
    siempre.
    """
    mala = dict(_PROPUESTA_FACTURA)
    mala["campos"] = [
        {"columna": "proveedor", "tipo": "texto", "patron": "Proveedor",
         "requerido": "si"},
        {"columna": "moneda_pago", "tipo": "texto",
         "patron": "Etiqueta Que No Existe"},
    ]
    _llm(monkeypatch, mala)
    r = P.proponer([{"cuerpo": FACTURA}], "correos")
    assert "moneda_pago" in r["prueba"]["siempre_vacias"]
    assert "proveedor" not in r["prueba"]["siempre_vacias"]


def test_los_rechazos_aparecen_en_la_prueba(monkeypatch):
    _llm(monkeypatch, _PROPUESTA_FACTURA)
    r = P.proponer(
        [{"cuerpo": FACTURA}, {"cuerpo": "esto no es una factura"}], "correos")
    assert len(r["prueba"]["filas"]) == 1
    assert len(r["prueba"]["rechazos"]) == 1


# --- salida ---------------------------------------------------------------

def test_el_tsv_trae_las_dos_pestanias_y_es_pegable(monkeypatch):
    """
    Tabuladores y no comas: al pegar, Sheets reparte los tabuladores en
    columnas solo, y las descripciones suelen traer comas.
    """
    _llm(monkeypatch, _PROPUESTA_FACTURA)
    r = P.proponer([{"cuerpo": FACTURA}], "correos", modelo_id="facturas")
    tsv = P.a_tsv(r["propuesta"])

    assert "_modelos" in tsv and "_campos" in tsv
    assert "\t" in tsv and "," not in tsv.splitlines()[1]
    filas = [l for l in tsv.splitlines()
             if l.startswith("facturas\t")]
    assert len(filas) == 1 + len(_PROPUESTA_FACTURA["campos"])


def test_el_prompt_advierte_sobre_los_identificadores(monkeypatch):
    capturado = {}
    _llm(monkeypatch, _PROPUESTA_FACTURA, capturado)
    P.proponer([{"cuerpo": FACTURA}], "correos")
    assert "ceros de la izquierda" in capturado["prompt"]
    assert "reenviado a mano" in capturado["prompt"]


def test_sin_texto_en_las_muestras_falla_claro(monkeypatch):
    with pytest.raises(RuntimeError, match="ninguna de las"):
        P.proponer([{"otra_columna": "x"}], "correos")
