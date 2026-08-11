"""
Pruebas de la construccion de tablas derivadas (persistencia).

Corren contra un DuckDB en memoria, no contra mocks: lo que se quiere probar es
justamente que el SQL generado funciona, que los tipos aterrizan bien y que
reconstruir no deja restos de la corrida anterior.
"""

import pandas as pd
import pytest

from modelo.construir import (
    FUENTE_METADATA_SEMANTICA,
    SUFIJO_RECHAZOS,
    _escribir,
    _publicar_metadata_semantica,
    nombre_esquema_semantico,
)
from modelo.motor import Modelo
from warehouse.duckdb_dest import DuckDBDestino


CUERPO = """Comercio: AM PM VEROLIZ
Ciudad y país: HEREDIA, Costa Rica
Fecha: Ago 9, 2026 , 16:22
MASTER: ************8774
Autorización: 328882
Referencia: 90709690
Tipo de Transacción: COMPRA
Monto: CRC 3,320.00
Este correo lo envia baccredomatic.cr"""


def _modelo(overrides=None):
    return Modelo(
        {"modelo_id": "bac", "tabla_origen": "correo_zoho__correos",
         "tabla_destino": "finanzas__transacciones",
         "extractor": "bac_transacciones", "columna_texto": "cuerpo"},
        {"campos": [], "clasificacion": [], "overrides": overrides or []},
    )


@pytest.fixture
def destino():
    d = DuckDBDestino(":memory:")
    d.asegurar_esquema("semantic_cliente_a")
    yield d
    d.cerrar()


# --- nombres de esquema ---------------------------------------------------

def test_el_esquema_semantico_es_hermano_del_raw():
    """
    Esquema aparte y no un prefijo dentro de raw: asi la reconstruccion puede
    hacer DROP con confianza, porque FISICAMENTE no puede tocar los datos
    ingestados aunque alguien escriba mal un nombre de tabla.
    """
    assert nombre_esquema_semantico("cliente_a") == "semantic_cliente_a"
    assert nombre_esquema_semantico("Cliente-B") == "semantic_cliente_b"


# --- creacion de tablas ---------------------------------------------------

def test_la_tabla_se_crea_aunque_no_haya_ni_una_fila(destino):
    """
    Si el esquema saliera de los datos, un modelo sin filas crearia una tabla
    sin columnas y el bot no la veria nunca. Peor: la tabla cambiaria de forma
    segun cuantas filas hubo ese dia.
    """
    modelo = _modelo()
    _escribir(destino, "semantic_cliente_a", modelo.tabla_destino,
              modelo.columnas(), [])
    cols = destino.conectar().execute(
        'DESCRIBE "semantic_cliente_a"."finanzas__transacciones"').fetchall()
    nombres = [c[0] for c in cols]
    for esperada in ("comercio", "monto", "monto_moneda", "cuenta_contable",
                     "_clave"):
        assert esperada in nombres


def test_los_tipos_aterrizan_como_numero_y_fecha_no_como_texto(destino):
    """Un monto guardado como texto convierte SUM() en una concatenacion."""
    modelo = _modelo()
    filas, _ = modelo.procesar([{"correo_id": "c1", "cuerpo": CUERPO}])
    _escribir(destino, "semantic_cliente_a", modelo.tabla_destino,
              modelo.columnas(), filas)

    tipos = dict(
        (c[0], c[1]) for c in destino.conectar().execute(
            'DESCRIBE "semantic_cliente_a"."finanzas__transacciones"'
        ).fetchall())
    assert "DOUBLE" in tipos["monto"].upper()
    assert "TIMESTAMP" in tipos["fecha_transaccion"].upper()

    total = destino.conectar().execute(
        'SELECT SUM(monto) FROM "semantic_cliente_a"."finanzas__transacciones"'
    ).fetchone()[0]
    assert total == 3320.0


def test_reconstruir_no_duplica_ni_deja_restos(destino):
    """
    La propiedad central: la derivada es una funcion pura de sus entradas, asi
    que correr dos veces con la misma entrada da exactamente lo mismo. Con
    UPSERT habria que razonar sobre estado previo; con DROP+CREATE no.
    """
    modelo = _modelo()
    filas, _ = modelo.procesar([{"correo_id": "c1", "cuerpo": CUERPO}])
    for _ in range(3):
        _escribir(destino, "semantic_cliente_a", modelo.tabla_destino,
                  modelo.columnas(), filas)
    n = destino.conectar().execute(
        'SELECT COUNT(*) FROM "semantic_cliente_a"."finanzas__transacciones"'
    ).fetchone()[0]
    assert n == 1


def test_menos_filas_que_ayer_no_deja_las_viejas(destino):
    """
    Reconstruir con menos filas tiene que DEJAR menos filas. Si se hiciera
    INSERT sin DROP, una fila borrada del origen seguiria viva en la derivada
    para siempre y nadie lo notaria.
    """
    modelo = _modelo()
    dos, _ = modelo.procesar([
        {"correo_id": "c1", "cuerpo": CUERPO},
        {"correo_id": "c2", "cuerpo": CUERPO.replace("3,320.00", "1,800.00")},
    ])
    _escribir(destino, "semantic_cliente_a", modelo.tabla_destino,
              modelo.columnas(), dos)
    una, _ = modelo.procesar([{"correo_id": "c1", "cuerpo": CUERPO}])
    _escribir(destino, "semantic_cliente_a", modelo.tabla_destino,
              modelo.columnas(), una)

    filas = destino.conectar().execute(
        'SELECT _clave FROM "semantic_cliente_a"."finanzas__transacciones"'
    ).fetchall()
    assert filas == [("c1",)]


# --- rechazos -------------------------------------------------------------

def test_los_rechazos_quedan_en_su_tabla_con_el_motivo(destino):
    """
    Rechazar no es descartar. Un correo que pasa el filtro pero no produce
    campos puede ser el cuerpo truncado, un cambio de formato del banco o un
    reenvio con cadena larga: los tres hay que verlos.
    """
    modelo = _modelo()
    filas, rechazos = modelo.procesar([
        {"correo_id": "c1", "cuerpo": CUERPO},
        {"correo_id": "c9", "cuerpo": "esto no es una notificacion"},
    ])
    _escribir(destino, "semantic_cliente_a",
              modelo.tabla_destino + SUFIJO_RECHAZOS,
              modelo.columnas_rechazos(), rechazos)

    fila = destino.conectar().execute(
        'SELECT _clave, motivo FROM '
        '"semantic_cliente_a"."finanzas__transacciones__rechazos"').fetchall()
    assert len(fila) == 1
    assert fila[0][0] == "c9" and "comercio" in fila[0][1]
    assert len(filas) == 1          # la buena si paso


# --- overrides ------------------------------------------------------------

def test_un_override_huerfano_se_detecta(destino):
    """
    Es el modo de falla mas traicionero del modulo: si la clave ya no existe,
    la correccion deja de aplicarse y los numeros vuelven a estar mal SIN que
    nadie se entere.
    """
    modelo = _modelo(overrides=[
        {"modelo_id": "bac", "clave": "c1", "columna": "cuenta_contable",
         "valor": "Regalos"},
        {"modelo_id": "bac", "clave": "YA_NO_EXISTE",
         "columna": "cuenta_contable", "valor": "Otra"},
    ])
    filas, _ = modelo.procesar([{"correo_id": "c1", "cuerpo": CUERPO}])
    claves = {f["_clave"] for f in filas}
    huerfanos = modelo.claves_de_override() - claves
    assert huerfanos == {"YA_NO_EXISTE"}
    assert filas[0]["cuenta_contable"] == "Regalos"


# --- publicacion de metadata semantica -----------------------------------

def test_modelos_publica_catalogo_y_kpis_sin_copiar_datos_a_raw(monkeypatch):
    escritos = []

    class Destino:
        def escribir_catalogo(self, esquema, fuente_id, filas):
            escritos.append(("catalogo", esquema, fuente_id, filas))

        def escribir_kpis(self, esquema, fuente_id, filas):
            escritos.append(("kpis", esquema, fuente_id, filas))

    monkeypatch.setattr(
        "modelo.construir.catalogo_cliente.leer",
        lambda cliente: (
            [
                {"tabla": "ventas", "columna": "*"},
                {"tabla": "transacciones", "columna": "*",
                 "instruccion": "si: la puede usar el bot"},
                {"tabla": "transacciones", "columna": "monto"},
            ],
            [
                {"kpi": "ventas_totales", "tabla": "ventas"},
                {"kpi": "gasto_neto", "tabla": "transacciones"},
            ],
        ),
    )

    _publicar_metadata_semantica(
        Destino(), {"cliente_id": "cliente_a"}, "raw_cliente_a",
        "semantic_cliente_a", {"finanzas__transacciones"})

    catalogo = escritos[0]
    kpis = escritos[1]
    assert catalogo[:3] == (
        "catalogo", "raw_cliente_a", FUENTE_METADATA_SEMANTICA)
    assert [f["tabla"] for f in catalogo[3]] == [
        "transacciones", "transacciones"]
    assert kpis[:3] == ("kpis", "raw_cliente_a", FUENTE_METADATA_SEMANTICA)
    assert [f["kpi"] for f in kpis[3]] == ["gasto_neto"]
    # No existe ninguna escritura de la tabla de negocio en raw: estas son
    # exclusivamente las dos tablas pequenas de metadata.
    assert {e[0] for e in escritos} == {"catalogo", "kpis"}
