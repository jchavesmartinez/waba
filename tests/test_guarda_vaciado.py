"""
Prueba de regresion de la guarda de borrado ante vaciados CONSECUTIVOS.

POR QUE EXISTE. La guarda estaba bien diseñada: si una tabla llega vacia y antes
tenia filas, no se escribe, porque la carga es full refresh y escribir cero filas
encima borraria la unica copia buena. Funcionaba perfecto... una vez.

El problema estaba en el punto de comparacion. Cuando la guarda bloqueaba, el
`continue` salteaba la linea que llena `corrida.detalle[tabla]`, asi que esa
corrida quedaba SIN registrar la forma de esa tabla. Y como el punto de
comparacion se buscaba en la ultima corrida con estado 'ok%' —que incluye
'ok_con_alertas'— la corrida siguiente no encontraba con que comparar, la guarda
no se disparaba, y escribia las 0 filas encima de los datos buenos.

    corrida #1 (lunes)   1.500 filas   ->  escribe. detalle: 1.500 filas.  BIEN
    corrida #2 (10:00)   0 filas       ->  BLOQUEA. detalle de esa tabla: VACIO.
    corrida #3 (10:15)   0 filas       ->  no hay con que comparar. ESCRIBE 0.

O sea: la guarda protegia quince minutos y despues se desarmaba sola. Y el
escenario que la dispara —alguien renombro una pestaña, le puso un filtro, o la
borro— no se resuelve en quince minutos: normalmente nadie se entera hasta el dia
siguiente. Peor, la corrida #2 dejaba una alerta en el log y la #3 no dejaba
ninguna: el registro de la PERDIDA de datos era mas silencioso que el del bloqueo
que la evito.

COMO SE PRUEBA. No se reimplementa la logica: se corre la funcion REAL
sync.sincronizar_fuente() contra un destino y una fuente de mentira. Si alguien
vuelve a sacar la linea que preserva el detalle, esta prueba se pone roja.
"""

import json

import pandas as pd
import pytest

import sync
from sources.base import Fragmento
from warehouse.base import nombre_tabla


CLIENTE = {"cliente_id": "demo"}
FUENTE = {"fuente_id": "sheet_ventas", "tipo": "google_sheets",
          "activo": True, "frescura_minutos": 0, "config": {}}
TABLA_REAL = nombre_tabla("sheet_ventas", "ventas")
COLUMNAS = ["fecha", "producto", "monto"]


class DestinoFalso:
    """
    Destino en memoria. Reproduce la semantica REAL de los destinos de verdad,
    incluido el detalle que hace de punto de comparacion:
      - registrar_corrida guarda la corrida serializada como en la bitacora;
      - ultimo_detalle FUSIONA las ultimas corridas OK (la mas reciente gana),
        que es la segunda mitad del arreglo de C-01.
    """

    def __init__(self):
        self.bitacora = []      # [{estado, detalle_json}]
        self.escrituras = []    # [(tabla, filas)] -> lo que de verdad se escribio

    def escribir_tabla(self, esquema, tabla, df):
        self.escrituras.append((tabla, len(df)))

    def escribir_catalogo(self, esquema, fuente_id, filas):
        pass

    def escribir_kpis(self, esquema, fuente_id, filas):
        pass

    def registrar_corrida(self, corrida):
        self.bitacora.append({"estado": corrida.estado,
                              "detalle": json.dumps(corrida.detalle)})

    def ultima_corrida_ok(self, cliente_id, fuente_id):
        return None

    def ultimo_detalle(self, cliente_id, fuente_id):
        fusionado = {}
        for c in [x for x in self.bitacora if x["estado"].startswith("ok")][-20:]:
            fusionado.update(json.loads(c["detalle"]))
        return fusionado

    # --- helpers para las aserciones ---

    @property
    def filas_escritas(self):
        return [f for _, f in self.escrituras]


class FuenteFalsa:
    """Fuente que devuelve la cantidad de filas que se le pida."""

    def __init__(self, filas: int, columnas=None):
        self.filas = filas
        self.columnas = columnas or COLUMNAS

    def cargar(self, con):
        # Una fila de ejemplo por cada columna pedida (para poder simular que
        # la hoja gano o perdio columnas entre corridas).
        filas = [
            [f"valor_{i}_{j}" for j in range(len(self.columnas))]
            for i in range(self.filas)
        ]
        df = pd.DataFrame(filas, columns=self.columnas)
        con.register("_tmp_ventas", df)
        con.execute("CREATE TABLE ventas AS SELECT * FROM _tmp_ventas")
        con.unregister("_tmp_ventas")
        return Fragmento(tablas=["ventas"])


@pytest.fixture
def destino():
    return DestinoFalso()


def corrida_con(destino, filas, columnas=None, monkeypatch=None):
    """Corre la funcion REAL de sync con una fuente que trae `filas` filas."""
    import sources
    original = sync.crear_fuente
    sync.crear_fuente = lambda tipo, fid, cfg: FuenteFalsa(filas, columnas)
    try:
        c = sync.sincronizar_fuente(destino, CLIENTE, FUENTE)
    finally:
        sync.crear_fuente = original
    destino.registrar_corrida(c)
    return c


# --------------------------------------------------------------------------


def test_primera_carga_escribe(destino):
    """Sin historial no hay nada que proteger."""
    c = corrida_con(destino, 1500)
    assert c.estado == "ok"
    assert destino.filas_escritas == [1500]
    assert c.detalle[TABLA_REAL]["filas"] == 1500


def test_una_tabla_vacia_bloquea_la_escritura(destino):
    """El caso que la guarda siempre atrapo bien."""
    corrida_con(destino, 1500)
    c = corrida_con(destino, 0)

    assert c.estado == "ok_con_bloqueo"
    assert any("VACIA" in a for a in c.alertas)
    assert destino.filas_escritas == [1500], "no se debio escribir la tabla vacia"


def test_dos_vaciados_seguidos_siguen_bloqueando(destino):
    """
    EL CASO QUE FALLABA. Es la prueba central de este archivo.

    Antes del arreglo, la corrida #2 no registraba el detalle de la tabla, la #3
    no encontraba con que comparar y ESCRIBIA 0 filas encima de los datos buenos.
    """
    corrida_con(destino, 1500)       # #1 carga buena
    corrida_con(destino, 0)          # #2 vacia -> bloquea
    c3 = corrida_con(destino, 0)     # #3 vacia otra vez

    assert c3.estado == "ok_con_bloqueo", (
        "LA GUARDA SE DESARMO: la segunda carga vacia consecutiva no bloqueo. "
        "Revisa que sync.py copie el detalle anterior antes del 'continue' "
        "cuando bloquea, y que ultimo_detalle() fusione las ultimas corridas OK."
    )
    assert destino.filas_escritas == [1500], (
        "SE PERDIERON LOS DATOS: se escribio una tabla vacia encima de la buena."
    )


def test_la_guarda_aguanta_un_fin_de_semana_entero(destino):
    """
    El escenario real: alguien rompio el Sheet un viernes a las 5pm y nadie lo
    mira hasta el lunes. Con el cron cada 15 minutos eso son ~250 corridas.
    """
    corrida_con(destino, 1500)
    for i in range(250):
        c = corrida_con(destino, 0)
        assert c.estado == "ok_con_bloqueo", (
            f"la guarda se desarmo en la corrida vacia numero {i + 1}"
        )
    assert destino.filas_escritas == [1500]


def test_cada_bloqueo_deja_rastro(destino):
    """
    El registro de la perdida de datos no puede ser mas silencioso que el del
    bloqueo que la evito. Toda corrida bloqueada tiene estado propio y alerta.
    """
    corrida_con(destino, 1500)
    for _ in range(5):
        c = corrida_con(destino, 0)
        assert c.alertas, "un bloqueo sin alerta es una perdida silenciosa"
        assert c.estado == "ok_con_bloqueo"

    bloqueadas = [x for x in destino.bitacora if x["estado"] == "ok_con_bloqueo"]
    assert len(bloqueadas) == 5, (
        "las corridas bloqueadas tienen que ser auditables de una: "
        "SELECT * FROM _meta.sync_corridas WHERE estado = 'ok_con_bloqueo'"
    )


def test_se_recupera_cuando_los_datos_vuelven(destino):
    """
    La guarda no puede quedarse trabada: cuando el cliente arregla la hoja, la
    carga vuelve a escribir normalmente.
    """
    corrida_con(destino, 1500)
    corrida_con(destino, 0)
    corrida_con(destino, 0)
    c = corrida_con(destino, 1480)

    assert c.estado in ("ok", "ok_con_alertas")
    assert destino.filas_escritas == [1500, 1480]


def test_caida_fuerte_alerta_pero_no_bloquea(destino):
    """
    Distincion deliberada del diseño: una caida de mas del 50% puede ser
    legitima (el cliente archivo el historico). Alerta, no bloquea.
    """
    corrida_con(destino, 1500)
    c = corrida_con(destino, 400)

    assert c.estado == "ok_con_alertas"
    assert any("cayeron" in a for a in c.alertas)
    assert destino.filas_escritas == [1500, 400]


def test_schema_drift_alerta_pero_no_bloquea(destino):
    """Una columna nueva no puede frenar la ingesta, pero deja rastro."""
    corrida_con(destino, 1500)
    c = corrida_con(destino, 1500, columnas=COLUMNAS + ["vendedor"])

    assert any("schema drift" in a for a in c.alertas)
    assert destino.filas_escritas == [1500, 1500]


def test_el_detalle_preservado_es_serializable(destino):
    """
    El detalle viaja a la bitacora como JSON. Si al copiarlo se colara algo no
    serializable, la corrida entera fallaria al registrarse.
    """
    corrida_con(destino, 1500)
    c = corrida_con(destino, 0)
    json.dumps(c.detalle)      # no debe lanzar


def test_la_fusion_de_detalles_tapa_huecos_del_historial(destino):
    """
    Segunda defensa de C-01, independiente de la primera: aunque una corrida no
    registre una tabla por el motivo que sea, la fusion de las corridas
    anteriores recupera su ultima forma conocida.
    """
    destino.bitacora = [
        {"estado": "ok", "detalle": json.dumps({
            "ventas": {"columnas": COLUMNAS, "filas": 1500},
            "inventario": {"columnas": ["sku"], "filas": 80},
        })},
        # Una corrida que solo registro una de las dos tablas.
        {"estado": "ok_con_alertas", "detalle": json.dumps({
            "ventas": {"columnas": COLUMNAS, "filas": 1520},
        })},
    ]

    previo = destino.ultimo_detalle("demo", "sheet_ventas")
    assert previo["ventas"]["filas"] == 1520, "debe ganar el valor mas reciente"
    assert "inventario" in previo, (
        "la tabla desaparecio del punto de comparacion por un hueco en el "
        "historial: la guarda de vaciado quedaria desarmada para esa tabla"
    )
    assert previo["inventario"]["filas"] == 80
