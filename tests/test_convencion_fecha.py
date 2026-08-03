"""
Pruebas de la deteccion de convencion de fecha (dia/mes vs mes/dia).

POR QUE EXISTE. Este es el error de datos mas caro que puede cometer la
ingesta, y el unico que no produce NI UNA señal por si solo:

  - No falla al parsear. '01/02/2026' es una fecha valida leida de las dos
    maneras, asi que _avisar_descartes() no se entera: no se descarto nada.
  - No rompe ningun total. La suma del mes cuadra; lo que no cuadra es de QUE
    mes es cada venta.
  - No se ve en una muestra chica. Los dias del 13 en adelante se parsean bien
    solos, asi que si mirás las primeras filas de un reporte podés no ver nada
    raro durante semanas.

Se descubrio en produccion: un cliente pidio ventas por dia y las del 1 de
febrero aparecieron contadas el 2 de enero.

La deteccion no es heuristica. En una serie de fechas reales de mas de dos
semanas siempre aparece un componente mayor a 12, y ese solo puede ser el dia.
"""

import pandas as pd
import pytest

from sources.base import (_a_fecha, detectar_convencion_fecha, dia_primero_de,
                          inferir_tipos)


# --- dayfirst NUNCA se aplica a ISO --------------------------------------
#
# El bug que originó todo esto. pandas le hace caso a dayfirst incluso sobre
# una fecha ISO: guess_datetime_format('2026-01-02', dayfirst=True) devuelve
# '%Y-%d-%m'. O sea que _a_fecha leía el 2 de enero como 1 de febrero, sin
# fallar y sin avisar, porque '%Y-%d-%m' produce una fecha perfectamente
# válida. Solo se salvaban los días mayores a 12, que no caben como mes.
#
# Golpea a toda fuente que entregue fechas REALES en vez de texto:
# sharepoint_excel/link (pandas convierte una celda de fecha a
# '2026-01-02 00:00:00'), api_rest (JSON con ISO), postgres (columnas date).

@pytest.mark.parametrize("dia_primero", [True, False])
@pytest.mark.parametrize("valor,esperado", [
    ("2026-01-02", "2026-01-02"),               # el caso del bug
    ("2026-02-01", "2026-02-01"),               # y su espejo
    ("2026-01-02 00:00:00", "2026-01-02"),      # como llega desde xlsx
    ("2026-01-16", "2026-01-16"),               # este se salvaba solo
    ("2026-01-07T14:30:00", "2026-01-07"),      # ISO de API
])
def test_iso_es_inequivoca_pase_lo_que_pase(valor, esperado, dia_primero):
    """
    ISO 8601 tiene el año adelante y el mes en el medio: no admite convención
    regional. El resultado debe ser el mismo con dia_primero en True o False.
    """
    r = _a_fecha(pd.Series([valor]), dia_primero=dia_primero)
    assert str(r.iloc[0])[:10] == esperado


def test_dmy_si_respeta_la_convencion():
    """El arreglo de ISO no debe romper el caso que dayfirst sí resuelve."""
    s = pd.Series(["01/02/2026"])
    assert str(_a_fecha(s, dia_primero=True).iloc[0])[:10] == "2026-02-01"
    assert str(_a_fecha(s, dia_primero=False).iloc[0])[:10] == "2026-01-02"


def test_serie_que_mezcla_iso_y_dmy():
    """Cada mitad con su regla: la ISO literal, el resto por convención."""
    s = pd.Series(["2026-01-02", "01/02/2026", "2026-02-01", "16/01/2026"])
    assert [str(v)[:10] for v in _a_fecha(s, dia_primero=True)] == [
        "2026-01-02", "2026-02-01", "2026-02-01", "2026-01-16"]


# --- El veredicto sale de los datos, no de la configuracion --------------

@pytest.mark.parametrize("valores,esperado", [
    (["01/01/2026", "16/01/2026", "21/01/2026"], "dia_primero"),
    (["01/01/2026", "01/16/2026", "01/21/2026"], "mes_primero"),
    # Un '16' en cada posicion: la columna mezcla formatos y no hay eleccion
    # correcta posible.
    (["16/01/2026", "01/16/2026"], "contradictorio"),
    # Nada supera el 12: indecidible, y decirlo es la respuesta honesta.
    (["01/01/2026", "02/01/2026", "03/01/2026"], "ambiguo"),
    # ISO y texto no son fechas d/m/a: no aplica.
    (["2026-01-01", "2026-02-01"], ""),
    (["ayer", "hoy", ""], ""),
])
def test_veredicto(valores, esperado):
    assert detectar_convencion_fecha(pd.Series(valores))[0] == esperado


def test_evidencia_cuenta_los_casos():
    _, ev = detectar_convencion_fecha(
        pd.Series(["01/01/2026", "16/01/2026", "21/01/2026", "05/02/2026"]))
    assert ev["total"] == 4
    assert ev["prueban_dia_primero"] == 2
    assert ev["prueban_mes_primero"] == 0
    assert ev["ambiguos"] == 2


# --- La evidencia le gana a la declaracion -------------------------------

def test_corrige_cuando_la_fuente_declara_al_reves():
    """
    El caso real: fuente declarada mes_primero, datos en DD/MM. Sin la
    correccion, el 1 de febrero entraba como 2 de enero.
    """
    df = pd.DataFrame({"fecha": ["01/02/2026", "16/01/2026", "21/01/2026"]})
    alertas = []
    r = inferir_tipos(df, alertas=alertas, contexto="t", dia_primero=False)
    assert str(r["fecha"].iloc[0])[:10] == "2026-02-01"   # 1 de febrero
    assert alertas, "corregir en silencio seria tan malo como equivocarse"
    assert "PRUEBAN" in alertas[0]


def test_corrige_en_la_direccion_contraria():
    """Sheet con locale en-US declarado como dia_primero."""
    df = pd.DataFrame({"fecha": ["2/1/2026", "1/16/2026", "1/21/2026"]})
    r = inferir_tipos(df, contexto="t", dia_primero=True)
    assert str(r["fecha"].iloc[0])[:10] == "2026-02-01"


def test_respeta_lo_declarado_si_no_hay_evidencia_en_contra():
    df = pd.DataFrame({"fecha": ["01/02/2026", "05/02/2026", "07/02/2026"]})
    r = inferir_tipos(df, contexto="t", dia_primero=True)
    assert str(r["fecha"].iloc[0])[:10] == "2026-02-01"


def test_avisa_cuando_no_se_puede_verificar():
    """Ninguna fecha supera el 12: se asume lo declarado PERO se dice."""
    df = pd.DataFrame({"fecha": [f"0{d}/01/2026" for d in range(1, 8)]})
    alertas = []
    inferir_tipos(df, alertas=alertas, contexto="t", dia_primero=True)
    assert any("NINGUNA supera el 12" in a for a in alertas)


def test_avisa_cuando_la_columna_mezcla_formatos():
    df = pd.DataFrame({"fecha": ["16/01/2026", "01/16/2026", "03/01/2026"]})
    alertas = []
    inferir_tipos(df, alertas=alertas, contexto="t", dia_primero=True)
    assert any("MEZCLA" in a for a in alertas)


def test_iso_no_dispara_alertas():
    df = pd.DataFrame({"fecha": ["2026-01-01", "2026-02-01", "2026-01-16"]})
    alertas = []
    inferir_tipos(df, alertas=alertas, contexto="t", dia_primero=True)
    assert not alertas


# --- dia_primero_de: no ignorar lo que el usuario escribio ---------------

@pytest.mark.parametrize("valor,esperado", [
    ("dia_primero", True), ("DIA_PRIMERO", True), ("dd/mm/yyyy", True),
    ("mes_primero", False), ("MES_PRIMERO", False), ("mm/dd/yyyy", False),
    ("MDY", False), ("us", False), ("month_first", False),
])
def test_reconoce_las_grafias_que_la_gente_escribe(valor, esperado):
    """
    'MM/DD/YYYY' es lo que cualquiera escribe cuando le pedis el formato. Antes
    caia fuera de la lista de cuatro strings y devolvia dia_primero en
    silencio: el usuario creia haber configurado mes-primero y obtenia lo
    contrario, sin una linea de log.
    """
    assert dia_primero_de({"formato_fecha": valor}) is esperado


def test_valor_desconocido_asume_tico_pero_no_calla(caplog):
    assert dia_primero_de({"formato_fecha": "quien sabe"}) is True
    assert "No entiendo el formato de fecha" in caplog.text
