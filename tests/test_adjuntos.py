"""
Pruebas de la capa de adjuntos (grafico / Excel / PDF por WhatsApp).

POR QUE EXISTE. Los errores de esta capa no se ven: producen un archivo que
igual se manda y que igual se abre. Cuatro casos concretos que ninguna prueba
manual atrapa a tiempo:

1. Los tipos que devuelve psycopg2 NO son los de Python. Un SUM() vuelve como
   decimal.Decimal y una fecha como datetime.date. matplotlib no grafica
   Decimal y xlsxwriter no lo serializa. En desarrollo, con datos escritos a
   mano en una lista de tuplas, todo funciona; contra Neon truena.

2. Varias columnas numericas en el mismo eje. Un SELECT tipico trae unidades Y
   monto. Dibujar 890 unidades al lado de ₡4.120.000 deja la primera barra en
   un pixel: el grafico se ve perfecto y dice lo contrario de lo que pasa. Es
   el error mas caro de los cuatro, porque el cliente le cree.

3. Datos que no se pueden graficar (una sola celda, sin columna numerica). La
   tentacion es mandar la imagen igual. Un grafico vacio es peor que ninguno.

4. El pie de una imagen o documento son 1024 caracteres, no 4096 como el texto
   suelto. Pasarse no recorta el caption: Meta rechaza el mensaje ENTERO.

Ninguna de estas cuatro cosas produce una excepcion en el camino feliz.
"""

import io
import zipfile
from datetime import date
from decimal import Decimal

import pytest

from bot import artefactos, formato, whatsapp

COLUMNAS = ["producto", "unidades", "monto"]
# Los tipos son los que devuelve psycopg2 de verdad, no int/float a mano (1).
FILAS = [
    ("Cemento gris 50kg", 890, Decimal("4120000.00")),
    ("Tornillo hexagonal 3/8", 1240, Decimal("1875400.50")),
    ("Varilla #3", 430, Decimal("2210500.75")),
    ("Lija de agua", 2100, Decimal("315000.00")),
]


# --- Deteccion del formato pedido ---------------------------------------

@pytest.mark.parametrize("pregunta,esperado", [
    ("¿cuánto vendimos ayer?", formato.TEXTO),
    ("dame el producto más vendido", formato.TEXTO),
    ("graficame las ventas por mes", formato.GRAFICO),
    ("quiero un gráfico de barras del top 10", formato.GRAFICO),
    ("pasame eso en Excel", formato.EXCEL),
    ("exportá el inventario", formato.EXCEL),
    ("dame el archivo con el detalle", formato.EXCEL),
    ("mandame el reporte en PDF", formato.PDF),
])
def test_detecta_formato(pregunta, esperado):
    assert formato.detectar(pregunta) == esperado


def test_pdf_gana_sobre_grafico():
    """'el gráfico en PDF' es un PDF: el contenedor manda sobre el contenido."""
    assert formato.detectar("mandame el gráfico en PDF") == formato.PDF


def test_csv_degrada_a_excel_por_defecto():
    """text/csv no esta en los MIME de documento que lista la Cloud API."""
    assert formato.detectar("mandame el csv") == formato.EXCEL


# --- Tipos de psycopg2 (1) ----------------------------------------------

def test_grafico_acepta_decimal():
    adj = artefactos.grafico_png(COLUMNAS, FILAS, titulo="Top productos")
    assert adj is not None
    assert adj.contenido[:4] == b"\x89PNG"
    assert adj.mime == "image/png"


def test_excel_acepta_decimal_y_fecha():
    columnas = ["fecha", "monto"]
    filas = [(date(2026, 7, 1), Decimal("125000.50")),
             (date(2026, 7, 2), Decimal("98000.00"))]
    adj = artefactos.excel_xlsx(columnas, filas, titulo="Ventas")
    # Un .xlsx es un zip; si xlsxwriter no serializo bien, no abre.
    assert zipfile.ZipFile(io.BytesIO(adj.contenido)).testzip() is None


def test_pdf_acepta_decimal():
    adj = artefactos.pdf_reporte(COLUMNAS, FILAS, titulo="Reporte",
                                 resumen="Resumen de prueba.")
    assert adj.contenido[:5] == b"%PDF-"


# --- Escalas incomparables (2) ------------------------------------------

def test_no_mezcla_series_de_escalas_distintas(monkeypatch):
    """
    unidades (miles) y monto (millones) no pueden compartir eje. Se comprueba
    contando las series que matplotlib termina dibujando.
    """
    trazadas = []
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    original = Axes.barh

    def espia(self, *a, **kw):
        trazadas.append(kw.get("label"))
        return original(self, *a, **kw)

    monkeypatch.setattr(Axes, "barh", espia)
    artefactos.grafico_png(COLUMNAS, FILAS, titulo="Top")
    assert len(trazadas) == 1, f"se dibujaron {len(trazadas)} series: {trazadas}"


def test_si_mezcla_series_comparables():
    """Dos años de ventas SI van juntos: mismo orden de magnitud."""
    columnas = ["mes", "ventas_2025", "ventas_2026"]
    filas = [(f"2026-{m:02d}", Decimal(str(900000 + m * 90000)),
              Decimal(str(1000000 + m * 137000))) for m in range(1, 13)]
    assert artefactos.grafico_png(columnas, filas) is not None


# --- Datos no graficables (3) -------------------------------------------

@pytest.mark.parametrize("columnas,filas", [
    (["total"], [(Decimal("15"),)]),                       # una sola celda
    (["a", "b"], [("x", "y"), ("z", "w")]),                # sin columna numerica
    (["producto", "monto"], [("Cemento", Decimal("1"))]),  # una sola fila
])
def test_no_grafica_lo_que_no_se_puede(columnas, filas):
    assert artefactos.grafico_png(columnas, filas) is None


def test_recorta_categorias_y_lo_dice(monkeypatch):
    """Con mas filas que el tope se grafica el top N, no las 200."""
    monkeypatch.setattr(artefactos.config, "BOT_ADJUNTO_MAX_BARRAS", 10,
                        raising=False)
    columnas = ["producto", "monto"]
    filas = [(f"Producto {i}", Decimal(str(i * 1000))) for i in range(1, 61)]
    adj = artefactos.grafico_png(columnas, filas, titulo="Ventas")
    assert adj is not None


# --- Tope del caption (4) ------------------------------------------------

def test_caption_se_recorta_a_1024():
    largo = "x" * 3000
    assert len(whatsapp._recortar(largo, whatsapp._tope_caption())) <= 1024


def test_texto_se_recorta_a_4096():
    assert len(whatsapp._recortar("x" * 9000)) <= 4096


# --- Nombres de archivo --------------------------------------------------

def test_nombre_de_archivo_sin_tildes_ni_espacios():
    n = artefactos.nombre_archivo("Ventas por sucursal — julio", "xlsx")
    assert n.endswith(".xlsx")
    assert " " not in n
    assert n.isascii()
