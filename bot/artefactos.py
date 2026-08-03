"""
Construye los ARCHIVOS que el bot manda por WhatsApp a partir del resultado de
la consulta (columnas, filas), sin tocar la base ni el modelo.

    grafico_png(...)  -> Adjunto(image/png)
    excel_xlsx(...)   -> Adjunto(.xlsx)
    csv_texto(...)    -> Adjunto(.csv)   [ver advertencia de MIME abajo]
    pdf_reporte(...)  -> Adjunto(.pdf)

Todo se genera en memoria (BytesIO) y se devuelve como bot.salida.Adjunto.

Tres decisiones que importan:

1. matplotlib SIN interfaz grafica. En un servidor no hay pantalla; hay que
   fijar el backend 'Agg' ANTES de importar pyplot o el import truena o se
   cuelga. Por eso el import de pyplot esta adentro de la funcion, despues del
   matplotlib.use("Agg").

2. Los tipos que devuelve psycopg2 no son los de Python. Un SUM() en Postgres
   vuelve como decimal.Decimal, una fecha como datetime.date. matplotlib no
   grafica Decimal y xlsxwriter no lo serializa: hay que convertir. Es la fuente
   #1 de errores silenciosos en este tipo de codigo.

3. El grafico se decide con REGLAS, no con el modelo. Pedirle a un LLM que elija
   el tipo de grafico cuesta una llamada mas por mensaje y acierta parecido a
   "una columna de texto en el eje X, una numerica en el eje Y". Si las reglas
   no encuentran nada graficable, se devuelve None y el bot responde en texto
   avisando — no inventa un grafico vacio.
"""

import csv
import io
import logging
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal

import config
from bot.salida import Adjunto

logger = logging.getLogger("fachavi.bot.artefactos")

MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIME_PDF = "application/pdf"
MIME_CSV = "text/csv"
MIME_PNG = "image/png"

# Cuanto pueden diferir dos series para compartir el mismo eje. Un factor de 20
# ya deja una de las dos como una linea pegada al piso.
_TOLERANCIA_ESCALA = 20.0


# --- Normalizacion de valores -------------------------------------------

def _a_numero(v):
    """Devuelve float si el valor es numerico; None si no lo es."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    return None


def _a_texto(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10] if not isinstance(v, datetime) else v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, Decimal):
        return f"{float(v):,.2f}"
    return str(v)


def _columnas_numericas(columnas, filas) -> list:
    """Indices de columnas donde TODOS los valores no nulos son numericos."""
    idx = []
    for i in range(len(columnas)):
        vistos = 0
        for f in filas:
            v = f[i]
            if v is None:
                continue
            if _a_numero(v) is None:
                vistos = -1
                break
            vistos += 1
        if vistos > 0:
            idx.append(i)
    return idx


def _es_temporal(columnas, filas, i) -> bool:
    """¿La columna i parece una serie de tiempo? (fechas o 'YYYY-MM')."""
    nombre = str(columnas[i]).lower()
    if any(p in nombre for p in ("fecha", "dia", "día", "mes", "semana",
                                 "periodo", "período", "año", "anio", "date")):
        return True
    for f in filas[:5]:
        v = f[i]
        if isinstance(v, (date, datetime)):
            return True
        if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v.strip()):
            return True
    return False


def nombre_archivo(base: str, extension: str) -> str:
    """Nombre seguro y con fecha: 'ventas por mes' -> ventas_por_mes_2026-08-03.xlsx"""
    txt = unicodedata.normalize("NFKD", str(base or "consulta"))
    txt = txt.encode("ascii", "ignore").decode("ascii").lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt).strip("_")[:40] or "consulta"
    return f"{txt}_{date.today().isoformat()}.{extension}"


# --- Grafico -------------------------------------------------------------

def grafico_png(columnas, filas, titulo: str = "", caption: str = ""):
    """
    Arma un grafico de barras o de linea. Devuelve un Adjunto o None si los
    datos no dan para graficar (una sola celda, sin columna numerica, etc.).
    """
    if not filas or len(columnas) < 2 or len(filas) < 2:
        return None

    num = _columnas_numericas(columnas, filas)
    if not num:
        return None

    # Eje X: la primera columna que NO sea numerica; si todas lo son, la primera.
    candidatas = [i for i in range(len(columnas)) if i not in num]
    i_x = candidatas[0] if candidatas else 0
    series = [i for i in num if i != i_x][:3]      # hasta 3 series
    if not series:
        return None

    # Un SELECT tipico trae varias columnas numericas de golpe (unidades Y
    # monto). Dibujarlas en el mismo eje es peor que no dibujarlas: 890 unidades
    # al lado de ₡4.120.000 es una barra de un pixel, y el grafico dice
    # exactamente lo contrario de lo que pasa. Si las magnitudes no son
    # comparables se grafica SOLO la primera; el resto igual va en el texto.
    def _escala(i):
        vals = [abs(_a_numero(f[i]) or 0.0) for f in filas]
        return max(vals) or 1.0

    base = _escala(series[0])
    series = [i for i in series
              if 1 / _TOLERANCIA_ESCALA <= _escala(i) / base <= _TOLERANCIA_ESCALA]

    # Tope de categorias. Una barra menos de ~15 px de alto no se lee, asi que
    # las barras se recortan agresivo; una LINEA con 90 puntos, en cambio, se lee
    # perfecto y recortarla a 25 mutila justo la tendencia que se pidio ver
    # ("ventas por dia" de un mes son 30 puntos, no 25).
    tope = int(getattr(config, "BOT_ADJUNTO_MAX_BARRAS", 25))
    temporal = _es_temporal(columnas, filas, i_x)
    if temporal:
        tope = int(getattr(config, "BOT_ADJUNTO_MAX_PUNTOS", 90))
    datos = list(filas)
    recortado = False
    if len(datos) > tope:
        recortado = True
        if temporal:
            datos = datos[-tope:]                  # los periodos mas recientes
        else:
            datos = sorted(
                datos,
                key=lambda f: abs(_a_numero(f[series[0]]) or 0),
                reverse=True,
            )[:tope]

    etiquetas = [_a_texto(f[i_x])[:22] or "(vacío)" for f in datos]
    valores = {i: [(_a_numero(f[i]) or 0.0) for f in datos] for i in series}

    import matplotlib
    matplotlib.use("Agg")                          # sin pantalla: obligatorio
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Barras horizontales cuando las etiquetas son largas: en vertical se
    # encabalgan y quedan ilegibles justo en el celular, que es donde se ven.
    horizontal = (not temporal) and max(len(e) for e in etiquetas) > 10
    alto = max(3.2, 0.38 * len(datos)) if horizontal else 4.2
    fig, ax = plt.subplots(figsize=(7.5, min(alto, 9.5)), dpi=140)

    if temporal:
        for i in series:
            ax.plot(etiquetas, valores[i], marker="o", linewidth=2,
                    markersize=4 if len(datos) > 20 else 6,
                    label=str(columnas[i]))
        # Con 30 fechas de 10 caracteres cada una, las etiquetas se encinman
        # hasta volverse una mancha. Se muestra una de cada N: la linea ya
        # comunica la forma, las fechas solo dan la referencia. El presupuesto
        # depende del LARGO: caben 12 etiquetas de '2026-01' pero solo 8 de
        # '2026-01-15'.
        largo = max(len(e) for e in etiquetas)
        presupuesto = 12 if largo <= 7 else 8
        paso = max(1, -(-len(datos) // presupuesto))    # division hacia arriba
        marcas = range(0, len(datos), paso)
        ax.set_xticks(list(marcas))
        ax.set_xticklabels([etiquetas[k] for k in marcas],
                           rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    elif horizontal:
        pos = range(len(datos))
        ancho = 0.8 / len(series)
        for k, i in enumerate(series):
            ax.barh([p + k * ancho for p in pos], valores[i], height=ancho,
                    label=str(columnas[i]))
        ax.set_yticks([p + ancho * (len(series) - 1) / 2 for p in pos])
        ax.set_yticklabels(etiquetas)
        ax.invert_yaxis()                          # el mayor arriba
        ax.grid(axis="x", alpha=0.25)
    else:
        pos = range(len(datos))
        ancho = 0.8 / len(series)
        for k, i in enumerate(series):
            ax.bar([p + k * ancho for p in pos], valores[i], width=ancho,
                   label=str(columnas[i]))
        ax.set_xticks([p + ancho * (len(series) - 1) / 2 for p in pos])
        ax.set_xticklabels(etiquetas, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)

    eje_valor = ax.xaxis if horizontal else ax.yaxis
    eje_valor.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    if len(series) > 1:
        ax.legend(fontsize=8)
    t = (titulo or str(columnas[series[0]])).strip()
    if recortado:
        t += f" (top {tope})" if not temporal else f" (últimos {tope})"
    ax.set_title(t[:80], fontsize=12, pad=12)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)                                 # sin esto la figura queda en
    buf.seek(0)                                    # memoria: fuga garantizada

    return Adjunto(
        tipo="image",
        contenido=buf.getvalue(),
        nombre=nombre_archivo(titulo or "grafico", "png"),
        mime=MIME_PNG,
        caption=caption,
    )


# --- Excel ---------------------------------------------------------------

def excel_xlsx(columnas, filas, titulo: str = "", caption: str = ""):
    """Hoja unica con encabezado congelado, autofiltro y anchos razonables."""
    import xlsxwriter

    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True,
                                   "default_date_format": "yyyy-mm-dd"})
    ws = wb.add_worksheet("datos")

    f_enc = wb.add_format({"bold": True, "bg_color": "#1F3864",
                           "font_color": "white", "border": 1})
    f_num = wb.add_format({"num_format": "#,##0.00"})
    f_fecha = wb.add_format({"num_format": "yyyy-mm-dd"})

    anchos = []
    for c, nombre in enumerate(columnas):
        ws.write(0, c, str(nombre), f_enc)
        anchos.append(len(str(nombre)) + 2)

    for r, fila in enumerate(filas, start=1):
        for c, v in enumerate(fila):
            if v is None:
                ws.write_blank(r, c, None)
                continue
            n = _a_numero(v)
            if n is not None:
                ws.write_number(r, c, n, f_num)
                anchos[c] = max(anchos[c], 14)
            elif isinstance(v, (datetime, date)):
                ws.write_datetime(r, c, v, f_fecha)
                anchos[c] = max(anchos[c], 12)
            else:
                s = str(v)
                ws.write_string(r, c, s)
                anchos[c] = max(anchos[c], min(len(s) + 2, 50))

    for c, w in enumerate(anchos):
        ws.set_column(c, c, w)
    ws.freeze_panes(1, 0)
    if filas:
        ws.autofilter(0, 0, len(filas), max(len(columnas) - 1, 0))
    wb.close()
    buf.seek(0)

    return Adjunto(
        tipo="document",
        contenido=buf.getvalue(),
        nombre=nombre_archivo(titulo or "consulta", "xlsx"),
        mime=MIME_XLSX,
        caption=caption,
    )


# --- CSV -----------------------------------------------------------------

def csv_texto(columnas, filas, titulo: str = "", caption: str = ""):
    """
    CSV con BOM (para que Excel en Windows abra las tildes bien).

    OJO: la lista de MIME de documento que acepta la Cloud API no incluye
    text/csv. Puede pasar o puede volver con un 400. Por eso config.BOT_ADJUNTO_CSV
    viene en "no" por defecto y un pedido de CSV se sirve como .xlsx.
    """
    salida = io.StringIO()
    w = csv.writer(salida)
    w.writerow([str(c) for c in columnas])
    for fila in filas:
        w.writerow([_a_texto(v) for v in fila])
    datos = ("\ufeff" + salida.getvalue()).encode("utf-8")

    return Adjunto(
        tipo="document",
        contenido=datos,
        nombre=nombre_archivo(titulo or "consulta", "csv"),
        mime=MIME_CSV,
        caption=caption,
    )


# --- PDF -----------------------------------------------------------------

def pdf_reporte(columnas, filas, titulo: str = "", resumen: str = "",
                incluir_grafico: bool = True, caption: str = ""):
    """
    Reporte de una pagina: titulo, resumen en texto, grafico (si sale) y tabla.
    Es el formato para "mandame el reporte" / "pasalo a PDF".
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=titulo or "Reporte")
    estilos = getSampleStyleSheet()
    hist = [Paragraph(titulo or "Reporte de datos", estilos["Title"]),
            Paragraph(date.today().strftime("Generado el %d/%m/%Y"),
                      estilos["Normal"]),
            Spacer(1, 12)]

    if resumen:
        hist.append(Paragraph(resumen.replace("\n", "<br/>"), estilos["BodyText"]))
        hist.append(Spacer(1, 12))

    if incluir_grafico:
        try:
            g = grafico_png(columnas, filas, titulo=titulo)
            if g:
                img = ImageReader(io.BytesIO(g.contenido))
                iw, ih = img.getSize()
                ancho = min(16 * cm, iw)
                hist.append(Image(io.BytesIO(g.contenido),
                                  width=ancho, height=ancho * ih / iw))
                hist.append(Spacer(1, 14))
        except Exception as e:  # noqa: BLE001
            logger.warning("no se pudo incrustar el grafico en el PDF: %s", e)

    tope_pdf = int(getattr(config, "BOT_ADJUNTO_PDF_MAX_FILAS", 60))
    datos = [[str(c) for c in columnas]]
    datos += [[_a_texto(v)[:40] for v in f] for f in filas[:tope_pdf]]
    tabla = Table(datos, repeatRows=1, hAlign="LEFT")
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F2F2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    hist.append(tabla)
    if len(filas) > tope_pdf:
        hist.append(Spacer(1, 8))
        hist.append(Paragraph(
            f"<i>Se muestran las primeras {tope_pdf} de {len(filas)} filas. "
            f"Pedí el archivo en Excel para verlas todas.</i>",
            estilos["Normal"]))

    doc.build(hist)
    buf.seek(0)

    return Adjunto(
        tipo="document",
        contenido=buf.getvalue(),
        nombre=nombre_archivo(titulo or "reporte", "pdf"),
        mime=MIME_PDF,
        caption=caption,
    )
