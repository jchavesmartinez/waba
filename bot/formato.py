"""
¿En que FORMATO quiere la respuesta el usuario?

    detectar("cuánto vendimos ayer")             -> "texto"
    detectar("graficame las ventas por mes")     -> "grafico"
    detectar("pasame eso en Excel")              -> "excel"
    detectar("mandame el reporte en PDF")        -> "pdf"

Se resuelve con REGLAS, no con el modelo. Dos razones:

  - Cuesta cero. El bot ya gasta entre 3 y 5 llamadas por mensaje (A-14); una
    sexta para decidir si el usuario dijo "gráfico" no se justifica.
  - Es un vocabulario cerrado y corto. Nadie pide un grafico con una perifrasis
    creativa: dice "gráfico", "graficá", "en Excel", "en PDF", "mandame el
    archivo". Una lista de palabras cubre practicamente todo, y cuando no
    cubre, el costo del error es bajo: se responde en texto, como siempre.

Va DESPUES del clasificador de intencion (bot/intencion.py), que decide si hace
falta ir a la base. Este modulo solo decide COMO se presenta lo que ya se
consulto: nunca habilita tablas ni cambia la gobernanza.
"""

import re
import unicodedata

import config

TEXTO = "texto"
GRAFICO = "grafico"
EXCEL = "excel"
PDF = "pdf"
CSV = "csv"


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", (s or "").lower())
                   if not unicodedata.combining(c))


# El orden importa: se evalua de arriba hacia abajo y gana el primero que
# aparece. "mandame el gráfico en PDF" -> pdf (el contenedor manda sobre el
# contenido; el PDF ya lleva el grafico adentro).
_REGLAS = (
    (PDF, r"\bpdf\b|\breporte\b|\binforme\b|\bdocumento\b"),
    (EXCEL, r"\bexcel\b|\bxlsx?\b|hoja de calculo|\bplanilla\b|\btabulad|"
            r"\bexcell\b"),
    (CSV, r"\bcsv\b"),
    (GRAFICO, r"\bgrafic\w*|\bgrafiqu\w*|\bchart\b|\bimagen\b|\bfoto\b|"
              r"\bvisualiza\w*|\bplote\w*|\bdiagrama\b|\bde barras\b|"
              r"\bde pastel\b|\bde lineas\b|\bpastel\b"),
    # "pasame el archivo", "exportá eso", "quiero descargarlo": quiere un
    # archivo pero no dijo cual. Excel es el que mas sirve para datos tabulares.
    (EXCEL, r"\barchivo\b|\bexporta\w*|\bexportar\b|\bdescarga\w*|\badjunt\w*|"
            r"\bbajar\w*\b"),
)

_COMPILADAS = tuple((f, re.compile(p)) for f, p in _REGLAS)

_AFIRMACION = re.compile(
    r"^\s*(?:si|sí|dale|claro|correcto|exacto|por supuesto|ok)(?:\b|[,!.])",
    re.IGNORECASE,
)

# Palabras de transporte/presentacion. Se quitan para decidir si el mensaje
# contiene una consulta de negocio nueva o solamente dice "eso en PDF".
_RUIDO_PEDIDO = re.compile(
    r"\b(?:pdf|reporte|informe|documento|excel|xlsx?|hoja de calculo|planilla|"
    r"csv|grafico|grafica|imagen|archivo|exporta\w*|descarga\w*|adjunta\w*|"
    r"manda\w*|pasa\w*|envia\w*|genera\w*|quiero|necesito|dame|en|a|"
    r"por favor|porfa|te|pedi|ya|eso|esto|ese|esa|mismo|misma|anterior|resultado|"
    r"detalle|datos|el|la|los|las|un|una|de|del|si)\b",
    re.IGNORECASE,
)


def detectar(pregunta: str) -> str:
    """Devuelve 'texto' | 'grafico' | 'excel' | 'pdf' | 'csv'."""
    if not getattr(config, "BOT_ADJUNTOS", True):
        return TEXTO

    t = _sin_tildes(pregunta)
    if not t.strip():
        return TEXTO

    for formato, rx in _COMPILADAS:
        if rx.search(t):
            # Degradaciones por configuracion. La Cloud API no lista text/csv
            # entre los MIME de documento aceptados, asi que por defecto un
            # pedido de CSV se sirve como Excel en vez de arriesgar un 400.
            if formato == CSV and not getattr(config, "BOT_ADJUNTO_CSV", False):
                return EXCEL
            if formato == PDF and not getattr(config, "BOT_ADJUNTO_PDF", True):
                return EXCEL
            return formato

    return TEXTO


def es_afirmacion(pregunta: str) -> bool:
    """True para respuestas cortas que aceptan una opcion ofrecida."""
    texto = (pregunta or "").strip()
    return len(texto.split()) <= 10 and bool(_AFIRMACION.search(texto))


def es_pedido_solo_formato(pregunta: str) -> bool:
    """
    Distingue "eso en PDF" de "ventas de agosto en PDF".

    El primero debe reutilizar la consulta anterior; el segundo contiene una
    pregunta de datos nueva y debe pasar por el planificador normal.
    """
    if detectar(pregunta) == TEXTO:
        return False
    restante = _RUIDO_PEDIDO.sub(" ", _sin_tildes(pregunta))
    restante = re.sub(r"\b\d+\b|[^a-z0-9]+", " ", restante).strip()
    return not restante


def detectar_con_contexto(pregunta: str, historial=None) -> str:
    """Hereda el formato ofrecido cuando el usuario responde afirmativamente."""
    actual = detectar(pregunta)
    if actual != TEXTO or not es_afirmacion(pregunta):
        return actual
    for turno in reversed(historial or []):
        if turno.get("rol") != "assistant":
            continue
        contenido = str(turno.get("contenido", ""))
        if "?" not in contenido:
            return TEXTO
        ofrecido = detectar(contenido)
        return ofrecido if ofrecido != TEXTO else TEXTO
    return TEXTO


def es_adjunto(formato: str) -> bool:
    return formato in (GRAFICO, EXCEL, PDF, CSV)
