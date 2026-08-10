"""
Tipos logicos: convierten el texto que devuelve un extractor a un valor tipado.

El vocabulario es CHICO Y CERRADO a proposito. Cada tipo nuevo es una promesa de
mantenimiento, y un vocabulario que crece sin freno termina siendo un lenguaje
de programacion escrito en columnas de Google Sheets: con condicionales y
variables, pero sin depurador, sin control de versiones y sin nadie que sepa
leerlo en seis meses.

Cada tipo declara cuantas columnas produce. Casi todos una; dos de ellos
producen DOS, y esa es justamente la razon de que existan:

  monto_con_moneda   'CRC 3,320.00' -> moneda='CRC', monto=3320.0
  etiqueta_de_lista  'AMEX: ***8715' -> marca='AMEX', valor='***8715'

Sin eso, un mapeo rigido de "una etiqueta -> una columna" no puede representar
los correos reales del BAC, donde la etiqueta de la tarjeta ES el dato (cambia
entre MASTER, VISA y AMEX segun con que se pago).
"""

import logging
import re
from datetime import datetime

logger = logging.getLogger("fachavi.modelo.tipos")

# Meses en espanol como los escribe el BAC ('Ago 9, 2026'). Se aceptan la forma
# corta y la larga porque no hay ninguna garantia de que un sistema use siempre
# la misma, y descubrirlo en produccion cuesta una corrida perdida.
_MESES = {
    "ene": 1, "enero": 1, "feb": 2, "febrero": 2, "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4, "may": 5, "mayo": 5, "jun": 6, "junio": 6,
    "jul": 7, "julio": 7, "ago": 8, "agosto": 8, "set": 9, "sep": 9,
    "setiembre": 9, "septiembre": 9, "oct": 10, "octubre": 10,
    "nov": 11, "noviembre": 11, "dic": 12, "diciembre": 12,
}

# 'Ago 9, 2026 , 16:22' y tambien 'Ago 9,2026 , 00:49'.
# El \s* despues de la coma NO es defensivo de mas: los tres correos de muestra
# del BAC traen las dos variantes, con y sin espacio, en el mismo formato de
# notificacion. Un strptime con formato fijo falla en una de las dos.
_RX_FECHA_ES = re.compile(
    r"([A-Za-zÁÉÍÓÚáéíóú]+)\s*(\d{1,2})\s*,\s*(\d{4})"
    r"(?:\s*,?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?)?"
)

_RX_MONTO = re.compile(
    r"(?P<moneda>[A-Z]{3}|[$₡€])?\s*"
    r"(?P<numero>-?[\d.,]+)\s*"
    r"(?P<moneda2>[A-Z]{3}|[$₡€])?"
)

_SIMBOLOS = {"$": "USD", "₡": "CRC", "€": "EUR"}


def columnas_de(tipo: str, columna: str) -> list:
    """
    Que columnas fisicas produce un campo. El motor lo necesita ANTES de
    procesar filas, para poder crear la tabla aunque no haya ni una fila.
    """
    tipo = (tipo or "texto").strip().lower()
    if tipo == "monto_con_moneda":
        return [(f"{columna}_moneda", "texto"), (columna, "decimal")]
    if tipo == "etiqueta_de_lista":
        return [(f"{columna}_tipo", "texto"), (columna, "texto")]
    return [(columna, tipo)]


def convertir(tipo: str, valor, contexto: dict = None) -> dict:
    """
    Convierte un valor crudo. Devuelve {columna_sufijo: valor} para que un tipo
    pueda producir mas de una columna.

    Un valor que no se puede convertir devuelve None, NO un cero ni una cadena
    vacia. La diferencia importa: 0 significa "gasto cero" y None significa "no
    se pudo leer el monto", y confundirlos convierte un problema visible en un
    reporte silenciosamente equivocado.
    """
    tipo = (tipo or "texto").strip().lower()
    contexto = contexto or {}

    if valor is None:
        return {"": None}

    if tipo == "texto":
        return {"": str(valor).strip()}
    if tipo == "entero":
        return {"": _a_entero(valor)}
    if tipo == "decimal":
        return {"": _a_numero(valor)}
    if tipo == "fecha_hora_es":
        return {"": _a_fecha_es(valor)}
    if tipo == "fecha_iso":
        return {"": _a_fecha_iso(valor)}
    if tipo == "monto_con_moneda":
        moneda, monto = _a_monto(valor)
        return {"_moneda": moneda, "": monto}
    if tipo == "etiqueta_de_lista":
        return {"_tipo": contexto.get("etiqueta", ""), "": str(valor).strip()}

    logger.warning("tipo desconocido '%s'; se trata como texto.", tipo)
    return {"": str(valor).strip()}


def tipos_disponibles() -> list:
    return ["texto", "entero", "decimal", "fecha_hora_es", "fecha_iso",
            "monto_con_moneda", "etiqueta_de_lista"]


# --------------------------------------------------------------------------

def _a_entero(valor):
    """
    Texto a entero SIN pasar por float.

    float64 solo representa enteros exactos hasta 2^53 (~9x10^15). Un
    consecutivo de Hacienda de 20 digitos o un numero de tarjeta caen justo
    despues de ese limite, y convertirlos con float() les cambia los ultimos
    digitos EN SILENCIO: '00100001010000000123' se guardaba como
    100001010000000128. El dato queda mal, se ve plausible y no hay ninguna
    alerta.

    (Un identificador NO deberia declararse como 'entero' en la metadata —
    pierde los ceros de la izquierda y a veces trae letras — pero que alguien
    lo declare mal no puede terminar en digitos alterados.)
    """
    if valor is None:
        return None
    crudo = str(valor).strip()
    solo_digitos = re.sub(r"[^\d\-]", "", crudo)

    # Con separador decimal o de miles hay que interpretarlo ('12.7' -> 13,
    # '3,320' -> 3320), asi que se pasa por el camino normal. Sin separadores
    # es un entero puro y se convierte DIRECTO, sin tocar float.
    if ("." in crudo or "," in crudo) or not solo_digitos or solo_digitos == "-":
        n = _a_numero(valor)
        return None if n is None else int(round(n))
    try:
        return int(solo_digitos)
    except ValueError:
        return None


def _a_numero(valor):
    """
    Texto a numero, resolviendo el separador de miles.

    '3,320.00' son tres mil trescientos veinte, no tres coma treinta y dos. Es
    la misma trampa que ya esta documentada en sources/base.py, y aca es peor
    porque el error no rompe nada: produce un gasto mil veces menor que se ve
    perfectamente plausible en un reporte.

    La regla: el separador que aparece MAS A LA DERECHA es el decimal, salvo
    que lo sigan exactamente tres digitos y no haya otro separador, en cuyo
    caso es de miles ('3,320' -> 3320).
    """
    if valor is None:
        return None
    texto = str(valor).strip().replace(" ", "")
    if not texto:
        return None
    texto = re.sub(r"[^\d.,\-]", "", texto)
    if not texto or texto in {"-", ".", ","}:
        return None

    puntos, comas = texto.count("."), texto.count(",")

    if puntos and comas:
        # Conviven los dos: el que aparece MAS A LA DERECHA es el decimal y el
        # otro es de miles. Cubre '3,320.00' y '3.320,00' sin suponer region.
        if texto.rfind(".") > texto.rfind(","):
            texto = texto.replace(",", "")
        else:
            texto = texto.replace(".", "").replace(",", ".")
    elif puntos or comas:
        sep = "." if puntos else ","
        veces = puntos or comas
        decimales = len(texto) - texto.rfind(sep) - 1
        # Un solo separador seguido de EXACTAMENTE tres digitos es ambiguo
        # ('3,320' podrian ser 3 con 320 milesimas). Se toma como miles, que es
        # como lo escribe todo el mundo aca: leerlo como decimal convertiria un
        # gasto de 3.320 colones en 3,32 y el reporte quedaria mil veces menor,
        # perfectamente plausible y sin ninguna alerta.
        if veces > 1 or decimales == 3:
            texto = texto.replace(sep, "")
        elif sep == ",":
            texto = texto.replace(",", ".")

    try:
        return float(texto)
    except ValueError:
        return None


def _a_monto(valor):
    """'CRC 3,320.00' -> ('CRC', 3320.0). Tambien '₡3.320,00' y '113.13 USD'."""
    texto = str(valor or "").strip()
    if not texto:
        return None, None
    m = _RX_MONTO.search(texto)
    if not m:
        return None, _a_numero(texto)
    moneda = m.group("moneda") or m.group("moneda2") or ""
    moneda = _SIMBOLOS.get(moneda, moneda).upper() or None
    return moneda, _a_numero(m.group("numero"))


def _a_fecha_es(valor):
    """
    'Ago 9, 2026 , 16:22' -> datetime(2026, 8, 9, 16, 22).

    Sin zona horaria a proposito: el correo no la declara, y asumir una seria
    inventar informacion. La hora es la local del banco (Costa Rica); si algun
    dia hace falta, se convierte con una regla explicita en la metadata, no
    adivinando aca.
    """
    texto = str(valor or "").strip()
    if not texto:
        return None
    m = _RX_FECHA_ES.search(texto)
    if not m:
        return _a_fecha_iso(texto)

    mes = _MESES.get(_sin_acentos(m.group(1)).lower()[:10])
    if mes is None:
        mes = _MESES.get(_sin_acentos(m.group(1)).lower()[:3])
    if mes is None:
        return None
    try:
        return datetime(
            int(m.group(3)), mes, int(m.group(2)),
            int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0),
        )
    except ValueError:
        return None


def _a_fecha_iso(valor):
    texto = str(valor or "").strip()
    if not texto:
        return None
    for formato in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                    "%d/%m/%Y %H:%M", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(texto[:19], formato)
        except ValueError:
            continue
    return None


def _sin_acentos(texto: str) -> str:
    import unicodedata
    plano = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in plano if not unicodedata.combining(c))
