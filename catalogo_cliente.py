"""
Catalogo y KPIs centrales de UN cliente.

Antes cada fuente traia su propio catalogo desde ADENTRO de si misma: una
pestania '_catalogo' en el mismo Sheet de datos, o una hoja '_catalogo' en el
mismo Excel de SharePoint. Eso dejaba de tener sentido en cuanto aparecio una
fuente que no tiene "adentro" en absoluto (google_calendar no tiene pestanias).

Ahora el catalogo vive en UN Sheet por cliente, separado de los datos, y
documenta TODAS las fuentes de ese cliente juntas — Sheets, SharePoint,
Calendar, lo que sea. El cliente declara su ID en la pestania 'clientes' del
Sheet maestro, columna 'catalogo_spreadsheet_id'.

sync.py llama leer(cliente) UNA VEZ POR CLIENTE (no una vez por fuente), deja
que TODAS sus fuentes activas carguen sus datos, y RECIEN AL FINAL escribe el
catalogo consolidado (ver sync.py:_escribir_catalogo_del_cliente).

Por que al final y no fuente por fuente (B-40): un KPI o una fila de catalogo
pueden mencionar tablas de VARIAS fuentes del mismo cliente -- el propio
'runway_inventario' de este proyecto cruza 'ventas' e 'inventario' en un JOIN,
y esas dos tablas perfectamente pueden venir de fuentes distintas. Escribir el
catalogo por fuente, con cada fuente viendo solo sus propias tablas, hacia
estructuralmente imposible que una fila asi encontrara sus dos tablas juntas.
El filtrado correcto compara contra la UNION de tablas de todas las fuentes
del cliente, y eso solo se conoce despues de que todas terminaron de correr.

FAIL-CLOSED a proposito: si el cliente no tiene catalogo_spreadsheet_id, o el
Sheet no tiene pestania '_catalogo', el resultado es catalogo vacio. Una tabla
sin fila de catalogo queda BLOQUEADA para el bot (ver bot/catalogo.py) — es la
misma regla de siempre, ahora aplicada a una fuente central en vez de a cada
fuente por separado.
"""

import logging

from gclient import abrir_libro

logger = logging.getLogger("fachavi.catalogo_cliente")

CATALOGO_SHEET = "_catalogo"
KPIS_SHEET = "_kpis"

_KPIS_COLS = ("kpi", "nombre", "descripcion", "preguntas_ejemplo",
              "formula_sql", "tabla", "dimensiones", "unidad",
              "supuestos", "minimo_datos", "instruccion")

# La metadata de edicion es opcional y vive junto a la documentacion del
# catalogo. Conservarla como datos (en vez de codificarla para una tabla en
# particular) permite que cada cliente habilite solo las tablas que conoce y
# mantiene como fuente de verdad.
_CATALOGO_COLS = (
    "fuente_id", "tabla", "columna", "descripcion", "instruccion",
    "sistema_origen", "frecuencia", "dueno",
    "editable", "acciones_permitidas", "origen_edicion", "clave_primaria",
    "anulacion_campo", "requerido", "editable_campo", "tipo_validacion",
    "valores_permitidos", "valor_por_defecto", "calculado_por_sistema",
    "etiqueta_usuario", "ejemplo",
)


def leer(cliente: dict) -> tuple:
    """
    Lee '_catalogo' y '_kpis' del Sheet central de un cliente.
    Devuelve (catalogo_filas, kpis_filas). Vacio y con warning si falta
    catalogo_spreadsheet_id o el Sheet no tiene esas pestanias.
    """
    cid = cliente.get("cliente_id", "")
    spreadsheet_id = str(cliente.get("catalogo_spreadsheet_id", "")).strip()
    if not spreadsheet_id:
        logger.error(
            "[%s] sin catalogo_spreadsheet_id: NINGUNA de sus tablas va a ser "
            "consultable por el bot hasta que se declare ese Sheet en la "
            "pestania 'clientes'.",
            cid,
        )
        return [], []

    try:
        libro = abrir_libro(spreadsheet_id)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[%s] no se pudo abrir el Sheet de catalogo (%s): %s. Revisa que "
            "este compartido con el service account, igual que el Sheet "
            "maestro.",
            cid, spreadsheet_id, e,
        )
        return [], []

    return _leer_catalogo(libro, cid), _leer_kpis(libro, cid)


def _leer_catalogo(libro, cid: str) -> list:
    try:
        ws = libro.worksheet(CATALOGO_SHEET)
    except Exception:  # noqa: BLE001
        logger.warning(
            "[%s] su Sheet de catalogo no tiene pestania '%s'. Ninguna tabla "
            "de este cliente va a ser consultable por el bot.",
            cid, CATALOGO_SHEET,
        )
        return []

    filas = ws.get_all_records()
    norm = []
    for f in filas:
        f = {str(k).strip().lower(): str(v).strip() for k, v in f.items()}
        fila = {
            # 'fuente_id' es NUEVO respecto al catalogo por-fuente de antes:
            # como una sola hoja documenta varias fuentes, hace falta saber a
            # cual pertenece cada fila para filtrar correctamente (ver
            # filtrar_para_fuente). Vacio = aplica a cualquier fuente que
            # tenga esa tabla (retrocompatible con catalogos que no la usan).
            "fuente_id": f.get("fuente_id", ""),
            "tabla": f.get("tabla", ""),
            "columna": f.get("columna", ""),
            "descripcion": f.get("descripcion", ""),
            "instruccion": f.get("instruccion", ""),
            "sistema_origen": f.get("sistema_origen", ""),
            "frecuencia": f.get("frecuencia", ""),
            "dueno": f.get("dueño", "") or f.get("dueno", ""),
        }
        # El resto de reglas es deliberadamente opcional y retrocompatible:
        # una hoja vieja sigue produciendo exactamente el mismo catalogo de
        # solo lectura, mientras que una hoja nueva puede declarar edicion.
        fila.update({c: f.get(c, "") for c in _CATALOGO_COLS if c not in fila})
        norm.append(fila)
    return norm


def _leer_kpis(libro, cid: str) -> list:
    try:
        ws = libro.worksheet(KPIS_SHEET)
    except Exception:  # noqa: BLE001
        logger.info("[%s] su Sheet de catalogo no tiene pestania '%s' (sin KPIs).",
                    cid, KPIS_SHEET)
        return []

    filas = ws.get_all_records()
    norm = []
    for f in filas:
        f = {str(k).strip().lower(): str(v).strip() for k, v in f.items()}
        if not f.get("kpi"):
            continue
        norm.append({c: f.get(c, "") for c in _KPIS_COLS})
    return norm


def tablas_de(valor_tabla: str) -> list:
    """
    Separa la columna 'tabla' de una fila en sus nombres individuales.
    Soporta el formato que ya usa 'runway_inventario' en este proyecto:
    varias tablas separadas por ';' cuando la fila (tipicamente un KPI) las
    cruza en un JOIN. Una sola tabla es el caso normal.
    """
    return [n.strip() for n in str(valor_tabla or "").split(";") if n.strip()]


# NOTA: filtrar_para_fuente() se elimino (B-40). El filtrado por tabla vive
# ahora en sync.py:_escribir_catalogo_del_cliente, porque necesita conocer la
# union de tablas de TODAS las fuentes del cliente -- algo que este modulo,
# que solo LEE el Sheet, no tiene forma de saber. tablas_de() de arriba es el
# unico helper de parseo que sigue haciendo falta aca.
