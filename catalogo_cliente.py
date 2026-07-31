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

sync.py llama leer(cliente) UNA VEZ POR CLIENTE (no una vez por fuente) y
reparte el resultado: cada fuente se queda solo con las filas de catalogo/kpis
que le corresponden a las tablas que ELLA cargo (ver filtrar_para_fuente()).
Sin ese filtrado, cada fuente pisaria en el warehouse el catalogo completo del
cliente, y la ultima fuente en correr "ganaria" sobre las demas.

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
        norm.append({
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
        })
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


def filtrar_para_fuente(filas: list, fuente_id: str, tablas_cargadas: set) -> list:
    """
    Se queda con las filas de catalogo/kpis que le corresponden a UNA fuente.

    Regla: si la fila trae 'fuente_id' y no coincide con esta fuente, se
    descarta (evita que dos fuentes con una tabla del mismo nombre —p.ej. dos
    hojas llamadas 'ventas' de fuentes distintas— se mezclen). Si la fila NO
    trae 'fuente_id' (catalogo viejo o alguien lo dejo vacio a proposito), se
    filtra solo por nombre de tabla, igual que se hacia antes.

    Esto es lo que impide que la ULTIMA fuente en correr "gane" y pise en el
    warehouse el catalogo completo del cliente: cada fuente escribe solo lo
    que es suyo.
    """
    cargadas = {t.strip().lower() for t in tablas_cargadas}
    resultado = []
    for f in filas:
        fid_fila = str(f.get("fuente_id", "")).strip()
        if fid_fila and fid_fila != fuente_id:
            continue
        tabla = str(f.get("tabla", "")).strip().lower()
        if tabla and tabla not in cargadas:
            continue
        resultado.append(f)
    return resultado
