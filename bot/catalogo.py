"""
Gobernanza en tiempo de request: QUE tablas puede leer el bot.

La ingesta ya dejo en `raw_<cliente>._catalogo` una fila por tabla/columna del
cliente, con una columna `instruccion` de texto libre (p.ej. "esta tabla puede
ser usada por el bot"). Este modulo:

  1. Lee ese catalogo del warehouse (no de Google Sheets).
  2. Decide, tabla por tabla, si el bot la puede consultar, mirando la
     'instruccion'. Regla: hay tablas que el bot NO debe leer (ventas internas,
     datos de RRHH, etc.) y otras que SI. La decision es DATO, editable desde el
     Sheet del cliente, sin tocar codigo.
  3. Arma el texto de schema de SOLO las tablas permitidas, para el prompt de
     text-to-SQL. Las tablas prohibidas ni siquiera aparecen: el modelo no sabe
     que existen.

Nombre real de la tabla en Neon:
    catalogo.tabla es el nombre LOGICO ('ventas'); en el warehouse la tabla se
    llama <fuente_id>__<tabla> ('sheet_ventas__ventas'). El catalogo guarda
    'fuente_id', asi que reconstruimos el nombre real con warehouse.nombre_tabla.
"""

import logging
from dataclasses import dataclass, field

import config
from warehouse.base import nombre_tabla
from bot import warehouse_ro

logger = logging.getLogger("fachavi.bot.catalogo")

# Columnas de linaje que agrega la ingesta a cada tabla. Utiles ("¿de cuando es
# este dato?" -> _ingestado_en) pero se marcan aparte para no confundir al modelo.
_COLS_LINAJE = {"_corrida_id", "_fuente_id", "_ingestado_en"}

# Reglas deterministas para leer la 'instruccion'. Se revisan los NEGATIVOS
# primero: "esta tabla NO debe ser usada por el bot" contiene "puede"/"bot" pero
# la intencion es prohibir, y el negativo tiene que ganar.
_NEG = (
    "no debe", "no se debe", "no usar", "no la use", "no leer", "no exponer",
    "no publicar", "no bot", "prohib", "restringid", "confidencial", "sensible",
    "uso interno", "solo interno", "interno", "privad", "oculta",
)
_POS = (
    "puede ser usada por el bot", "puede usarse por el bot", "puede usar el bot",
    "puede ser usada", "puede usar", "usar por el bot", "disponible para el bot",
    "expone", "publica", "si bot", "habilitad",
)


def _puede_bot(instruccion: str) -> bool:
    """
    True si la 'instruccion' habilita a la tabla para el bot.
    Vacia/ambigua -> config.BOT_PERMITIR_SIN_INSTRUCCION (por defecto False).
    """
    t = (instruccion or "").strip().lower()
    if not t:
        return config.BOT_PERMITIR_SIN_INSTRUCCION
    if any(n in t for n in _NEG):
        return False
    if any(p in t for p in _POS):
        return True
    return config.BOT_PERMITIR_SIN_INSTRUCCION


@dataclass
class TablaPermitida:
    tabla_logica: str            # 'ventas'
    tabla_real: str              # 'sheet_ventas__ventas'
    fuente_id: str               # 'sheet_ventas'
    descripcion: str = ""        # de la fila columna='*'
    instruccion: str = ""
    # {columna: descripcion} de las filas por-columna del catalogo
    columnas_doc: dict = field(default_factory=dict)


@dataclass
class Contexto:
    schema_text: str                       # bloque para el prompt de nl2sql
    tablas_reales: set = field(default_factory=set)   # lista blanca para el validador
    permitidas: list = field(default_factory=list)    # [TablaPermitida]
    catalogo_tiene_instruccion: bool = True


def _leer_filas_catalogo(cliente: dict) -> tuple[list, bool]:
    """
    Devuelve (filas, hay_columna_instruccion). Robusto a un _catalogo viejo que
    todavia no tenga la columna 'instruccion' (SELECT * y se detecta si vino).
    """
    esquema_tabla = '"_catalogo"'  # search_path ya apunta al esquema del cliente
    try:
        filas = warehouse_ro.leer_interno(cliente, f"SELECT * FROM {esquema_tabla}")
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo leer _catalogo: %s", cliente.get("cliente_id"), e)
        return [], True
    hay_instr = bool(filas) and ("instruccion" in filas[0])
    return filas, hay_instr


def resolver_tablas(cliente: dict) -> list:
    """
    Aplica la regla de gobernanza y devuelve la lista de TablaPermitida.
    Agrupa el catalogo por (fuente_id, tabla) y decide con la fila '*' si existe;
    si no hay fila '*', exige que TODAS las filas de esa tabla habiliten (fail-closed).
    """
    filas, hay_instr = _leer_filas_catalogo(cliente)
    if not hay_instr:
        logger.warning(
            "[%s] el _catalogo no tiene columna 'instruccion'. Re-corre la "
            "ingesta (python sync.py --forzar) para que viaje desde el Sheet.",
            cliente.get("cliente_id"),
        )

    grupos: dict = {}
    for f in filas:
        f = {str(k).strip().lower(): ("" if v is None else str(v).strip())
             for k, v in f.items()}
        tabla = f.get("tabla", "")
        fuente = f.get("fuente_id", "")
        if not tabla or not fuente:
            continue
        grupos.setdefault((fuente, tabla), []).append(f)

    permitidas = []
    for (fuente, tabla), rows in grupos.items():
        estrella = next((r for r in rows if r.get("columna", "") in ("*", "")), None)
        if estrella is not None:
            ok = _puede_bot(estrella.get("instruccion", ""))
            instr = estrella.get("instruccion", "")
            desc = estrella.get("descripcion", "")
        else:
            # Sin fila de tabla completa: se permite solo si NINGUNA fila prohibe
            # y al menos una habilita.
            ok = all(_puede_bot(r.get("instruccion", "")) for r in rows) and bool(rows)
            instr = rows[0].get("instruccion", "")
            desc = ""

        if not ok:
            logger.info("[%s] tabla '%s' NO habilitada para el bot (instruccion=%r)",
                        cliente.get("cliente_id"), tabla, instr[:60])
            continue

        cols_doc = {r["columna"]: r.get("descripcion", "")
                    for r in rows if r.get("columna", "") not in ("*", "")}
        permitidas.append(TablaPermitida(
            tabla_logica=tabla,
            tabla_real=nombre_tabla(fuente, tabla),
            fuente_id=fuente,
            descripcion=desc,
            instruccion=instr,
            columnas_doc=cols_doc,
        ))

    return permitidas


def construir_contexto(cliente: dict) -> Contexto:
    """
    Arma el Contexto para nl2sql: texto de schema de las tablas permitidas
    (columnas reales de information_schema + descripcion del catalogo) y la
    lista blanca de nombres reales para el validador.
    """
    permitidas = resolver_tablas(cliente)
    if not permitidas:
        return Contexto(schema_text="", tablas_reales=set(), permitidas=[])

    reales = {t.tabla_real for t in permitidas}
    cols_por_tabla = warehouse_ro.listar_columnas(cliente, reales) \
        if hasattr(warehouse_ro, "listar_columnas") else {}

    bloques = []
    for t in permitidas:
        cols = cols_por_tabla.get(t.tabla_real, [])
        lineas_col = []
        for col, tipo in cols:
            if col in _COLS_LINAJE:
                doc = "linaje (columna tecnica de la ingesta)"
            else:
                doc = t.columnas_doc.get(col, "")
            lineas_col.append(f"    - {col} ({tipo}){(': ' + doc) if doc else ''}")
        encabezado = f"Tabla {t.tabla_real}"
        if t.descripcion:
            encabezado += f" — {t.descripcion}"
        bloques.append(encabezado + "\n" + "\n".join(lineas_col))

    return Contexto(
        schema_text="\n\n".join(bloques),
        tablas_reales=reales,
        permitidas=permitidas,
    )
