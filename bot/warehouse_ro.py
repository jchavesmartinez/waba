"""
Acceso de SOLO LECTURA al warehouse (Neon / Postgres) de un cliente.

El bot NUNCA escribe. Esta capa:
  - resuelve a que proyecto de Neon va cada cliente (reusa
    config.dsn_de_cliente, la MISMA logica que usa la ingesta),
  - cachea un engine de SQLAlchemy por DSN,
  - ejecuta SQL dentro de una transaccion READ ONLY, con:
      * search_path fijado a los esquemas del cliente (semantic_ y raw_), para que
        el modelo pueda nombrar las tablas sin prefijo y NO pueda saltar a otro
        esquema por descuido;
      * statement_timeout, que corta consultas que se pasen de tiempo;
      * un tope de filas aplicado por la app (fetchmany), por si el SQL no trae
        LIMIT.

Esto es la ULTIMA linea de defensa, no la unica: el SQL ya viene validado por
bot/nl2sql.py (solo SELECT, solo tablas permitidas). Aca la garantia es que,
aunque algo se colara, la transaccion es de solo lectura y Postgres aborta
cualquier escritura.
"""

import logging
import threading

from sqlalchemy import create_engine, text

import config
from warehouse.base import nombre_esquema
from modelo.construir import nombre_esquema_semantico

logger = logging.getLogger("fachavi.bot.warehouse_ro")

_engines: dict = {}
# B-28: el diccionario de engines es global y el bot atiende varios mensajes a
# la vez (BackgroundTasks). Sin lock, dos mensajes simultaneos del mismo cliente
# podian crear dos engines y dejar uno huerfano con su pool de conexiones
# abierto. Con Neon, que cobra por conexion-hora, eso no es gratis.
_lock_engines = threading.Lock()


def _engine(cliente: dict):
    """Devuelve (y cachea) un engine para el DSN de este cliente."""
    dsn = config.dsn_de_cliente(cliente)
    eng = _engines.get(dsn)
    if eng is not None:
        return eng
    with _lock_engines:
        if dsn not in _engines:      # doble chequeo: otro hilo pudo crearlo
            # pool_pre_ping: las bases serverless de Neon se duermen; sin esto la
            # primera consulta tras un rato falla con "connection closed".
            _engines[dsn] = create_engine(dsn, pool_pre_ping=True)
            logger.info("engine RO abierto para cliente '%s'",
                        cliente.get("cliente_id"))
    return _engines[dsn]


def _prep(cx, esquema: str, cliente_id: str = ""):
    """Configura la sesion de solo lectura sobre una transaccion ya abierta."""
    # READ ONLY tiene que ser lo primero de la transaccion. Aunque el SQL se
    # cuele con un INSERT/UPDATE, Postgres lo rechaza.
    cx.execute(text("SET TRANSACTION READ ONLY"))
    cx.execute(text(f"SET LOCAL statement_timeout = {int(config.BOT_TIMEOUT_MS)}"))
    # Aisla al cliente: solo ve SUS esquemas. Sin 'public' ni los de otros
    # clientes. Son dos porque la capa semantica escribe aparte:
    #   semantic_<cliente>  tablas derivadas (listas para consumo)
    #   raw_<cliente>       tablas ingestadas tal cual
    # El aislamiento ENTRE CLIENTES no cambia: los dos esquemas son del mismo.
    # El orden importa: si una derivada y una raw se llamaran igual, gana la
    # derivada, que es la que tiene la logica de negocio aplicada.
    cx.execute(text(
        f'SET LOCAL search_path = "{nombre_esquema_semantico(cliente_id)}", '
        f'"{esquema}"'))


def ejecutar(cliente: dict, sql: str, limite: int | None = None):
    """
    Ejecuta SQL (ya validado como solo-lectura) y devuelve (columnas, filas).

    `filas` es una lista de tuplas; se corta a `limite` filas (por defecto
    config.BOT_MAX_FILAS) aunque el SELECT devolviera mas.

    A-18 — que hay que entender de este tope: se aplica al TRAER los datos, no
    al calcularlos. Una consulta puede procesar millones de filas en el servidor
    de base de datos y devolver 200. La proteccion efectiva del warehouse es el
    statement_timeout de _prep(), no este limite; este solo protege la memoria
    del proceso del bot y el tamaño del prompt de redaccion.
    """
    limite = limite or config.BOT_MAX_FILAS
    esquema = nombre_esquema(cliente["cliente_id"])
    with _engine(cliente).connect() as cx:
        trans = cx.begin()
        try:
            _prep(cx, esquema, cliente["cliente_id"])
            res = cx.execute(text(sql))
            columnas = list(res.keys())
            filas = [tuple(r) for r in res.fetchmany(limite)]
            return columnas, filas
        finally:
            # Nunca commit: es solo lectura, no hay nada que confirmar.
            trans.rollback()


def listar_tablas(cliente: dict) -> list:
    """
    Devuelve los nombres REALES (fisicos) de todas las tablas de datos del
    esquema de un cliente, leyendo information_schema. Excluye las de
    metadata (_catalogo, _kpis) y cualquier otra que empiece con '_'.

    B-40: el catalogo ahora es consolidado por cliente y sus filas ya NO
    guardan de forma confiable el fuente_id real de cada tabla (ver
    bot/catalogo.py:resolver_tablas). En vez de reconstruir el nombre fisico
    a partir de fuente_id + tabla logica (que ya no es seguro), se resuelve
    al reves: se listan las tablas reales que existen de verdad en el
    warehouse, y se busca cual TERMINA en '__<tabla_logica>'.
    """
    cid = cliente["cliente_id"]
    sql = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = ANY(:esq) AND table_type = 'BASE TABLE'
    """
    filas = leer_interno(cliente, sql, {
        "esq": [nombre_esquema_semantico(cid), nombre_esquema(cid)]})
    # Se excluyen las que empiezan con '_' (metadata: _catalogo, _kpis, _mapeo)
    # y las de rechazos: son diagnostico de la ingesta, no data de negocio, y
    # solo gastarian espacio del prompt.
    return sorted({
        f["table_name"] for f in filas
        if not f["table_name"].startswith("_")
        and not f["table_name"].endswith("__rechazos")
    })


def listar_columnas(cliente: dict, tablas_reales) -> dict:
    """
    Devuelve {tabla_real: [(columna, tipo), ...]} leyendo information_schema,
    SOLO de las tablas que se le pasan (las permitidas por el catalogo).
    """
    tablas = list(tablas_reales or [])
    if not tablas:
        return {}
    cid = cliente["cliente_id"]
    sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = ANY(:esq) AND table_name = ANY(:tablas)
        ORDER BY table_name, ordinal_position
    """
    filas = leer_interno(cliente, sql, {
        "esq": [nombre_esquema_semantico(cid), nombre_esquema(cid)],
        "tablas": tablas})
    out: dict = {}
    for f in filas:
        out.setdefault(f["table_name"], []).append((f["column_name"], f["data_type"]))
    return out


def leer_interno(cliente: dict, sql: str, params: dict | None = None):
    """
    Consulta CONFIABLE de la propia app (catalogo, information_schema). No pasa
    por el validador de nl2sql porque el SQL lo escribimos nosotros, con
    parametros ligados. Igual corre READ ONLY y con search_path del cliente.
    """
    esquema = nombre_esquema(cliente["cliente_id"])
    with _engine(cliente).connect() as cx:
        trans = cx.begin()
        try:
            _prep(cx, esquema, cliente["cliente_id"])
            res = cx.execute(text(sql), params or {})
            columnas = list(res.keys())
            filas = [dict(zip(columnas, r)) for r in res.fetchall()]
            return filas
        finally:
            trans.rollback()
