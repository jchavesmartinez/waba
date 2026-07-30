"""
Memoria conversacional del bot.

Guarda los turnos de cada numero para dar continuidad entre mensajes (y entre
dias). A diferencia de bot/warehouse_ro.py, esta capa SI escribe — pero con dos
limites duros de gobernanza:

  1. Escribe UNICAMENTE en el esquema `_bot` (tabla `_bot.conversaciones`).
     Nunca toca los esquemas de datos del cliente (raw_<cliente_id>). Los datos
     del cliente siguen siendo de solo-lectura para el bot.
  2. Todo esta particionado por (cliente_id, numero). Un numero jamas ve la
     conversacion de otro, ni de otro cliente.

Diseño a prueba de fallos: si la memoria falla (base dormida, permisos, lo que
sea), NO puede tumbar la respuesta. Toda funcion publica atrapa sus errores,
loguea y sigue como si no hubiera memoria. Un bot sin memoria responde igual;
un bot que revienta por la memoria, no.

Persistencia: vive en el MISMO Neon del cliente (config.dsn_de_cliente), asi
que sobrevive redeploys de Render y funciona con varios workers. Un dict en
memoria no serviria (muere en cada restart y no se comparte entre procesos).
"""

import logging
import threading

from sqlalchemy import create_engine, text

import config

logger = logging.getLogger("fachavi.bot.memoria")

_engines: dict = {}      # DSN -> engine (escritura)
_tabla_lista: set = set()  # DSNs donde ya VERIFICAMOS que la tabla existe
_lock = threading.Lock()   # serializa la creacion (evita la carrera de IF NOT EXISTS)

_DDL = (
    'CREATE SCHEMA IF NOT EXISTS "_bot"',
    """
    CREATE TABLE IF NOT EXISTS "_bot".conversaciones (
        id          BIGSERIAL PRIMARY KEY,
        cliente_id  TEXT        NOT NULL,
        numero      TEXT        NOT NULL,
        rol         TEXT        NOT NULL,
        contenido   TEXT        NOT NULL,
        creado_en   TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_conv_cliente_numero_fecha
        ON "_bot".conversaciones (cliente_id, numero, creado_en DESC)
    """,
)


def _engine(cliente: dict):
    dsn = config.dsn_de_cliente(cliente)
    if dsn not in _engines:
        _engines[dsn] = create_engine(dsn, pool_pre_ping=True)
        logger.info("engine de memoria abierto para '%s'", cliente.get("cliente_id"))
    return _engines[dsn]


def _asegurar_tabla(cliente: dict) -> bool:
    """
    Crea esquema/tabla/indice si faltan y CONFIRMA que la tabla existe antes de
    cachear el DSN como listo. Seguro ante concurrencia:
      - un lock en proceso serializa a los BackgroundTasks del mismo worker;
      - un pg_advisory_xact_lock serializa entre varios workers/procesos;
      - se verifica con to_regclass DENTRO de la transaccion: si por la carrera
        de 'CREATE IF NOT EXISTS' la tabla no quedo, NO se cachea y se reintenta.
    Devuelve True solo si la tabla existe de verdad.
    """
    dsn = config.dsn_de_cliente(cliente)
    if dsn in _tabla_lista:
        return True
    eng = _engine(cliente)
    with _lock:
        if dsn in _tabla_lista:      # doble chequeo: otro hilo ya la creo
            return True
        with eng.begin() as cx:      # begin() = transaccion con commit al salir
            # Lock de asesor: dos workers no pueden crear la tabla a la vez.
            cx.execute(text("SELECT pg_advisory_xact_lock(hashtext('fachavi_bot_memoria'))"))
            for sentencia in _DDL:
                cx.execute(text(sentencia))
            existe = cx.execute(
                text("SELECT to_regclass('\"_bot\".conversaciones')")
            ).scalar()
        if not existe:
            # No cacheamos: se reintenta en el proximo mensaje. Si esto se repite,
            # el problema es de permisos (el rol no puede CREATE en la base).
            raise RuntimeError(
                "la tabla _bot.conversaciones no quedo creada tras el DDL "
                "(revisá que el rol de Neon tenga permiso CREATE)"
            )
        _tabla_lista.add(dsn)
        logger.info("tabla de memoria lista para '%s'", cliente.get("cliente_id"))
    return True


def cargar_historial(cliente: dict, numero: str) -> list:
    """
    Devuelve los ultimos turnos como [{'rol': 'user'|'assistant', 'contenido': str}]
    en orden cronologico (viejo -> nuevo). Lista vacia si memoria apagada o falla.
    """
    if not config.BOT_MEMORIA:
        return []
    try:
        _asegurar_tabla(cliente)
        sql = text(
            """
            SELECT rol, contenido
            FROM "_bot".conversaciones
            WHERE cliente_id = :cid
              AND numero = :num
              AND creado_en > now() - make_interval(hours => :h)
            ORDER BY creado_en DESC
            LIMIT :lim
            """
        )
        with _engine(cliente).connect() as cx:
            filas = cx.execute(sql, {
                "cid": cliente["cliente_id"],
                "num": numero,
                "h": config.BOT_MEMORIA_VENTANA_HORAS,
                "lim": config.BOT_MEMORIA_MAX_TURNOS,
            }).fetchall()
        # Vinieron en DESC (nuevo->viejo); los damos vuelta a cronologico.
        return [{"rol": r[0], "contenido": r[1]} for r in reversed(filas)]
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo cargar historial: %s",
                       cliente.get("cliente_id"), e)
        return []


def guardar_intercambio(cliente: dict, numero: str,
                        pregunta: str, respuesta: str) -> None:
    """Guarda el turno del usuario y el del asistente. Nunca relanza."""
    if not config.BOT_MEMORIA:
        return
    if not (pregunta and respuesta):
        return
    try:
        _asegurar_tabla(cliente)
        ins = text(
            """
            INSERT INTO "_bot".conversaciones (cliente_id, numero, rol, contenido)
            VALUES (:cid, :num, :rol, :cont)
            """
        )
        purga = text(
            """
            DELETE FROM "_bot".conversaciones
            WHERE cliente_id = :cid AND numero = :num
              AND creado_en < now() - make_interval(days => :d)
            """
        )
        cid = cliente["cliente_id"]
        with _engine(cliente).begin() as cx:
            cx.execute(ins, {"cid": cid, "num": numero, "rol": "user", "cont": pregunta})
            cx.execute(ins, {"cid": cid, "num": numero, "rol": "assistant", "cont": respuesta})
            # Purga de retencion: barata porque va por (cliente, numero) indexado.
            cx.execute(purga, {"cid": cid, "num": numero, "d": config.BOT_MEMORIA_TTL_DIAS})
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo guardar intercambio: %s",
                       cliente.get("cliente_id"), e)


def olvidar(cliente: dict, numero: str) -> None:
    """Borra toda la memoria de un numero (para un comando 'reset'/'olvidá')."""
    try:
        _asegurar_tabla(cliente)
        with _engine(cliente).begin() as cx:
            cx.execute(
                text('DELETE FROM "_bot".conversaciones '
                     'WHERE cliente_id = :cid AND numero = :num'),
                {"cid": cliente["cliente_id"], "num": numero},
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] no se pudo olvidar: %s", cliente.get("cliente_id"), e)
