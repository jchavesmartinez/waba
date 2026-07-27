"""
Memoria de conversacion (SQLite).
Guarda los ultimos turnos por usuario para permitir preguntas de seguimiento
tipo "y el mes pasado?" que dependen del contexto anterior.
"""

import sqlite3
import time
from contextlib import contextmanager

import config


@contextmanager
def _db():
    con = sqlite3.connect(config.MEMORY_DB_PATH, check_same_thread=False)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _db() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS historial(
                user_id TEXT, rol TEXT, texto TEXT, ts INTEGER
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS ix_hist ON historial(user_id, ts)")


def guardar(user_id: str, rol: str, texto: str):
    with _db() as con:
        con.execute(
            "INSERT INTO historial(user_id, rol, texto, ts) VALUES (?,?,?,?)",
            (user_id, rol, texto, int(time.time())),
        )


def contexto(user_id: str) -> str:
    """Devuelve los ultimos turnos como texto para inyectar en el prompt."""
    with _db() as con:
        cur = con.execute(
            "SELECT rol, texto FROM historial WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, config.MEMORY_MAX_TURNS * 2),
        )
        filas = cur.fetchall()
    if not filas:
        return ""
    filas.reverse()
    lineas = [f"{'Usuario' if r == 'user' else 'Bot'}: {t}" for r, t in filas]
    return "Conversacion reciente (para contexto de preguntas de seguimiento):\n" + "\n".join(lineas)
