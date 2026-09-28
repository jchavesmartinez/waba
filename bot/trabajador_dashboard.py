"""Worker durable para materializar cambios del dashboard sin bloquear la UI.

Se ejecuta como un Background Worker de Render. Los cambios ya están guardados
en Google Sheets y en la cola de Neon antes de que este proceso los tome; por
eso un reinicio no pierde correcciones ni duplica pagos.
"""

from __future__ import annotations

import logging
import signal
import threading

import config
import registry
from bot import dashboard_edicion

logger = logging.getLogger("fachavi.bot.trabajador_dashboard")
_DETENER = threading.Event()


def _pedir_detencion(*_args) -> None:
    _DETENER.set()


def ejecutar() -> None:
    """Consume lotes hasta recibir SIGTERM; seguro para un único worker."""
    signal.signal(signal.SIGTERM, _pedir_detencion)
    signal.signal(signal.SIGINT, _pedir_detencion)
    logger.info("worker durable de dashboard iniciado")
    while not _DETENER.is_set():
        procesadas = 0
        try:
            for cliente in registry.listar_clientes():
                if _DETENER.is_set():
                    break
                procesadas += dashboard_edicion.procesar_reconstrucciones(
                    # Respeta la misma ventana de agrupación que usa el web:
                    # varios toques consecutivos se materializan juntos.
                    cliente, maximo=100, agrupar=True,
                )
        except Exception:  # el siguiente ciclo vuelve a intentar la cola durable
            logger.exception("fallo al buscar ediciones pendientes del dashboard")
        espera = 0 if procesadas else config.DASHBOARD_EDICION_WORKER_SEGUNDOS
        _DETENER.wait(espera)
    logger.info("worker durable de dashboard detenido")


if __name__ == "__main__":
    ejecutar()
