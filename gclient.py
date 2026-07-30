"""
Cliente de Google Sheets compartido.
Un solo service account autentica y lee TODO: el Sheet maestro (registro)
y los Sheets de datos de cada cliente. Cada cliente comparte su hoja con el
mismo client_email del service account.
"""

import json
import logging
import threading

import gspread
from google.oauth2.service_account import Credentials

import config

logger = logging.getLogger("fachavi.gclient")

# Solo lectura, a proposito: aunque la credencial se filtre, nadie puede
# modificar ni borrar las hojas de los clientes. Es la defensa que hace que C-06
# sea "critico en impacto, bajo en probabilidad" y no simplemente critico.
#
# C-06 — lo que este archivo NO puede resolver solo: una sola cuenta de servicio
# lee la hoja maestra y las hojas de datos de TODOS los clientes. Si se filtra,
# se filtra el acceso de lectura a todos a la vez, no hay forma de revocarle el
# acceso a un cliente sin afectar a los demas, y las credenciales de cuenta de
# servicio de Google NO VENCEN: la de hoy sigue siendo valida en cinco años.
# Separar en una cuenta por cliente destruye la ventaja comercial ("compartí tu
# hoja con este correo") y es una decision de negocio, no tecnica. Las
# mitigaciones que SI corresponden son operativas:
#   1. Rotar esta credencial con periodicidad definida (ponelo en el calendario;
#      nada en el sistema te lo va a recordar).
#   2. Restringir quien entra al panel de Render con la misma seriedad con que
#      restringirias el acceso a la base de datos: quien ve el panel, ve la
#      credencial completa y con ella las hojas de todos los clientes.
#   3. Activar la auditoria de acceso de Google Cloud para esta cuenta.
_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_gc = None
# B-10: el cliente se guardaba en una global sin proteccion. En el bot, dos
# mensajes simultaneos podian construirlo dos veces (dos handshakes OAuth).
_lock = threading.Lock()


def _cliente():
    """Devuelve un cliente gspread autenticado (se crea una sola vez)."""
    global _gc
    if _gc is not None:
        return _gc
    with _lock:
        if _gc is not None:          # doble chequeo
            return _gc
        if not config.GOOGLE_CREDENTIALS_JSON:
            raise RuntimeError("Falta GOOGLE_CREDENTIALS_JSON.")
        creds_info = json.loads(config.GOOGLE_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(creds_info, scopes=_SCOPES)
        _gc = gspread.authorize(creds)
    return _gc


def abrir_libro(spreadsheet_id: str):
    """Abre un libro de Google Sheets por su ID."""
    return _cliente().open_by_key(spreadsheet_id)
