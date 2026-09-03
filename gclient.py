"""
Clientes de Google con lectura predeterminada y escritura confirmada.

Un mismo service account autentica:
  - Google Sheets: Sheet maestro, fuentes y catalogos.
  - Google Calendar: calendarios compartidos por los clientes.
  - Vertex AI: llamadas a Gemini, si tiene ``roles/aiplatform.user``.

Cada API recibe un token con el alcance minimo que necesita. Compartir un
calendario con el service account NO le da acceso a las hojas, ni al reves.
Sheets y Calendar exigen compartir cada recurso; Vertex exige permisos IAM en
el proyecto, ademas del scope del token.
"""

import json
import logging
import threading

import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

import config

logger = logging.getLogger("fachavi.gclient")

# Sheets de ingesta y Calendar son de solo lectura, a propósito. La única
# excepción es abrir_libro_escritura(), llamado por el adaptador confirmado de
# edición. Vertex usa el scope de
# Google Cloud, pero las acciones reales quedan limitadas por los roles IAM del
# service account; en este proyecto solo debe recibir Vertex AI User.
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
_SCOPES_SHEETS = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_SCOPES_SHEETS_ESCRITURA = ["https://www.googleapis.com/auth/spreadsheets"]
_SCOPES_CALENDAR = ["https://www.googleapis.com/auth/calendar.readonly"]
_SCOPES_VERTEX = ["https://www.googleapis.com/auth/cloud-platform"]
_gc = None
_gc_escritura = None
_calendar_creds = None
_vertex_creds = None
# B-10: el cliente se guardaba en una global sin proteccion. En el bot, dos
# mensajes simultaneos podian construirlo dos veces (dos handshakes OAuth).
_lock = threading.RLock()


def _info_credenciales() -> dict:
    """Lee y valida el JSON del service account sin mostrar secretos."""
    if not config.GOOGLE_CREDENTIALS_JSON:
        raise RuntimeError("Falta GOOGLE_CREDENTIALS_JSON.")
    try:
        info = json.loads(config.GOOGLE_CREDENTIALS_JSON)
    except json.JSONDecodeError as e:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON no contiene JSON valido.") from e
    if not isinstance(info, dict):
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON debe contener un objeto JSON.")
    if info.get("type") != "service_account":
        raise RuntimeError(
            "GOOGLE_CREDENTIALS_JSON debe corresponder a un service account."
        )
    return info


def _cliente():
    """Devuelve un cliente gspread autenticado (se crea una sola vez)."""
    global _gc
    if _gc is not None:
        return _gc
    with _lock:
        if _gc is not None:          # doble chequeo
            return _gc
        creds = Credentials.from_service_account_info(
            _info_credenciales(), scopes=_SCOPES_SHEETS
        )
        _gc = gspread.authorize(creds)
    return _gc


def abrir_libro(spreadsheet_id: str):
    """Abre un libro de Google Sheets por su ID."""
    return _cliente().open_by_key(spreadsheet_id)


def abrir_libro_escritura(spreadsheet_id: str):
    """Abre un Sheet con el scope de escritura, solo para adaptadores aprobados.

    Reutiliza la misma identidad de servicio, pero no el cliente readonly. Que
    una hoja este compartida como editor no cambia el minimo privilegio del
    lector de ingesta; este camino se invoca exclusivamente despues de la
    confirmacion explicita de una edicion.
    """
    global _gc_escritura
    if _gc_escritura is None:
        with _lock:
            if _gc_escritura is None:
                creds = Credentials.from_service_account_info(
                    _info_credenciales(), scopes=_SCOPES_SHEETS_ESCRITURA
                )
                _gc_escritura = gspread.authorize(creds)
    return _gc_escritura.open_by_key(spreadsheet_id)


def credenciales_calendar():
    """
    Devuelve credenciales validas para Google Calendar en modo solo lectura.

    Se usa un objeto separado del cliente de Sheets para que cada token tenga
    solamente su scope correspondiente. El refresh queda protegido por el lock:
    si dos fuentes intentan leer Calendar al mismo tiempo, no refrescan el mismo
    token concurrentemente.
    """
    global _calendar_creds
    with _lock:
        if _calendar_creds is None:
            _calendar_creds = Credentials.from_service_account_info(
                _info_credenciales(), scopes=_SCOPES_CALENDAR
            )
        if not _calendar_creds.valid:
            _calendar_creds.refresh(Request())
        return _calendar_creds


def project_id_service_account() -> str:
    """Project ID declarado en el JSON; puede pisarse con GEMINI_PROJECT_ID."""
    return str(_info_credenciales().get("project_id", "") or "").strip()


def credenciales_vertex():
    """Credenciales separadas con el scope de Google Cloud para Vertex AI."""
    global _vertex_creds
    with _lock:
        if _vertex_creds is None:
            _vertex_creds = Credentials.from_service_account_info(
                _info_credenciales(), scopes=_SCOPES_VERTEX
            )
        if not _vertex_creds.valid:
            _vertex_creds.refresh(Request())
        return _vertex_creds
