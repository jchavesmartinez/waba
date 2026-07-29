"""
Cliente de Google Sheets compartido.
Un solo service account autentica y lee TODO: el Sheet maestro (registro)
y los Sheets de datos de cada cliente. Cada cliente comparte su hoja con el
mismo client_email del service account.
"""

import json
import logging

import gspread
from google.oauth2.service_account import Credentials

import config

logger = logging.getLogger("fachavi.gclient")

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_gc = None


def _cliente():
    """Devuelve un cliente gspread autenticado (se crea una sola vez)."""
    global _gc
    if _gc is not None:
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
