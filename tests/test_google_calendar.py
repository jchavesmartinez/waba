"""
Pruebas del conector de Google Calendar.

No llaman a Google ni a Neon: simulan la respuesta de Calendar y comprueban
que la fuente este registrada, fusione calendarios y produzca la tabla que el
pipeline general escribira en el warehouse.
"""

from datetime import timezone

import duckdb

import gclient
from sources import crear_fuente, tipos_disponibles
from sources.google_calendar import (
    GoogleCalendarSource,
    _entero_no_negativo,
    _es_verdadero,
    _evento_a_fila,
)


def _evento(evento_id: str, inicio: str, fin: str, titulo: str = "Corte") -> dict:
    return {
        "id": evento_id,
        "summary": titulo,
        "description": "Cliente de prueba",
        "status": "confirmed",
        "start": {"dateTime": inicio},
        "end": {"dateTime": fin},
        "created": "2026-07-30T10:00:00Z",
        "updated": "2026-07-30T10:05:00Z",
    }


def test_google_calendar_esta_registrado():
    assert "google_calendar" in tipos_disponibles()
    fuente = crear_fuente(
        "google_calendar", "agenda", {"calendar_id": "agenda@example.com"}
    )
    assert isinstance(fuente, GoogleCalendarSource)


def test_credencial_calendar_usa_scope_de_solo_lectura(monkeypatch):
    capturado = {}

    class CredencialFalsa:
        valid = False
        token = None

        def refresh(self, request):
            self.valid = True
            self.token = "token-de-prueba"

    def crear(info, scopes):
        capturado["info"] = info
        capturado["scopes"] = scopes
        return CredencialFalsa()

    monkeypatch.setattr(
        gclient.config,
        "GOOGLE_CREDENTIALS_JSON",
        '{"type":"service_account","client_email":"bot@example.com"}',
    )
    monkeypatch.setattr(gclient, "_calendar_creds", None)
    monkeypatch.setattr(
        gclient.Credentials, "from_service_account_info", staticmethod(crear)
    )

    credencial = gclient.credenciales_calendar()

    assert credencial.token == "token-de-prueba"
    assert capturado["scopes"] == [
        "https://www.googleapis.com/auth/calendar.readonly"
    ]


def test_fusiona_un_calendario_por_barbero(monkeypatch):
    fuente = GoogleCalendarSource(
        "agenda",
        {
            "calendarios": {
                "Carlos": "cal-carlos",
                "Andres": "cal-andres",
            },
            "dias_atras": 30,
            "dias_adelante": 60,
        },
    )
    respuestas = {
        "cal-carlos": [
            _evento(
                "evt-1",
                "2026-08-03T09:00:00-06:00",
                "2026-08-03T09:45:00-06:00",
            )
        ],
        "cal-andres": [
            _evento(
                "evt-2",
                "2026-08-03T10:00:00-06:00",
                "2026-08-03T10:30:00-06:00",
                "Barba",
            )
        ],
    }

    monkeypatch.setattr(
        fuente,
        "_listar_eventos",
        lambda calendar_id, desde, hasta, incluir: respuestas[calendar_id],
    )

    con = duckdb.connect(database=":memory:")
    try:
        fragmento = fuente.cargar(con)
        filas = con.execute(
            "SELECT recurso, evento_id, titulo, duracion_min "
            "FROM citas ORDER BY evento_id"
        ).fetchall()
    finally:
        con.close()

    assert fragmento.tablas == ["citas"]
    assert filas == [
        ("Carlos", "evt-1", "Corte", 45),
        ("Andres", "evt-2", "Barba", 30),
    ]


def test_normaliza_fechas_con_offset_a_utc():
    fila = _evento_a_fila(
        "Carlos",
        _evento(
            "evt-1",
            "2026-08-03T09:00:00-06:00",
            "2026-08-03T09:45:00-06:00",
        ),
    )
    assert fila["inicio"].tzinfo == timezone.utc
    assert fila["inicio"].hour == 15
    assert fila["duracion_min"] == 45


def test_evento_de_dia_completo():
    fila = _evento_a_fila(
        "Carlos",
        {
            "id": "vacaciones",
            "summary": "Vacaciones",
            "status": "confirmed",
            "start": {"date": "2026-08-03"},
            "end": {"date": "2026-08-04"},
        },
    )
    assert fila["todo_el_dia"] is True
    assert fila["duracion_min"] is None
    assert fila["inicio"].tzinfo == timezone.utc


def test_booleano_false_en_texto_no_incluye_cancelados():
    assert _es_verdadero(False) is False
    assert _es_verdadero("false") is False
    assert _es_verdadero("si") is True


def test_rango_de_dias_invalido_se_rechaza():
    try:
        _entero_no_negativo("-1", "dias_atras")
    except RuntimeError as e:
        assert "no puede ser negativo" in str(e)
    else:
        raise AssertionError("un rango negativo debio rechazarse")
