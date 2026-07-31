"""Pruebas de Google Calendar con una sola tabla acumulativa."""

from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

import gclient
import sync as modulo_sync
from sources import google_calendar as modulo_calendar
from sources import crear_fuente, tipos_disponibles
from sources.base import Fragmento
from sources.google_calendar import (
    CAMPOS_EVENTO,
    GoogleCalendarSource,
    _entero_no_negativo,
    _es_verdadero,
    _evento_a_fila,
)
from warehouse.base import Corrida
from warehouse.duckdb_dest import DuckDBDestino


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


def _corrida(numero: int = 1) -> Corrida:
    return Corrida(
        corrida_id=f"corrida-{numero}",
        cliente_id="cliente_a",
        fuente_id="agenda",
        tipo="google_calendar",
        inicio=datetime.now(timezone.utc),
    )


def _df_evento(
    evento_id: str = "evt-1",
    titulo: str = "Corte",
    estado: str = "confirmed",
) -> pd.DataFrame:
    if estado == "cancelled":
        evento = {
            "id": evento_id,
            "status": "cancelled",
            "updated": "2026-07-31T12:00:00Z",
        }
    else:
        evento = _evento(
            evento_id,
            "2026-08-03T09:00:00-06:00",
            "2026-08-03T09:45:00-06:00",
            titulo,
        )
    fila = _evento_a_fila("Carlos", "cal-carlos", evento)
    return pd.DataFrame([fila], columns=list(CAMPOS_EVENTO) + ["version_hash"])


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

    assert gclient.credenciales_calendar().token == "token-de-prueba"
    assert capturado["scopes"] == [
        "https://www.googleapis.com/auth/calendar.readonly"
    ]


def test_fuente_fusiona_calendarios_y_respeta_ventana(monkeypatch):
    fuente = GoogleCalendarSource(
        "agenda",
        {
            "calendarios": {"Carlos": "cal-carlos", "Andres": "cal-andres"},
            "dias_atras": 30,
            "dias_adelante": 60,
        },
    )
    respuestas = {
        "cal-carlos": [_evento(
            "evt-1", "2026-08-03T09:00:00-06:00",
            "2026-08-03T09:45:00-06:00",
        )],
        "cal-andres": [_evento(
            "evt-2", "2026-08-03T10:00:00-06:00",
            "2026-08-03T10:30:00-06:00", "Barba",
        )],
    }
    rangos = []

    def listar(calendar_id, desde, hasta):
        rangos.append((hasta - desde).days)
        return respuestas[calendar_id]

    monkeypatch.setattr(fuente, "_listar_eventos", listar)
    con = duckdb.connect(database=":memory:")
    try:
        frag = fuente.cargar(con)
        filas = con.execute(
            "SELECT recurso, evento_id, titulo, duracion_min "
            "FROM citas ORDER BY evento_id"
        ).fetchall()
    finally:
        con.close()

    assert frag.tablas == ["citas"]
    assert rangos == [90, 90]
    assert filas == [
        ("Carlos", "evt-1", "Corte", 45),
        ("Andres", "evt-2", "Barba", 30),
    ]


def test_peticion_api_usa_30_60_y_trae_cancelados(monkeypatch):
    llamadas = []

    class Respuesta:
        status_code = 200

        def json(self):
            return {"items": []}

        def raise_for_status(self):
            return None

    class Cliente:
        def __init__(self, **_kwargs): ...
        def __enter__(self): return self
        def __exit__(self, *_args): return False

        def get(self, url, headers, params):
            llamadas.append(dict(params))
            return Respuesta()

    fuente = GoogleCalendarSource("agenda", {"calendar_id": "cal-carlos"})
    monkeypatch.setattr(fuente, "_token", lambda: "access-token")
    monkeypatch.setattr(modulo_calendar.httpx, "Client", Cliente)
    desde = datetime(2026, 7, 1, tzinfo=timezone.utc)
    hasta = datetime(2026, 9, 29, tzinfo=timezone.utc)

    assert fuente._listar_eventos("cal-carlos", desde, hasta) == []
    assert llamadas == [{
        "timeMin": desde.isoformat(),
        "timeMax": hasta.isoformat(),
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": 2500,
        "showDeleted": "true",
    }]


def test_normaliza_fechas_y_eventos_de_dia_completo():
    fila = _evento_a_fila(
        "Carlos", "cal-carlos",
        _evento(
            "evt-1", "2026-08-03T09:00:00-06:00",
            "2026-08-03T09:45:00-06:00",
        ),
    )
    assert fila["inicio"].tzinfo == timezone.utc
    assert fila["inicio"].hour == 15
    assert fila["duracion_min"] == 45

    dia = _evento_a_fila("Carlos", "cal-carlos", {
        "id": "vacaciones",
        "summary": "Vacaciones",
        "status": "confirmed",
        "start": {"date": "2026-08-03"},
        "end": {"date": "2026-08-04"},
    })
    assert dia["todo_el_dia"] is True
    assert dia["duracion_min"] is None


def test_lectura_repetida_no_duplica():
    destino = DuckDBDestino(":memory:")
    try:
        primero = destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento(), _corrida(1)
        )
        segundo = destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento(), _corrida(2)
        )
        total = destino.conectar().execute(
            'SELECT count(*) FROM "raw_cliente_a"."agenda__citas"'
        ).fetchone()[0]
    finally:
        destino.cerrar()

    assert primero["nuevos"] == 1
    assert segundo["sin_cambios"] == 1
    assert total == 1


def test_citas_fuera_de_la_ventana_no_se_borran():
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento("evt-antigua"), _corrida(1)
        )
        destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento("evt-nueva"), _corrida(2)
        )
        ids = destino.conectar().execute(
            'SELECT evento_id FROM "raw_cliente_a"."agenda__citas" '
            "ORDER BY evento_id"
        ).fetchall()
    finally:
        destino.cerrar()
    assert ids == [("evt-antigua",), ("evt-nueva",)]


def test_ventana_vacia_tampoco_borra_lo_acumulado():
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento(), _corrida(1)
        )
        vacio = pd.DataFrame(
            columns=list(CAMPOS_EVENTO) + ["version_hash"]
        )
        stats = destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", vacio, _corrida(2)
        )
    finally:
        destino.cerrar()
    assert stats["actual_total"] == 1


def test_cambio_actualiza_la_misma_fila():
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento(titulo="Corte"), _corrida(1)
        )
        stats = destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas",
            _df_evento(titulo="Corte y barba"), _corrida(2),
        )
        fila = destino.conectar().execute(
            'SELECT count(*), max(titulo) FROM "raw_cliente_a"."agenda__citas"'
        ).fetchone()
    finally:
        destino.cerrar()
    assert stats["actualizados"] == 1
    assert fila == (1, "Corte y barba")


def test_cancelacion_actualiza_estado_y_preserva_detalles():
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento(), _corrida(1)
        )
        destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas",
            _df_evento(estado="cancelled"), _corrida(2),
        )
        fila = destino.conectar().execute(
            'SELECT count(*), max(titulo), max(estado), max(inicio) '
            'FROM "raw_cliente_a"."agenda__citas"'
        ).fetchone()
    finally:
        destino.cerrar()
    assert fila[0] == 1
    assert fila[1] == "Corte"
    assert fila[2] == "cancelled"
    assert fila[3] is not None


def test_tabla_snapshot_existente_se_amplia_sin_crear_otras():
    destino = DuckDBDestino(":memory:")
    try:
        destino.escribir_tabla(
            "raw_cliente_a",
            "agenda__citas",
            pd.DataFrame([{"recurso": "Carlos", "evento_id": "evt-1",
                           "titulo": "Corte viejo"}]),
        )
        destino.actualizar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", _df_evento(), _corrida(1)
        )
        tablas = destino.conectar().execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='raw_cliente_a' ORDER BY table_name"
        ).fetchall()
        fila = destino.conectar().execute(
            'SELECT count(*), max(calendar_id), max(titulo) '
            'FROM "raw_cliente_a"."agenda__citas"'
        ).fetchone()
    finally:
        destino.cerrar()

    assert tablas == [("agenda__citas",)]
    assert fila == (1, "cal-carlos", "Corte")


def test_sync_escribe_y_reporta_una_sola_tabla(monkeypatch):
    class FuenteFalsa:
        def cargar(self, con):
            df = _df_evento()
            con.register("_eventos_prueba", df)
            con.execute("CREATE TABLE citas AS SELECT * FROM _eventos_prueba")
            con.unregister("_eventos_prueba")
            return Fragmento(schema=[], tablas=["citas"], alertas=[])

    monkeypatch.setattr(
        modulo_sync, "crear_fuente",
        lambda tipo, fuente_id, config: FuenteFalsa(),
    )
    destino = DuckDBDestino(":memory:")
    try:
        corrida = modulo_sync.sincronizar_fuente(
            destino,
            {"cliente_id": "cliente_a"},
            {"fuente_id": "agenda", "tipo": "google_calendar",
             "frescura_minutos": 0, "config": {}},
            forzar=True,
        )
        tablas = destino.conectar().execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='raw_cliente_a' ORDER BY table_name"
        ).fetchall()
    finally:
        destino.cerrar()

    assert corrida.estado == "ok"
    assert corrida.tablas_logicas == {"citas"}
    assert corrida.tablas == ["raw_cliente_a.agenda__citas"]
    assert tablas == [("agenda__citas",)]


def test_config_invalida_se_rechaza():
    assert _es_verdadero("false") is False
    assert _es_verdadero("si") is True
    with pytest.raises(RuntimeError, match="no puede ser negativo"):
        _entero_no_negativo("-1", "dias_atras")

    fuente = GoogleCalendarSource(
        "agenda", {"calendarios": {"Carlos": "cal-uno", "Andres": "cal-uno"}}
    )
    with pytest.raises(RuntimeError, match="mas de una vez"):
        fuente._calendarios()


def test_si_fallan_todos_los_calendarios_la_fuente_falla(monkeypatch):
    fuente = GoogleCalendarSource(
        "agenda", {"calendarios": {"Carlos": "cal-carlos"}}
    )
    monkeypatch.setattr(
        fuente, "_listar_eventos",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("sin permiso")),
    )
    con = duckdb.connect(database=":memory:")
    try:
        with pytest.raises(RuntimeError, match="ninguno"):
            fuente.cargar(con)
    finally:
        con.close()
