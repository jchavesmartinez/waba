"""
Pruebas del conector de Google Calendar.

No llaman a Google ni a Neon: simulan la respuesta de Calendar y comprueban
que la fuente este registrada, fusione calendarios y produzca la tabla que el
pipeline general escribira en el warehouse.
"""

from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

import gclient
import sync as modulo_sync
from sources.google_calendar import CAMPOS_EVENTO
from sources import google_calendar as modulo_calendar
from sources import crear_fuente, tipos_disponibles
from sources.google_calendar import (
    GoogleCalendarSource,
    _entero_no_negativo,
    _es_verdadero,
    _evento_a_fila,
)
from warehouse.base import Corrida
from warehouse.duckdb_dest import DuckDBDestino
from sources.base import Fragmento


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
        lambda calendar_id, desde, sync_token="": (
            respuestas[calendar_id],
            f"token-{calendar_id}",
            False,
        ),
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
    assert fuente.sync_tokens_siguientes == {
        "cal-carlos": "token-cal-carlos",
        "cal-andres": "token-cal-andres",
    }
    assert filas == [
        ("Carlos", "evt-1", "Corte", 45),
        ("Andres", "evt-2", "Barba", 30),
    ]


def test_normaliza_fechas_con_offset_a_utc():
    fila = _evento_a_fila(
        "Carlos",
        "cal-carlos",
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
        "cal-carlos",
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


def test_historico_no_duplica_una_lectura_identica():
    destino = DuckDBDestino(":memory:")
    try:
        primero = destino.fusionar_eventos_calendar(
            "raw_cliente_a",
            "agenda__citas",
            "agenda__citas_historial",
            _df_evento(),
            _corrida(1),
        )
        segundo = destino.fusionar_eventos_calendar(
            "raw_cliente_a",
            "agenda__citas",
            "agenda__citas_historial",
            _df_evento(),
            _corrida(2),
        )
        con = destino.conectar()
        conteos = con.execute(
            'SELECT (SELECT count(*) FROM "raw_cliente_a"."agenda__citas"), '
            '(SELECT count(*) FROM '
            '"raw_cliente_a"."agenda__citas_historial")'
        ).fetchone()
    finally:
        destino.cerrar()

    assert primero["nuevos"] == 1
    assert segundo["sin_cambios"] == 1
    assert conteos == (1, 1)


def test_cambio_crea_version_y_cierra_la_anterior():
    destino = DuckDBDestino(":memory:")
    try:
        destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            _df_evento(titulo="Corte"), _corrida(1),
        )
        stats = destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            _df_evento(titulo="Corte y barba"), _corrida(2),
        )
        con = destino.conectar()
        titulo = con.execute(
            'SELECT titulo FROM "raw_cliente_a"."agenda__citas"'
        ).fetchone()[0]
        versiones = con.execute(
            'SELECT count(*), sum(CASE WHEN es_version_actual THEN 1 ELSE 0 END) '
            'FROM "raw_cliente_a"."agenda__citas_historial"'
        ).fetchone()
    finally:
        destino.cerrar()

    assert stats["cambiados"] == 1
    assert titulo == "Corte y barba"
    assert versiones == (2, 1)


def test_cancelacion_preserva_detalles_del_tombstone():
    destino = DuckDBDestino(":memory:")
    try:
        destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            _df_evento(), _corrida(1),
        )
        destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            _df_evento(estado="cancelled"), _corrida(2),
        )
        fila = destino.conectar().execute(
            'SELECT titulo, inicio, estado FROM '
            '"raw_cliente_a"."agenda__citas"'
        ).fetchone()
        historia = destino.conectar().execute(
            'SELECT count(*) FROM "raw_cliente_a"."agenda__citas_historial"'
        ).fetchone()[0]
    finally:
        destino.cerrar()

    assert fila[0] == "Corte"
    assert fila[1] is not None
    assert fila[2] == "cancelled"
    assert historia == 2


def test_un_evento_nuevo_no_borra_los_anteriores():
    destino = DuckDBDestino(":memory:")
    try:
        destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            _df_evento("evt-1"), _corrida(1),
        )
        destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            _df_evento("evt-2"), _corrida(2),
        )
        ids = destino.conectar().execute(
            'SELECT evento_id FROM "raw_cliente_a"."agenda__citas" '
            "ORDER BY evento_id"
        ).fetchall()
    finally:
        destino.cerrar()
    assert ids == [("evt-1",), ("evt-2",)]


def test_sync_tokens_se_guardan_por_calendario():
    destino = DuckDBDestino(":memory:")
    try:
        assert destino.leer_estado_calendar("raw_cliente_a", "agenda") == {}
        destino.guardar_estado_calendar(
            "raw_cliente_a",
            "agenda",
            {"cal-carlos": "token-1", "cal-andres": "token-2"},
        )
        destino.guardar_estado_calendar(
            "raw_cliente_a", "agenda", {"cal-carlos": "token-3"}
        )
        tokens = destino.leer_estado_calendar("raw_cliente_a", "agenda")
    finally:
        destino.cerrar()
    assert tokens == {"cal-carlos": "token-3", "cal-andres": "token-2"}


def test_reinicio_410_reconstruye_actual_y_conserva_historial():
    destino = DuckDBDestino(":memory:")
    try:
        inicial = pd.concat(
            [_df_evento("evt-1"), _df_evento("evt-2")],
            ignore_index=True,
        )
        destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            inicial, _corrida(1),
        )
        stats = destino.fusionar_eventos_calendar(
            "raw_cliente_a",
            "agenda__citas",
            "agenda__citas_historial",
            _df_evento("evt-2"),
            _corrida(2),
            ["cal-carlos"],
        )
        actuales = destino.conectar().execute(
            'SELECT evento_id FROM "raw_cliente_a"."agenda__citas"'
        ).fetchall()
        versiones = destino.conectar().execute(
            'SELECT count(*) FROM "raw_cliente_a"."agenda__citas_historial"'
        ).fetchone()[0]
    finally:
        destino.cerrar()

    assert stats["retirados_actual"] == 1
    assert actuales == [("evt-2",)]
    assert versiones == 2


def test_si_fallan_todos_los_calendarios_la_fuente_falla(monkeypatch):
    fuente = GoogleCalendarSource(
        "agenda", {"calendarios": {"Carlos": "cal-carlos"}}
    )

    def falla(*_args, **_kwargs):
        raise RuntimeError("sin permiso")

    monkeypatch.setattr(fuente, "_listar_eventos", falla)
    con = duckdb.connect(database=":memory:")
    try:
        with pytest.raises(RuntimeError, match="ninguno"):
            fuente.cargar(con)
    finally:
        con.close()


def test_calendar_id_duplicado_se_rechaza():
    fuente = GoogleCalendarSource(
        "agenda",
        {"calendarios": {"Carlos": "cal-uno", "Andres": "cal-uno"}},
    )
    with pytest.raises(RuntimeError, match="mas de una vez"):
        fuente._calendarios()


def test_listado_incremental_envia_sync_token_y_recibe_el_siguiente(monkeypatch):
    llamadas = []

    class Respuesta:
        status_code = 200

        def json(self):
            return {"items": [], "nextSyncToken": "token-nuevo"}

        def raise_for_status(self):
            return None

    class Cliente:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, headers, params):
            llamadas.append(dict(params))
            return Respuesta()

    fuente = GoogleCalendarSource("agenda", {"calendar_id": "cal-carlos"})
    monkeypatch.setattr(fuente, "_token", lambda: "access-token")
    monkeypatch.setattr(modulo_calendar.httpx, "Client", Cliente)

    eventos, token, reiniciado = fuente._listar_eventos(
        "cal-carlos", datetime.now(timezone.utc), "token-anterior"
    )

    assert eventos == []
    assert token == "token-nuevo"
    assert reiniciado is False
    assert llamadas == [{
        "singleEvents": "true",
        "maxResults": 2500,
        "showDeleted": "true",
        "syncToken": "token-anterior",
    }]


def test_sync_token_vencido_hace_carga_completa_automaticamente(monkeypatch):
    llamadas = []

    class Respuesta:
        def __init__(self, status_code, datos=None):
            self.status_code = status_code
            self._datos = datos or {}

        def json(self):
            return self._datos

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    respuestas = iter([
        Respuesta(410),
        Respuesta(200, {"items": [], "nextSyncToken": "token-recreado"}),
    ])

    class Cliente:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, headers, params):
            llamadas.append(dict(params))
            return next(respuestas)

    fuente = GoogleCalendarSource("agenda", {"calendar_id": "cal-carlos"})
    monkeypatch.setattr(fuente, "_token", lambda: "access-token")
    monkeypatch.setattr(modulo_calendar.httpx, "Client", Cliente)
    desde = datetime(2026, 7, 1, tzinfo=timezone.utc)

    _, token, reiniciado = fuente._listar_eventos(
        "cal-carlos", desde, "token-vencido"
    )

    assert token == "token-recreado"
    assert reiniciado is True
    assert llamadas[0]["syncToken"] == "token-vencido"
    assert "timeMin" not in llamadas[0]
    assert llamadas[1]["timeMin"] == desde.isoformat()
    assert "syncToken" not in llamadas[1]


def test_migracion_conserva_la_tabla_snapshot_anterior():
    destino = DuckDBDestino(":memory:")
    try:
        destino.escribir_tabla(
            "raw_cliente_a",
            "agenda__citas",
            pd.DataFrame([{"recurso": "Carlos", "evento_id": "viejo"}]),
        )
        destino.fusionar_eventos_calendar(
            "raw_cliente_a", "agenda__citas", "agenda__citas_historial",
            _df_evento(), _corrida(1),
        )
        tablas = {
            fila[0]
            for fila in destino.conectar().execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='raw_cliente_a'"
            ).fetchall()
        }
        legado = destino.conectar().execute(
            'SELECT evento_id FROM '
            '"raw_cliente_a"."agenda__citas__snapshot_legacy"'
        ).fetchone()[0]
    finally:
        destino.cerrar()

    assert "agenda__citas" in tablas
    assert "agenda__citas_historial" in tablas
    assert "agenda__citas__snapshot_legacy" in tablas
    assert legado == "viejo"


def test_sync_completo_persiste_token_despues_de_fusionar(monkeypatch):
    configuraciones = []

    class FuenteFalsa:
        def __init__(self, config):
            self.config = config
            self.alertas = []
            anterior = config.get("_sync_tokens", {}).get("cal-carlos")
            self.sync_tokens_siguientes = {
                "cal-carlos": "token-2" if anterior else "token-1"
            }

        def cargar(self, con):
            configuraciones.append(self.config)
            df = _df_evento()
            con.register("_eventos_prueba", df)
            con.execute("CREATE TABLE citas AS SELECT * FROM _eventos_prueba")
            con.unregister("_eventos_prueba")
            return Fragmento(schema=[], tablas=["citas"], alertas=[])

    monkeypatch.setattr(
        modulo_sync,
        "crear_fuente",
        lambda tipo, fuente_id, config: FuenteFalsa(config),
    )
    destino = DuckDBDestino(":memory:")
    cliente = {"cliente_id": "cliente_a"}
    fuente = {
        "fuente_id": "agenda",
        "tipo": "google_calendar",
        "frescura_minutos": 0,
        "config": {"calendar_id": "cal-carlos"},
    }
    try:
        primera = modulo_sync.sincronizar_fuente(
            destino, cliente, fuente, forzar=True
        )
        segunda = modulo_sync.sincronizar_fuente(
            destino, cliente, fuente, forzar=True
        )
        tokens = destino.leer_estado_calendar("raw_cliente_a", "agenda")
        historia = destino.conectar().execute(
            'SELECT count(*) FROM "raw_cliente_a"."agenda__citas_historial"'
        ).fetchone()[0]
    finally:
        destino.cerrar()

    assert primera.estado == "ok"
    assert segunda.estado == "ok"
    assert configuraciones[0]["_sync_tokens"] == {}
    assert configuraciones[1]["_sync_tokens"] == {"cal-carlos": "token-1"}
    assert tokens == {"cal-carlos": "token-2"}
    assert historia == 1
