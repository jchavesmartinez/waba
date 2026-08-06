"""Regresiones de la persistencia acumulativa de Zoho IMAP."""

from datetime import datetime, timezone

import pandas as pd

import sync as modulo_sync
from sources.base import Fragmento
from sources.zoho_imap import CAMPOS_CORREO
from warehouse.base import Corrida
from warehouse.duckdb_dest import DuckDBDestino


def _corrida(numero: int) -> Corrida:
    return Corrida(
        corrida_id=f"run-{numero}",
        cliente_id="cliente_a",
        fuente_id="correo_zoho",
        tipo="zoho_imap",
        inicio=datetime.now(timezone.utc),
    )


def _df(correo_id="correo-1", uid="1", asunto="Factura", destinatario=None):
    destinatario = destinatario or "compras.uno@gmail.com"
    return pd.DataFrame([{
        "correo_id": correo_id,
        "message_id": f"<{correo_id}@example.com>",
        "buzon": "central@fachavi.com",
        "carpeta": "INBOX/Cliente A",
        "uid": uid,
        "fecha": datetime(2026, 8, 3, 12, 0),
        "remitente_nombre": "Proveedor",
        "remitente_correo": "ventas@proveedor.com",
        "destinatarios": destinatario,
        "cc": "",
        "asunto": asunto,
        "cuerpo": "Contenido",
        "n_adjuntos": 0,
        "adjuntos": "",
    }], columns=list(CAMPOS_CORREO))


def test_append_conserva_correos_fuera_de_la_ventana_y_no_duplica():
    destino = DuckDBDestino(":memory:")
    try:
        primero = destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos",
            _df("correo-antiguo", "1"), _corrida(1),
        )
        segundo = destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos",
            _df("correo-nuevo", "2"), _corrida(2),
        )
        repetido = destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos",
            _df("correo-nuevo", "2"), _corrida(3),
        )
        ids = destino.conectar().execute(
            'SELECT correo_id FROM "raw_cliente_a"."correo_zoho__correos" '
            "ORDER BY correo_id"
        ).fetchall()
    finally:
        destino.cerrar()

    assert primero == {
        "nuevos": 1, "actualizados": 0, "sin_cambios": 0, "actual_total": 1,
    }
    assert segundo["actual_total"] == 2
    assert repetido["sin_cambios"] == 1
    assert repetido["actual_total"] == 2
    assert ids == [("correo-antiguo",), ("correo-nuevo",)]


def test_mismo_correo_actualiza_la_fila_existente():
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos",
            _df(asunto="Factura pendiente"), _corrida(1),
        )
        stats = destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos",
            _df(asunto="Factura pagada"), _corrida(2),
        )
        fila = destino.conectar().execute(
            'SELECT count(*), max(asunto) FROM '
            '"raw_cliente_a"."correo_zoho__correos"'
        ).fetchone()
    finally:
        destino.cerrar()

    assert stats["actualizados"] == 1
    assert fila == (1, "Factura pagada")


def test_ventana_vacia_no_borra_el_historial():
    destino = DuckDBDestino(":memory:")
    try:
        destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos", _df(), _corrida(1)
        )
        vacio = pd.DataFrame(columns=list(CAMPOS_CORREO))
        stats = destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos", vacio, _corrida(2)
        )
    finally:
        destino.cerrar()

    assert stats["actual_total"] == 1


def test_migra_snapshot_legado_en_la_misma_tabla_sin_perder_filas():
    destino = DuckDBDestino(":memory:")
    legado = pd.DataFrame([
        {
            "uid": "1", "fecha": datetime(2026, 8, 1),
            "remitente_nombre": "Proveedor", "remitente_correo": "a@b.com",
            "asunto": "Version anterior", "cuerpo": "x",
            "n_adjuntos": 0, "adjuntos": "",
        },
        {
            "uid": "99", "fecha": datetime(2026, 6, 1),
            "remitente_nombre": "Historico", "remitente_correo": "h@b.com",
            "asunto": "Fuera de ventana", "cuerpo": "viejo",
            "n_adjuntos": 0, "adjuntos": "",
        },
    ])
    try:
        destino.escribir_tabla(
            "raw_cliente_a", "correo_zoho__correos", legado
        )
        stats = destino.actualizar_correos_zoho(
            "raw_cliente_a", "correo_zoho__correos",
            _df("correo-1", "1", "Version nueva"), _corrida(1),
        )
        filas = destino.conectar().execute(
            'SELECT uid, asunto FROM "raw_cliente_a"."correo_zoho__correos" '
            "ORDER BY uid"
        ).fetchall()
        tablas = destino.conectar().execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='raw_cliente_a' ORDER BY table_name"
        ).fetchall()
    finally:
        destino.cerrar()

    assert stats["actualizados"] == 1
    assert stats["actual_total"] == 2
    assert filas == [("1", "Version nueva"), ("99", "Fuera de ventana")]
    assert tablas == [("correo_zoho__correos",)]


def test_sync_zoho_usa_la_ruta_acumulativa(monkeypatch):
    datos = [_df("correo-antiguo", "1")]

    class FuenteFalsa:
        def cargar(self, con):
            con.register("_correos_prueba", datos[0])
            con.execute("CREATE TABLE correos AS SELECT * FROM _correos_prueba")
            con.unregister("_correos_prueba")
            return Fragmento(schema="", tablas=["correos"], alertas=[])

    monkeypatch.setattr(
        modulo_sync, "crear_fuente",
        lambda tipo, fuente_id, config: FuenteFalsa(),
    )
    destino = DuckDBDestino(":memory:")
    fuente = {
        "fuente_id": "correo_zoho",
        "tipo": "zoho_imap",
        "frescura_minutos": 0,
        "config": {"tabla": "correos"},
    }
    try:
        primera = modulo_sync.sincronizar_fuente(
            destino, {"cliente_id": "cliente_a"}, fuente, forzar=True
        )
        datos[0] = _df("correo-nuevo", "2")
        segunda = modulo_sync.sincronizar_fuente(
            destino, {"cliente_id": "cliente_a"}, fuente, forzar=True
        )
        total = destino.conectar().execute(
            'SELECT count(*) FROM "raw_cliente_a"."correo_zoho__correos"'
        ).fetchone()[0]
    finally:
        destino.cerrar()

    assert primera.estado == segunda.estado == "ok"
    assert segunda.detalle["correo_zoho__correos"]["filas"] == 2
    assert total == 2
