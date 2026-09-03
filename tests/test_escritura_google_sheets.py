from bot.edicion import CampoEdicion, PoliticaEdicion
from bot import escritura_google_sheets as escritor


def _politica():
    return PoliticaEdicion(
        tabla="gastos_manuales", origen="Google Sheets", clave_primaria="movimiento_id",
        anulacion_campo="activo", origen_tipo="google_sheets",
        origen_fuente_id="finanzas", hoja_origen="gastos_manuales",
        acciones=("crear", "modificar", "anular"),
        campos={
            "movimiento_id": CampoEdicion("movimiento_id", "ID", calculado=True,
                                            generador="id_aleatorio_fecha"),
            "fecha": CampoEdicion("fecha", "Fecha"),
            "activo": CampoEdicion("activo", "Activo"),
        },
    )


class HojaFalsa:
    def __init__(self):
        self.encabezados = ["movimiento_id", "fecha", "activo"]
        self.appendidas = []

    def row_values(self, _fila):
        return self.encabezados

    def col_values(self, _columna):
        return ["movimiento_id"]

    def append_row(self, valores, **_kwargs):
        self.appendidas.append(valores)


class LibroFalso:
    def __init__(self, hoja):
        self.hoja = hoja

    def worksheet(self, nombre):
        assert nombre == "gastos_manuales"
        return self.hoja


def test_crear_usa_origen_de_metadata_y_genera_id(monkeypatch):
    hoja = HojaFalsa()
    monkeypatch.setattr(escritor, "abrir_libro_escritura", lambda _id: LibroFalso(hoja))
    cliente = {"fuentes": [{"fuente_id": "finanzas", "tipo": "google_sheets",
                             "activo": True, "config": {"spreadsheet_id": "abc"}}]}
    r = escritor.aplicar_confirmado(cliente, _politica(), "crear", {"fecha": "2026-09-03", "activo": "si"})
    assert r["clave"].startswith("MAN-")
    assert hoja.appendidas[0][1:] == ["2026-09-03", "si"]


def test_escritura_rechaza_fuente_distinta(monkeypatch):
    cliente = {"fuentes": [{"fuente_id": "finanzas", "tipo": "sharepoint",
                             "activo": True, "config": {}}]}
    try:
        escritor.aplicar_confirmado(cliente, _politica(), "crear", {})
    except escritor.ErrorEscritura as e:
        assert "Google Sheets" in str(e)
    else:
        raise AssertionError("debió rechazar la fuente")
