from bot.edicion import CampoEdicion, PoliticaEdicion, validar_borrador
import bot.edicion as edicion


def _politica():
    return PoliticaEdicion(
        tabla="gastos_manuales", origen="Google Sheets", clave_primaria="movimiento_id",
        anulacion_campo="activo", origen_tipo="google_sheets",
        origen_fuente_id="finanzas", hoja_origen="gastos_manuales",
        acciones=("crear", "modificar", "anular"),
        campos={
            "movimiento_id": CampoEdicion("movimiento_id", "ID", requerido=True,
                                            editable=False, calculado=True),
            "fecha": CampoEdicion("fecha", "Fecha", requerido=True, tipo="fecha_iso"),
            "monto": CampoEdicion("monto", "Monto", requerido=True, tipo="monto_positivo"),
            "moneda": CampoEdicion("moneda", "Moneda", requerido=True, tipo="lista",
                                    valores=("CRC", "USD"), defecto="CRC"),
        },
    )


def test_borrador_normaliza_fecha_y_monto_local():
    r = validar_borrador(_politica(), "crear", {
        "fecha": "2026-09-03", "monto": "₡1.234,50", "moneda": "crc",
    })
    assert r.listo_para_confirmar
    assert r.valores == {"fecha": "2026-09-03", "monto": "1234.50", "moneda": "CRC"}


def test_borrador_rechaza_fecha_ambigua_y_monto_no_positivo():
    r = validar_borrador(_politica(), "crear", {
        "fecha": "03/04/2026", "monto": "0", "moneda": "CRC",
    })
    assert not r.listo_para_confirmar
    assert any("Fecha" in e for e in r.errores)
    assert any("Monto" in e for e in r.errores)


def test_modificar_exige_id_pero_no_aplica_valores_por_defecto():
    r = validar_borrador(_politica(), "modificar", {"movimiento_id": "MAN-01"})
    assert r.listo_para_confirmar
    assert r.valores == {"movimiento_id": "MAN-01"}


def test_extraer_natural_usa_solo_campos_de_metadata(monkeypatch):
    class Respuesta:
        texto = '{"cambios":{"fecha":"2026-09-02","monto":"12500"}}'
    monkeypatch.setattr(edicion.llm, "generar_texto", lambda *a, **k: Respuesta())
    r = edicion._extraer_natural(_politica(), "El 2026-09-02 gasté 12.500")
    assert r == ({"fecha": "2026-09-02", "monto": "12500"}, {})
