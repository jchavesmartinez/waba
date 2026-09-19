from modelo.movimientos_canonicos import (
    _aplicar_reclasificaciones,
    _indexar_reclasificaciones,
)
from modelo.reclasificaciones import clave_grupo


def test_regla_grupal_se_aplica_y_override_puntual_gana():
    regla = {
        "modelo_id": "movimientos",
        "clave": clave_grupo("descripcion", "Walmart Heredia"),
        "columna": "linea_presupuesto_id",
        "valor": "gas_comedera",
    }
    exacto = {
        "modelo_id": "movimientos",
        "clave": "bac:2",
        "columna": "linea_presupuesto_id",
        "valor": "gas_otros",
    }
    puntuales, grupales = _indexar_reclasificaciones([regla, exacto])

    fila = {"_clave": "bac:1", "descripcion": "WALMART HEREDIA", "linea_presupuesto_id": "gas_viejo"}
    _aplicar_reclasificaciones(fila, puntuales, grupales)
    assert fila["linea_presupuesto_id"] == "gas_comedera"

    fila = {"_clave": "bac:2", "descripcion": "Walmart Heredia", "linea_presupuesto_id": "gas_viejo"}
    _aplicar_reclasificaciones(fila, puntuales, grupales)
    assert fila["linea_presupuesto_id"] == "gas_otros"
