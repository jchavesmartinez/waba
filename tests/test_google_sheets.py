from sources.google_sheets import (
    _leer_hojas_en_lotes,
    _rango_hoja,
    _rellenar_filas,
)


class _Hoja:
    def __init__(self, title):
        self.title = title


class _Libro:
    def __init__(self):
        self.llamadas = []

    def values_batch_get(self, rangos, params=None):
        self.llamadas.append((rangos, params))
        return {
            "valueRanges": [
                {"range": rango, "values": [["columna"], [rango]]}
                for rango in rangos
            ]
        }


def test_escapa_titulo_para_rango_a1():
    assert _rango_hoja("O'Brien 2021") == "'O''Brien 2021'"


def test_lee_47_hojas_en_dos_solicitudes_batch():
    libro = _Libro()
    hojas = [_Hoja(f"Hoja {numero}") for numero in range(47)]

    resultado = _leer_hojas_en_lotes(libro, hojas)

    assert len(libro.llamadas) == 2
    assert len(libro.llamadas[0][0]) == 25
    assert len(libro.llamadas[1][0]) == 22
    assert libro.llamadas[0][1] == {"majorDimension": "ROWS"}
    assert resultado["Hoja 0"] == [["columna"], ["'Hoja 0'"]]
    assert resultado["Hoja 46"] == [["columna"], ["'Hoja 46'"]]


def test_no_hace_solicitudes_si_no_hay_hojas():
    libro = _Libro()

    assert _leer_hojas_en_lotes(libro, []) == {}
    assert libro.llamadas == []


def test_rellena_celdas_finales_omitidas_por_batch_get():
    registros = [
        ["encabezado"],
        ["dato 1", "dato 2", "dato 3"],
        ["otro 1", "otro 2"],
    ]

    assert _rellenar_filas(registros) == [
        ["encabezado", "", ""],
        ["dato 1", "dato 2", "dato 3"],
        ["otro 1", "otro 2", ""],
    ]
