"""
Prueba de conformidad del contrato de destinos.

POR QUE EXISTE. El decorador @abstractmethod de `ultimo_detalle` habia quedado
pegado al final de la linea del docstring del metodo anterior. Python NO lo
tomo como error de sintaxis —desde 3.5 el simbolo @ es tambien el operador de
multiplicacion de matrices, asi que la linea se interpretaba como la expresion
valida «texto @ funcion»— y ademas esa expresion nunca se ejecutaba, porque
quedo dentro del cuerpo de un metodo que los dos destinos concretos
reescriben.

Resultado: `ultimo_detalle` dejo de ser obligatorio en silencio. Alguien podia
escribir un destino nuevo sin implementarlo, Python lo dejaba instanciar sin una
sola queja, y el sistema reventaba durante una corrida REAL con un error de
atributo inexistente que apunta al lugar equivocado.

Nada de eso produce un mensaje de error, aparece en una prueba manual ni se nota
en el uso normal. Esta prueba es lo que lo atrapa.
"""

import inspect

import pytest

from warehouse import base as wbase
from warehouse.base import METODOS_CONTRATO, Destino


def _destinos_concretos():
    """Importa los destinos que esten disponibles en este entorno."""
    encontrados = []
    try:
        from warehouse.postgres_dest import PostgresDestino
        encontrados.append(PostgresDestino)
    except Exception:  # noqa: BLE001  (falta sqlalchemy/driver en este entorno)
        pass
    try:
        from warehouse.duckdb_dest import DuckDBDestino
        encontrados.append(DuckDBDestino)
    except Exception:  # noqa: BLE001
        pass
    return encontrados


def test_el_contrato_declara_los_ocho_metodos():
    """
    Los ocho metodos tienen que estar en __abstractmethods__ de verdad, no solo
    aparentar estarlo por tener el decorador escrito arriba.
    """
    abstractos = set(Destino.__abstractmethods__)
    faltan = set(METODOS_CONTRATO) - abstractos
    assert not faltan, (
        f"Estos metodos NO son abstractos aunque deberian serlo: {sorted(faltan)}. "
        "Revisa que cada @abstractmethod este en su propia linea y no pegado al "
        "final del docstring del metodo anterior."
    )


def test_ultimo_detalle_sigue_siendo_obligatorio():
    """El caso puntual de C-07, aislado para que el mensaje de error sea claro."""
    assert "ultimo_detalle" in Destino.__abstractmethods__, (
        "ultimo_detalle dejo de ser obligatorio: el decorador @abstractmethod se "
        "perdio o quedo mal ubicado. Es exactamente el error C-07."
    )


def test_registrar_corrida_conserva_su_documentacion():
    """
    Señal visible de que el decorador esta bien puesto: si vuelve a quedar
    pegado al docstring, la expresion se lo come y la funcion se queda sin
    descripcion.
    """
    doc = inspect.getdoc(Destino.registrar_corrida)
    assert doc and "sync_corridas" in doc, (
        "registrar_corrida perdio su docstring. Sintoma clasico de un decorador "
        "absorbido por el texto de documentacion."
    )


@pytest.mark.parametrize("clase", _destinos_concretos(),
                         ids=lambda c: getattr(c, "tipo", str(c)))
def test_destino_concreto_implementa_todo_el_contrato(clase):
    """Cada destino real implementa los ocho metodos, no siete."""
    faltan = [m for m in METODOS_CONTRATO
              if getattr(clase, m, None) is getattr(Destino, m, object())]
    assert not faltan, (
        f"El destino '{clase.tipo}' no implementa: {faltan}. Va a fallar durante "
        "una corrida real con un AttributeError que apunta al lugar equivocado."
    )
    assert not clase.__abstractmethods__, (
        f"El destino '{clase.tipo}' tiene metodos abstractos sin implementar: "
        f"{sorted(clase.__abstractmethods__)}"
    )


def test_un_destino_incompleto_no_se_puede_instanciar():
    """
    La prueba de que el contrato de verdad protege: una clase a la que le falta
    ultimo_detalle tiene que ser IMPOSIBLE de instanciar. Mientras C-07 estuvo
    presente, esta clase se creaba sin problema.
    """
    class DestinoIncompleto(Destino):
        tipo = "incompleto"

        def conectar(self): ...
        def cerrar(self): ...
        def asegurar_esquema(self, esquema): ...
        def escribir_tabla(self, esquema, tabla, df): ...
        def escribir_catalogo(self, esquema, fuente_id, filas): ...
        def registrar_corrida(self, corrida): ...
        def ultima_corrida_ok(self, cliente_id, fuente_id): ...
        # falta ultimo_detalle a proposito

    with pytest.raises(TypeError):
        DestinoIncompleto("dsn-de-mentira")


def test_el_modulo_base_no_arrastra_pandas():
    """
    B-05: el bot importa este modulo solo por nombre_esquema/nombre_tabla, que
    son manipulacion de strings. Cargar pandas ahi eran decenas de MB de RAM y
    segundos de arranque para nada.
    """
    fuente = inspect.getsource(wbase)
    cabecera = fuente.split("class Corrida")[0]
    assert "\nimport pandas" not in cabecera, (
        "warehouse/base.py volvio a importar pandas en el top-level."
    )
