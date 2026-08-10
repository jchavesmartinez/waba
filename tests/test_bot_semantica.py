"""
Pruebas del acceso del bot a la capa semantica.

warehouse_ro.py es el archivo que garantiza que un cliente NO vea datos de
otro. Al agregarle el esquema semantico se toca justo esa frontera, asi que lo
que se prueba aca no es tanto que el bot vea las tablas nuevas (eso es lo
facil) sino que el aislamiento entre clientes siga intacto.
"""

from unittest.mock import patch

import bot.warehouse_ro as W
from modelo.construir import nombre_esquema_semantico
from warehouse.base import nombre_esquema


CLIENTE = {"cliente_id": "cliente_a"}


def test_los_dos_esquemas_son_del_mismo_cliente():
    """
    El search_path lleva dos esquemas, pero los dos pertenecen al MISMO
    cliente. El aislamiento no se debilita: se sigue sin ver 'public' ni el
    esquema de nadie mas.
    """
    assert nombre_esquema("cliente_a") == "raw_cliente_a"
    assert nombre_esquema_semantico("cliente_a") == "semantic_cliente_a"
    for otro in ("cliente_b", "cliente_c"):
        assert nombre_esquema_semantico(otro) != nombre_esquema_semantico("cliente_a")


def test_el_search_path_pone_primero_lo_semantico():
    """
    Si una tabla derivada y una raw se llamaran igual, tiene que ganar la
    derivada: es la que tiene la logica de negocio aplicada. Y no puede
    aparecer 'public' ni ningun esquema de otro cliente.
    """
    ejecutados = []

    class _Cx:
        def execute(self, sentencia, *a, **k):
            ejecutados.append(str(sentencia))

    W._prep(_Cx(), "raw_cliente_a", "cliente_a")
    path = next(s for s in ejecutados if "search_path" in s)

    assert path.index("semantic_cliente_a") < path.index("raw_cliente_a")
    assert "public" not in path
    assert "cliente_b" not in path


def test_la_transaccion_sigue_siendo_de_solo_lectura():
    """READ ONLY tiene que ir primero, antes de cualquier otra cosa."""
    ejecutados = []

    class _Cx:
        def execute(self, sentencia, *a, **k):
            ejecutados.append(str(sentencia))

    W._prep(_Cx(), "raw_cliente_a", "cliente_a")
    assert "READ ONLY" in ejecutados[0]


def test_se_listan_tablas_de_los_dos_esquemas():
    consultas = {}

    def _falso(cliente, sql, params):
        consultas["sql"] = sql
        consultas["params"] = params
        return [{"table_name": "correo_zoho__correos"},
                {"table_name": "finanzas__transacciones"},
                {"table_name": "finanzas__transacciones__rechazos"},
                {"table_name": "_mapeo"}]

    with patch.object(W, "leer_interno", _falso):
        tablas = W.listar_tablas(CLIENTE)

    assert consultas["params"]["esq"] == ["semantic_cliente_a", "raw_cliente_a"]
    assert "finanzas__transacciones" in tablas
    assert "correo_zoho__correos" in tablas


def test_las_tablas_de_metadata_y_rechazos_no_llegan_al_bot():
    """
    '_mapeo' es interno y los rechazos son diagnostico de la ingesta, no data
    de negocio. Dejarlos entrar solo gastaria espacio del prompt y le daria al
    modelo tablas que no sabe interpretar.
    """
    def _falso(cliente, sql, params):
        return [{"table_name": "_mapeo"},
                {"table_name": "_catalogo"},
                {"table_name": "finanzas__transacciones__rechazos"},
                {"table_name": "finanzas__transacciones"}]

    with patch.object(W, "leer_interno", _falso):
        tablas = W.listar_tablas(CLIENTE)

    assert tablas == ["finanzas__transacciones"]


def test_las_columnas_se_buscan_en_los_dos_esquemas():
    consultas = {}

    def _falso(cliente, sql, params):
        consultas["params"] = params
        return [{"table_name": "finanzas__transacciones",
                 "column_name": "monto", "data_type": "double precision"}]

    with patch.object(W, "leer_interno", _falso):
        cols = W.listar_columnas(CLIENTE, ["finanzas__transacciones"])

    assert consultas["params"]["esq"] == ["semantic_cliente_a", "raw_cliente_a"]
    assert cols["finanzas__transacciones"] == [("monto", "double precision")]
