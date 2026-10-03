"""Presupuesto: versionado, escrituras atómicas y aislamiento de cliente."""
from contextlib import contextmanager
from datetime import date
import copy
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
import pytest

from bot import presupuesto as p
from bot import app as app_mod

HEADERS = ["linea_id", "tipo", "categoria", "subcategoria", "clave", "concepto",
           "monto_quincenal", "monto_mensual", "rastreable", "medio_pago",
           "vigencia_desde", "vigencia_hasta", "nota", "pagable", ""]
CLIENTE = {"cliente_id": "cliente_a", "fuentes": [{"fuente_id": "google",
           "tipo": "google_sheets", "activo": True,
           "config": {"spreadsheet_id": "libro_real", "hojas": ["presupuesto"]}}]}
ID = "2af93ca8-9c4e-49be-8d22-7c4d65e93c75"


def fila(linea="gas_a", **kwargs):
    valores = {"linea_id": linea, "tipo": "gasto", "categoria": "Servicios",
        "concepto": "Internet", "monto_mensual": "29000", "monto_quincenal": "14501",
        "vigencia_desde": "2026-09-01", "vigencia_hasta": "2026-12-31",
        "nota": "Conservar esta nota", "pagable": "si", "medio_pago": "tarjeta_bac"} | kwargs
    return [valores.get(h, "") for h in HEADERS]


def solicitud(grilla, **kwargs):
    return {"inicio": "2026-10-01", "revision": p._revision(grilla), "operacion_id": ID,
            "confirmado": True, "cambios": [{"linea_id": "gas_a", "monto_mensual": "30000.50"}],
            "nuevas": []} | kwargs


@pytest.fixture(autouse=True)
def reloj(monkeypatch):
    monkeypatch.setattr(p, "fecha_local", lambda: date(2026, 10, 2))


def test_divide_vigencia_sin_perder_ids_o_columnas():
    grilla = [HEADERS, fila(), fila("ing_a", tipo="ingreso", concepto="Salario", monto_mensual="100000")]
    original = copy.deepcopy(grilla)
    plan = p.planificar(HEADERS, grilla, solicitud(grilla), ID)
    assert grilla == original
    assert plan["escrituras"] == [(1, 11, "2026-09-30")]
    assert plan["grilla"][1][7] == "29000"  # septiembre intacto
    nueva = dict(zip(HEADERS, plan["nuevas_filas"][0]))
    assert nueva["linea_id"] == "gas_a"
    assert nueva["monto_mensual"] == "30000.50"
    assert nueva["monto_quincenal"] == "15000.25"
    assert nueva["vigencia_desde"] == "2026-10-01"
    assert nueva["vigencia_hasta"] == "2026-12-31"
    assert nueva["nota"] == "Conservar esta nota"
    assert nueva["medio_pago"] == "tarjeta_bac"
    assert plan["grilla"][2] == original[2]
    assert len(p._vista(HEADERS, plan["grilla"], date(2026, 9, 1))) == 2
    assert len(p._vista(HEADERS, plan["grilla"], date(2026, 10, 1))) == 2


def test_edicion_en_mismo_inicio_no_duplica_y_respeta_futuro():
    grilla = [HEADERS, fila(vigencia_desde="2026-10-01", vigencia_hasta="2026-10-31"),
              fila(vigencia_desde="2026-11-01", monto_mensual="50000")]
    plan = p.planificar(HEADERS, grilla, solicitud(grilla), ID)
    assert len(plan["escrituras"]) == 2
    assert not plan["nuevas_filas"]
    assert plan["grilla"][2] == grilla[2]


def test_creacion_categoria_y_subpartidas_cero_y_ingreso():
    grilla = [HEADERS, fila()]
    nuevas = [{"tipo": "gasto", "categoria": "Educación", "concepto": "Curso", "monto_mensual": "0"},
              {"tipo": "ingreso", "categoria": "Emprendimiento", "concepto": "Ventas", "monto_mensual": "1000.01"}]
    plan = p.planificar(HEADERS, grilla, solicitud(grilla, cambios=[], nuevas=nuevas), ID)
    assert not plan["escrituras"]
    assert len(plan["nuevas_filas"]) == 2
    assert plan["nuevas_filas"][0][13] == "si"
    assert plan["nuevas_filas"][1][6] == "500.01"
    assert plan["nuevas_filas"][1][13] == "no"
    assert plan == p.planificar(HEADERS, grilla, solicitud(grilla, cambios=[], nuevas=nuevas), ID)
    assert len({f[0] for f in plan["grilla"][1:]}) == 3
    assert p._vista(HEADERS, plan["grilla"], date(2026, 10, 1))[0]["tipo"] == "gasto"


@pytest.mark.parametrize("monto", ["-1", "NaN", "Infinity", "1e15", "1.234", "", None])
def test_rechaza_montos_invalidos(monto):
    grilla = [HEADERS, fila()]
    with pytest.raises(p.ErrorPresupuesto):
        p.planificar(HEADERS, grilla, solicitud(grilla, cambios=[{"linea_id": "gas_a", "monto_mensual": monto}]), ID)


@pytest.mark.parametrize("inicio", ["2026-09-01", "2026-10-02", "2032-01-01", "x"])
def test_no_modifica_pasado_o_rango_arbitrario(inicio):
    grilla = [HEADERS, fila()]
    with pytest.raises(p.ErrorPresupuesto):
        p.planificar(HEADERS, grilla, solicitud(grilla, inicio=inicio), ID)


def test_rechaza_repetidos_solapamientos_lineas_ajenas_y_renombrados():
    grilla = [HEADERS, fila()]
    for cambios in [[{"linea_id": "otra", "monto_mensual": "1"}],
                    [{"linea_id": "gas_a", "monto_mensual": "1", "categoria": "Otra"}],
                    [{"linea_id": "gas_a", "monto_mensual": "1"}] * 2]:
        with pytest.raises(p.ErrorPresupuesto):
            p.planificar(HEADERS, grilla, solicitud(grilla, cambios=cambios), ID)
    with pytest.raises(p.ErrorPresupuesto, match="superpuestas"):
        p.planificar(HEADERS, [HEADERS, fila(), fila()], solicitud(grilla), ID)


def test_nueva_categoria_reutiliza_nombre_y_rechaza_duplicacion():
    grilla = [HEADERS, fila()]
    nuevas = [{"tipo": "gasto", "categoria": "servícios", "concepto": "Celular", "monto_mensual": "0"}]
    plan = p.planificar(HEADERS, grilla, solicitud(grilla, cambios=[], nuevas=nuevas), ID)
    assert plan["nuevas_filas"][0][2] == "Servicios"
    nuevas[0]["concepto"] = "internet"
    with pytest.raises(p.ErrorPresupuesto, match="Ya existe"):
        p.planificar(HEADERS, grilla, solicitud(grilla, cambios=[], nuevas=nuevas), ID)


def test_no_sobrescribe_formulas():
    grilla = [HEADERS, fila(monto_mensual="=SUM(A1:A2)")]
    with pytest.raises(p.ErrorPresupuesto):
        p.planificar(HEADERS, grilla, solicitud(grilla), ID)
    grilla = [HEADERS, fila(nota="=A2")]
    with pytest.raises(p.ErrorPresupuesto, match="fórmulas"):
        p.planificar(HEADERS, grilla, solicitud(grilla), ID)


def test_lee_vigencias_nativas_y_revision_numeros():
    assert p._fecha_celda(46266) == date(2026, 9, 1)
    assert p._fecha_celda("01/10/2026") == date(2026, 10, 1)
    assert p._revision([HEADERS, fila(monto_mensual="30000.00")]) == p._revision([HEADERS, fila(monto_mensual=30000)])


class Libro:
    id = 12

    def __init__(self, grilla):
        self.grilla = copy.deepcopy(grilla)
        self.llamadas = []
        self.fallar = False

    def get_all_values(self, **kwargs):
        assert kwargs["value_render_option"] == "FORMULA"
        return copy.deepcopy(self.grilla)

    def worksheet(self, nombre):
        assert nombre == "presupuesto"
        return self

    def batch_update(self, cuerpo):
        self.llamadas.append(cuerpo)
        for req in cuerpo["requests"]:
            if "updateCells" in req:
                datos = req["updateCells"]; start = datos["start"]
                celda = datos["rows"][0]["values"][0].get("userEnteredValue", {})
                self.grilla[start["rowIndex"]][start["columnIndex"]] = next(iter(celda.values()), "")
            else:
                for fila_ in req["appendCells"]["rows"]:
                    self.grilla.append([next(iter(c.get("userEnteredValue", {}).values()), "") for c in fila_["values"]])
        if self.fallar:
            raise TimeoutError("respuesta perdida después de aplicar")


class Auditoria:
    def __init__(self):
        self.filas = {}
        self.commits = 0

    def commit(self):
        self.commits += 1

    def execute(self, sql, params):
        consulta = str(sql)
        llave = (params["cid"], params.get("id"))
        if consulta.startswith("SELECT *"):
            return SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: self.filas.get(llave)))
        if consulta.startswith("INSERT"):
            assert llave not in self.filas
            self.filas[llave] = {"solicitud_hash": params["firma"], "estado": "preparado", "revision_despues": params["revision"]}
        if "SET estado='guardado'" in consulta:
            self.filas[llave]["estado"] = "guardado"
        if "SET estado='encolado'" in consulta:
            self.filas[llave]["estado"] = "encolado"
        if "SET estado='incierto'" in consulta:
            self.filas[llave]["estado"] = "incierto"


@pytest.fixture
def sistema(monkeypatch):
    libro = Libro([HEADERS, fila()])
    auditoria = Auditoria()
    cola = []
    @contextmanager
    def bloqueo(_):
        yield auditoria
    monkeypatch.setattr(p, "_bloqueo", bloqueo)
    monkeypatch.setattr(p, "_origen", lambda _: CLIENTE["fuentes"][0])
    monkeypatch.setattr(p, "abrir_libro", lambda _: libro)
    monkeypatch.setattr(p, "abrir_libro_escritura", lambda _: libro)
    monkeypatch.setattr(p.dashboard_edicion, "_encolar_reconstruccion", lambda *args: cola.append(args))
    return libro, auditoria, cola


def test_guardado_atomico_auditable_e_idempotente(sistema):
    libro, auditoria, cola = sistema
    datos = solicitud(libro.grilla)
    result = p.guardar(CLIENTE, datos)
    assert result["guardado"]
    assert len(libro.llamadas) == 1  # cierra vieja vigencia y añade nueva en el MISMO lote
    assert len(libro.llamadas[0]["requests"]) == 2
    assert "numberValue" in libro.llamadas[0]["requests"][1]["appendCells"]["rows"][0]["values"][7]["userEnteredValue"]
    assert cola[0][1:] == ("presupuesto." + ID, "google")
    assert auditoria.filas[("cliente_a", ID)]["estado"] == "encolado"
    assert p.guardar(CLIENTE, datos)["guardado"]
    assert len(libro.llamadas) == len(cola) == 1


def test_revision_obsoleta_no_escribe(sistema):
    libro, auditoria, cola = sistema
    datos = solicitud(libro.grilla, revision="obsoleta")
    with pytest.raises(p.ErrorPresupuesto, match="cambió"):
        p.guardar(CLIENTE, datos)
    assert not libro.llamadas and not auditoria.filas and not cola


def test_timeout_despues_de_guardar_no_duplica(sistema):
    libro, _, cola = sistema
    libro.fallar = True
    datos = solicitud(libro.grilla)
    assert p.guardar(CLIENTE, datos)["guardado"]
    assert p.guardar(CLIENTE, datos)["guardado"]
    assert len(libro.grilla) == 3
    assert len(libro.llamadas) == len(cola) == 1


def test_timeout_sin_aplicar_no_inventa_exito(sistema, monkeypatch):
    libro, auditoria, cola = sistema
    def falla(_):
        raise TimeoutError()
    monkeypatch.setattr(libro, "batch_update", falla)
    datos = solicitud(libro.grilla)
    with pytest.raises(p.ErrorPresupuesto, match="confirmar"):
        p.guardar(CLIENTE, datos)
    assert auditoria.filas[("cliente_a", ID)]["estado"] == "incierto"
    assert not cola and len(libro.grilla) == 2
    with pytest.raises(p.ErrorPresupuesto, match="anterior"):
        p.guardar(CLIENTE, datos)


def test_no_confirmado_no_abre_fuente(sistema):
    libro, _, _ = sistema
    with pytest.raises(p.ErrorPresupuesto, match="confirme"):
        p.guardar(CLIENTE, solicitud(libro.grilla, confirmado=False))
    assert not libro.llamadas


def test_fuente_sale_del_cliente_no_del_navegador(monkeypatch):
    tabla = SimpleNamespace(tabla_logica="presupuesto", tabla_real="google__presupuesto")
    monkeypatch.setattr(p.catalogo, "construir_contexto", lambda _: SimpleNamespace(error_lectura=False, permitidas=[tabla]))
    assert p._origen(CLIENTE)["config"]["spreadsheet_id"] == "libro_real"
    with pytest.raises(p.ErrorPresupuesto):
        p._origen({"cliente_id": "cliente_b", "fuentes": []})


def test_api_exige_sesion_y_aisla_cliente(monkeypatch):
    def sin_sesion(_):
        raise app_mod.dashboard.EnlaceInvalido("ausente")
    monkeypatch.setattr(app_mod, "_sesion_dashboard", sin_sesion)
    web = TestClient(app_mod.app)
    assert web.get("/api/dashboard/presupuesto?inicio=2026-10-01").status_code == 401
    assert web.post("/api/dashboard/presupuesto", json={"confirmado": True}).status_code == 401
    recibido = []
    monkeypatch.setattr(app_mod, "_sesion_dashboard", lambda _: ({"cid": "cliente_a"}, CLIENTE))
    monkeypatch.setattr(p, "guardar", lambda c, d: recibido.append(c) or {"ok": True})
    monkeypatch.setattr(app_mod.dashboard_edicion, "procesar_reconstrucciones_cliente", lambda _: 0)
    assert web.post("/api/dashboard/presupuesto", json={"cliente_id": "cliente_b"}).status_code == 200
    assert recibido == [CLIENTE]
