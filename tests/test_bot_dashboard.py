import time

import pytest
from fastapi.testclient import TestClient

import config
from bot import catalogo, dashboard
from bot import app as app_mod
from bot.salida import Adjunto, Respuesta


@pytest.fixture
def dashboard_configurado(monkeypatch):
    monkeypatch.setattr(config, "BOT_DASHBOARD", True)
    monkeypatch.setattr(config, "APP_PUBLIC_URL", "https://app.example.com")
    monkeypatch.setattr(config, "DASHBOARD_SECRET", "s" * 48)
    monkeypatch.setattr(config, "DASHBOARD_TOKEN_TTL_MINUTOS", 30)
    monkeypatch.setattr(config, "DASHBOARD_SESION_DIAS", 180)
    return {"cliente_id": "cliente_a", "nombre": "Cliente A"}


def test_detecta_solicitudes_de_dashboard():
    assert dashboard.es_solicitud("Mándame mi dashboard")
    assert dashboard.es_solicitud("Quiero ver mis KPIs")
    assert dashboard.es_solicitud("Abre el panel financiero")
    assert not dashboard.es_solicitud("¿Cuánto gasté este mes?")


def test_enlace_firmado_conserva_cliente_numero_y_periodo(
    monkeypatch, dashboard_configurado,
):
    cliente = dashboard_configurado
    monkeypatch.setattr(dashboard.registry, "resolver", lambda numero: cliente)

    url, etiqueta = dashboard.crear_enlace(
        cliente, "+506 8888-9999", "dashboard de agosto 2026",
    )
    token = url.rsplit("/", 1)[-1]
    payload, resuelto = dashboard.validar_enlace(token)

    assert etiqueta == "agosto 2026"
    assert payload["cid"] == "cliente_a"
    assert payload["num"] == "50688889999"
    assert payload["inicio"] == "2026-08-01"
    assert payload["fin"] == "2026-09-01"
    assert resuelto == cliente


def test_enlace_alterado_o_vencido_se_rechaza(monkeypatch, dashboard_configurado):
    cliente = dashboard_configurado
    monkeypatch.setattr(dashboard.registry, "resolver", lambda numero: cliente)
    url, _ = dashboard.crear_enlace(cliente, "50688889999", "dashboard")
    token = url.rsplit("/", 1)[-1]

    with pytest.raises(dashboard.EnlaceInvalido):
        dashboard.validar_enlace(token + "x")
    with pytest.raises(dashboard.EnlaceInvalido):
        dashboard.validar_enlace(token, ahora=int(time.time()) + 3600)


def test_render_incrusta_snapshot_sin_llamadas_del_frontend(
    monkeypatch, dashboard_configurado,
):
    cliente = dashboard_configurado
    monkeypatch.setattr(dashboard.registry, "resolver", lambda numero: cliente)
    monkeypatch.setattr(
        dashboard,
        "generar_snapshot",
        lambda cliente, periodo: {
            "cliente": {"id": "cliente_a", "nombre": "Cliente A"},
            "periodo": {**periodo, "etiqueta": "agosto 2026"},
            "actualizado_en": "2026-09-02T02:00-06:00",
            "kpis": [],
        },
    )
    url, _ = dashboard.crear_enlace(
        cliente, "50688889999", "dashboard de agosto 2026",
    )
    html = dashboard.renderizar(url.rsplit("/", 1)[-1])

    assert "__DASHBOARD_DATA__" not in html
    assert "__DASHBOARD_ASSET_VERSION__" not in html
    assert '"nombre":"Cliente A"' in html
    assert "/dashboard-assets/app.js" in html
    assert 'id="chat-mensajes"' in html


def test_enlace_activa_sesion_y_redirige_a_url_permanente(
    monkeypatch, dashboard_configurado,
):
    cliente = dashboard_configurado
    monkeypatch.setattr(dashboard.registry, "resolver", lambda _numero: cliente)
    url, _ = dashboard.crear_enlace(cliente, "50688889999", "dashboard")

    respuesta = TestClient(app_mod.app).get(
        url.replace("https://app.example.com", ""), follow_redirects=False,
    )

    assert respuesta.status_code == 303
    assert respuesta.headers["location"] == "/dashboard"
    assert f"{dashboard.DASHBOARD_COOKIE}=" in respuesta.headers["set-cookie"]
    assert "HttpOnly" in respuesta.headers["set-cookie"]


def test_app_permanente_entrega_shell_y_datos_por_api(
    monkeypatch, dashboard_configurado,
):
    cliente = dashboard_configurado
    sesion = {"v": 2, "cid": "cliente_a", "num": "50688889999", "exp": int(time.time()) + 600}
    snapshot = {
        "cliente": {"id": "cliente_a", "nombre": "Cliente A"},
        "periodo": {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
        "kpis": [], "movimientos": [], "lineas_presupuesto": [],
    }
    monkeypatch.setattr(app_mod.dashboard, "validar_sesion", lambda _: (sesion, cliente))
    monkeypatch.setattr(app_mod.dashboard, "obtener_snapshot", lambda *_: (snapshot, False))
    web = TestClient(app_mod.app)
    web.cookies.set(dashboard.DASHBOARD_COOKIE, "sesion")

    pagina = web.get("/dashboard")
    datos = web.get("/api/dashboard/datos?inicio=2026-09-01")

    assert pagina.status_code == 200
    assert 'id="cargando-dashboard"' in pagina.text
    assert '"nombre":"Cliente A"' not in pagina.text
    assert datos.status_code == 200
    assert datos.json()["dashboard"] == snapshot


def test_snapshot_reaplica_ultima_edicion_pendiente_despues_de_recargar(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    periodo = {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"}
    snapshot = {
        "kpis": [
            {
                "kpi": "gasto_por_concepto",
                "columnas": [
                    "linea_id", "categoria", "concepto", "presupuesto",
                    "gastado", "disponible", "porcentaje",
                ],
                "filas": [
                    ["linea_anterior", "Otros", "Anterior", 100, 40, 60, 40],
                    ["linea_nueva", "Vivienda", "Nueva", 200, 10, 190, 5],
                ],
            },
            {
                "kpi": "presupuesto_disponible",
                "columnas": ["presupuesto", "gastado", "disponible"],
                "filas": [[300, 50, 250]],
            },
        ],
        "movimientos": [{
            "movimiento_clave": "bac:1", "linea_id": "linea_anterior",
            "categoria": "Otros", "concepto": "Anterior", "monto": 40,
            "moneda": "CRC", "medio_pago": "8774", "fecha": "2026-09-20",
        }],
    }
    monkeypatch.setattr(
        dashboard, "leer_snapshot_persistente", lambda *_: (snapshot, False),
    )
    monkeypatch.setattr(dashboard, "_leer_proyecciones_pendientes", lambda *_: [{
        "tipo": "editar", "movimiento_clave": "bac:1", "version": 7,
        "linea_id": "linea_nueva", "categoria": "Vivienda", "concepto": "Nueva",
        "monto": "25", "moneda": "CRC", "medio_pago": "SINPE",
    }])

    proyectado, sucio = dashboard.obtener_snapshot(cliente, periodo)

    movimiento = proyectado["movimientos"][0]
    assert sucio is False
    assert movimiento["linea_id"] == "linea_nueva"
    assert movimiento["monto"] == 25
    assert movimiento["medio_pago"] == "SINPE"
    assert movimiento["pendiente_sincronizacion"] is True
    assert movimiento["version_pendiente"] == 7
    assert proyectado["kpis"][0]["filas"][0][4:] == [0.0, 100.0, 0.0]
    assert proyectado["kpis"][0]["filas"][1][4:] == [35.0, 165.0, 17.5]
    assert proyectado["kpis"][1]["filas"][0] == [300, 35.0, 265.0]
    # La proyección no contamina el snapshot canónico que se reutiliza en Neon.
    assert snapshot["movimientos"][0]["linea_id"] == "linea_anterior"
    assert snapshot["kpis"][1]["filas"][0] == [300, 50, 250]


def test_snapshot_muestra_creacion_pendiente_solo_en_su_mes(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    snapshot = {"kpis": [], "movimientos": []}
    monkeypatch.setattr(
        dashboard, "_leer_proyecciones_pendientes", lambda *_: [{
            "tipo": "crear", "movimiento_clave": "manual:MAN-1", "version": 1,
            "movimiento": {
                "linea_id": "linea_1", "categoria": "Vivienda", "concepto": "Cuota",
                "fecha": "2026-09-15", "descripcion": "Pago", "monto": "75000",
                "moneda": "CRC", "medio_pago": "SINPE",
            },
        }],
    )

    septiembre = dashboard._aplicar_proyecciones(
        snapshot, cliente, {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
    )
    octubre = dashboard._aplicar_proyecciones(
        snapshot, cliente, {"inicio": "2026-10-01", "fin_exclusivo": "2026-11-01"},
    )

    assert septiembre["movimientos"][0]["movimiento_clave"] == "manual:MAN-1"
    assert septiembre["movimientos"][0]["monto"] == 75000
    assert septiembre["sincronizaciones_pendientes"] == 1
    assert octubre["movimientos"] == []


def test_api_permanente_edita_con_periodo_de_la_vista(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    sesion = {"cid": "cliente_a", "num": "50688889999"}
    recibido = {}
    monkeypatch.setattr(app_mod, "_sesion_dashboard", lambda _request: (sesion, cliente))
    monkeypatch.setattr(
        app_mod.dashboard_edicion, "reclasificar",
        lambda token, clave, linea, medio, alcance, monto: recibido.update(
            token=token, clave=clave, linea=linea, monto=monto,
        ) or {"ok": True, "estado": "pendiente"},
    )
    monkeypatch.setattr(app_mod.dashboard_edicion, "procesar_reconstrucciones_cliente", lambda *_: None)

    respuesta = TestClient(app_mod.app).post(
        "/api/dashboard/movimientos/reclasificar",
        json={
            "movimiento_clave": "bac:1", "linea_id": "gas_comedera",
            "medio_pago": "VISA", "alcance": "individual", "monto": "4900",
            "periodo_inicio": "2026-09-01",
        },
    )

    assert respuesta.status_code == 200
    payload = dashboard._decodificar_payload(recibido["token"])
    assert payload["inicio"] == "2026-09-01"
    assert payload["fin"] == "2026-10-01"
    assert recibido["clave"] == "bac:1"


def test_chat_dashboard_reutiliza_numero_e_historial_de_whatsapp(monkeypatch):
    cliente = {"cliente_id": "cliente_a", "nombre": "Cliente A"}
    llamadas = []
    monkeypatch.setattr(
        app_mod.dashboard, "validar_enlace",
        lambda _token: ({"num": "50688889999"}, cliente),
    )
    monkeypatch.setattr(
        app_mod.memoria, "cargar_historial",
        lambda c, numero: [
            {"rol": "user", "contenido": "¿Cuánto gasté ayer?", "sql": "", "estado": {}},
            {"rol": "assistant", "contenido": "Gastaste ₡2.000.", "sql": "SELECT", "estado": {}},
        ],
    )
    monkeypatch.setattr(
        app_mod, "responder",
        lambda numero, pregunta: llamadas.append((numero, pregunta)) or Respuesta(
            "En alimentación.", botones=[{"id": "edicion:confirmar", "title": "Confirmar"}],
        ),
    )

    web = TestClient(app_mod.app)
    historial = web.get("/dashboard/token/chat")
    respuesta = web.post("/dashboard/token/chat", json={"mensaje": "¿Y en cuál categoría?"})

    assert historial.status_code == 200
    assert historial.json()["mensajes"][1]["contenido"] == "Gastaste ₡2.000."
    assert respuesta.status_code == 200
    assert llamadas == [("50688889999", "¿Y en cuál categoría?")]
    assert respuesta.json()["mensaje"]["contenido"] == "En alimentación."
    # El navegador solo recibe el texto controlado que debe enviar de vuelta,
    # no el id interno de las acciones interactivas de WhatsApp.
    assert respuesta.json()["botones"] == [{"title": "Confirmar"}]


def test_chat_dashboard_entrega_adjunto_pequeno_como_descarga(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    monkeypatch.setattr(
        app_mod.dashboard, "validar_enlace",
        lambda _token: ({"num": "50688889999"}, cliente),
    )
    monkeypatch.setattr(
        app_mod, "responder",
        lambda *_: Respuesta(
            "Listo.", adjuntos=[Adjunto("document", b"contenido", "detalle.csv", "text/csv")],
        ),
    )

    respuesta = TestClient(app_mod.app).post("/dashboard/token/chat", json={"mensaje": "Dámelo en CSV"})

    assert respuesta.status_code == 200
    assert respuesta.json()["adjuntos"] == [{
        "nombre": "detalle.csv", "mime": "text/csv", "contenido_b64": "Y29udGVuaWRv",
    }]


def test_endpoint_pago_delega_en_movimiento_manual_normal(monkeypatch):
    cliente = {"cliente_id": "cliente_a"}
    recibido = {}
    monkeypatch.setattr(
        app_mod.dashboard, "validar_enlace",
        lambda _token: ({"cid": "cliente_a"}, cliente),
    )
    monkeypatch.setattr(app_mod.dashboard_edicion, "procesar_reconstrucciones_cliente", lambda *_: None)
    monkeypatch.setattr(
        app_mod.dashboard_edicion, "registrar_pago",
        lambda token, linea, monto, fecha: recibido.update(
            token=token, linea=linea, monto=monto, fecha=fecha,
        ) or {"ok": True, "movimiento_id": "MAN-77"},
    )

    respuesta = TestClient(app_mod.app).post(
        "/dashboard/token/conceptos/gas_hipoteca/pagar",
        json={"monto": "75000", "fecha": "2026-09-15"},
    )

    assert respuesta.status_code == 200
    assert respuesta.json() == {"ok": True, "movimiento_id": "MAN-77"}
    assert recibido == {
        "token": "token", "linea": "gas_hipoteca", "monto": "75000", "fecha": "2026-09-15",
    }


def test_endpoint_editar_movimiento_pasa_monto_al_mismo_flujo(monkeypatch):
    recibido = {}
    monkeypatch.setattr(
        app_mod.dashboard_edicion, "reclasificar",
        lambda token, clave, linea, medio, alcance, monto: recibido.update(
            token=token, clave=clave, linea=linea, medio=medio, alcance=alcance, monto=monto,
        ) or {"ok": True, "estado": "pendiente", "version": 1},
    )
    monkeypatch.setattr(
        app_mod.dashboard, "validar_enlace",
        lambda _token: ({"cid": "cliente_a"}, {"cliente_id": "cliente_a"}),
    )
    monkeypatch.setattr(app_mod.dashboard_edicion, "procesar_reconstrucciones_cliente", lambda *_: None)

    respuesta = TestClient(app_mod.app).post(
        "/dashboard/token/movimientos/reclasificar",
        json={
            "movimiento_clave": "bac:1", "linea_id": "gas_comedera",
            "medio_pago": "VISA 1234", "alcance": "individual", "monto": "23000",
        },
    )

    assert respuesta.status_code == 200
    assert recibido == {
        "token": "token", "clave": "bac:1", "linea": "gas_comedera",
        "medio": "VISA 1234", "alcance": "individual", "monto": "23000",
    }


def test_formulario_manual_solo_expone_campos_habilitados_por_metadata():
    manuales = catalogo.TablaPermitida(
        tabla_logica="gastos_manuales", tabla_real="gastos_manuales", fuente_id="finanzas",
        configuracion={
            "editable": "si", "acciones_permitidas": "crear,modificar",
            "origen_tipo": "google_sheets", "hoja_origen": "gastos_manuales",
            "origen_fuente_id": "finanzas", "clave_primaria": "movimiento_id",
        },
        columnas_config={
            "movimiento_id": {"calculado_por_sistema": "si", "generador": "id_aleatorio_fecha"},
            "fecha": {"requerido": "si", "tipo_validacion": "fecha_iso", "etiqueta_usuario": "Fecha"},
            "linea_presupuesto_id": {"requerido": "si", "generador": "concepto_a_linea_id", "etiqueta_usuario": "Concepto"},
            "categoria": {"etiqueta_usuario": "Categoría"},
        },
    )
    ctx = catalogo.Contexto(schema_text="", permitidas=[manuales])

    formulario = dashboard._formulario_creacion_manual(ctx)

    assert formulario["tabla"] == "gastos_manuales"
    campos = {campo["nombre"]: campo for campo in formulario["campos"]}
    assert "movimiento_id" not in campos
    assert campos["linea_presupuesto_id"]["seleccion_linea"] is True
    assert campos["categoria"]["derivado_de_linea"] is True


def test_lineas_presupuesto_respetan_periodo_y_pagable_desde_metadata(monkeypatch):
    presupuesto = catalogo.TablaPermitida(
        tabla_logica="presupuesto", tabla_real="presupuesto", fuente_id="finanzas",
        columnas_config={
            "linea_id": {}, "categoria": {}, "concepto": {}, "tipo": {},
            "pagable": {}, "vigencia_desde": {}, "vigencia_hasta": {},
        },
    )
    ctx = catalogo.Contexto(schema_text="", tablas_reales={"presupuesto"}, permitidas=[presupuesto])
    capturado = {}

    def ejecutar(_cliente, sql, limite):
        capturado.update(sql=sql, limite=limite)
        return (["linea_id", "categoria", "concepto", "pagable"], [
            ("gas_hipoteca", "Vivienda", "Hipoteca", True),
        ])

    monkeypatch.setattr(dashboard.nl2sql, "validar_sql", lambda *_: (True, ""))
    monkeypatch.setattr(dashboard.warehouse_ro, "ejecutar", ejecutar)
    lineas = dashboard._lineas_presupuesto(
        {"cliente_id": "cliente_a"}, ctx,
        {"inicio": "2026-08-01", "fin_exclusivo": "2026-09-01"},
    )

    assert "DATE '2026-08-01'" in capturado["sql"]
    assert "AS pagable" in capturado["sql"]
    assert lineas == [{"linea_id": "gas_hipoteca", "categoria": "Vivienda", "concepto": "Hipoteca", "pagable": True}]


def test_lineas_presupuesto_sin_pagable_sigue_siendo_compatible(monkeypatch):
    presupuesto = catalogo.TablaPermitida(
        tabla_logica="presupuesto", tabla_real="presupuesto", fuente_id="finanzas",
        columnas_config={"linea_id": {}, "categoria": {}, "concepto": {}},
    )
    ctx = catalogo.Contexto(schema_text="", tablas_reales={"presupuesto"}, permitidas=[presupuesto])
    capturado = {}
    monkeypatch.setattr(dashboard.nl2sql, "validar_sql", lambda *_: (True, ""))
    monkeypatch.setattr(
        dashboard.warehouse_ro, "ejecutar",
        lambda _cliente, sql, limite: capturado.update(sql=sql) or (
            ["linea_id", "categoria", "concepto", "pagable"], [("gas", "Otros", "No pagable", False)]),
    )

    assert dashboard._lineas_presupuesto({"cliente_id": "cliente_a"}, ctx) == [{
        "linea_id": "gas", "categoria": "Otros", "concepto": "No pagable", "pagable": False,
    }]
    assert "FALSE AS pagable" in capturado["sql"]


def test_jerarquia_prefiere_movimientos_canonicos_para_detalle(monkeypatch):
    """El árbol no debe perder cargos bancarios al existir la tabla canónica."""
    presupuesto = catalogo.TablaPermitida(
        tabla_logica="presupuesto", tabla_real="presupuesto", fuente_id="sheet",
        columnas_config={"linea_id": {}, "categoria": {}, "concepto": {}},
    )
    canonicos = catalogo.TablaPermitida(
        tabla_logica="movimientos", tabla_real="finanzas__movimientos", fuente_id="modelo",
        columnas_config={
            "linea_presupuesto_id": {}, "fecha": {}, "descripcion": {},
            "moneda": {}, "monto_neto": {}, "medio_pago": {}, "tipo_movimiento": {},
        },
    )
    manuales = catalogo.TablaPermitida(
        tabla_logica="gastos_manuales", tabla_real="gastos_manuales", fuente_id="sheet",
        columnas_config={
            "linea_presupuesto_id": {}, "fecha": {}, "descripcion": {},
            "monto": {}, "moneda": {}, "tipo_movimiento": {}, "activo": {},
            "incluir_en_gasto": {},
        },
    )
    ctx = catalogo.Contexto(
        schema_text="", tablas_reales={"presupuesto", "finanzas__movimientos", "gastos_manuales"},
        permitidas=[presupuesto, canonicos, manuales],
    )
    capturado = {}

    def ejecutar(_cliente, sql, limite):
        capturado["sql"] = sql
        capturado["limite"] = limite
        return (
            ["linea_id", "categoria", "concepto", "fecha", "descripcion", "moneda", "monto", "medio_pago"],
            [("gas_comedera", "Alimentacion", "Comedera", "2026-09-05",
              "WALMART", "CRC", 23148, "VISA 1234")],
        )

    monkeypatch.setattr(dashboard.nl2sql, "validar_sql", lambda *_: (True, ""))
    monkeypatch.setattr(dashboard.warehouse_ro, "ejecutar", ejecutar)

    filas = dashboard._movimientos_jerarquia(
        {"cliente_id": "cliente_a"}, ctx,
        {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
    )

    assert 'FROM "finanzas__movimientos" m' in capturado["sql"]
    assert 'FROM "gastos_manuales" g' not in capturado["sql"]
    assert filas == [{
        "linea_id": "gas_comedera", "categoria": "Alimentacion",
        "concepto": "Comedera", "fecha": "2026-09-05",
        "descripcion": "WALMART", "moneda": "CRC", "monto": 23148,
        "medio_pago": "VISA 1234",
    }]
    assert "m.medio_pago" in capturado["sql"]


def test_jerarquia_resuelve_canonica_desde_metadata_si_no_esta_en_catalogo(monkeypatch):
    """Los modelos derivados no necesitan duplicarse en ``_catalogo``."""
    presupuesto = catalogo.TablaPermitida(
        tabla_logica="presupuesto", tabla_real="presupuesto", fuente_id="sheet",
        columnas_config={"linea_id": {}, "categoria": {}, "concepto": {},
                         "vigencia_desde": {}, "vigencia_hasta": {}},
    )
    ctx = catalogo.Contexto(
        schema_text="", tablas_reales={"presupuesto"}, permitidas=[presupuesto],
    )
    monkeypatch.setattr(dashboard.metadata_modelos, "leer", lambda _cliente: {
        "modelos": [{"extractor": "movimientos_canonicos",
                     "tabla_destino": "finanzas__movimientos", "activo": "si"}],
    })
    monkeypatch.setattr(dashboard.warehouse_ro, "listar_tablas",
                        lambda _cliente: ["presupuesto", "finanzas__movimientos"])
    monkeypatch.setattr(
        dashboard.warehouse_ro, "listar_columnas",
        lambda _cliente, _tablas: {"finanzas__movimientos": [
            ("linea_presupuesto_id", "text"), ("fecha", "date"),
            ("descripcion", "text"), ("moneda", "text"),
            ("monto_neto", "numeric"), ("_clave", "text"),
            ("_modelo_id", "text"), ("fuente", "text"),
            ("clave_origen", "text"), ("medio_pago", "text"),
        ]},
    )
    capturado = {}

    def ejecutar(_cliente, sql, limite):
        capturado["sql"] = sql
        return (
            ["linea_id", "categoria", "concepto", "fecha", "descripcion",
             "moneda", "monto", "medio_pago", "movimiento_clave", "fuente",
             "clave_origen", "modelo_canonico"],
            [("gas_comedera", "Alimentacion", "Comedera", "2026-09-05",
              "WALMART", "CRC", 23148, "VISA 1234", "bac:1", "bac", "1",
              "movimientos_consolidados")],
        )

    monkeypatch.setattr(dashboard.nl2sql, "validar_sql", lambda *_: (True, ""))
    monkeypatch.setattr(dashboard.warehouse_ro, "ejecutar", ejecutar)

    filas = dashboard._movimientos_jerarquia(
        {"cliente_id": "cliente_a"}, ctx,
        {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
    )

    assert 'FROM "finanzas__movimientos" m' in capturado["sql"]
    assert filas[0]["movimiento_clave"] == "bac:1"


def test_jerarquia_no_oculta_movimientos_sin_linea_de_presupuesto(monkeypatch):
    """Una línea no mapeada aparece explícitamente como sin clasificar."""
    presupuesto = catalogo.TablaPermitida(
        tabla_logica="presupuesto", tabla_real="presupuesto", fuente_id="sheet",
        columnas_config={"linea_id": {}, "categoria": {}, "concepto": {}},
    )
    canonicos = catalogo.TablaPermitida(
        tabla_logica="movimientos", tabla_real="movimientos", fuente_id="modelo",
        columnas_config={
            "linea_presupuesto_id": {}, "fecha": {}, "descripcion": {},
            "moneda": {}, "monto_neto": {},
        },
    )
    ctx = catalogo.Contexto(
        schema_text="", tablas_reales={"presupuesto", "movimientos"},
        permitidas=[presupuesto, canonicos],
    )

    def ejecutar(_cliente, _sql, limite):
        assert limite > 0
        return (
            ["linea_id", "categoria", "concepto", "fecha", "descripcion", "moneda", "monto"],
            [("sin_clasificar", "Sin clasificar", "Gastos sin identificar",
              "2026-09-05", "Comercio pendiente", "CRC", 1000)],
        )

    monkeypatch.setattr(dashboard.nl2sql, "validar_sql", lambda *_: (True, ""))
    monkeypatch.setattr(dashboard.warehouse_ro, "ejecutar", ejecutar)
    filas = dashboard._movimientos_jerarquia(
        {"cliente_id": "cliente_a"}, ctx,
        {"inicio": "2026-09-01", "fin_exclusivo": "2026-10-01"},
    )
    assert filas[0]["categoria"] == "Sin clasificar"
    assert filas[0]["concepto"] == "Gastos sin identificar"
