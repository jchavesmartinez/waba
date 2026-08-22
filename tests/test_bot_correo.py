"""Regresiones de Google OAuth y correo saliente confirmado."""

from email import policy
from email.parser import BytesParser
from urllib.parse import parse_qs, urlparse

import config
from fastapi.testclient import TestClient

from bot import correo
from bot import app as app_mod
from bot import responder as responder_mod
from bot.salida import Adjunto, Respuesta


CLIENTE = {"cliente_id": "cliente_prueba"}
CONEXION = correo.Conexion(
    "sheila@empresa.com", "token-cifrado", correo.GMAIL_SEND_SCOPE, "google-sub",
)
ARTEFACTO = correo.Artefacto(
    7, "ventas_marzo.pdf", "application/pdf", b"%PDF-prueba",
)


def test_detecta_pedido_de_correo_con_archivo():
    assert correo.es_pedido("Envíe el PDF anterior a gerente@empresa.com")
    assert not correo.es_pedido("¿Cuánto vendimos ayer?")


def test_extrae_destinatario_asunto_y_texto():
    destino, asunto, cuerpo = correo._extraer_campos(
        "Envía el PDF a Gerente@Empresa.com asunto: Marzo y texto: Hola, adjunto el reporte",
        "ventas.pdf",
    )
    assert destino == "gerente@empresa.com"
    assert asunto == "Marzo"
    assert cuerpo == "Hola, adjunto el reporte"


def test_separa_generacion_de_pdf_de_la_instruccion_de_correo():
    pedido = (
        "Crea un PDF con las últimas excursiones y sus reservaciones, "
        "envíaselo a gerente@empresa.com y pon en el cuerpo: Hecho por el bot"
    )
    assert correo.es_generacion_y_correo(pedido)
    pregunta = correo.pregunta_para_generar(pedido)
    assert "excursiones" in pregunta
    assert "PDF" not in pregunta
    assert "Crea" not in pregunta
    assert "gerente@empresa.com" not in pregunta
    assert "cuerpo" not in pregunta


def test_turno_compuesto_genera_archivo_y_luego_prepara_borrador(monkeypatch):
    pedido = (
        "Crea un PDF con las últimas excursiones, envíaselo a "
        "gerente@empresa.com y pon en el cuerpo: Prueba"
    )
    adjunto = Adjunto(
        tipo="document", contenido=b"%PDF-prueba", nombre="excursiones.pdf",
        mime="application/pdf",
    )
    contexto = type("Ctx", (), {"permitidas": [], "tablas_reales": {"excursiones"}})()
    eventos = []

    monkeypatch.setattr(responder_mod.registry, "resolver", lambda _n: CLIENTE)
    monkeypatch.setattr(responder_mod, "_pasa_tope_diario", lambda _cid: True)
    monkeypatch.setattr(responder_mod.memoria, "cargar_historial", lambda *_: [])
    monkeypatch.setattr(responder_mod.memoria, "guardar_intercambio", lambda *_a, **_k: None)
    monkeypatch.setattr(responder_mod.catalogo, "construir_contexto", lambda _c: contexto)
    monkeypatch.setattr(responder_mod.catalogo, "nombres_habilitados", lambda _c: ["excursiones"])
    monkeypatch.setattr(responder_mod.catalogo, "resumir_habilitados", lambda _c: "excursiones")
    monkeypatch.setattr(
        responder_mod.formato, "detectar_con_contexto",
        lambda pregunta, _historial: (
            responder_mod.formato.PDF if "PDF" in pregunta else responder_mod.formato.TEXTO
        ),
    )

    def responder_datos(_cliente, _numero, pregunta, _historial, **_kwargs):
        eventos.append(("generar", pregunta))
        return Respuesta("Preparé el PDF.", adjuntos=[adjunto])

    def guardar(_cliente, _numero, adjuntos):
        eventos.append(("guardar", adjuntos[0].nombre))

    def preparar(_cliente, _numero, texto):
        eventos.append(("correo", texto))
        return Respuesta("Confirme el envío a gerente@empresa.com respondiendo *sí*.")

    monkeypatch.setattr(responder_mod, "_responder_datos", responder_datos)
    monkeypatch.setattr(responder_mod.correo, "guardar_artefactos", guardar)
    monkeypatch.setattr(responder_mod.correo, "procesar_mensaje", preparar)

    respuesta = responder_mod.responder("5061111", pedido)

    assert [evento[0] for evento in eventos] == ["generar", "guardar", "correo"]
    assert "gerente@empresa.com" not in eventos[0][1]
    assert eventos[0][1] == "las últimas excursiones"
    assert respuesta.adjuntos == [adjunto]
    assert "Confirme el envío" in respuesta.texto


def test_refresh_token_se_cifra_con_contexto(monkeypatch):
    monkeypatch.setattr(config, "OAUTH_TOKEN_KEY", "x" * 48)
    cifrado = correo._cifrar_token("refresh-secreto", "cliente_a", "5061111")
    assert "refresh-secreto" not in cifrado
    assert correo._descifrar_token(cifrado, "cliente_a", "5061111") == "refresh-secreto"
    try:
        correo._descifrar_token(cifrado, "cliente_b", "5061111")
        assert False, "el token no debe poder moverse de cliente"
    except Exception:
        pass


def test_state_firmado_resuelve_solo_el_cliente_correcto(monkeypatch):
    monkeypatch.setattr(config, "OAUTH_TOKEN_KEY", "y" * 48)
    monkeypatch.setattr(correo.registry, "listar_clientes", lambda: [CLIENTE])
    state = correo._crear_state("cliente_prueba")
    assert correo._cliente_desde_state(state) == CLIENTE
    assert correo._cliente_desde_state(state + "x") is None


def test_url_google_pide_solo_identidad_y_gmail_send(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setattr(config, "GOOGLE_OAUTH_REDIRECT_URI", "https://app.example.com/oauth/google/callback")
    monkeypatch.setattr(config, "APP_TERMS_VERSION", "v1")
    monkeypatch.setattr(correo, "_cliente_desde_state", lambda _s: CLIENTE)
    monkeypatch.setattr(correo, "_sesion_oauth", lambda *_args, **_kwargs: ("506", "verifier", ""))

    class ConexionDB:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, *_args, **_kwargs): return None

    class Engine:
        def begin(self): return ConexionDB()

    monkeypatch.setattr(correo, "_engine", lambda *_: Engine())
    url = correo.url_autorizacion_google("state")
    query = parse_qs(urlparse(url).query)
    scopes = set(query["scope"][0].split())
    assert scopes == {"openid", "email", "profile", correo.GMAIL_SEND_SCOPE}
    assert "gmail.readonly" not in query["scope"][0]
    assert query["access_type"] == ["offline"]
    assert query["code_challenge_method"] == ["S256"]


def test_conectar_correo_devuelve_enlace_sin_llm(monkeypatch):
    monkeypatch.setattr(config, "BOT_EMAIL", True)
    monkeypatch.setattr(config, "OAUTH_ENLACE_TTL_MINUTOS", 10)
    monkeypatch.setattr(
        correo, "crear_enlace_conexion",
        lambda *_: "https://app.example.com/oauth/google/iniciar?token=abc",
    )
    respuesta = correo.procesar_mensaje(CLIENTE, "5061111", "Conectar mi correo")
    assert "https://app.example.com" in respuesta.texto
    assert "no podrá leer" in respuesta.texto


def test_preguntar_cual_correo_esta_conectado_no_genera_otro_enlace(monkeypatch):
    monkeypatch.setattr(config, "BOT_EMAIL", True)
    monkeypatch.setattr(correo, "_conexion", lambda *_: CONEXION)
    monkeypatch.setattr(
        correo, "crear_enlace_conexion",
        lambda *_: (_ for _ in ()).throw(AssertionError("no debe crear enlace")),
    )
    respuesta = correo.procesar_mensaje(
        CLIENTE, "5061111", "¿Cuál correo tengo conectado?",
    )
    assert "sheila@empresa.com" in respuesta.texto


def test_pedido_crea_vista_previa_y_no_envia(monkeypatch):
    monkeypatch.setattr(config, "BOT_EMAIL", True)
    monkeypatch.setattr(correo, "_conexion", lambda *_: CONEXION)
    monkeypatch.setattr(correo, "_ultimo_artefacto", lambda *_: ARTEFACTO)
    creados = []

    def crear(_cliente, _numero, conexion, artefacto, destino, asunto, cuerpo):
        borrador = correo.Borrador("uuid", conexion.correo, destino, asunto, cuerpo, artefacto)
        creados.append(borrador)
        return borrador

    monkeypatch.setattr(correo, "_crear_borrador", crear)
    respuesta = correo.procesar_mensaje(
        CLIENTE, "5061111",
        "Envía el PDF anterior a gerente@empresa.com asunto: Reporte marzo y texto: Hola",
    )
    assert len(creados) == 1
    assert "De: sheila@empresa.com" in respuesta.texto
    assert "Para: gerente@empresa.com" in respuesta.texto
    assert "ventas_marzo.pdf" in respuesta.texto


def test_si_confirma_una_sola_vez(monkeypatch):
    monkeypatch.setattr(config, "BOT_EMAIL", True)
    borrador = correo.Borrador(
        "uuid", CONEXION.correo, "gerente@empresa.com", "Reporte", "Hola", ARTEFACTO,
    )
    monkeypatch.setattr(correo, "_pendiente", lambda *_: borrador)
    monkeypatch.setattr(correo, "_conexion", lambda *_: CONEXION)
    reclamos = iter([(True, ""), (False, "")])
    monkeypatch.setattr(correo, "_reclamar", lambda *_: next(reclamos))
    envios = []
    monkeypatch.setattr(
        correo, "_enviar_google",
        lambda *_: (envios.append("enviado") is None, "gmail-id", False),
    )
    primera = correo.procesar_mensaje(CLIENTE, "5061111", "sí")
    segunda = correo.procesar_mensaje(CLIENTE, "5061111", "sí")
    assert len(envios) == 1
    assert "Gmail aceptó" in primera.texto
    assert "ya fue procesado" in segunda.texto


def test_mime_contiene_texto_y_adjunto_exactos():
    borrador = correo.Borrador(
        "uuid", "sheila@empresa.com", "gerente@empresa.com",
        "Reporte", "Hola, adjunto el reporte.", ARTEFACTO,
    )
    crudo = correo._b64url_decode(correo._mensaje_mime(borrador))
    mensaje = BytesParser(policy=policy.default).parsebytes(crudo)
    assert mensaje["From"] == "sheila@empresa.com"
    assert mensaje["To"] == "gerente@empresa.com"
    assert mensaje["Subject"] == "Reporte"
    adjunto = next(mensaje.iter_attachments())
    assert adjunto.get_filename() == "ventas_marzo.pdf"
    assert adjunto.get_payload(decode=True) == ARTEFACTO.contenido


def test_timeout_despues_de_llamar_gmail_es_ambiguo_y_no_se_reintenta(monkeypatch):
    monkeypatch.setattr(config, "BOT_EMAIL_TIMEOUT_SEGUNDOS", 30)
    monkeypatch.setattr(correo, "_access_token", lambda *_: "access")
    monkeypatch.setattr(correo.httpx, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(correo.httpx.TimeoutException("timeout")))
    ejecutado = []

    class ConexionDB:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, *_args, **_kwargs): ejecutado.append(True)

    class Engine:
        def begin(self): return ConexionDB()

    monkeypatch.setattr(correo, "_engine", lambda *_: Engine())
    borrador = correo.Borrador(
        "uuid", CONEXION.correo, "gerente@empresa.com", "Reporte", "Hola", ARTEFACTO,
    )
    ok, _, ambiguo = correo._enviar_google(CLIENTE, "506", CONEXION, borrador)
    assert not ok and ambiguo
    assert ejecutado


def test_pagina_oauth_exige_aceptacion_post(monkeypatch):
    monkeypatch.setattr(correo, "validar_enlace_oauth", lambda _token: True)
    monkeypatch.setattr(config, "APP_TERMS_URL", "https://fachavi.example/terminos")
    monkeypatch.setattr(config, "APP_PRIVACY_URL", "https://fachavi.example/privacidad")
    cliente = TestClient(app_mod.app)
    pagina = cliente.get("/oauth/google/iniciar?token=abc")
    assert pagina.status_code == 200
    assert 'method="post"' in pagina.text
    assert 'name="acepto"' in pagina.text
    assert 'required' in pagina.text

    sin_aceptar = cliente.post("/oauth/google/autorizar", data={"token": "abc"})
    assert sin_aceptar.status_code == 400


def test_callback_oauth_conecta_y_avisa_por_whatsapp(monkeypatch):
    resultado = correo.OAuthCompletado(CLIENTE, "5061111", "sheila@empresa.com")
    monkeypatch.setattr(correo, "completar_oauth_google", lambda *_: resultado)
    avisos = []
    monkeypatch.setattr(app_mod.whatsapp, "enviar_texto", lambda *args: avisos.append(args))
    respuesta = TestClient(app_mod.app).get(
        "/oauth/google/callback?state=state&code=code"
    )
    assert respuesta.status_code == 200
    assert "sheila@empresa.com" in respuesta.text
    assert avisos and avisos[0][0] == "5061111"
