"""Confirmación asíncrona de entrega para PDF, Excel y gráficos."""

import asyncio

from fastapi import BackgroundTasks

from bot import app as app_mod
from bot import entregas, whatsapp
from bot.salida import Adjunto, Respuesta


CLIENTE = {"cliente_id": "cliente_a"}


class _RespuestaHTTP:
    status_code = 200
    text = '{"messages":[{"id":"wamid.aceptado"}]}'

    @staticmethod
    def json():
        return {"messages": [{"id": "wamid.aceptado"}]}


def test_post_de_meta_devuelve_el_message_id(monkeypatch):
    monkeypatch.setattr(whatsapp.httpx, "post", lambda *a, **k: _RespuestaHTTP())

    message_id = whatsapp._post_mensaje_id(
        {"to": "50611112222", "type": "document"}, "documento",
        "phone-id",
    )

    assert message_id == "wamid.aceptado"


def test_enviar_adjunto_devuelve_message_y_media_id(monkeypatch):
    monkeypatch.setattr(whatsapp, "subir_media",
                        lambda *a, **k: "media-123")
    monkeypatch.setattr(whatsapp, "enviar_documento",
                        lambda *a, **k: "wamid.documento")
    adjunto = Adjunto(
        tipo="document", contenido=b"%PDF-prueba", nombre="reporte.pdf",
        mime="application/pdf",
    )

    envio = whatsapp.enviar_adjunto("50611112222", adjunto, "phone-id")

    assert envio == whatsapp.EnvioAdjunto(
        message_id="wamid.documento", media_id="media-123",
    )


def test_atender_registra_el_adjunto_aceptado(monkeypatch):
    adjunto = Adjunto(
        tipo="document", contenido=b"PK-xlsx", nombre="ventas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    monkeypatch.setattr(app_mod, "responder",
                        lambda numero, texto: Respuesta("Listo", [adjunto]))
    monkeypatch.setattr(app_mod.registry, "resolver", lambda numero: CLIENTE)
    monkeypatch.setattr(app_mod.whatsapp, "enviar_texto", lambda *a, **k: True)
    monkeypatch.setattr(
        app_mod.whatsapp, "enviar_adjunto",
        lambda *a, **k: whatsapp.EnvioAdjunto("wamid.xlsx", "media.xlsx"),
    )
    guardado = {}

    def registrar(*args, **kwargs):
        guardado.update(args=args, kwargs=kwargs)

    monkeypatch.setattr(app_mod.entregas, "registrar", registrar)

    app_mod._atender("50611112222", "dámelo en Excel", "phone-id")

    assert guardado["args"][:5] == (
        CLIENTE, "50611112222", "phone-id", "wamid.xlsx", "media.xlsx",
    )
    assert guardado["args"][5:] == (
        "document", "ventas.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def test_webhook_enruta_status_sin_necesitar_un_mensaje_entrante(monkeypatch):
    monkeypatch.setattr(app_mod, "_firma_valida", lambda cuerpo, firma: True)

    class Solicitud:
        headers = {}

        async def body(self):
            return b"{}"

        async def json(self):
            return {
                "entry": [{"changes": [{"value": {
                    "metadata": {"phone_number_id": "phone-id"},
                    "statuses": [{
                        "id": "wamid.documento",
                        "status": "delivered",
                        "recipient_id": "50611112222",
                    }],
                }}]}],
            }

    tareas = BackgroundTasks()
    respuesta = asyncio.run(app_mod.webhook(Solicitud(), tareas))

    assert respuesta.status_code == 200
    assert len(tareas.tasks) == 1
    assert tareas.tasks[0].func is app_mod._procesar_estado_salida
    assert tareas.tasks[0].args[0]["status"] == "delivered"
    assert tareas.tasks[0].args[1] == "phone-id"


def test_status_failed_reintenta_una_vez_y_registra_el_nuevo_id(monkeypatch):
    pendiente = entregas.Reintento(
        message_id="wamid.original", numero="50611112222",
        phone_number_id="phone-id", media_id="media-1", tipo="document",
        nombre="ventas.xlsx", mime="application/xlsx", intentos=1,
    )
    monkeypatch.setattr(app_mod.registry, "resolver", lambda numero: CLIENTE)
    monkeypatch.setattr(
        app_mod.entregas, "actualizar_estado",
        lambda cliente, estado: entregas.ResultadoEstado(reintento=pendiente),
    )
    monkeypatch.setattr(app_mod.whatsapp, "reintentar_adjunto",
                        lambda *a, **k: "wamid.reintento")
    vinculos = []
    registros = []
    monkeypatch.setattr(app_mod.entregas, "vincular_reintento",
                        lambda *a, **k: vinculos.append((a, k)))
    monkeypatch.setattr(app_mod.entregas, "registrar",
                        lambda *a, **k: registros.append((a, k)))

    app_mod._procesar_estado_salida({
        "id": "wamid.original", "status": "failed",
        "recipient_id": "50611112222",
    }, "phone-id")

    assert vinculos[0][0][2] == "wamid.reintento"
    assert registros[0][0][3] == "wamid.reintento"
    assert registros[0][1]["intentos"] == 2
    assert registros[0][1]["reintento_de"] == "wamid.original"


def test_fallo_del_segundo_intento_avisa_una_vez(monkeypatch):
    monkeypatch.setattr(app_mod.registry, "resolver", lambda numero: CLIENTE)
    monkeypatch.setattr(
        app_mod.entregas, "actualizar_estado",
        lambda cliente, estado: entregas.ResultadoEstado(
            fallo_final=True, numero="50611112222", nombre="reporte.pdf",
        ),
    )
    enviados = []
    monkeypatch.setattr(app_mod.whatsapp, "enviar_texto",
                        lambda *a, **k: enviados.append((a, k)))

    app_mod._procesar_estado_salida({
        "id": "wamid.reintento", "status": "failed",
        "recipient_id": "50611112222",
    }, "phone-id")

    assert len(enviados) == 1
    assert "reporte.pdf" in enviados[0][0][1]
    assert "después de reintentarlo" in enviados[0][0][1]
