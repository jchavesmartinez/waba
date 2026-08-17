"""Pruebas de notas de voz entrantes y transcripcion."""

import asyncio

from fastapi import BackgroundTasks

from bot import audio
from bot import app as app_mod


def _habilitar_audio(monkeypatch):
    monkeypatch.setattr(audio.config, "BOT_AUDIO_ENTRANTE", True)
    monkeypatch.setattr(audio.config, "BOT_AUDIO_MODELO",
                        "gemini-3.6-flash")
    monkeypatch.setattr(audio.config, "BOT_AUDIO_PROMPT", "Consulta en espanol")
    monkeypatch.setattr(audio.config, "BOT_AUDIO_MAX_MB", 5)
    monkeypatch.setattr(audio.config, "BOT_AUDIO_TIMEOUT_SEGUNDOS", 90)


def test_transcribir_wav_manda_el_audio_directo(monkeypatch):
    _habilitar_audio(monkeypatch)
    llamada = {}

    def transcribir(modelo, contenido, mime, prompt, **kwargs):
        llamada.update({
            "modelo": modelo,
            "contenido": contenido,
            "mime": mime,
            "prompt": prompt,
        })
        return "cual es el presupuesto de vivienda"

    monkeypatch.setattr(audio.llm, "transcribir_audio", transcribir)

    texto = audio.transcribir(b"RIFF-audio-de-prueba", "audio/wav; rate=16000")

    assert texto == "cual es el presupuesto de vivienda"
    assert llamada["modelo"] == "gemini-3.6-flash"
    assert llamada["contenido"] == b"RIFF-audio-de-prueba"
    assert llamada["mime"] == "audio/wav"
    assert "Transcribi literalmente" in llamada["prompt"]


def test_transcribir_ogg_lo_convierte_antes_de_enviar(monkeypatch):
    _habilitar_audio(monkeypatch)
    monkeypatch.setattr(audio, "_convertir_a_mp3",
                        lambda contenido, mime: b"mp3-convertido")
    llamada = {}

    def transcribir(modelo, contenido, mime, prompt, **kwargs):
        llamada.update({"contenido": contenido, "mime": mime})
        return "cuanto vendimos ayer"

    monkeypatch.setattr(audio.llm, "transcribir_audio", transcribir)

    audio.transcribir(b"ogg-opus", "audio/ogg; codecs=opus")

    assert llamada == {"contenido": b"mp3-convertido", "mime": "audio/mpeg"}


def test_audio_demasiado_grande_no_llama_a_gemini(monkeypatch):
    _habilitar_audio(monkeypatch)
    monkeypatch.setattr(audio.config, "BOT_AUDIO_MAX_MB", 0.000001)
    llamado = {"gemini": False}

    def transcribir(*args, **kwargs):
        llamado["gemini"] = True
        return "texto"

    monkeypatch.setattr(audio.llm, "transcribir_audio", transcribir)

    try:
        audio.transcribir(b"demasiados-bytes", "audio/wav")
    except audio.AudioDemasiadoGrande:
        pass
    else:
        raise AssertionError("Debio rechazar el audio grande")

    assert llamado["gemini"] is False


def test_numero_no_registrado_no_descarga_ni_transcribe(monkeypatch):
    monkeypatch.setattr(app_mod.registry, "resolver", lambda numero: None)
    llamado = {"descarga": False, "transcripcion": False, "mensajes": []}

    def descargar(media_id):
        llamado["descarga"] = True
        return b"audio", "audio/ogg"

    def transcribir(contenido, mime):
        llamado["transcripcion"] = True
        return "consulta"

    monkeypatch.setattr(app_mod.whatsapp, "descargar_media", descargar)
    monkeypatch.setattr(app_mod.audio, "transcribir", transcribir)
    monkeypatch.setattr(
        app_mod.whatsapp,
        "enviar_texto",
        lambda numero, texto, origen="": llamado["mensajes"].append(texto),
    )

    app_mod._atender_audio("50600000000", "media-1")

    assert llamado["descarga"] is False
    assert llamado["transcripcion"] is False
    assert "no está registrado" in llamado["mensajes"][0]


def test_audio_transcrito_usa_el_mismo_flujo_del_texto(monkeypatch):
    monkeypatch.setattr(app_mod.registry, "resolver",
                        lambda numero: {"cliente_id": "cliente_a"})
    monkeypatch.setattr(app_mod.whatsapp, "descargar_media",
                        lambda media_id: (b"ogg", "audio/ogg"))
    monkeypatch.setattr(app_mod.audio, "transcribir",
                        lambda contenido, mime: "cuanto vendimos ayer")
    atendido = []
    monkeypatch.setattr(
        app_mod,
        "_atender",
        lambda numero, texto, origen="": atendido.append((numero, texto, origen)),
    )

    app_mod._atender_audio("50611112222", "media-2", "audio/ogg", "phone-id")

    assert atendido == [
        ("50611112222", "cuanto vendimos ayer", "phone-id")
    ]


def test_webhook_enruta_audio_a_la_tarea_de_transcripcion(monkeypatch):
    monkeypatch.setattr(app_mod, "_firma_valida", lambda cuerpo, firma: True)
    monkeypatch.setattr(app_mod.config, "BOT_AUDIO_ENTRANTE", True)
    monkeypatch.setattr(app_mod, "_pasa_limite", lambda numero: True)

    class Solicitud:
        headers = {}

        async def body(self):
            return b"{}"

        async def json(self):
            return {
                "entry": [{
                    "changes": [{
                        "value": {
                            "metadata": {"phone_number_id": "phone-id"},
                            "messages": [{
                                "id": "wamid-prueba-audio-unica",
                                "from": "50611112222",
                                "type": "audio",
                                "audio": {
                                    "id": "media-3",
                                    "mime_type": "audio/ogg; codecs=opus",
                                    "voice": True,
                                },
                            }],
                        },
                    }],
                }],
            }

    tareas = BackgroundTasks()
    respuesta = asyncio.run(app_mod.webhook(Solicitud(), tareas))

    assert respuesta.status_code == 200
    assert len(tareas.tasks) == 1
    assert tareas.tasks[0].func is app_mod._atender_audio
    assert tareas.tasks[0].args == (
        "50611112222",
        "media-3",
        "audio/ogg; codecs=opus",
        "phone-id",
    )
