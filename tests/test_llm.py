"""Contrato del cliente compartido de Gemini/Vertex AI."""

from types import SimpleNamespace

import llm


class _Modelos:
    def __init__(self, respuesta):
        self.respuesta = respuesta
        self.llamada = None

    def generate_content(self, **kwargs):
        self.llamada = kwargs
        return self.respuesta


def _respuesta(texto="ok", razon="STOP"):
    candidato = SimpleNamespace(finish_reason=razon)
    return SimpleNamespace(text=texto, candidates=[candidato])


def test_generar_texto_traduce_roles_y_configura_json(monkeypatch):
    modelos = _Modelos(_respuesta('{"accion":"sql_libre"}'))
    monkeypatch.setattr(
        llm, "cliente", lambda: SimpleNamespace(models=modelos),
    )
    esquema = {
        "type": "object",
        "properties": {"accion": {"type": "string"}},
        "required": ["accion"],
    }

    respuesta = llm.generar_texto(
        "gemini-3.6-flash",
        [
            {"role": "user", "content": "pregunta"},
            {"role": "assistant", "content": "respuesta anterior"},
            {"role": "user", "content": "seguimiento"},
        ],
        system="reglas",
        max_tokens=321,
        thinking_level="low",
        response_schema=esquema,
    )

    llamada = modelos.llamada
    assert respuesta.texto == '{"accion":"sql_libre"}'
    assert [c.role for c in llamada["contents"]] == ["user", "model", "user"]
    assert llamada["config"].system_instruction == "reglas"
    assert llamada["config"].max_output_tokens == 321
    assert llamada["config"].response_mime_type == "application/json"
    assert llamada["config"].response_json_schema == esquema
    assert "LOW" in str(
        llamada["config"].thinking_config.thinking_level
    ).upper()


def test_generar_texto_detecta_respuesta_cortada(monkeypatch):
    modelos = _Modelos(_respuesta("json incompleto", "MAX_TOKENS"))
    monkeypatch.setattr(
        llm, "cliente", lambda: SimpleNamespace(models=modelos),
    )

    respuesta = llm.generar_texto("gemini-3.6-flash", "consulta")

    assert respuesta.truncada is True
    assert respuesta.finish_reason == "MAX_TOKENS"


def test_rechaza_un_nombre_de_modelo_heredado_de_claude():
    try:
        llm.generar_texto("claude-haiku-anterior", "consulta")
    except RuntimeError as exc:
        assert "no es de Gemini" in str(exc)
        assert "BOT_MODELO_*" in str(exc)
    else:
        raise AssertionError("Debio rechazar el modelo del proveedor anterior")


def test_transcribir_audio_envia_bytes_y_mime_sin_responder(monkeypatch):
    modelos = _Modelos(_respuesta("cuanto vendimos ayer"))
    monkeypatch.setattr(
        llm, "cliente", lambda: SimpleNamespace(models=modelos),
    )

    texto = llm.transcribir_audio(
        "gemini-3.6-flash",
        b"RIFF-audio",
        "audio/wav",
        "Transcribi literalmente.",
    )

    llamada = modelos.llamada
    parte = llamada["contents"][1]
    assert texto == "cuanto vendimos ayer"
    assert llamada["contents"][0] == "Transcribi literalmente."
    assert parte.inline_data.data == b"RIFF-audio"
    assert parte.inline_data.mime_type == "audio/wav"
    assert "sin responder la consulta" in llamada["config"].system_instruction


def test_cliente_vertex_usa_service_account_y_proyecto_del_json(monkeypatch):
    capturado = {}
    _, types = llm._sdk()

    class _GenAI:
        @staticmethod
        def Client(**kwargs):
            capturado.update(kwargs)
            return "cliente-vertex"

    credenciales = object()
    monkeypatch.setattr(llm, "_sdk", lambda: (_GenAI, types))
    monkeypatch.setattr(llm.config, "GEMINI_BACKEND", "vertex")
    monkeypatch.setattr(llm.config, "GEMINI_PROJECT_ID", "")
    monkeypatch.setattr(llm.config, "GEMINI_LOCATION", "global")
    monkeypatch.setattr(llm.config, "GEMINI_TIMEOUT_SEGUNDOS", 90)
    monkeypatch.setattr(
        llm.gclient, "project_id_service_account", lambda: "proyecto-prueba",
    )
    monkeypatch.setattr(
        llm.gclient, "credenciales_vertex", lambda: credenciales,
    )

    cliente = llm._crear_cliente()

    assert cliente == "cliente-vertex"
    assert capturado["vertexai"] is True
    assert capturado["project"] == "proyecto-prueba"
    assert capturado["location"] == "global"
    assert capturado["credentials"] is credenciales
