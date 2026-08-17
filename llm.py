"""Cliente unico de Gemini para el bot y los jobs semanticos.

Todo el repositorio pasa por estas funciones. Asi la autenticacion de Vertex
AI, los limites de salida, el nivel de razonamiento y el parseo de respuestas
no se implementan de una forma distinta en cada modulo.

El backend predeterminado es Vertex AI con el mismo service account JSON que
usa Google Sheets. Tambien se admite ``GEMINI_BACKEND=api_key`` para desarrollo
local, pero produccion no necesita una segunda credencial si el service account
tiene permisos de Vertex AI.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import config
import gclient

logger = logging.getLogger("fachavi.llm")

_cliente = None
_lock = threading.RLock()


@dataclass(frozen=True)
class RespuestaGemini:
    texto: str
    truncada: bool = False
    finish_reason: str = ""


def _sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "Falta google-genai. Reinstala requirements-bot.txt o "
            "requirements-clasificar.txt."
        ) from e
    return genai, types


def _crear_cliente():
    genai, types = _sdk()
    backend = str(config.GEMINI_BACKEND or "vertex").strip().lower()
    opciones_http = types.HttpOptions(
        api_version="v1",
        timeout=max(int(config.GEMINI_TIMEOUT_SEGUNDOS), 10) * 1000,
    )

    if backend == "api_key":
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_BACKEND=api_key pero falta GEMINI_API_KEY."
            )
        return genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=opciones_http,
        )

    if backend != "vertex":
        raise RuntimeError(
            "GEMINI_BACKEND debe ser 'vertex' o 'api_key', no "
            f"'{config.GEMINI_BACKEND}'."
        )

    proyecto = str(config.GEMINI_PROJECT_ID or "").strip()
    if not proyecto:
        proyecto = gclient.project_id_service_account()
    if not proyecto:
        raise RuntimeError(
            "No se pudo resolver el proyecto de Vertex AI. Configura "
            "GEMINI_PROJECT_ID o revisa project_id en GOOGLE_CREDENTIALS_JSON."
        )
    return genai.Client(
        vertexai=True,
        project=proyecto,
        location=config.GEMINI_LOCATION,
        credentials=gclient.credenciales_vertex(),
        http_options=opciones_http,
    )


def cliente():
    """Cliente perezoso y compartido por proceso."""
    global _cliente
    if _cliente is not None:
        return _cliente
    with _lock:
        if _cliente is None:
            _cliente = _crear_cliente()
    return _cliente


def _contenidos(valor: Any, types):
    """Convierte el rol interno ``assistant`` al rol ``model`` de Gemini."""
    if isinstance(valor, str):
        return valor
    if not isinstance(valor, (list, tuple)):
        return str(valor)

    contenidos = []
    for mensaje in valor:
        if not isinstance(mensaje, dict):
            contenidos.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=str(mensaje))],
                )
            )
            continue
        rol = "model" if mensaje.get("role") == "assistant" else "user"
        texto = str(mensaje.get("content", "") or "")
        contenidos.append(
            types.Content(
                role=rol,
                parts=[types.Part.from_text(text=texto)],
            )
        )
    return contenidos


def _razon_fin(respuesta) -> str:
    candidatos = getattr(respuesta, "candidates", None) or []
    if not candidatos:
        return ""
    razon = getattr(candidatos[0], "finish_reason", "") or ""
    if hasattr(razon, "value"):
        razon = razon.value
    elif hasattr(razon, "name"):
        razon = razon.name
    return str(razon).strip()


def _texto(respuesta) -> str:
    try:
        directo = respuesta.text
    except (AttributeError, ValueError):
        directo = ""
    if directo:
        return str(directo).strip()

    piezas = []
    for candidato in getattr(respuesta, "candidates", None) or []:
        contenido = getattr(candidato, "content", None)
        for parte in getattr(contenido, "parts", None) or []:
            if getattr(parte, "thought", False):
                continue
            texto = getattr(parte, "text", "")
            if texto:
                piezas.append(str(texto))
    return "".join(piezas).strip()


def _configuracion(types, *, system: str, max_tokens: int,
                   thinking_level: str, response_schema=None):
    kwargs = {
        "max_output_tokens": max(int(max_tokens), 1),
    }
    if system:
        kwargs["system_instruction"] = system

    # thinking_level pertenece a Gemini 3.x. Si alguien configura temporalmente
    # un modelo 2.x, omitirlo es mas seguro que enviar un parametro incompatible.
    if thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
        )
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_json_schema"] = response_schema
    return types.GenerateContentConfig(**kwargs)


def _validar_modelo(modelo: str) -> str:
    nombre = str(modelo or "").strip()
    if not nombre:
        raise RuntimeError("No se configuro el modelo de Gemini.")
    if not nombre.lower().startswith("gemini-"):
        raise RuntimeError(
            f"El modelo '{nombre}' no es de Gemini. Revisa GEMINI_MODELO y "
            "las variables BOT_MODELO_* heredadas del despliegue anterior."
        )
    return nombre


def generar_texto(modelo: str, contenido, *, system: str = "",
                   max_tokens: int = 1000, thinking_level: str = "low",
                   response_schema=None) -> RespuestaGemini:
    """Genera texto o JSON estructurado y normaliza el estado de terminacion."""
    modelo = _validar_modelo(modelo)
    _, types = _sdk()
    nivel = str(thinking_level or "").strip().lower()
    if not str(modelo).lower().startswith("gemini-3"):
        nivel = ""
    respuesta = cliente().models.generate_content(
        model=modelo,
        contents=_contenidos(contenido, types),
        config=_configuracion(
            types,
            system=system,
            max_tokens=max_tokens,
            thinking_level=nivel,
            response_schema=response_schema,
        ),
    )
    razon = _razon_fin(respuesta)
    texto = _texto(respuesta)
    truncada = "MAX_TOKENS" in razon.upper()
    if not texto and not truncada:
        raise RuntimeError(
            "Gemini no devolvio texto"
            + (f" (finish_reason={razon})" if razon else "")
            + "."
        )
    return RespuestaGemini(texto=texto, truncada=truncada,
                           finish_reason=razon)


def transcribir_audio(modelo: str, contenido: bytes, mime: str,
                      prompt: str, max_tokens: int = 700) -> str:
    """Transcribe audio inline con Gemini y devuelve solo el texto visible."""
    modelo = _validar_modelo(modelo)
    _, types = _sdk()
    parte = types.Part.from_bytes(data=contenido, mime_type=mime)
    respuesta = cliente().models.generate_content(
        model=modelo,
        contents=[prompt, parte],
        config=_configuracion(
            types,
            system=(
                "Transcribis consultas de voz para un bot de datos. Devolve "
                "unicamente la transcripcion literal, sin responder la consulta, "
                "sin explicaciones, sin comillas y sin markdown."
            ),
            max_tokens=max_tokens,
            thinking_level=(
                "minimal" if str(modelo).lower().startswith("gemini-3") else ""
            ),
        ),
    )
    razon = _razon_fin(respuesta)
    if "MAX_TOKENS" in razon.upper():
        raise RuntimeError("La transcripcion se corto por longitud.")
    texto = _texto(respuesta)
    if not texto:
        raise RuntimeError("Gemini no devolvio una transcripcion.")
    return texto
