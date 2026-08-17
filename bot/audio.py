"""Transcripcion de notas de voz entrantes para el bot de WhatsApp.

El audio nunca se guarda de forma permanente. WhatsApp entrega OGG/Opus para
las notas de voz; Gemini documenta OGG/Vorbis, no Opus, por lo que esas notas
se convierten temporalmente a MP3 antes de enviarlas. Los formatos oficialmente
admitidos se mandan sin recodificar.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import config
import llm

logger = logging.getLogger("fachavi.bot.audio")

# La documentacion de Gemini reserva las solicitudes inline para cargas de
# menos de 20 MB. Dejamos un MB de margen para el prompt y el envoltorio JSON.
_LIMITE_INLINE_BYTES = 19 * 1024 * 1024

_FORMATOS_DIRECTOS = {
    "audio/mpeg": ("nota.mp3", "audio/mpeg"),
    "audio/mp3": ("nota.mp3", "audio/mpeg"),
    "audio/wav": ("nota.wav", "audio/wav"),
    "audio/x-wav": ("nota.wav", "audio/wav"),
    "audio/aiff": ("nota.aiff", "audio/aiff"),
    "audio/x-aiff": ("nota.aiff", "audio/aiff"),
    "audio/aac": ("nota.aac", "audio/aac"),
    "audio/flac": ("nota.flac", "audio/flac"),
    "audio/x-flac": ("nota.flac", "audio/flac"),
}


class ErrorAudio(RuntimeError):
    """Error controlado que se puede comunicar sin filtrar datos."""


class AudioDemasiadoGrande(ErrorAudio):
    """La nota supera el tope propio o el de la API."""


class AudioNoConfigurado(ErrorAudio):
    """Falta una credencial o dependencia para transcribir."""


class AudioNoEntendido(ErrorAudio):
    """La API no devolvio texto util."""


def _mime_base(mime: str) -> str:
    """Quita parametros como ``; codecs=opus`` y normaliza a minusculas."""
    return str(mime or "").split(";", 1)[0].strip().lower()


def _tope_bytes() -> int:
    configurado = max(float(config.BOT_AUDIO_MAX_MB or 0), 0) * 1024 * 1024
    return int(min(configurado or _LIMITE_INLINE_BYTES, _LIMITE_INLINE_BYTES))


def _validar_tamano(contenido: bytes) -> None:
    if not contenido:
        raise AudioNoEntendido("El audio esta vacio.")
    if len(contenido) > _tope_bytes():
        raise AudioDemasiadoGrande(
            f"El audio pesa {len(contenido)} bytes y supera el tope configurado."
        )


def _convertir_a_mp3(contenido: bytes, mime: str) -> bytes:
    """Convierte un formato no admitido (normalmente OGG/Opus) a MP3 mono."""
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise AudioNoConfigurado(
            "Falta imageio-ffmpeg para convertir las notas de voz."
        ) from e

    extensiones = {
        "audio/ogg": ".ogg",
        "application/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/aac": ".aac",
        "audio/amr": ".amr",
        "audio/3gpp": ".3gp",
    }
    extension = extensiones.get(_mime_base(mime), ".audio")

    with tempfile.TemporaryDirectory(prefix="fachavi-audio-") as carpeta:
        entrada = Path(carpeta) / f"entrada{extension}"
        salida = Path(carpeta) / "nota.mp3"
        entrada.write_bytes(contenido)
        comando = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(entrada),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-codec:a", "libmp3lame",
            "-b:a", "48k",
            "-y", str(salida),
        ]
        try:
            resultado = subprocess.run(
                comando,
                capture_output=True,
                check=False,
                timeout=max(int(config.BOT_AUDIO_TIMEOUT_SEGUNDOS), 10),
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise AudioNoEntendido("No se pudo convertir la nota de voz.") from e

        if resultado.returncode != 0 or not salida.exists():
            # No se registra stderr: algunos codecs incluyen metadatos del
            # archivo. El codigo de salida alcanza para diagnosticar el fallo.
            logger.warning("ffmpeg no pudo convertir audio; codigo=%s",
                           resultado.returncode)
            raise AudioNoEntendido("No se pudo decodificar la nota de voz.")

        convertido = salida.read_bytes()

    _validar_tamano(convertido)
    return convertido


def _preparar(contenido: bytes, mime: str) -> tuple[bytes, str, str]:
    _validar_tamano(contenido)
    mime_limpio = _mime_base(mime)
    directo = _FORMATOS_DIRECTOS.get(mime_limpio)
    if directo:
        nombre, mime_salida = directo
        return contenido, nombre, mime_salida

    if not (mime_limpio.startswith("audio/") or mime_limpio == "application/ogg"):
        raise AudioNoEntendido("El archivo recibido no es audio.")

    convertido = _convertir_a_mp3(contenido, mime_limpio)
    return convertido, "nota.mp3", "audio/mpeg"


def transcribir(contenido: bytes, mime: str) -> str:
    """Transcribe una nota y devuelve solo el texto reconocido."""
    if not config.BOT_AUDIO_ENTRANTE:
        raise AudioNoConfigurado("La transcripcion de voz no esta configurada.")

    datos, _nombre, mime_salida = _preparar(contenido, mime)
    prompt = str(config.BOT_AUDIO_PROMPT or "").strip()
    try:
        texto = llm.transcribir_audio(
            config.BOT_AUDIO_MODELO,
            datos,
            mime_salida,
            prompt=(
                (prompt + "\n\n") if prompt else ""
            ) + "Transcribi literalmente esta nota de voz en su idioma original.",
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Fallo la transcripcion con Gemini: %s", type(e).__name__)
        raise ErrorAudio("No se pudo transcribir la nota de voz.") from e

    texto = str(texto or "").strip()
    if not texto:
        raise AudioNoEntendido("La transcripcion no devolvio texto.")
    return texto
