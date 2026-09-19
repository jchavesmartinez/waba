"""Formato interno para reglas de reclasificación persistentes.

Las reglas se guardan en la misma pestaña ``_overrides`` que las correcciones
puntuales. No dependen de nombres de clientes ni de comercios concretos: la
clave codifica la dimensión canónica y el valor normalizado que debe coincidir.
"""

from __future__ import annotations

import base64
import unicodedata


_PREFIJO = "__grupo__"


def normalizar(valor: object) -> str:
    """Normaliza texto para comparar valores de una regla sin sensibilidad a acentos."""
    texto = str(valor or "").strip().casefold()
    return "".join(
        caracter for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )


def clave_grupo(campo: str, valor: object) -> str:
    """Construye una clave compacta y segura para una regla de grupo."""
    dimension = str(campo or "").strip().lower()
    normalizado = normalizar(valor)
    if not dimension or not normalizado:
        raise ValueError("una regla de grupo requiere campo y valor")
    token = base64.urlsafe_b64encode(normalizado.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{_PREFIJO}:{dimension}:{token}"


def grupo_desde_clave(clave: object) -> tuple[str, str] | None:
    """Devuelve ``(campo, valor_normalizado)`` o ``None`` para claves puntuales."""
    partes = str(clave or "").split(":", 2)
    if len(partes) != 3 or partes[0] != _PREFIJO or not partes[1] or not partes[2]:
        return None
    try:
        padding = "=" * (-len(partes[2]) % 4)
        valor = base64.urlsafe_b64decode((partes[2] + padding).encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeError):
        return None
    valor = normalizar(valor)
    return (partes[1].strip().lower(), valor) if valor else None

