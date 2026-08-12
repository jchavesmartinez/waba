"""
Tipos de SALIDA del bot.

Hasta ahora responder() devolvia un str y app.py lo mandaba con
whatsapp.enviar_texto(). Con adjuntos (graficos, Excel, PDF) una respuesta ya no
es solo texto: es un texto MAS cero o mas archivos.

Este modulo define esos dos tipos y nada mas. Vive aparte para que
bot/whatsapp.py (transporte) y bot/responder.py (orquestacion) puedan hablar el
mismo idioma sin importarse entre si.

Nota de diseño: el archivo viaja en MEMORIA (bytes), no en disco. Render tiene
filesystem efimero y el proceso puede atender varios mensajes a la vez; escribir
temporales seria pedir una colision de nombres o un archivo huerfano. Los
adjuntos son chicos (un grafico ronda 60 KB, un Excel de 5.000 filas ~200 KB).
"""

from dataclasses import dataclass, field


@dataclass
class Adjunto:
    """Un archivo listo para subir a Meta y mandar por WhatsApp."""

    tipo: str          # "image" | "document"  (el 'type' del mensaje de Meta)
    contenido: bytes
    nombre: str        # nombre de archivo que ve el usuario (ventas_2026-08-03.xlsx)
    mime: str          # image/png, application/pdf, ...
    caption: str = ""  # pie de foto / de documento (tope 1024 chars en Meta)

    @property
    def tamano_mb(self) -> float:
        return len(self.contenido) / (1024 * 1024)


@dataclass
class Respuesta:
    """Lo que responder() devuelve: el texto y, si aplica, los adjuntos."""

    texto: str
    adjuntos: list = field(default_factory=list)
    # Consulta que produjo la respuesta. No se muestra al usuario; se guarda en
    # memoria para que "cuales son esas" conserve exactamente los filtros.
    sql: str = ""

    def __str__(self) -> str:      # compatibilidad con codigo viejo/tests
        return self.texto
