"""
Registro de TIPOS de fuente + factory.

Para agregar una fuente nueva:
  1. Crear sources/mi_fuente.py con una subclase de Source (tipo = "mi_fuente").
  2. Registrarla abajo con `registrar(MiFuenteSource)`.
  3. En el registro (Sheet maestro), las filas con tipo="mi_fuente" ya la usan.

No se toca nl2sql ni main: el pipeline solo ve (con, schema, catalogo).
"""

import logging

from .base import Source, Fragmento

logger = logging.getLogger("fachavi.sources")

_REGISTRO = {}


def registrar(cls):
    """Registra una clase de fuente por su atributo `tipo`."""
    if not getattr(cls, "tipo", None) or cls.tipo == "base":
        raise ValueError(f"La fuente {cls} debe definir un 'tipo' propio.")
    _REGISTRO[cls.tipo] = cls
    return cls


def crear_fuente(tipo: str, fuente_id: str, config: dict) -> Source:
    """Instancia la fuente del tipo pedido. Lanza si el tipo no esta registrado."""
    tipo = (tipo or "").strip().lower()
    if tipo not in _REGISTRO:
        raise RuntimeError(
            f"Tipo de fuente desconocido: '{tipo}'. Disponibles: {tipos_disponibles()}"
        )
    return _REGISTRO[tipo](fuente_id, config)


def tipos_disponibles() -> list:
    return sorted(_REGISTRO)


# --- Registro de fuentes incluidas ---------------------------------------
from .google_sheets import GoogleSheetsSource  # noqa: E402
from .google_calendar import GoogleCalendarSource  # noqa: E402
from .csv_url import CSVURLSource               # noqa: E402
from .api_rest import ApiRestSource             # noqa: E402

registrar(GoogleSheetsSource)
registrar(GoogleCalendarSource)
registrar(CSVURLSource)
registrar(ApiRestSource)

# SharePoint/Excel es opcional: necesita openpyxl para leer .xlsx. Si no esta
# instalado, los tipos simplemente no se registran y las demas fuentes siguen
# funcionando (mismo criterio que postgres).
#
# Hay dos variantes, y cual usar depende del cliente:
#   - sharepoint_link  : para PYMEs. El cliente pega un «link de compartir»
#                        publico y listo. Sin Azure, sin admin. Los datos
#                        quedan accesibles a quien tenga el link.
#   - sharepoint_excel : para clientes con IT. Registro de aplicacion en Azure
#                        AD del cliente, con permiso Sites.Read.All y
#                        consentimiento de admin. Acceso privado y auditable,
#                        pero implica configuracion tecnica del lado del cliente.
try:  # noqa: SIM105
    from .sharepoint_link import SharePointLinkSource  # noqa: E402
    registrar(SharePointLinkSource)
except Exception as e:  # noqa: BLE001
    logger.info("Fuente 'sharepoint_link' no disponible en este entorno: %s", e)

try:  # noqa: SIM105
    from .sharepoint_excel import SharePointExcelSource  # noqa: E402
    registrar(SharePointExcelSource)
except Exception as e:  # noqa: BLE001
    logger.info("Fuente 'sharepoint_excel' no disponible en este entorno: %s", e)

# Postgres es opcional: se registra solo si la extension/entorno lo permite.
try:  # noqa: SIM105
    from .postgres import PostgresSource  # noqa: E402
    registrar(PostgresSource)
except Exception as e:  # noqa: BLE001
    logger.info("Fuente 'postgres' no disponible en este entorno: %s", e)

logger.info("Tipos de fuente registrados: %s", tipos_disponibles())
