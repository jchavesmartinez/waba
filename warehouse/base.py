"""
Contrato base de un DESTINO (warehouse).

Un destino es cualquier almacenamiento persistente donde aterriza la data cruda
que la capa de ingesta extrae de las fuentes. El job de sync no sabe si detras
hay DuckDB, MotherDuck o Postgres: solo llama estos metodos.

Convencion de nombres en el warehouse:
    esquema:  raw_<cliente_id>          -> aislamiento por cliente
    tabla:    <fuente_id>__<tabla>      -> de que fuente vino cada tabla

Toda tabla escrita lleva columnas de trazabilidad (Fase 0 - linaje):
    _corrida_id     id de la corrida de sync que la escribio
    _fuente_id      fuente logica de origen
    _ingestado_en   timestamp UTC de la ingesta

Ademas cada corrida se registra en la tabla de control _meta.sync_corridas,
que es lo que despues permite: medir frescura, detectar syncs rotos y
responder "¿de cuando es este dato?".

Nota de dependencias (B-05): este modulo NO importa pandas en el top. El bot lo
importa solo por nombre_esquema()/nombre_tabla(), que son manipulacion de
strings; cargar pandas ahi eran ~40 MB de RAM y varios segundos de arranque
para nada. pandas se importa adentro de agregar_trazabilidad(), que es lo unico
que de verdad lo necesita (y eso solo corre en la ingesta).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # solo para type checkers; no se ejecuta en runtime
    import pandas as pd

logger = logging.getLogger("fachavi.warehouse")

ESQUEMA_META = "_meta"
TABLA_CORRIDAS = "sync_corridas"


@dataclass
class Corrida:
    """Registro de una corrida de sync de UNA fuente de UN cliente."""
    corrida_id: str
    cliente_id: str
    fuente_id: str
    tipo: str
    inicio: datetime
    fin: Optional[datetime] = None
    # corriendo | ok | ok_con_alertas | bloqueado | error | omitido
    estado: str = "corriendo"
    filas: int = 0
    tablas: list = field(default_factory=list)
    # Nombres LOGICOS de las tablas cargadas (sin el prefijo fuente_id__), tal
    # como los declara el catalogo del cliente. 'tablas' arriba tiene el
    # nombre FISICO completo (esquema.fuente__tabla); esto es lo que sync.py
    # necesita para saber si el catalogo/KPIs de un cliente ya pueden
    # escribirse (B-40).
    tablas_logicas: set = field(default_factory=set)
    error: str = ""
    alertas: list = field(default_factory=list)
    # detalle: {tabla: {"columnas": [...], "filas": N}} -> base del drift check
    detalle: dict = field(default_factory=dict)

    @property
    def duracion_seg(self) -> float:
        if not self.fin:
            return 0.0
        return round((self.fin - self.inicio).total_seconds(), 3)


def ahora_utc() -> datetime:
    return datetime.now(timezone.utc)


def _sanear(nombre: str) -> str:
    """
    Normaliza un identificador para el warehouse: minusculas y solo
    [a-z0-9_]. Evita el clasico dolor de Postgres, donde un esquema
    "raw_Demo" queda sensible a mayusculas y obliga a poner comillas en
    TODA consulta para siempre.
    """
    limpio = re.sub(r"[^0-9a-zA-Z_]", "_", str(nombre).strip().lower())
    if not limpio or limpio[0].isdigit():
        limpio = "t_" + limpio
    return limpio


def nombre_esquema(cliente_id: str) -> str:
    return f"raw_{_sanear(cliente_id)}"


def nombre_tabla(fuente_id: str, tabla: str) -> str:
    return f"{_sanear(fuente_id)}__{_sanear(tabla)}"


def agregar_trazabilidad(df: "pd.DataFrame", corrida: Corrida) -> "pd.DataFrame":
    """Agrega las columnas de linaje a cada fila antes de escribirla."""
    df = df.copy()
    df["_corrida_id"] = corrida.corrida_id
    df["_fuente_id"] = corrida.fuente_id
    df["_ingestado_en"] = ahora_utc()
    return df


class Destino(ABC):
    """Contrato de un warehouse de aterrizaje."""

    tipo: str = "base"

    def __init__(self, dsn: str = ""):
        self.dsn = dsn

    # --- ciclo de vida ---

    @abstractmethod
    def conectar(self):
        """Abre la conexion al warehouse."""

    @abstractmethod
    def cerrar(self):
        """Cierra la conexion."""

    # --- escritura de datos ---

    @abstractmethod
    def asegurar_esquema(self, esquema: str):
        """Crea el esquema si no existe (aislamiento por cliente)."""

    @abstractmethod
    def escribir_tabla(self, esquema: str, tabla: str, df: "pd.DataFrame"):
        """
        Escribe el DataFrame reemplazando la tabla completa (full refresh).
        Para el volumen de una PYME esto es lo correcto y lo mas simple:
        sin estado incremental que se pueda desincronizar.

        El reemplazo debe ser ATOMICO: en ningun momento puede existir una
        ventana en la que la tabla no exista, porque el bot consulta en vivo.
        """

    # --- control / metadata ---

    @abstractmethod
    def escribir_catalogo(self, esquema: str, fuente_id: str, filas: list):
        """
        Persiste el catalogo (governance) de UNA fuente en <esquema>._catalogo.
        Semantica: borra solo las filas de esa fuente e inserta las nuevas, para
        no pisar el catalogo de las otras fuentes del mismo cliente (que pueden
        haberse omitido por frescura en esta corrida).
        """

    @abstractmethod
    def registrar_corrida(self, corrida: Corrida):
        """Persiste el resultado de una corrida en _meta.sync_corridas."""

    # OJO (C-07): este decorador estaba pegado al final del docstring de
    # registrar_corrida. Python NO lo tomaba como error de sintaxis (desde 3.5
    # '@' es tambien el operador de multiplicacion de matrices), asi que
    # ultimo_detalle habia dejado de ser abstracto en silencio y quedaba una
    # expresion 'str @ function' latente. El salto de linea es el arreglo.
    @abstractmethod
    def ultimo_detalle(self, cliente_id: str, fuente_id: str) -> dict:
        """
        Devuelve el 'detalle' de la ultima corrida exitosa (forma previa de las
        tablas). Es contra esto que se detecta schema drift y colapso de filas.

        Contrato reforzado (C-01): la implementacion NO debe devolver
        simplemente el detalle de la ultima corrida, sino la FUSION de las
        ultimas corridas OK (la mas reciente gana por tabla). Si no, un hueco
        en el historial —por ejemplo una corrida que bloqueo la escritura y no
        registro esa tabla— desarma la guarda de vaciado.
        """

    @abstractmethod
    def ultima_corrida_ok(self, cliente_id: str, fuente_id: str) -> Optional[datetime]:
        """
        Devuelve el 'fin' de la ultima corrida exitosa de esa fuente,
        o None si nunca corrio bien. Es la base del control de frescura.
        """

    # Concreto (no abstracto) para no romper destinos que no soporten KPIs: por
    # defecto no hace nada. El destino Postgres lo sobreescribe para persistir
    # <esquema>._kpis. La capa semantica es opcional.
    def escribir_kpis(self, esquema: str, fuente_id: str, filas: list):
        """Persiste los KPIs de una fuente en <esquema>._kpis (default: no-op)."""
        logger.info("escribir_kpis no soportado en destino '%s'; se omite.", self.tipo)

    def actualizar_eventos_calendar(
        self,
        esquema: str,
        tabla_actual: str,
        df: "pd.DataFrame",
        corrida: Corrida,
    ) -> dict:
        """
        Inserta citas nuevas y actualiza las existentes sin borrar las antiguas.

        Es concreto para no romper destinos externos que no implementen
        Google Calendar; el job da un error explicito si intenta usarlo.
        """
        raise NotImplementedError(
            f"El destino '{self.tipo}' no soporta UPSERT de Google Calendar."
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} tipo={self.tipo}>"


# Los OCHO metodos que todo destino concreto DEBE implementar (eran siete
# mientras C-07 tenia a ultimo_detalle fuera del contrato). Se usa en la prueba
# de conformidad (tests/test_contrato_destinos.py) para que un error como ese no
# vuelva a pasar desapercibido.
METODOS_CONTRATO = (
    "conectar", "cerrar", "asegurar_esquema", "escribir_tabla",
    "escribir_catalogo", "registrar_corrida", "ultimo_detalle",
    "ultima_corrida_ok",
)
