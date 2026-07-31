"""
Fuente: Google Calendar (solo lectura).

Trae los EVENTOS de uno o varios calendarios de Google hacia el warehouse como
una tabla de citas. Piensa en esto como el equivalente de google_sheets pero
para agendas: el cliente comparte su calendario con el mismo service account
que ya usa para Sheets (mismo client_email, mismo GOOGLE_CREDENTIALS_JSON), y
esta fuente lee los eventos via la API REST de Calendar.

Uso tipico: una barberia con un calendario por barbero. Cada barbero anota sus
citas en su Google Calendar del celular, como ya hace con cualquier cita
personal — no hay software nuevo que aprender. Esta fuente convierte esos
eventos en una tabla consultable: quien, cuando, con quien, cuanto dura.

Config esperada (columna 'config' del registro, como JSON):

  Un solo calendario:
    {"calendar_id": "barbero1@contoso.com"}

  Varios calendarios (varios barberos), fusionados en UNA tabla con una
  columna extra 'recurso' para distinguir de cual vino cada fila:
    {"calendarios": {
        "Carlos": "carlos.barbero@gmail.com",
        "Andres": "c_abc123...@group.calendar.google.com"
     }}

  El "calendar_id" de un calendario compartido normalmente ES el correo de la
  cuenta (para el calendario principal) o un id largo terminado en
  '@group.calendar.google.com' (para un calendario secundario creado a
  proposito). Se encuentra en Configuracion del calendario > Integrar
  calendario > "ID de calendario", en Google Calendar.

Opcionales:
  "dias_atras": 30      -> cuantos dias hacia atras traer (default 30)
  "dias_adelante": 60   -> cuantos dias hacia adelante traer (default 60)
  "incluir_cancelados": false -> si se incluyen eventos con status=cancelled

CADA FILA de la tabla resultante es UNA cita, con estas columnas:
  recurso        -> nombre del calendario/barbero (de la config; "" si es uno solo)
  evento_id      -> id del evento en Google Calendar (para dedupe/trazabilidad)
  titulo         -> el "summary" del evento (lo que el barbero escribio)
  descripcion    -> el "description" del evento, si tiene
  inicio         -> fecha/hora de inicio
  fin            -> fecha/hora de fin
  duracion_min   -> fin - inicio, en minutos (calculado, no viene de Google)
  todo_el_dia    -> true si es un evento de dia completo (sin hora)
  estado         -> confirmed | tentative | cancelled
  creado_en      -> cuando se creo el evento en Calendar
  actualizado_en -> ultima modificacion del evento en Calendar

GOBERNANZA: catalogo y KPIS ya NO se leen ni se generan aca. Viven en UN Sheet
central por cliente (ver catalogo_cliente.py) que documenta TODAS sus fuentes
juntas, incluida esta. sync.py lo lee una vez por cliente y lo reparte.

Importante para quien de alta esta fuente por primera vez: como el catalogo
central es responsabilidad del cliente y no de este conector, la tabla
'citas' NO queda bloqueada por codigo como antes — queda bloqueada por la
regla de siempre (bot/catalogo.py): una tabla sin fila en el catalogo del
cliente es fail-closed, invisible para el bot. Hay que agregar
explicitamente una fila para 'citas' en la pestania '_catalogo' del Sheet
central de ese cliente para habilitarla. Dado que los eventos suelen traer
nombres reales de clientes en el titulo, conviene revisar esa fila con
cuidado antes de habilitarla.

Requiere: httpx (ya es dependencia) + las credenciales de gclient.py, que ya
declaran el scope calendar.readonly.
"""

import logging
from datetime import datetime, timedelta, timezone

import httpx

from gclient import credenciales_calendar
from .base import (
    Source,
    Fragmento,
    registrar_df,
    describir_tabla,
    limpiar_nombre,
    normalizar_columnas,
)

import pandas as pd

logger = logging.getLogger("fachavi.sources.google_calendar")

CALENDAR_API = "https://www.googleapis.com/calendar/v3"
TIMEOUT_SEG = 30.0

DIAS_ATRAS_DEFECTO = 30
DIAS_ADELANTE_DEFECTO = 60


class GoogleCalendarSource(Source):
    tipo = "google_calendar"

    def cargar(self, con) -> Fragmento:
        calendarios = self._calendarios()
        if not calendarios:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (google_calendar) sin 'calendar_id' "
                "ni 'calendarios' en config."
            )

        dias_atras = _entero_no_negativo(
            self.config.get("dias_atras", DIAS_ATRAS_DEFECTO), "dias_atras"
        )
        dias_adelante = _entero_no_negativo(
            self.config.get("dias_adelante", DIAS_ADELANTE_DEFECTO),
            "dias_adelante",
        )
        incluir_cancelados = _es_verdadero(
            self.config.get("incluir_cancelados", False)
        )

        ahora = datetime.now(timezone.utc)
        desde = ahora - timedelta(days=dias_atras)
        hasta = ahora + timedelta(days=dias_adelante)

        alertas = []
        filas = []
        for nombre, calendar_id in calendarios.items():
            try:
                eventos = self._listar_eventos(calendar_id, desde, hasta, incluir_cancelados)
            except RuntimeError as e:
                # Un calendario que falla (no compartido, id mal escrito) no
                # tiene por que tumbar a los demas del mismo cliente. B-12: se
                # avisa fuerte, no se traga en silencio.
                msg = f"[{self.fuente_id}] calendario '{nombre}' ({calendar_id}): {e}"
                logger.warning(msg)
                alertas.append(msg)
                continue

            for ev in eventos:
                filas.append(_evento_a_fila(nombre, ev))

        if not filas:
            msg = (f"[{self.fuente_id}] ningun calendario devolvio eventos en el "
                   f"rango [-{dias_atras}, +{dias_adelante}] dias. Si ya hubo "
                   "citas cargadas antes, la guarda de vaciado va a proteger esos "
                   "datos; revisa que el calendario siga compartido con el service "
                   "account y que el rango de dias sea el correcto.")
            logger.warning(msg)
            alertas.append(msg)

        df = pd.DataFrame(filas, columns=[
            "recurso", "evento_id", "titulo", "descripcion", "inicio", "fin",
            "duracion_min", "todo_el_dia", "estado", "creado_en", "actualizado_en",
        ])
        df.columns = normalizar_columnas(df.columns)
        # Las columnas ya vienen tipadas desde _evento_a_fila (datetime, int,
        # bool); a diferencia de google_sheets, aca NO hace falta inferir_tipos
        # porque el dato no pasa por texto en ningun momento.

        tabla = registrar_df(con, df, "citas", self.fuente_id)
        logger.info("[%s] tabla %s (%d filas, %d calendario(s))",
                    self.fuente_id, tabla, len(df), len(calendarios))

        # Catalogo y KPIs ya NO se generan aca: vienen del Sheet central del
        # cliente (catalogo_cliente.py). La tabla 'citas' queda BLOQUEADA para
        # el bot hasta que alguien agregue su fila en ese Sheet — es la regla
        # normal de fail-closed (bot/catalogo.py), no algo especial de esta
        # fuente.
        return Fragmento(
            schema=describir_tabla(con, tabla),
            tablas=[tabla],
            alertas=alertas,
        )

    # -------- config --------

    def _calendarios(self) -> dict:
        """
        Normaliza la config a {nombre: calendar_id}. Con un solo calendar_id
        (sin 'calendarios'), el nombre queda vacio y la columna 'recurso' sale
        vacia en toda la tabla — no hace falta distinguir si solo hay uno.
        """
        if self.config.get("calendarios"):
            crudo = self.config["calendarios"]
            if not isinstance(crudo, dict):
                raise RuntimeError(
                    f"Fuente '{self.fuente_id}': 'calendarios' debe ser un "
                    "objeto JSON de nombre a calendar_id."
                )
            resultado = {
                str(nombre).strip(): str(calendar_id).strip()
                for nombre, calendar_id in crudo.items()
                if str(calendar_id).strip()
            }
            if len(resultado) != len(crudo):
                logger.warning(
                    "[%s] se ignoraron calendarios con calendar_id vacio.",
                    self.fuente_id,
                )
            return resultado
        cal_id = str(self.config.get("calendar_id", "")).strip()
        return {"": cal_id} if cal_id else {}

    # -------- Google Calendar API --------

    def _token(self) -> str:
        """
        Token de acceso a partir de las credenciales del service account
        compartido. gclient.py crea y refresca un token exclusivo con scope
        calendar.readonly.
        """
        creds = credenciales_calendar()
        return creds.token

    def _listar_eventos(self, calendar_id: str, desde: datetime, hasta: datetime,
                        incluir_cancelados: bool) -> list:
        """
        Trae todos los eventos de un calendario en el rango [desde, hasta],
        paginando con pageToken hasta agotar los resultados.
        """
        headers = {"Authorization": f"Bearer {self._token()}"}
        params = {
            "timeMin": desde.isoformat(),
            "timeMax": hasta.isoformat(),
            "singleEvents": "true",      # expande eventos recurrentes a instancias
            "orderBy": "startTime",
            "maxResults": 2500,           # tope de Google por pagina
            "showDeleted": "true" if incluir_cancelados else "false",
        }
        url = f"{CALENDAR_API}/calendars/{_url_quote(calendar_id)}/events"

        eventos = []
        with httpx.Client(timeout=TIMEOUT_SEG) as cli:
            while True:
                r = cli.get(url, headers=headers, params=params)
                if r.status_code == 404:
                    raise RuntimeError(
                        "no existe o el service account no tiene acceso. "
                        "Comparte el calendario con el client_email del service "
                        "account (Configuracion del calendario > Compartir con "
                        "determinadas personas)."
                    )
                if r.status_code == 403:
                    raise RuntimeError(
                        "el service account no tiene permiso de lectura sobre "
                        "este calendario. Revisa que este compartido con al "
                        "menos permiso 'Ver todos los detalles del evento'."
                    )
                r.raise_for_status()
                datos = r.json()
                eventos.extend(datos.get("items", []))

                token = datos.get("nextPageToken")
                if not token:
                    break
                params["pageToken"] = token

        if not incluir_cancelados:
            eventos = [e for e in eventos if e.get("status") != "cancelled"]
        return eventos


def _url_quote(texto: str) -> str:
    import urllib.parse
    return urllib.parse.quote(texto, safe="")


def _evento_a_fila(recurso: str, ev: dict) -> dict:
    """
    Convierte un evento crudo de la API de Calendar en una fila de la tabla.

    Un evento de Calendar tiene dos formas posibles de fecha: 'dateTime' (hora
    exacta) o 'date' (solo fecha, para eventos de dia completo tipo "Carlos de
    vacaciones"). Se detectan y normalizan las dos al mismo esquema de columnas
    para que la tabla resultante sea consistente.
    """
    inicio_raw = ev.get("start", {})
    fin_raw = ev.get("end", {})
    todo_el_dia = "date" in inicio_raw and "dateTime" not in inicio_raw

    inicio = _parsear_fecha(inicio_raw.get("dateTime") or inicio_raw.get("date"))
    fin = _parsear_fecha(fin_raw.get("dateTime") or fin_raw.get("date"))

    duracion_min = None
    if inicio is not None and fin is not None and not todo_el_dia:
        duracion_min = round((fin - inicio).total_seconds() / 60)

    return {
        "recurso": recurso,
        "evento_id": ev.get("id", ""),
        "titulo": ev.get("summary", ""),
        "descripcion": ev.get("description", ""),
        "inicio": inicio,
        "fin": fin,
        "duracion_min": duracion_min,
        "todo_el_dia": todo_el_dia,
        "estado": ev.get("status", ""),
        "creado_en": _parsear_fecha(ev.get("created")),
        "actualizado_en": _parsear_fecha(ev.get("updated")),
    }


def _parsear_fecha(valor):
    if not valor:
        return None
    try:
        # 'date' llega como "2026-08-03"; 'dateTime' como
        # "2026-08-03T15:00:00-06:00". Se normaliza todo a UTC para evitar una
        # columna object cuando un calendario mezcla eventos con offsets
        # distintos o eventos de dia completo (sin zona).
        ts = pd.Timestamp(valor)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _entero_no_negativo(valor, campo: str) -> int:
    """Convierte opciones de rango y rechaza valores negativos o invalidos."""
    try:
        numero = int(valor)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"Fuente google_calendar: '{campo}' debe ser un entero."
        ) from e
    if numero < 0:
        raise RuntimeError(
            f"Fuente google_calendar: '{campo}' no puede ser negativo."
        )
    return numero


def _es_verdadero(valor) -> bool:
    """
    Interpreta booleanos del JSON sin caer en bool("false") == True.

    El Sheet normalmente guarda false como booleano JSON real, pero aceptar
    tambien strings evita incluir cancelados por un error de tipeo/formato.
    """
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in {"1", "si", "sí", "true", "yes", "on"}
