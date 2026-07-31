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
  "dias_atras": 30 -> limite inferior de la primera carga (default 30)

Despues de la primera carga se usa nextSyncToken: Google entrega solamente
eventos creados, modificados o cancelados desde la corrida anterior. Los
tokens se guardan en el warehouse, nunca en la Hoja Maestra.

CADA FILA de la tabla resultante es UNA cita, con estas columnas:
  calendar_id     -> calendario origen; junto con evento_id forma la clave
  recurso        -> nombre del calendario/barbero (de la config; "" si es uno solo)
  evento_id      -> id del evento en Google Calendar (para dedupe/trazabilidad)
  ical_uid        -> identificador interoperable iCalendar
  titulo         -> el "summary" del evento (lo que el barbero escribio)
  descripcion    -> el "description" del evento, si tiene
  ubicacion       -> lugar del evento
  inicio         -> fecha/hora de inicio
  fin            -> fecha/hora de fin
  zona_horaria   -> zona declarada por Calendar
  duracion_min   -> fin - inicio, en minutos (calculado, no viene de Google)
  todo_el_dia    -> true si es un evento de dia completo (sin hora)
  estado         -> confirmed | tentative | cancelled
  tipo_evento    -> default | outOfOffice | focusTime | etc.
  evento_recurrente_id / inicio_original -> identidad de recurrencias
  creado_en      -> cuando se creo el evento en Calendar
  actualizado_en -> ultima modificacion del evento en Calendar
  organizador / invitados / propiedades -> JSON de campos comunes
  raw_evento     -> JSON completo de la respuesta para no perder campos

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

import hashlib
import json
import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

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


# Modelo del evento compartido con los destinos. Vive junto al conector porque
# toda esta semantica es exclusiva de Google Calendar: columnas, deduplicacion
# y conservacion de detalles cuando Google devuelve un tombstone cancelado.
CAMPOS_EVENTO = (
    "calendar_id",
    "recurso",
    "evento_id",
    "ical_uid",
    "titulo",
    "descripcion",
    "ubicacion",
    "inicio",
    "fin",
    "zona_horaria",
    "duracion_min",
    "todo_el_dia",
    "estado",
    "transparencia",
    "visibilidad",
    "tipo_evento",
    "evento_recurrente_id",
    "inicio_original",
    "creado_en",
    "actualizado_en",
    "enlace_evento",
    "organizador",
    "invitados",
    "propiedades",
    "raw_evento",
)

_CONSERVAR_EN_CANCELACION = (
    "recurso",
    "ical_uid",
    "titulo",
    "descripcion",
    "ubicacion",
    "inicio",
    "fin",
    "zona_horaria",
    "duracion_min",
    "todo_el_dia",
    "transparencia",
    "visibilidad",
    "tipo_evento",
    "evento_recurrente_id",
    "inicio_original",
    "creado_en",
    "enlace_evento",
    "organizador",
    "invitados",
    "propiedades",
)


def limpiar_valor(valor):
    """Convierte valores pandas/numpy a tipos simples aceptados por SQL/JSON."""
    if valor is None:
        return None
    try:
        if valor != valor:  # NaN / NaT
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(valor, "to_pydatetime"):
        try:
            return valor.to_pydatetime()
        except (TypeError, ValueError):
            pass
    if hasattr(valor, "item"):
        try:
            return valor.item()
        except (TypeError, ValueError):
            pass
    return valor


def fila_evento(fila) -> dict:
    """Extrae solamente las columnas estables del evento."""
    getter = fila.get
    return {campo: limpiar_valor(getter(campo)) for campo in CAMPOS_EVENTO}


def fusionar_evento(anterior: dict | None, nuevo: dict) -> dict:
    """Preserva detalles anteriores si una cancelacion llega incompleta."""
    actual = fila_evento(nuevo)
    if not anterior or actual.get("estado") != "cancelled":
        return actual

    previo = fila_evento(anterior)
    for campo in _CONSERVAR_EN_CANCELACION:
        valor = actual.get(campo)
        if valor is None or valor == "" or valor == "[]" or valor == "{}":
            actual[campo] = previo.get(campo)
    return actual


def _json_default(valor):
    valor = limpiar_valor(valor)
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    if isinstance(valor, Decimal):
        return str(valor)
    return str(valor)


def hash_evento(evento: dict) -> str:
    """Hash estable: el mismo evento leido otra vez no crea otra version."""
    texto = json.dumps(
        fila_evento(evento),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

logger = logging.getLogger("fachavi.sources.google_calendar")

CALENDAR_API = "https://www.googleapis.com/calendar/v3"
TIMEOUT_SEG = 30.0

DIAS_ATRAS_DEFECTO = 30


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
        ahora = datetime.now(timezone.utc)
        desde = ahora - timedelta(days=dias_atras)

        alertas = []
        filas = []
        exitosos = 0
        self.sync_tokens_siguientes = {}
        self.calendarios_reiniciados = []
        tokens_previos = self.config.get("_sync_tokens", {})
        if not isinstance(tokens_previos, dict):
            tokens_previos = {}

        if "dias_adelante" in self.config:
            logger.info(
                "[%s] 'dias_adelante' ya no limita Calendar: con syncToken la "
                "primera carga trae todos los eventos futuros para no perder "
                "citas cuando el horizonte avanza.",
                self.fuente_id,
            )
        if not _es_verdadero(self.config.get("incluir_cancelados", True)):
            logger.warning(
                "[%s] se ignora incluir_cancelados=false: los tombstones son "
                "necesarios para conservar un historico correcto.",
                self.fuente_id,
            )

        for nombre, calendar_id in calendarios.items():
            try:
                eventos, token_nuevo, reiniciado = self._listar_eventos(
                    calendar_id,
                    desde,
                    str(tokens_previos.get(calendar_id, "") or ""),
                )
            except (RuntimeError, httpx.HTTPError) as e:
                # Un calendario que falla (no compartido, id mal escrito) no
                # tiene por que tumbar a los demas del mismo cliente. B-12: se
                # avisa fuerte, no se traga en silencio.
                msg = f"[{self.fuente_id}] calendario '{nombre}' ({calendar_id}): {e}"
                logger.warning(msg)
                alertas.append(msg)
                continue

            exitosos += 1
            self.sync_tokens_siguientes[calendar_id] = token_nuevo
            if reiniciado:
                self.calendarios_reiniciados.append(calendar_id)
                alertas.append(
                    f"[{self.fuente_id}] el syncToken de '{nombre}' vencio; "
                    "Google solicito y completo una resincronizacion."
                )
            for ev in eventos:
                filas.append(_evento_a_fila(nombre, calendar_id, ev))

        if exitosos == 0:
            raise RuntimeError(
                f"[{self.fuente_id}] no se pudo leer ninguno de los "
                f"{len(calendarios)} calendarios configurados."
            )
        if not filas:
            logger.info(
                "[%s] Google Calendar no devolvio cambios en esta corrida.",
                self.fuente_id,
            )

        df = pd.DataFrame(
            filas, columns=list(CAMPOS_EVENTO) + ["version_hash"]
        )
        df.columns = normalizar_columnas(df.columns)
        # Las columnas ya vienen tipadas desde _evento_a_fila (datetime, int,
        # bool); a diferencia de google_sheets, aca NO hace falta inferir_tipos
        # porque el dato no pasa por texto en ningun momento.

        tabla = registrar_df(con, df, "citas", self.fuente_id)
        logger.info("[%s] tabla %s (%d filas, %d calendario(s))",
                    self.fuente_id, tabla, len(df), exitosos)

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
                if calendar_id is not None and str(calendar_id).strip()
            }
            if len(resultado) != len(crudo):
                logger.warning(
                    "[%s] se ignoraron calendarios con calendar_id vacio.",
                    self.fuente_id,
                )
            ids = list(resultado.values())
            if len(ids) != len(set(ids)):
                raise RuntimeError(
                    f"Fuente '{self.fuente_id}': un mismo calendar_id aparece "
                    "mas de una vez en 'calendarios'. Cada ID debe tener un "
                    "solo nombre de barbero."
                )
            return resultado
        valor = self.config.get("calendar_id", "")
        cal_id = "" if valor is None else str(valor).strip()
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

    def _listar_eventos(
        self, calendar_id: str, desde: datetime, sync_token: str = ""
    ) -> tuple[list, str, bool]:
        """
        Primera vez: carga completa desde ``desde`` y obtiene nextSyncToken.
        Siguientes veces: envia el token y recibe solo cambios/cancelaciones.

        Si Google responde 410, el token vencio. Se repite automaticamente una
        carga completa; el historico local NO se borra.
        """
        headers = {"Authorization": f"Bearer {self._token()}"}
        url = f"{CALENDAR_API}/calendars/{_url_quote(calendar_id)}/events"

        def parametros(token: str) -> dict:
            base = {
                "singleEvents": "true",
                "maxResults": 2500,
                # Con syncToken Google exige incluir eliminados.
                "showDeleted": "true",
            }
            if token:
                base["syncToken"] = token
            else:
                # No se usa timeMax: un limite futuro quedaria congelado dentro
                # del token y eventualmente haria perder citas nuevas.
                base["timeMin"] = desde.isoformat()
            return base

        params = parametros(sync_token)
        eventos = []
        reiniciado = False
        with httpx.Client(timeout=TIMEOUT_SEG) as cli:
            while True:
                r = cli.get(url, headers=headers, params=params)
                if r.status_code == 410 and sync_token and not reiniciado:
                    logger.warning(
                        "[%s] syncToken vencido para %s; se hace carga completa.",
                        self.fuente_id,
                        calendar_id,
                    )
                    eventos = []
                    sync_token = ""
                    reiniciado = True
                    params = parametros("")
                    continue
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
                    siguiente = str(datos.get("nextSyncToken", "") or "")
                    if not siguiente:
                        raise RuntimeError(
                            "Google no devolvio nextSyncToken en la ultima "
                            "pagina; no se guarda una sincronizacion incompleta."
                        )
                    break
                params["pageToken"] = token

        return eventos, siguiente, reiniciado


def _url_quote(texto: str) -> str:
    import urllib.parse
    return urllib.parse.quote(texto, safe="")


def _json_texto(valor) -> str:
    return json.dumps(
        valor if valor is not None else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _evento_a_fila(recurso: str, calendar_id: str, ev: dict) -> dict:
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

    original = ev.get("originalStartTime", {})
    fila = {
        "calendar_id": calendar_id,
        "recurso": recurso,
        "evento_id": ev.get("id", ""),
        "ical_uid": ev.get("iCalUID", ""),
        "titulo": ev.get("summary", ""),
        "descripcion": ev.get("description", ""),
        "ubicacion": ev.get("location", ""),
        "inicio": inicio,
        "fin": fin,
        "zona_horaria": inicio_raw.get("timeZone") or fin_raw.get("timeZone") or "",
        "duracion_min": duracion_min,
        "todo_el_dia": todo_el_dia,
        "estado": ev.get("status", ""),
        "transparencia": ev.get("transparency", ""),
        "visibilidad": ev.get("visibility", ""),
        "tipo_evento": ev.get("eventType", "default"),
        "evento_recurrente_id": ev.get("recurringEventId", ""),
        "inicio_original": _parsear_fecha(
            original.get("dateTime") or original.get("date")
        ),
        "creado_en": _parsear_fecha(ev.get("created")),
        "actualizado_en": _parsear_fecha(ev.get("updated")),
        "enlace_evento": ev.get("htmlLink", ""),
        "organizador": _json_texto(ev.get("organizer", {})),
        "invitados": _json_texto(ev.get("attendees", [])),
        "propiedades": _json_texto(ev.get("extendedProperties", {})),
        # Conserva cualquier campo presente o agregado por Google aunque no
        # tenga una columna dedicada en esta version del conector.
        "raw_evento": _json_texto(ev),
    }
    fila["version_hash"] = hash_evento(fila)
    return fila


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
