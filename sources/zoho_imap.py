"""
Fuente: bandeja de correo Zoho por IMAP.

Convierte los correos de una carpeta en una tabla consultable, para que el bot
pueda responder cosas como "cuantos correos me mando el proveedor X este mes" o
"buscame el correo del pedido 1053".

Config esperada (JSON en la columna 'config' del registro):

  {"usuario": "jose@fachavi.com",
   "password_env": "ZOHO_PASSWORD_FACHAVI",
   "carpeta": "INBOX",
   "dias": 30,
   "tabla": "correos"}

Opcionales: "host" (default imap.zoho.com), "puerto" (993), "tope" (500),
"criterio" (sintaxis IMAP cruda, pisa a "dias"), "max_chars_cuerpo" (2000).

LA CONTRASEÑA NO VA EN EL REGISTRO. `password_env` es el NOMBRE de una variable
de entorno, no la contraseña. El registro es una hoja de Google que se comparte,
se exporta y se respalda; una credencial de correo ahi es una fuga esperando
turno. El unico lugar donde vive el secreto es el panel de Render.

Y si tenes verificacion en dos pasos —deberias— la contraseña normal NO sirve:
hay que generar una especifica de aplicacion en accounts.zoho.com -> Seguridad.

NOTA SOBRE EL PLAN. Zoho deshabilito el acceso IMAP en las cuentas gratuitas
nuevas. Si el login falla con credenciales correctas, revisar primero que IMAP
este habilitado en Zoho Mail -> Configuracion -> Correo -> IMAP.

POR QUE EL PARSEO ES MAS LARGO DE LO QUE PARECE. Conectarse a IMAP son cuatro
lineas; leer un correo de verdad no:

  1. Los encabezados vienen codificados. Un asunto con tilde llega como
     '=?UTF-8?B?RmFjdHVyYQ==?=' y decode_header() devuelve una LISTA de
     fragmentos, cada uno con su charset. Concatenar sin decodificar da basura.
  2. El cuerpo casi nunca esta en el primer nivel: multipart/alternative con
     texto y HTML, o multipart/mixed con adjuntos. Hay que recorrer el arbol.
  3. El charset miente o falta. Muchos servidores declaran iso-8859-1 y mandan
     UTF-8. Por eso todo decode va con errors="replace": un byte raro no debe
     tumbar la corrida entera del cliente.
"""

import email
import imaplib
import logging
import os
import re
from datetime import date, datetime, timedelta
from email.header import decode_header, make_header
from email.utils import getaddresses, parsedate_to_datetime

import pandas as pd

from .base import (
    Fragmento,
    Source,
    construir_catalogo,
    describir_tabla,
    registrar_df,
)

logger = logging.getLogger("fachavi.sources.zoho_imap")

HOST_DEFECTO = "imap.zoho.com"
PUERTO_DEFECTO = 993
TOPE_DEFECTO = 500
MAX_CHARS_CUERPO = 2000

_MESES_IMAP = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# --- Parseo MIME ---------------------------------------------------------

def _texto_encabezado(valor) -> str:
    """Decodifica un encabezado MIME a texto legible."""
    if not valor:
        return ""
    try:
        return str(make_header(decode_header(valor)))
    except Exception:  # noqa: BLE001
        return str(valor)          # malformado: crudo antes que perder el correo


def _decodificar(parte) -> str:
    datos = parte.get_payload(decode=True) or b""
    charset = parte.get_content_charset() or "utf-8"
    try:
        return datos.decode(charset, errors="replace")
    except LookupError:
        return datos.decode("utf-8", errors="replace")   # charset inventado


def _cuerpo_y_adjuntos(msg):
    """Recorre el arbol MIME. Prefiere text/plain; cae a HTML si no hay."""
    texto, html, adjuntos = "", "", []

    for parte in msg.walk():
        if parte.get_content_maintype() == "multipart":
            continue
        disp = str(parte.get("Content-Disposition") or "")
        nombre = _texto_encabezado(parte.get_filename())

        if "attachment" in disp.lower() or nombre:
            adjuntos.append(nombre or "sin_nombre")
            continue

        tipo = parte.get_content_type()
        if tipo == "text/plain" and not texto:
            texto = _decodificar(parte)
        elif tipo == "text/html" and not html:
            html = _decodificar(parte)

    if texto:
        return texto.strip(), adjuntos
    if html:
        limpio = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                        flags=re.S | re.I)
        return " ".join(re.sub(r"<[^>]+>", " ", limpio).split()), adjuntos
    return "", adjuntos


def _partir_remitente(valor: str):
    """
    'Ferretería Solano <ventas@solano.co.cr>' -> (nombre, correo)

    El orden importa y no es el obvio: hay que DECODIFICAR el encabezado antes
    de partirlo. Un nombre con tilde viaja como '=?utf-8?q?Ferrer=C3=ADa?=', y
    getaddresses() no decodifica MIME — ve los '?' y los '=' como parte de la
    direccion y devuelve el nombre vacio. Decodificar despues no sirve: para
    entonces el dato ya se perdio.
    """
    pares = getaddresses([_texto_encabezado(valor)])
    if not pares:
        return "", ""
    nombre, correo = pares[0]
    return nombre.strip(), (correo or "").strip().lower()


def _seleccionar(con, carpeta: str):
    """
    Abre la carpeta en solo lectura.

    DOS TRAMPAS DE IMAP QUE ESTO RESUELVE:

    1. imaplib NO entrecomilla el nombre. Una carpeta con espacio ("Jose
       Chaves") se manda como `EXAMINE Jose Chaves` y el servidor lee "Chaves"
       como un segundo argumento:
           BAD [CLIENTBUG] syntax: expecting '(', found 'c'
       El error no menciona el espacio ni la carpeta, asi que no se parece en
       nada a su causa. Se entrecomilla siempre.

    2. Las subcarpetas van con ruta completa. Una carpeta que en la interfaz
       aparece anidada bajo Inbox se llama "Inbox/Jose Chaves" en IMAP, no
       "Jose Chaves". Por eso, si el primer intento falla, se prueba con el
       prefijo antes de darse por vencido.

    Y si igual no abre, se listan las carpetas REALES en el error. Adivinar
    nombres de carpeta contra un servidor remoto es de las cosas mas tediosas
    de depurar; que el mensaje traiga la lista ahorra la ronda de tanteo.
    """
    candidatas = [carpeta]
    if "/" not in carpeta and carpeta.upper() != "INBOX":
        candidatas.append(f"INBOX/{carpeta}")

    ultimo = None
    for nombre in candidatas:
        try:
            ok, datos = con.select(f'"{nombre}"', readonly=True)
            if ok == "OK":
                if nombre != carpeta:
                    logger.info("Carpeta '%s' encontrada como '%s'.", carpeta, nombre)
                return
            ultimo = datos
        except Exception as e:  # noqa: BLE001
            ultimo = e

    disponibles = []
    try:
        ok, filas = con.list()
        if ok == "OK":
            for f in filas or []:
                texto = f.decode(errors="replace") if isinstance(f, bytes) else str(f)
                # Formato: (\HasNoChildren) "/" "Nombre De La Carpeta"
                partes = texto.split(' "')
                disponibles.append(partes[-1].rstrip('"') if partes else texto)
    except Exception:  # noqa: BLE001
        pass

    raise RuntimeError(
        f"No se pudo abrir la carpeta '{carpeta}' ({ultimo}). "
        + (f"Carpetas disponibles: {', '.join(disponibles)}"
           if disponibles else "No se pudo listar las carpetas del buzon.")
    )


# --- La fuente -----------------------------------------------------------

class ZohoIMAPSource(Source):
    tipo = "zoho_imap"

    def cargar(self, con) -> Fragmento:
        usuario = (self.config.get("usuario") or "").strip()
        if not usuario:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (zoho_imap) sin 'usuario' en config."
            )

        var = (self.config.get("password_env") or "").strip()
        if not var:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': falta 'password_env' con el NOMBRE "
                "de la variable de entorno que guarda la contraseña. La "
                "contraseña no va escrita en el registro."
            )
        password = os.environ.get(var, "")
        if not password:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': la variable de entorno '{var}' esta "
                "vacia. Configurala en Render (contraseña especifica de "
                "aplicacion si tenes verificacion en dos pasos)."
            )

        alertas = []
        correos, truncado = self._bajar(usuario, password, alertas)

        df = pd.DataFrame(correos, columns=[
            "uid", "fecha", "remitente_nombre", "remitente_correo",
            "asunto", "cuerpo", "n_adjuntos", "adjuntos",
        ])
        # inferir_tipos NO se usa aca a proposito: los tipos ya salen correctos
        # del parseo (fecha es datetime real, n_adjuntos es int) y pasar el
        # cuerpo de un correo por el detector de numeros y fechas solo puede
        # empeorarlo — un "01/02" dentro de un texto no es una fecha.
        if not df.empty:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", utc=True)
            df["fecha"] = df["fecha"].dt.tz_localize(None)

        nombre_deseado = self.config.get("tabla") or "correos"
        tabla = registrar_df(con, df, nombre_deseado, self.fuente_id)
        logger.info("[zoho_imap:%s] tabla %s (%d correos)",
                    self.fuente_id, tabla, len(df))

        if truncado:
            alertas.append(
                f"[{self.fuente_id}] se alcanzo el tope de "
                f"{self._tope()} correos; hay mas en la carpeta y NO se "
                "cargaron. Subi 'tope' o reduci 'dias' en la config de la fuente."
            )

        catalogo_filas = self.config.get("catalogo") or self._catalogo_defecto(
            nombre_deseado)

        return Fragmento(
            schema=describir_tabla(con, tabla),
            catalogo=construir_catalogo(catalogo_filas),
            tablas=[tabla],
            catalogo_filas=catalogo_filas,
            alertas=alertas,
        )

    # --- IMAP ---

    def _tope(self) -> int:
        return int(self.config.get("tope", TOPE_DEFECTO))

    def _criterio(self) -> str:
        """Criterio IMAP. 'dias' es el atajo amable; 'criterio' pisa todo."""
        crudo = (self.config.get("criterio") or "").strip()
        if crudo:
            return crudo
        dias = int(self.config.get("dias", 30))
        if dias <= 0:
            return "ALL"
        desde = date.today() - timedelta(days=dias)
        # IMAP exige el formato DD-Mon-YYYY con el mes en ingles. Un
        # strftime("%d-%b-%Y") depende del locale del contenedor y en un
        # servidor en español devuelve "03-ago-2026", que el servidor rechaza.
        return f'(SINCE "{desde.day:02d}-{_MESES_IMAP[desde.month - 1]}-{desde.year}")'

    def _bajar(self, usuario: str, password: str, alertas: list):
        host = self.config.get("host") or HOST_DEFECTO
        puerto = int(self.config.get("puerto", PUERTO_DEFECTO))
        carpeta = self.config.get("carpeta") or "INBOX"
        criterio = self._criterio()
        tope = self._tope()
        max_chars = int(self.config.get("max_chars_cuerpo", MAX_CHARS_CUERPO))

        filas, truncado = [], False
        con = imaplib.IMAP4_SSL(host, puerto)
        try:
            con.login(usuario, password)
            # readonly=True: leer la bandeja para ingesta NO debe marcar los
            # correos como leidos. Sin esto, la primera corrida del cron le
            # vacia la cola de pendientes al cliente y nadie entiende por que.
            _seleccionar(con, carpeta)

            ok, datos = con.search(None, criterio)
            if ok != "OK":
                raise RuntimeError(
                    f"Busqueda IMAP rechazada ({criterio}): {datos}"
                )

            ids = (datos[0] or b"").split()
            total = len(ids)
            if total > tope:
                truncado = True
                ids = ids[-tope:]              # los mas recientes
            logger.info("[zoho_imap:%s] %d coinciden con %s; se leen %d",
                        self.fuente_id, total, criterio, len(ids))

            for i in ids:
                ok, cruda = con.fetch(i, "(RFC822)")
                if ok != "OK" or not cruda or not isinstance(cruda[0], tuple):
                    continue
                msg = email.message_from_bytes(cruda[0][1])

                try:
                    fecha = parsedate_to_datetime(msg.get("Date"))
                except Exception:  # noqa: BLE001
                    fecha = None

                cuerpo, adj = _cuerpo_y_adjuntos(msg)
                nombre, correo = _partir_remitente(msg.get("From"))
                filas.append({
                    "uid": i.decode(),
                    "fecha": fecha,
                    "remitente_nombre": nombre,
                    "remitente_correo": correo,
                    "asunto": _texto_encabezado(msg.get("Subject")),
                    # El cuerpo se recorta: una cadena de respuestas de 40 KB
                    # infla el warehouse y, peor, entra al prompt de redaccion.
                    "cuerpo": cuerpo[:max_chars],
                    "n_adjuntos": len(adj),
                    "adjuntos": ", ".join(adj)[:500],
                })
        finally:
            try:
                con.logout()
            except Exception:  # noqa: BLE001
                pass

        return filas, truncado

    def _catalogo_defecto(self, tabla: str) -> list:
        def fila(col, desc):
            return {"tabla": tabla, "columna": col, "descripcion": desc,
                    "sistema_origen": "Zoho Mail", "frecuencia": "Segun cron",
                    "dueño": "Administracion"}
        return [
            fila("*", f"Correos de la carpeta {self.config.get('carpeta','INBOX')} "
                      f"de {self.config.get('usuario','')}. El cuerpo esta "
                      f"recortado y NO incluye el contenido de los adjuntos."),
            fila("fecha", "Fecha de envio del correo, sin zona horaria."),
            fila("remitente_correo", "Direccion del remitente, en minusculas. "
                                     "Usar esta y no remitente_nombre para agrupar."),
            fila("asunto", "Asunto ya decodificado."),
            fila("cuerpo", "Texto plano del correo, recortado."),
            fila("n_adjuntos", "Cantidad de archivos adjuntos."),
            fila("adjuntos", "Nombres de los adjuntos, separados por coma."),
        ]
