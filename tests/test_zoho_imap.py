"""
Pruebas de la fuente zoho_imap.

POR QUE EXISTE. El parseo de correo falla de formas silenciosas: no tira
excepcion, devuelve texto corrupto o vacio. Y la parte IMAP no se puede probar
a mano sin dejar la bandeja del cliente marcada como leida, que es justo lo que
no queremos que pase. Un IMAP falso permite verificar el camino completo
—buscar, bajar, parsear, cargar a DuckDB— sin tocar una cuenta real.

Los cuatro casos que romperian esto en produccion:

1. Encabezados codificados. Un asunto en base64 con tildes es lo normal, no la
   excepcion, y sin decode_header() llega como '=?UTF-8?B?...?='.
2. Cuerpo anidado. multipart/mixed con un multipart/alternative adentro y un
   PDF al lado: el texto esta dos niveles abajo.
3. Solo HTML. Muchos sistemas (ERPs, notificaciones) no mandan texto plano.
4. readonly. Si la ingesta marca los correos como leidos, la primera corrida
   del cron le vacia la cola de pendientes al cliente.
"""

import email
from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import imaplib

import duckdb
import pytest

from sources.zoho_imap import (ZohoIMAPSource, _cuerpo_y_adjuntos,
                               _partir_remitente, _texto_encabezado)


def _correo_completo():
    m = MIMEMultipart("mixed")
    m["From"] = str(Header("Ferretería Solano <Ventas@Solano.CO.CR>", "utf-8"))
    m["Subject"] = str(Header("Factura de julio — pedido N° 1053", "utf-8"))
    m["Date"] = "Mon, 03 Aug 2026 14:22:11 -0600"
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText("Buenas, adjunto la factura.\nTotal: ₡1.470.000",
                        "plain", "utf-8"))
    alt.attach(MIMEText("<p>Versión <b>HTML</b></p>", "html", "utf-8"))
    m.attach(alt)
    pdf = MIMEApplication(b"%PDF-1.4 falso", _subtype="pdf")
    pdf.add_header("Content-Disposition", "attachment", filename="factura_ñ.pdf")
    m.attach(pdf)
    return m.as_bytes()


def _correo_solo_html():
    m = MIMEText("<html><script>x=1</script><p>Pedido <b>listo</b></p></html>",
                 "html", "utf-8")
    m["From"] = "ERP <noreply@erp.cr>"
    m["Subject"] = "Notificación"
    m["Date"] = "Tue, 04 Aug 2026 09:00:00 -0600"
    return m.as_bytes()


class IMAPFalso:
    """Sustituto de imaplib.IMAP4_SSL. Registra que se le pidio."""

    creado_con = None
    ultimo = None

    def __init__(self, host, puerto):
        IMAPFalso.creado_con = (host, puerto)
        IMAPFalso.ultimo = self
        self.mensajes = {b"1": _correo_completo(), b"2": _correo_solo_html()}
        self.readonly = None
        self.criterio = None
        self.deslogueado = False

    def login(self, u, p):
        return "OK", [b""]

    # Carpetas que el servidor falso "tiene". Como Zoho: la subcarpeta vive
    # bajo INBOX y el nombre lleva espacio.
    CARPETAS = {"INBOX", "INBOX/Jose Chaves"}

    def select(self, carpeta, readonly=False):
        # Un servidor IMAP de verdad parte por espacios: sin comillas, un
        # nombre con espacio llega partido y es error de sintaxis.
        if not (carpeta.startswith('"') and carpeta.endswith('"')):
            if " " in carpeta:
                raise imaplib.IMAP4.error(
                    b"[CLIENTBUG] syntax: expecting '(', found 'c'")
            nombre = carpeta
        else:
            nombre = carpeta[1:-1]
        if nombre not in self.CARPETAS:
            return "NO", [b"folder does not exist"]
        self.readonly = readonly
        self.seleccionada = nombre
        return "OK", [b"2"]

    def list(self, *a, **k):
        return "OK", [f'(\\HasNoChildren) "/" "{c}"'.encode()
                      for c in sorted(self.CARPETAS)]

    def search(self, charset, criterio):
        self.criterio = criterio
        return "OK", [b" ".join(self.mensajes)]

    def fetch(self, i, spec):
        return "OK", [(b"1 (RFC822 {n}", self.mensajes[i]), b")"]

    def logout(self):
        self.deslogueado = True
        return "BYE", [b""]


@pytest.fixture
def fuente(monkeypatch):
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    monkeypatch.setenv("ZOHO_TEST_PASS", "clave-de-aplicacion")
    return ZohoIMAPSource("correo_fachavi", {
        "usuario": "jose@fachavi.com",
        "password_env": "ZOHO_TEST_PASS",
        "tabla": "correos",
        "dias": 30,
    })


# --- Camino completo -----------------------------------------------------

def test_carga_la_tabla_en_duckdb(fuente):
    con = duckdb.connect()
    frag = fuente.cargar(con)
    assert len(frag.tablas) == 1
    filas = con.execute(f"SELECT count(*) FROM {frag.tablas[0]}").fetchone()[0]
    assert filas == 2


def test_decodifica_encabezados_y_cuerpo_anidado(fuente):
    con = duckdb.connect()
    frag = fuente.cargar(con)
    r = con.execute(
        f"SELECT asunto, remitente_correo, remitente_nombre, cuerpo, "
        f"n_adjuntos, adjuntos FROM {frag.tablas[0]} WHERE n_adjuntos > 0"
    ).fetchone()
    asunto, correo, nombre, cuerpo, n_adj, adjuntos = r

    assert asunto == "Factura de julio — pedido N° 1053"
    assert nombre == "Ferretería Solano"
    # En minusculas: agrupar por remitente falla si el mismo buzon aparece
    # escrito de tres formas distintas.
    assert correo == "ventas@solano.co.cr"
    assert "₡1.470.000" in cuerpo
    assert "HTML" not in cuerpo          # gana text/plain sobre text/html
    assert n_adj == 1 and "factura_ñ.pdf" in adjuntos


def test_cae_a_html_cuando_no_hay_texto_plano(fuente):
    con = duckdb.connect()
    frag = fuente.cargar(con)
    cuerpo = con.execute(
        f"SELECT cuerpo FROM {frag.tablas[0]} WHERE n_adjuntos = 0"
    ).fetchone()[0]
    assert cuerpo == "Pedido listo"      # sin etiquetas y sin el <script>


def test_la_fecha_queda_como_timestamp(fuente):
    con = duckdb.connect()
    frag = fuente.cargar(con)
    tipo = con.execute(
        f"SELECT typeof(fecha) FROM {frag.tablas[0]} LIMIT 1").fetchone()[0]
    assert "TIMESTAMP" in tipo.upper()


# --- No romper la bandeja del cliente ------------------------------------

def test_abre_en_solo_lectura(fuente):
    fuente.cargar(duckdb.connect())
    assert IMAPFalso.ultimo.readonly is True, \
        "la ingesta no debe marcar los correos como leidos"


def test_cierra_la_sesion(fuente):
    fuente.cargar(duckdb.connect())
    assert IMAPFalso.ultimo.deslogueado is True


# --- Criterio de busqueda ------------------------------------------------

def test_criterio_por_dias_usa_meses_en_ingles(fuente):
    """
    IMAP exige DD-Mon-YYYY con el mes en ingles. Un strftime('%b') depende del
    locale del contenedor y en un servidor en español devuelve 'ago', que el
    servidor rechaza con un error que no explica nada.
    """
    fuente.cargar(duckdb.connect())
    crit = IMAPFalso.ultimo.criterio
    assert crit.startswith("(SINCE ")
    assert not any(m in crit for m in ("ene", "ago", "dic"))


def test_criterio_crudo_pisa_a_dias(monkeypatch):
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    monkeypatch.setenv("P", "x")
    f = ZohoIMAPSource("f", {"usuario": "a@b.cr", "password_env": "P",
                             "dias": 7, "criterio": '(FROM "prov@x.cr")'})
    f.cargar(duckdb.connect())
    assert IMAPFalso.ultimo.criterio == '(FROM "prov@x.cr")'


def test_avisa_cuando_trunca(monkeypatch):
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    monkeypatch.setenv("P", "x")
    f = ZohoIMAPSource("f", {"usuario": "a@b.cr", "password_env": "P", "tope": 1})
    frag = f.cargar(duckdb.connect())
    assert any("tope" in a for a in frag.alertas)


# --- La credencial no vive en el registro --------------------------------

def test_exige_password_env_no_la_clave(monkeypatch):
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    f = ZohoIMAPSource("f", {"usuario": "a@b.cr", "password": "secreto-en-el-sheet"})
    with pytest.raises(RuntimeError, match="password_env"):
        f.cargar(duckdb.connect())


def test_falla_claro_si_la_variable_esta_vacia(monkeypatch):
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    monkeypatch.delenv("NO_EXISTE", raising=False)
    f = ZohoIMAPSource("f", {"usuario": "a@b.cr", "password_env": "NO_EXISTE"})
    with pytest.raises(RuntimeError, match="NO_EXISTE"):
        f.cargar(duckdb.connect())


# --- Unidades del parseo -------------------------------------------------

@pytest.mark.parametrize("crudo,esperado", [
    ("=?UTF-8?B?RmFjdHVyYSBkZSBqdWxpbw==?=", "Factura de julio"),
    ("Asunto simple", "Asunto simple"),
    ("", ""),
])
def test_texto_encabezado(crudo, esperado):
    assert _texto_encabezado(crudo) == esperado


def test_partir_remitente_sin_nombre():
    assert _partir_remitente("ventas@x.cr") == ("", "ventas@x.cr")


def test_cuerpo_vacio_no_revienta():
    msg = email.message_from_bytes(b"From: a@b.cr\r\nSubject: x\r\n\r\n")
    assert _cuerpo_y_adjuntos(msg) == ("", [])


# --- Nombres de carpeta: espacios y anidamiento --------------------------
#
# Zoho muestra las subcarpetas anidadas bajo Inbox, pero en IMAP se llaman
# "INBOX/Nombre". Y imaplib no entrecomilla el nombre, asi que una carpeta con
# espacio produce un error de sintaxis cuyo texto no menciona ni el espacio ni
# la carpeta. Las dos cosas juntas son la trampa mas comun de este conector.

def test_carpeta_con_espacio_y_anidada(monkeypatch):
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    monkeypatch.setenv("P", "x")
    f = ZohoIMAPSource("f", {"usuario": "a@b.cr", "password_env": "P",
                             "carpeta": "Jose Chaves"})
    f.cargar(duckdb.connect())
    assert IMAPFalso.ultimo.seleccionada == "INBOX/Jose Chaves"


def test_ruta_completa_tambien_funciona(monkeypatch):
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    monkeypatch.setenv("P", "x")
    f = ZohoIMAPSource("f", {"usuario": "a@b.cr", "password_env": "P",
                             "carpeta": "INBOX/Jose Chaves"})
    f.cargar(duckdb.connect())
    assert IMAPFalso.ultimo.seleccionada == "INBOX/Jose Chaves"


def test_carpeta_inexistente_lista_las_reales(monkeypatch):
    """El error debe traer los nombres, no dejarte adivinando contra el servidor."""
    monkeypatch.setattr("sources.zoho_imap.imaplib.IMAP4_SSL", IMAPFalso)
    monkeypatch.setenv("P", "x")
    f = ZohoIMAPSource("f", {"usuario": "a@b.cr", "password_env": "P",
                             "carpeta": "Ferreteria"})
    with pytest.raises(RuntimeError, match="INBOX/Jose Chaves"):
        f.cargar(duckdb.connect())
