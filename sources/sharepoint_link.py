"""
Fuente: Excel de Microsoft alojado en SharePoint u OneDrive, leido por URL de
sharing publico.

Este conector se saltea Azure ENTERO. En vez de registrar una aplicacion y
pedir tokens, usa el «link de compartir» que cualquier usuario de SharePoint
u OneDrive puede generar en un clic:

    Clic derecho al archivo  ->  Compartir  ->  «Cualquiera con el enlace
                                                 puede ver»  ->  Copiar enlace

El link resultante se ve asi:

    https://contoso-my.sharepoint.com/:x:/g/personal/reportes_contoso_com/
    EX_ABC123...?e=abcdef

y con ese link, sin credenciales, el cliente lee el archivo. Ese link es lo
unico que hay que pegar en la config de la fuente.

Config esperada (JSON en la columna 'config' del registro):

  {"url": "https://contoso.sharepoint.com/:x:/g/...?e=xxxxx"}

Opcionales (mismos nombres que google_sheets, a proposito):
  "hojas":   ["ventas"]           -> SOLO estas (lista blanca)
  "excluir": ["notas"]            -> todas menos estas (lista negra)
  "max_filas": 200000             -> tope de filas por hoja (0 = sin tope)
  "formato_fecha": "mes_primero"  -> convencion de fecha de ESTE archivo
  "formato_numero": "punto_decimal" -> convencion numerica de ESTE archivo
  "timeout_seg": 60               -> tiempo maximo de descarga
  "max_mb": 100                   -> tope de tamaño del archivo

COMPORTAMIENTO — identico al de google_sheets:
  - Cada hoja que NO empiece con '_' se carga como tabla consultable.
  - Cualquier hoja que empiece con '_' se excluye de las tablas (por si el
    archivo trae una '_catalogo' o '_kpis' vieja; ya no se leen de aca).

CATALOGO Y KPIS: ya NO se leen de este archivo. Viven en UN Sheet central por
cliente (ver catalogo_cliente.py) que documenta todas sus fuentes juntas;
sync.py lo lee una vez por cliente y lo reparte.

LO QUE HAY QUE ENTENDER DE ESTA FORMA DE ACCESO (importante):

Un link «cualquiera con el enlace puede ver» es exactamente eso: CUALQUIERA que
tenga el link puede leer el archivo. No hay contraseña, no hay auditoria, no
hay forma de saber quien mas lo tiene. Si el link se filtra —por correo, chat,
un log— quien lo reciba puede leer el Excel completo. Para datos comerciales de
una PYME (ventas, inventario, clientes) suele estar bien; para nomina, datos
personales o cualquier cosa regulada NO.

Si el cliente cambia el link (regenera el sharing) o lo revoca, la ingesta
empieza a fallar con 404 o 401. Cuando eso pase, pedir un link nuevo y
actualizar la celda 'config' de la fuente.

Requiere: httpx (ya es dependencia) + openpyxl para leer .xlsx.
"""

import io
import logging
import urllib.parse

import httpx
import pandas as pd

from .base import (
    Source,
    Fragmento,
    coma_decimal_de,
    dia_primero_de,
    inferir_tipos,
    registrar_df,
    describir_tabla,
    limpiar_nombre,
    normalizar_columnas,
)

logger = logging.getLogger("fachavi.sources.sharepoint_link")

TIMEOUT_DEFECTO_SEG = 60.0
MAX_MB_DEFECTO = 100


class SharePointLinkSource(Source):
    tipo = "sharepoint_link"

    def cargar(self, con) -> Fragmento:
        url = str(self.config.get("url", "")).strip()
        if not url:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (sharepoint_link) sin 'url'. "
                "En SharePoint u OneDrive: clic derecho al archivo -> "
                "Compartir -> 'Cualquiera con el enlace puede ver' -> Copiar."
            )

        solo = {str(h).strip().lower() for h in self.config.get("hojas", []) if str(h).strip()}
        excluir = {str(h).strip().lower() for h in self.config.get("excluir", []) if str(h).strip()}

        # Misma guarda que google_sheets: pedir '_catalogo' o '_kpis' en 'hojas'
        # no tiene efecto (son metadata, se excluyen siempre) y la fuente
        # cargaria CERO tablas sin que el motivo sea obvio.
        ocultas = {h for h in solo if h.startswith("_")}
        if ocultas:
            logger.error(
                "[%s] 'hojas' pide pestanias de metadata (%s). Esas se excluyen "
                "siempre y esta fuente cargaria CERO tablas. El catalogo ya se "
                "guarda solo; borra esta fila de la pestania 'fuentes'.",
                self.fuente_id, ", ".join(sorted(ocultas)),
            )

        max_filas = int(self.config.get("max_filas", 0) or 0)
        dia_primero = dia_primero_de(self.config)
        coma_dec = coma_decimal_de(self.config)
        alertas = []

        crudo = self._descargar(url)

        # sheet_name=None -> dict {nombre_hoja: DataFrame} de TODAS las hojas.
        # dtype=str: se leen como texto y despues inferir_tipos() decide, igual
        # que en google_sheets. Si dejaramos que pandas infiriera, los dos
        # conectores darian tipos distintos para el mismo dato.
        try:
            libro = pd.read_excel(io.BytesIO(crudo), sheet_name=None, header=0,
                                  dtype=str, engine="openpyxl")
        except Exception as e:  # noqa: BLE001
            # Casi siempre esto significa que la URL devolvio HTML (la pagina de
            # login o de "no autorizado") en vez del binario del .xlsx. El
            # mensaje crudo de openpyxl no lo dice de forma obvia.
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': el archivo descargado no es un "
                ".xlsx valido. Casi siempre significa que el link YA NO ES "
                "PUBLICO (fue revocado, o el permiso quedo como 'personas de la "
                "organizacion' en vez de 'cualquiera con el enlace') o que la "
                f"URL apunta a una carpeta y no a un archivo. Detalle: {e}"
            ) from e

        schema_parts = []
        tablas = []
        vistas = list(libro.keys())

        for titulo, hoja in libro.items():
            if titulo.startswith("_"):
                continue
            if solo and titulo.strip().lower() not in solo:
                continue
            if titulo.strip().lower() in excluir:
                logger.info("[%s] hoja '%s' excluida por config", self.fuente_id, titulo)
                continue

            df = hoja.dropna(how="all")     # filas completamente vacias de Excel
            if df.empty:
                # B-12: warning, no info. Si la tabla YA existia con datos,
                # saltarla significa que la guarda de vaciado ni siquiera llega a
                # evaluarse.
                msg = (f"[{self.fuente_id}] la hoja '{titulo}' llego SIN FILAS "
                       "(solo encabezados o vacia); se salta y no se actualiza "
                       "nada de esa tabla.")
                logger.warning(msg)
                alertas.append(msg)
                continue

            if max_filas and len(df) > max_filas:
                msg = (f"[{self.fuente_id}] la hoja '{titulo}' tiene {len(df)} filas "
                       f"y el tope configurado es {max_filas}: se cargan las "
                       f"primeras {max_filas}. HAY DATOS QUE NO ENTRARON.")
                logger.warning(msg)
                alertas.append(msg)
                df = df.head(max_filas)

            df = df.fillna("")
            df.columns = normalizar_columnas(df.columns)
            df = inferir_tipos(df, alertas=alertas,
                               contexto=f"{self.fuente_id}/{titulo}",
                               dia_primero=dia_primero, coma_decimal=coma_dec)

            tabla = registrar_df(con, df, titulo, self.fuente_id)
            tablas.append(tabla)
            schema_parts.append(describir_tabla(con, tabla))
            logger.info("[%s] tabla %s (%d filas)", self.fuente_id, tabla, len(df))

        if solo:
            faltan = solo - {v.strip().lower() for v in vistas}
            if faltan:
                logger.warning(
                    "[%s] pedidas en 'hojas' pero NO existen en el archivo: %s",
                    self.fuente_id, ", ".join(sorted(faltan)),
                )

        # Catalogo y KPIs ya NO se leen de este archivo: vienen del Sheet
        # central del cliente (catalogo_cliente.py), que sync.py lee una vez y
        # reparte a cada fuente despues de cargar().
        return Fragmento(
            schema="\n\n".join(schema_parts),
            tablas=tablas,
            alertas=alertas,
        )

    # -------- descarga --------

    def _descargar(self, url: str) -> bytes:
        """
        Baja el archivo con limite de tiempo y de tamaño (mismo A-09 que csv_url).

        Detalle importante de SharePoint/OneDrive: el link de compartir apunta a
        una pagina intermedia y hay que agregarle 'download=1' para que devuelva
        el binario en vez de HTML. Se hace transparente para el usuario: pega el
        link tal como lo copio, aca se le arma la variante de descarga.
        """
        url_descarga = _forzar_descarga(url)
        timeout = float(self.config.get("timeout_seg", TIMEOUT_DEFECTO_SEG))
        max_bytes = int(self.config.get("max_mb", MAX_MB_DEFECTO)) * 1024 * 1024

        acumulado = bytearray()
        with httpx.Client(timeout=timeout, follow_redirects=True) as cli:
            with cli.stream("GET", url_descarga) as r:
                if r.status_code == 401 or r.status_code == 403:
                    raise RuntimeError(
                        f"Fuente '{self.fuente_id}': SharePoint respondio "
                        f"{r.status_code}. El link YA NO ES PUBLICO. Causas: "
                        "el cliente revoco el link, o el permiso quedo como "
                        "'personas de <su organizacion>' en vez de 'cualquiera "
                        "con el enlace'. Pedile un link nuevo y actualiza la "
                        "config de la fuente."
                    )
                if r.status_code == 404:
                    raise RuntimeError(
                        f"Fuente '{self.fuente_id}': SharePoint respondio 404. "
                        "El archivo se movio, se renombro o se borro. Pedile "
                        "al cliente un link nuevo al archivo actual."
                    )
                r.raise_for_status()
                for pedazo in r.iter_bytes():
                    acumulado += pedazo
                    if max_bytes and len(acumulado) > max_bytes:
                        raise RuntimeError(
                            f"Fuente '{self.fuente_id}': el archivo pasa de "
                            f"{max_bytes // (1024*1024)} MB. Subi 'max_mb' en la "
                            "config si de verdad es asi de grande."
                        )
        logger.info("[%s] descargados %.1f MB desde SharePoint (link publico)",
                    self.fuente_id, len(acumulado) / 1024 / 1024)
        return bytes(acumulado)

def _forzar_descarga(url: str) -> str:
    """
    Convierte un link de sharing de SharePoint/OneDrive en su variante de
    descarga directa. La regla depende del formato:

      - Link personal de OneDrive (contiene '/personal/'):
            ...?e=xxx  ->  ...?e=xxx&download=1

      - Link empresarial de SharePoint (contiene '/sites/'):
            ...?e=xxx  ->  ...?e=xxx&download=1

      - Link corto (1drv.ms u onedrive.live.com/embed): se deja tal cual;
        SharePoint hace la redireccion despues.

    Si el usuario YA pego un link con download=1, no se toca.
    """
    if not url:
        return url
    partes = urllib.parse.urlsplit(url)
    parametros = dict(urllib.parse.parse_qsl(partes.query, keep_blank_values=True))
    if "download" in parametros:
        return url
    parametros["download"] = "1"
    query = urllib.parse.urlencode(parametros)
    return urllib.parse.urlunsplit(
        (partes.scheme, partes.netloc, partes.path, query, partes.fragment)
    )
