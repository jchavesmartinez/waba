"""
Fuente: Excel de Microsoft alojado en SharePoint u OneDrive (Microsoft Graph).

Se comporta EXACTAMENTE igual que el conector de Google Sheets, para que el
cliente no tenga que aprender dos modelos mentales:

  - Cada pestania (hoja) que NO empiece con '_' se carga como tabla consultable.
  - La pestania '_catalogo' (opcional) aporta la metadata y la gobernanza.
  - La pestania '_kpis' (opcional) aporta la capa semantica.
  - Cualquier pestania que empiece con '_' se excluye de las tablas.

Config esperada (columna 'config' del registro, como JSON). Tres formas de
apuntar al archivo, de la mas comoda a la mas precisa:

  1. Por ruta dentro de un sitio de SharePoint (lo normal):
     {"tenant_env": "MS_TENANT_FERRETERIA_A",
      "cliente_env": "MS_CLIENTE_FERRETERIA_A",
      "secreto_env": "MS_SECRETO_FERRETERIA_A",
      "sitio": "contoso.sharepoint.com:/sites/Ventas",
      "ruta": "Documentos compartidos/Reportes/ventas.xlsx"}

  2. Por identificadores de Graph, si ya los tenes:
     {"...credenciales...", "drive_id": "b!xxx", "item_id": "01ABC..."}

  3. Desde el OneDrive de un usuario:
     {"...credenciales...", "usuario": "reportes@contoso.com",
      "ruta": "Documentos/ventas.xlsx"}

Opcionales (mismos nombres que en google_sheets, a proposito):
     "hojas":   ["ventas"]          -> SOLO estas (lista blanca)
     "excluir": ["notas"]           -> todas menos estas (lista negra)
     "max_filas": 200000            -> tope de filas por hoja (0 = sin tope)
     "formato_fecha": "mes_primero" -> convencion de fecha de ESTE archivo

SEGURIDAD — C-04. Las credenciales de Azure NUNCA van en la hoja maestra: la
celda 'config' lleva el NOMBRE de la variable de entorno que las guarda
(tenant_env, cliente_env, secreto_env). Es el mismo mecanismo que ya usan los
conectores postgres (dsn_env) y api_rest (headers_env). Un client_secret de
Azure escrito en un Sheet lo ve cualquiera con permiso de lectura, y el
historial de versiones de Google lo guarda para siempre aunque se borre la
celda.

COMO SE HABILITA DEL LADO DEL CLIENTE (una vez por cliente):
  1. En el portal de Azure del cliente: Microsoft Entra ID > Registros de
     aplicaciones > Nuevo registro. Anotar el 'Id. de aplicacion (cliente)' y
     el 'Id. de directorio (inquilino)'.
  2. Certificados y secretos > Nuevo secreto de cliente. Copiarlo AHI MISMO:
     Azure no lo vuelve a mostrar. OJO: los secretos de cliente VENCEN (24
     meses como maximo). Anotar la fecha; cuando venza, la ingesta falla con
     un error de autenticacion y nada avisa antes.
  3. Permisos de API > Microsoft Graph > Permisos de APLICACION (no delegados)
     > Sites.Read.All (o Files.Read.All para OneDrive). Despues:
     "Conceder consentimiento de administrador". Sin ese consentimiento el
     token se obtiene igual pero cada lectura devuelve 403.

Solo lectura, a proposito: con Sites.Read.All la aplicacion no puede modificar
ni borrar nada del SharePoint del cliente, igual que la cuenta de servicio de
Google es de solo lectura.

Requiere: httpx (ya es dependencia) y openpyxl para leer .xlsx.
"""

import io
import logging
import time
import urllib.parse

import httpx
import pandas as pd

import config
from .base import (
    Source,
    Fragmento,
    coma_decimal_de,
    dia_primero_de,
    inferir_tipos,
    registrar_df,
    describir_tabla,
    construir_catalogo,
    limpiar_nombre,
    normalizar_columnas,
)

logger = logging.getLogger("fachavi.sources.sharepoint_excel")

CATALOGO_HOJA = "_catalogo"
KPIS_HOJA = "_kpis"

GRAPH = "https://graph.microsoft.com/v1.0"
LOGIN = "https://login.microsoftonline.com"
TIMEOUT_SEG = 60.0

# Mismo criterio que csv_url (A-09): un archivo enorme no puede agotar la
# memoria del contenedor y matar la corrida de TODOS los clientes.
MAX_MB_DEFECTO = 100

# Cache de tokens por (tenant, cliente). Un token de Graph dura una hora; sin
# esto, una fuente con varias hojas pediria uno nuevo por archivo.
_TOKENS: dict = {}


class SharePointExcelSource(Source):
    tipo = "sharepoint_excel"

    def cargar(self, con) -> Fragmento:
        solo = {str(h).strip().lower() for h in self.config.get("hojas", []) if str(h).strip()}
        excluir = {str(h).strip().lower() for h in self.config.get("excluir", []) if str(h).strip()}

        # Misma guarda que google_sheets: pedir una pestania '_' no tiene efecto
        # porque son metadata y se excluyen siempre; la fuente cargaria CERO
        # tablas y el motivo no seria obvio.
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

        crudo = self._descargar_xlsx()
        # sheet_name=None -> dict {nombre_hoja: DataFrame} de TODAS las hojas.
        # header=0 y dtype=str: se leen como texto y despues inferir_tipos()
        # decide, igual que en Google Sheets. Si dejaramos que pandas infiriera,
        # los dos conectores darian tipos distintos para el mismo dato y el
        # sistema perderia su propiedad mas util: que la misma hoja produce
        # siempre el mismo resultado sin importar de donde venga.
        libro = pd.read_excel(io.BytesIO(crudo), sheet_name=None, header=0,
                              dtype=str, engine="openpyxl")

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

            df = hoja.dropna(how="all")          # filas totalmente vacias de Excel
            if df.empty:
                # B-12: warning, no info. Si la tabla YA existia con datos,
                # saltarla significa que la guarda de vaciado ni siquiera llega a
                # evaluarse, y el dato viejo se queda ahi callado, envejeciendo.
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
                               dia_primero=dia_primero,
                           coma_decimal=coma_dec)

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

        catalogo, catalogo_filas = self._leer_catalogo(libro)

        # Mismo filtrado que google_sheets: el catalogo documenta TODAS las
        # hojas del archivo; si esta fuente cargo solo algunas, se queda con las
        # filas de esas. Sin esto, dos fuentes sobre el mismo archivo duplicarian
        # el catalogo entero.
        cargadas = {t.strip().lower() for t in tablas}
        if catalogo_filas and cargadas:
            antes = len(catalogo_filas)
            catalogo_filas = [
                f for f in catalogo_filas
                if str(f.get("tabla", "")).strip().lower() in cargadas
            ]
            if len(catalogo_filas) != antes:
                logger.info("[%s] catalogo filtrado a sus tablas: %d de %d filas",
                            self.fuente_id, len(catalogo_filas), antes)

        kpis_filas = self._leer_kpis(libro) if tablas else []

        return Fragmento(
            schema="\n\n".join(schema_parts),
            catalogo=catalogo,
            tablas=tablas,
            catalogo_filas=catalogo_filas,
            kpis_filas=kpis_filas,
            alertas=alertas,
        )

    # ---------------- Microsoft Graph ----------------

    def _credenciales(self):
        """Resuelve las tres credenciales de Azure desde variables de entorno (C-04)."""
        def leer(clave_env, clave_directa, etiqueta):
            nombre = str(self.config.get(clave_env, "")).strip()
            if nombre:
                return config.secreto_de_env(
                    nombre, para=f"La fuente '{self.fuente_id}' (sharepoint_excel)"
                )
            valor = str(self.config.get(clave_directa, "")).strip()
            if valor and clave_directa == "secreto":
                logger.warning(
                    "[%s] el client_secret de Azure esta escrito en la hoja de "
                    "calculo. Cualquiera con acceso de lectura lo ve, y el "
                    'historial de versiones lo guarda para siempre. Usa "secreto_env".',
                    self.fuente_id,
                )
            return valor

        tenant = leer("tenant_env", "tenant", "tenant")
        cliente = leer("cliente_env", "cliente", "client_id")
        secreto = leer("secreto_env", "secreto", "client_secret")

        faltan = [n for n, v in
                  (("tenant_env", tenant), ("cliente_env", cliente), ("secreto_env", secreto))
                  if not v]
        if faltan:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (sharepoint_excel) sin {', '.join(faltan)}. "
                "Se necesitan las tres credenciales del registro de aplicacion de Azure."
            )
        return tenant, cliente, secreto

    def _token(self) -> str:
        """
        Token de aplicacion (client credentials). Se cachea hasta un minuto
        antes de que venza, para no pedir uno por hoja.
        """
        tenant, cliente, secreto = self._credenciales()
        clave = (tenant, cliente)
        guardado = _TOKENS.get(clave)
        if guardado and guardado["vence"] > time.time() + 60:
            return guardado["token"]

        try:
            r = httpx.post(
                f"{LOGIN}/{tenant}/oauth2/v2.0/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": cliente,
                    "client_secret": secreto,
                    "scope": "https://graph.microsoft.com/.default",
                },
                timeout=TIMEOUT_SEG,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # El error mas comun aca es un secreto VENCIDO, y el mensaje crudo de
            # Azure (AADSTS7000215) no lo dice de forma obvia.
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': Azure rechazo las credenciales "
                f"[{e.response.status_code}]. Causas tipicas: el client_secret "
                "vencio (duran 24 meses como maximo), el tenant o el client_id "
                f"estan mal, o el secreto se copio incompleto. Respuesta: "
                f"{e.response.text[:300]}"
            ) from e

        datos = r.json()
        token = datos.get("access_token", "")
        if not token:
            raise RuntimeError(f"Fuente '{self.fuente_id}': Azure no devolvio access_token.")
        _TOKENS[clave] = {"token": token,
                          "vence": time.time() + int(datos.get("expires_in", 3600))}
        return token

    def _url_contenido(self, cli: httpx.Client) -> str:
        """Arma la URL de Graph que devuelve el binario del .xlsx."""
        drive_id = str(self.config.get("drive_id", "")).strip()
        item_id = str(self.config.get("item_id", "")).strip()
        if drive_id and item_id:
            return f"{GRAPH}/drives/{drive_id}/items/{item_id}/content"

        ruta = str(self.config.get("ruta", "")).strip().strip("/")
        if not ruta:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (sharepoint_excel) necesita 'ruta' "
                "(o 'drive_id' + 'item_id')."
            )
        ruta_url = urllib.parse.quote(ruta)

        usuario = str(self.config.get("usuario", "")).strip()
        if usuario:
            return f"{GRAPH}/users/{usuario}/drive/root:/{ruta_url}:/content"

        sitio = str(self.config.get("sitio", "")).strip()
        if not sitio:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}' (sharepoint_excel) necesita 'sitio' "
                "(dominio:/sites/NOMBRE) o 'usuario' para OneDrive."
            )
        # Resolver el sitio a su id: Graph no acepta el nombre en la ruta del
        # contenido. Una llamada extra, cacheada implicitamente por corrida.
        r = cli.get(f"{GRAPH}/sites/{sitio}")
        if r.status_code == 404:
            raise RuntimeError(
                f"Fuente '{self.fuente_id}': el sitio '{sitio}' no existe o la "
                "aplicacion no tiene acceso. El formato correcto es "
                "'contoso.sharepoint.com:/sites/Ventas'."
            )
        r.raise_for_status()
        site_id = r.json()["id"]
        return f"{GRAPH}/sites/{site_id}/drive/root:/{ruta_url}:/content"

    def _descargar_xlsx(self) -> bytes:
        """Baja el archivo con limite de tiempo y de tamaño."""
        max_bytes = int(self.config.get("max_mb", MAX_MB_DEFECTO)) * 1024 * 1024
        cabeceras = {"Authorization": f"Bearer {self._token()}"}

        with httpx.Client(timeout=TIMEOUT_SEG, follow_redirects=True,
                          headers=cabeceras) as cli:
            url = self._url_contenido(cli)
            acumulado = bytearray()
            with cli.stream("GET", url) as r:
                if r.status_code == 403:
                    raise RuntimeError(
                        f"Fuente '{self.fuente_id}': Graph respondio 403. Casi "
                        "siempre significa que al permiso Sites.Read.All le falta "
                        "el «consentimiento de administrador» en Azure, o que se "
                        "configuro como permiso DELEGADO en vez de de APLICACION."
                    )
                if r.status_code == 404:
                    raise RuntimeError(
                        f"Fuente '{self.fuente_id}': Graph respondio 404. Revisa la "
                        "'ruta' dentro del sitio; suele fallar por el nombre de la "
                        "biblioteca ('Documentos compartidos' en tenants en espanol, "
                        "'Shared Documents' en ingles)."
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
        logger.info("[%s] descargados %.1f MB desde SharePoint",
                    self.fuente_id, len(acumulado) / 1024 / 1024)
        return bytes(acumulado)

    # ---------------- metadata: _catalogo y _kpis ----------------

    @staticmethod
    def _filas_de(hoja: pd.DataFrame) -> list:
        """Convierte una hoja de metadata en lista de dicts con claves normalizadas."""
        if hoja is None or hoja.empty:
            return []
        h = hoja.fillna("")
        h.columns = [str(c).strip().lower() for c in h.columns]
        return [{k: str(v).strip() for k, v in fila.items()}
                for fila in h.to_dict("records")]

    def _leer_catalogo(self, libro: dict):
        """Lee '_catalogo' si existe. Si falta, la fuente queda sin documentar."""
        hoja = next((v for k, v in libro.items()
                     if k.strip().lower() == CATALOGO_HOJA), None)
        if hoja is None:
            logger.warning(
                "Fuente '%s' sin hoja '%s' (sin catalogo/governance). Sin ella, "
                "el bot no va a poder consultar ninguna de sus tablas.",
                self.fuente_id, CATALOGO_HOJA,
            )
            return "", []

        filas = self._filas_de(hoja)
        if not filas:
            return "", []
        norm = []
        for f in filas:
            norm.append({
                "tabla": f.get("tabla", ""),
                "columna": f.get("columna", ""),
                "descripcion": f.get("descripcion", ""),
                "instruccion": f.get("instruccion", ""),
                "sistema_origen": f.get("sistema_origen", ""),
                "frecuencia": f.get("frecuencia", ""),
                "dueno": f.get("dueño", "") or f.get("dueno", ""),
            })
        return construir_catalogo(filas), norm

    # Mismas columnas que en google_sheets, a proposito: el cliente puede mover
    # una pestania '_kpis' de un Sheet a un Excel sin cambiar nada.
    _KPIS_COLS = ("kpi", "nombre", "descripcion", "preguntas_ejemplo",
                  "formula_sql", "tabla", "dimensiones", "unidad",
                  "supuestos", "minimo_datos", "instruccion")

    def _leer_kpis(self, libro: dict):
        """Lee la hoja '_kpis' si existe; si falta, no hay capa semantica."""
        hoja = next((v for k, v in libro.items()
                     if k.strip().lower() == KPIS_HOJA), None)
        if hoja is None:
            logger.info("Fuente '%s' sin hoja '%s' (sin KPIs).",
                        self.fuente_id, KPIS_HOJA)
            return []
        norm = []
        for f in self._filas_de(hoja):
            if not f.get("kpi"):
                continue
            norm.append({c: f.get(c, "") for c in self._KPIS_COLS})
        return norm
