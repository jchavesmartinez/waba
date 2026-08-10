"""
Lectura de la metadata de modelos semanticos desde el Sheet del cliente.

Vive en el MISMO Sheet central donde ya estan '_catalogo' y '_kpis', en tres
pestanias nuevas:

  _modelos         que tabla derivada se construye, desde donde y con que filtro
  _campos          que columnas tiene y como se extrae cada una
  _clasificacion   reglas por defecto (patron de comercio -> cuenta contable)
  _categorias      valores VALIDOS de clasificacion, con ejemplos
  _overrides       correcciones por transaccion puntual

POR QUE EN EL SHEET Y NO EN LA BASE. Es la misma decision que ya tomo el
catalogo: el Sheet es la superficie que una persona edita sin credenciales de
produccion, tiene historial de versiones y se puede compartir con el cliente.
Escribir la metadata directo en Neon la volveria invisible, no versionada y solo
accesible para quien tenga el DSN.

LOS OVERRIDES TAMBIEN VAN ACA, y esto merece una nota. Es tentador escribirlos
a mano en el editor SQL de Neon, que es mas rapido. Pero entonces dejan de ser
un INSUMO del modelo y pasan a ser estado que vive solo en la base: si la base
se pierde o se migra, las correcciones no se reconstruyen desde ningun lado, y
la propiedad que sostiene todo este modulo —que la tabla derivada es una
funcion pura de sus entradas— se rompe en silencio.

QUE NO HACE ESTE MODULO. No interpreta la metadata ni la valida contra la base:
solo la lee y la normaliza. La validacion vive en motor.py, que es quien sabe
que tipos existen. Separarlo permite escribir pruebas del motor sin tocar
Google.
"""

import logging

from gclient import abrir_libro

logger = logging.getLogger("fachavi.modelo.metadata")

MODELOS_SHEET = "_modelos"
CAMPOS_SHEET = "_campos"
CLASIFICACION_SHEET = "_clasificacion"
CATEGORIAS_SHEET = "_categorias"
OVERRIDES_SHEET = "_overrides"

_MODELOS_COLS = (
    "modelo_id",        # identificador unico dentro del cliente
    "tabla_origen",     # tabla raw de la que se lee (nombre fisico)
    "tabla_destino",    # tabla derivada que se crea en semantic_<cliente>
    "extractor",        # como se leen los campos (ver modelo/extractores.py)
    "columna_texto",    # de que columna del origen sale el texto a extraer
    "filtro",           # SQL: que filas del origen aplican
    "activo",
)

_CAMPOS_COLS = (
    "modelo_id",
    "columna",          # nombre de la columna en la tabla derivada
    "tipo",             # tipo logico (ver modelo/tipos.py)
    "patron",           # que buscar: etiqueta, regex o ruta, segun extractor
    "requerido",        # 'si' -> sin este campo la fila se RECHAZA
    # Nombre de la columna donde cae la CLASIFICACION de este campo. Vacio =
    # el campo no se clasifica. Ej: columna 'comercio' con
    # clasifica_en='cuenta_contable' crea esa columna y le aplica las tres
    # capas (override > mapeo > regla).
    "clasifica_en",
    # Columnas adicionales, separadas por coma, que se presentan junto al
    # campo principal al clasificador. Pueden ser campos extraidos por el
    # modelo o columnas de la tabla raw (p.ej. asunto,tipo_transaccion).
    "clasifica_con",
    "descripcion",
)

_CLASIFICACION_COLS = (
    "modelo_id",
    # Columnas contra las que se evalua ESTA regla, separadas por coma. Vacio
    # conserva el comportamiento historico: solo el campo que clasifica.
    "columnas",
    "patron",           # patron SQL LIKE sobre la columna clasificada
    "valor",            # que se asigna cuando calza
    "prioridad",        # menor gana; permite reglas especificas antes que generales
)

_CATEGORIAS_COLS = (
    "modelo_id",
    "categoria",        # el valor que se le puede asignar a una fila
    "descripcion",      # que entra en esta categoria y que no
    "ejemplos",         # casos concretos, separados por coma
)

_OVERRIDES_COLS = (
    "modelo_id",
    "clave",            # identidad de la fila corregida (normalmente correo_id)
    "columna",          # que columna se corrige
    "valor",            # con que valor
    "nota",             # por que (para auditoria; el motor no lo usa)
)


def leer(cliente: dict) -> dict:
    """
    Lee las cuatro pestanias del Sheet del cliente.

    Devuelve {"modelos": [...], "campos": [...], "clasificacion": [...],
              "overrides": [...]}.

    Un cliente sin pestania '_modelos' simplemente no tiene capa semantica: se
    devuelve vacio y NO es un error. Es el caso de todos los clientes actuales.
    """
    cid = cliente.get("cliente_id", "")
    spreadsheet_id = str(cliente.get("catalogo_spreadsheet_id", "")).strip()
    vacio = {"modelos": [], "campos": [], "clasificacion": [],
             "categorias": [], "overrides": []}

    if not spreadsheet_id:
        return vacio

    try:
        libro = abrir_libro(spreadsheet_id)
    except Exception as e:  # noqa: BLE001
        # Mismo criterio que catalogo_cliente: no se puede distinguir "no tiene
        # capa semantica" de "no se pudo abrir el Sheet", asi que se avisa
        # fuerte y se sigue sin modelos en vez de tumbar la corrida.
        logger.error(
            "[%s] no se pudo abrir el Sheet para leer la metadata de modelos "
            "(%s): %s. No se va a construir NINGUNA tabla derivada en esta "
            "corrida; las que ya existan quedan como estaban.",
            cid, spreadsheet_id, e,
        )
        return vacio

    modelos = _leer_pestania(libro, cid, MODELOS_SHEET, _MODELOS_COLS,
                             clave="modelo_id", silencioso=True)
    if not modelos:
        return vacio

    return {
        "modelos": [m for m in modelos if _es_si(m.get("activo", "si"))],
        "campos": _leer_pestania(libro, cid, CAMPOS_SHEET, _CAMPOS_COLS,
                                 clave="columna"),
        "clasificacion": _leer_pestania(libro, cid, CLASIFICACION_SHEET,
                                        _CLASIFICACION_COLS, clave="patron",
                                        silencioso=True),
        "categorias": _leer_pestania(libro, cid, CATEGORIAS_SHEET,
                                     _CATEGORIAS_COLS, clave="categoria",
                                     silencioso=True),
        "overrides": _leer_pestania(libro, cid, OVERRIDES_SHEET,
                                    _OVERRIDES_COLS, clave="clave",
                                    silencioso=True),
    }


def _leer_pestania(libro, cid: str, nombre: str, columnas: tuple,
                   clave: str, silencioso: bool = False) -> list:
    """
    Lee una pestania y normaliza sus filas al juego de columnas esperado.

    Las cabeceras se normalizan a minusculas y sin espacios porque una hoja
    editada a mano termina teniendo 'Modelo_id', 'modelo id ' y 'MODELO_ID' en
    clientes distintos, y eso no deberia ser un error de configuracion.
    """
    try:
        ws = libro.worksheet(nombre)
    except Exception:  # noqa: BLE001
        if not silencioso:
            logger.warning("[%s] falta la pestania '%s' del Sheet.", cid, nombre)
        return []

    # No se usa get_all_records(): varias plantillas traen una fila explicativa
    # encima del encabezado y gspread la interpreta como headers vacios
    # duplicados. Se detecta la fila que contiene la columna clave y desde ahi
    # se arma el registro, permitiendo notas editables antes de la tabla.
    valores = ws.get_all_values()
    encabezado_i = next(
        (i for i, fila in enumerate(valores)
         if clave in {str(c).strip().lower().replace(" ", "_") for c in fila}),
        None,
    )
    if encabezado_i is None:
        if not silencioso:
            logger.warning("[%s] la pestania '%s' no tiene encabezado '%s'.",
                           cid, nombre, clave)
        return []

    encabezados = [str(c).strip().lower().replace(" ", "_")
                   for c in valores[encabezado_i]]
    filas = []
    for cruda in valores[encabezado_i + 1:]:
        f = {encabezados[i]: str(v).strip()
             for i, v in enumerate(cruda)
             if i < len(encabezados) and encabezados[i]}
        # Una fila sin su columna clave es una fila en blanco o a medio
        # escribir; se ignora sin ruido. Google devuelve las filas vacias que
        # quedan debajo de los datos.
        if not f.get(clave):
            continue
        filas.append({c: f.get(c, "") for c in columnas})
    return filas


def campos_de(metadata: dict, modelo_id: str) -> list:
    return [c for c in metadata.get("campos", [])
            if c.get("modelo_id") == modelo_id]


def clasificacion_de(metadata: dict, modelo_id: str) -> list:
    reglas = [r for r in metadata.get("clasificacion", [])
              if r.get("modelo_id") == modelo_id]
    # Prioridad ascendente: la regla mas especifica se declara con un numero
    # menor. Sin orden estable, dos patrones que calzan con el mismo comercio
    # ('AM PM%' y 'AM%') darian resultados distintos segun como Google
    # devolviera las filas.
    return sorted(reglas, key=lambda r: _a_entero(r.get("prioridad"), 100))


def categorias_de(metadata: dict, modelo_id: str) -> list:
    """
    Valores validos de clasificacion para un modelo.

    Existe para que el LLM tenga un conjunto CERRADO al cual responder. Sin
    esto, un modelo de lenguaje inventa categorias nuevas con total soltura
    ('Compras varias', 'Gastos personales', 'Otros') y el reporte del cliente
    termina con quince cuentas contables que nadie definio, todas plausibles y
    ninguna comparable entre meses.
    """
    return [c for c in metadata.get("categorias", [])
            if c.get("modelo_id") == modelo_id]


def overrides_de(metadata: dict, modelo_id: str) -> list:
    return [o for o in metadata.get("overrides", [])
            if o.get("modelo_id") == modelo_id]


def _a_entero(valor, defecto: int) -> int:
    try:
        return int(str(valor).strip())
    except (TypeError, ValueError):
        return defecto


def _es_si(valor) -> bool:
    """Vacio cuenta como SI: una fila nueva sin llenar 'activo' deberia correr."""
    texto = str(valor).strip().lower()
    if texto == "":
        return True
    return texto in {"1", "si", "sí", "true", "yes", "x"}
