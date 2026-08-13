"""
Job de clasificacion: llena la tabla de mapeo con ayuda del LLM.

    valores sin clasificar  ->  [LLM en lotes]  ->  semantic_<cliente>._mapeo

CORRE APARTE DE LA CONSTRUCCION, Y ESO ES EL DISEÑO ENTERO.

La tabla derivada se reconstruye completa en cada corrida, y eso solo es
sostenible porque el LLM NO participa de esa ruta. Lo que hace este job es
producir una ENTRADA mas —la tabla de mapeo— que despues construir_modelos.py
consume con un JOIN. Reconstruir 100.000 transacciones son cero llamadas al
modelo; este job solo mira los valores que todavia NO estan mapeados.

En la practica: la primera corrida clasifica todos los comercios del cliente;
la segunda, los tres o cuatro nuevos de la semana; y si no hay ninguno nuevo,
cero llamadas. El costo tiende a cero en vez de crecer con el historial.

POR LOTES Y NO FILA POR FILA, por tres razones y solo una es el costo:

  1. Menos llamadas: 50 comercios en una en vez de 50 llamadas.
  2. CONSISTENCIA: viendo los valores juntos, el modelo nota que
     'AUTOMERCADO ESCAZU' y 'AUTOMERC #12' son el mismo negocio y los manda a
     la misma cuenta. Uno por uno, cada llamada es independiente y pueden
     salir distintos.
  3. Un lote se puede REVISAR antes de aplicarlo; una decision por fila en
     tiempo real, no.

SALIDA CERRADA. El modelo solo puede responder con una categoria declarada en
la pestania '_categorias'. Cualquier cosa fuera de esa lista se descarta y el
valor queda sin clasificar. Sin esta validacion, un LLM inventa categorias
nuevas sin ningun esfuerzo ('Gastos varios', 'Otros', 'Compras personales') y
el cliente termina con quince cuentas contables que nadie definio, todas
plausibles y ninguna comparable entre meses.

Y PUEDE DECIR QUE NO SABE. 'sin_clasificar' es una respuesta valida y
explicita. Un modelo obligado a elegir siempre elige, y adivinar con seguridad
es peor que no responder: un gasto mal clasificado con confianza no se detecta
nunca, mientras que uno sin clasificar aparece en una lista para revisar.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import config
import registry
from warehouse import crear_destino
from .metadata import categorias_de, leer as leer_metadata
from .motor import SIN_CLASIFICAR, Modelo, _normalizar

logger = logging.getLogger("fachavi.modelo.clasificar")

TABLA_MAPEO = "_mapeo"
TAMANO_LOTE = 25
MAX_TOKENS = 2000

COLUMNAS_MAPEO = [
    ("modelo_id", "texto"),
    ("clasifica_en", "texto"),       # dimension independiente
    ("valor_normalizado", "texto"),   # llave del JOIN
    ("valor_original", "texto"),      # como lo escribio el banco
    ("valor_asignado", "texto"),
    ("confianza", "texto"),
    ("modelo_llm", "texto"),
    ("clasificado_en", "texto"),
]

_cliente = None


def _anthropic():
    """Cliente perezoso: importar este modulo no debe exigir la API key."""
    global _cliente
    if _cliente is None:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "falta ANTHROPIC_API_KEY: el job de clasificacion no puede "
                "correr. La construccion de tablas SI puede: los valores "
                "nuevos quedan 'sin_clasificar' hasta que este job corra."
            )
        _cliente = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _cliente


def clasificar_modelo(destino, esquema_sem: str, modelo, metadata: dict,
                      probar: bool = False, tamano_lote: int = TAMANO_LOTE,
                      forzar: bool = False) -> dict:
    total = {"pendientes": 0, "clasificados": 0, "sin_resolver": 0,
             "llamadas": 0, "alertas": []}
    for campo in (c for c in modelo.campos if c.get("clasifica_en")):
        r = _clasificar_dimension(destino, esquema_sem, modelo, metadata,
                                  campo, probar, tamano_lote, forzar)
        for k in ("pendientes", "clasificados", "sin_resolver", "llamadas"):
            total[k] += r[k]
        total["alertas"] += r["alertas"]
    return total


def _clasificar_dimension(destino, esquema_sem: str, modelo, metadata: dict,
                           campo: dict, probar: bool, tamano_lote: int,
                           forzar: bool) -> dict:
    """
    Clasifica los valores pendientes de un modelo y actualiza la tabla de mapeo.

    Devuelve {"pendientes", "clasificados", "sin_resolver", "llamadas",
              "alertas"}.
    """
    resultado = {"pendientes": 0, "clasificados": 0, "sin_resolver": 0,
                 "llamadas": 0, "alertas": []}

    dimension = campo.get("clasifica_en", "")
    categorias = categorias_de(metadata, modelo.modelo_id, dimension)
    principal = next((c.get("clasifica_en") for c in modelo.campos
                      if c.get("clasifica_en")), "")
    # Las filas historicas sin clasifica_en pertenecen solo al destino
    # principal; no se reutilizan accidentalmente para concepto u otra dimension.
    if dimension != principal:
        categorias = [c for c in categorias if c.get("clasifica_en") == dimension]
    if not categorias:
        msg = (f"[{modelo.modelo_id}] no hay categorias declaradas en la "
               "pestania '_categorias'; no se clasifica nada. Sin un conjunto "
               "cerrado de valores validos, el modelo inventaria categorias "
               "nuevas en cada corrida.")
        logger.warning(msg)
        resultado["alertas"].append(msg)
        return resultado

    mapeo_actual = _leer_mapeo(destino, esquema_sem, modelo.modelo_id,
                               dimension, aceptar_legacy=(dimension == principal))
    pendientes = _pendientes(destino, esquema_sem, modelo, campo,
                             set() if forzar else set(mapeo_actual))
    resultado["pendientes"] = len(pendientes)
    if not pendientes:
        logger.info("[%s] nada pendiente de clasificar.", modelo.modelo_id)
        return resultado

    logger.info("[%s] %d valor(es) sin clasificar; %d lote(s) de %d.",
                modelo.modelo_id, len(pendientes),
                (len(pendientes) + tamano_lote - 1) // tamano_lote, tamano_lote)

    validas = {c["categoria"] for c in categorias if c.get("categoria")}
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    nuevos = {}

    for lote in _lotes(sorted(pendientes.items()), tamano_lote):
        originales = {norm: orig for norm, orig in lote}
        if probar:
            logger.info("[PRUEBA] lote de %d: %s", len(lote),
                        ", ".join(list(originales.values())[:5]))
            resultado["llamadas"] += 1
            continue

        try:
            respuestas = _preguntar(list(originales.values()), categorias)
        except Exception as e:  # noqa: BLE001
            # Un lote que falla no tumba a los demas: los valores de ese lote
            # quedan pendientes y se reintentan en la proxima corrida, que es
            # exactamente el comportamiento deseado de un job idempotente.
            msg = f"[{modelo.modelo_id}] fallo un lote de clasificacion: {e}"
            logger.error(msg)
            resultado["alertas"].append(msg)
            continue

        resultado["llamadas"] += 1
        for original, propuesta in respuestas.items():
            clave = _normalizar(original)
            if clave not in originales:
                # El modelo devolvio un valor que no estaba en el lote. Se
                # ignora: aceptarlo escribiria en el mapeo una entrada que
                # nadie pidio.
                continue
            categoria = str(propuesta.get("categoria", "")).strip()
            if categoria not in validas:
                if categoria and categoria != SIN_CLASIFICAR:
                    logger.info(
                        "[%s] '%s': se descarta la categoria inventada '%s'.",
                        modelo.modelo_id, original, categoria)
                resultado["sin_resolver"] += 1
                continue
            nuevos[clave] = {
                "modelo_id": modelo.modelo_id,
                "clasifica_en": dimension,
                "valor_normalizado": clave,
                "valor_original": originales[clave],
                "valor_asignado": categoria,
                "confianza": str(propuesta.get("confianza", "")).strip(),
                "modelo_llm": config.BOT_MODELO_KPIS,
                "clasificado_en": ahora,
            }

        # Lo que el modelo no devolvio queda pendiente para la proxima; no se
        # inventa nada por completar el lote.
        faltantes = set(originales) - {_normalizar(o) for o in respuestas}
        resultado["sin_resolver"] += len(faltantes)

    resultado["clasificados"] = len(nuevos)
    if nuevos and not probar:
        _guardar_mapeo(destino, esquema_sem, mapeo_actual, nuevos)

    if resultado["sin_resolver"]:
        logger.info(
            "[%s] %d valor(es) siguen sin clasificar; van a aparecer como "
            "'%s' en la tabla derivada hasta que alguien los resuelva con una "
            "regla o un override.",
            modelo.modelo_id, resultado["sin_resolver"], SIN_CLASIFICAR)
    return resultado


# --------------------------------------------------------------------------

def _pendientes(destino, esquema_sem: str, modelo, campo: dict,
                ya_mapeados: set) -> dict:
    """
    {valor_normalizado: valor_original} de lo que falta clasificar.

    Se leen los valores DISTINTOS, no las filas: si 300 transacciones son del
    mismo comercio, al modelo se le manda ese comercio UNA vez. Es la
    diferencia entre 300 llamadas y una, y ademas garantiza que las 300 filas
    reciban la misma categoria.
    """
    pendientes = {}
    columnas = modelo.columnas_de_clasificacion(campo)
    destino_col = campo.get("clasifica_en")
    seleccion = ", ".join(f'"{c}"' for c in columnas)
    sql = (f'SELECT DISTINCT {seleccion} FROM "{esquema_sem}"."'
           f'{modelo.tabla_destino}" WHERE "{destino_col}" = :sc')
    try:
        filas = destino.leer_filas(sql, {"sc": SIN_CLASIFICAR})
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[%s] no se pudo leer la tabla derivada '%s' (%s). Corre "
            "primero construir_modelos.py.",
            modelo.modelo_id, modelo.tabla_destino, e)
        return {}
    for fila in filas:
        original = modelo.valor_clasificacion(fila, campo)
        clave = _normalizar(original)
        if clave and clave not in ya_mapeados:
            pendientes[clave] = original
    return pendientes


def _lotes(items: list, tamano: int):
    """
    Lotes chicos a proposito.

    Es tentador mandar 500 de una, pero los modelos pierden precision en listas
    largas y tienden a apurar el final. Con 25 la respuesta entra comoda y si
    un lote sale mal se pierde poco.
    """
    for i in range(0, len(items), max(1, tamano)):
        yield items[i:i + tamano]


def _preguntar(valores: list, categorias: list) -> dict:
    """Una llamada. Devuelve {valor_original: {"categoria", "confianza"}}."""
    lineas = []
    for c in categorias:
        linea = f"- {c['categoria']}"
        if c.get("descripcion"):
            linea += f": {c['descripcion']}"
        if c.get("ejemplos"):
            linea += f" (ejemplos: {c['ejemplos']})"
        lineas.append(linea)

    prompt = f"""Clasifica cada valor en UNA de estas categorias.

CATEGORIAS VALIDAS:
{chr(10).join(lineas)}

VALORES A CLASIFICAR:
{chr(10).join(f"{i + 1}. {v}" for i, v in enumerate(valores))}

REGLAS:
- Usa SOLO las categorias de la lista, copiadas tal cual. No inventes.
- Si no estas razonablemente seguro, responde "{SIN_CLASIFICAR}". Es una
  respuesta valida y preferible a adivinar.
- Devuelve TODOS los valores, con el texto del valor EXACTAMENTE como se te
  dio.
- Valores que sean el mismo negocio con distinto nombre o sucursal van a la
  misma categoria.
- confianza: "alta", "media" o "baja".

Responde SOLO un array JSON, sin texto alrededor ni bloques de codigo:
[{{"valor": "...", "categoria": "...", "confianza": "alta"}}]"""

    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_KPIS,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    texto = "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    if getattr(resp, "stop_reason", "") == "max_tokens":
        # La respuesta quedo cortada: el JSON esta incompleto y parsear la
        # mitad daria un lote a medias sin avisar.
        raise RuntimeError(
            "la respuesta se corto por longitud; reduci el tamano del lote.")

    texto = texto.strip().removeprefix("```json").removeprefix("```")
    texto = texto.removesuffix("```").strip()
    datos = json.loads(texto)
    if not isinstance(datos, list):
        raise RuntimeError("la respuesta no es un array JSON")

    # Se indexa POR VALOR y no por posicion. Si el modelo se saltara un
    # elemento, un mapeo posicional correria todo lo demas y asignaria
    # categorias equivocadas a valores que nunca se revisaron.
    salida = {}
    for item in datos:
        if isinstance(item, dict) and item.get("valor"):
            salida[str(item["valor"]).strip()] = item
    return salida


def _leer_mapeo(destino, esquema_sem: str, modelo_id: str,
                 clasifica_en: str = "", aceptar_legacy: bool = False) -> dict:
    sql = (f'SELECT * FROM "{esquema_sem}"."{TABLA_MAPEO}" '
           "WHERE modelo_id = :m")
    try:
        filas = destino.leer_filas(sql, {"m": modelo_id})
        return {f["valor_normalizado"]: dict(f) for f in filas
                if (f.get("clasifica_en", "") == clasifica_en or
                    (not f.get("clasifica_en") and
                     (not clasifica_en or aceptar_legacy)))}
    except Exception:  # noqa: BLE001
        return {}      # primera corrida: la tabla todavia no existe


def _guardar_mapeo(destino, esquema_sem: str, actual: dict, nuevos: dict):
    """
    Reescribe la tabla de mapeo con lo viejo mas lo nuevo.

    Se lee todo, se fusiona y se reescribe entero en vez de hacer INSERT: la
    tabla tiene tantas filas como comercios distintos (cientos, no millones),
    y asi el destino no necesita un metodo de UPSERT mas. Como contrapartida,
    dos corridas simultaneas podrian pisarse; es un cron y no corre en
    paralelo consigo mismo.
    """
    # Se leen TODOS los modelos, no solo el que se acaba de clasificar, porque
    # la reescritura es de la tabla completa y perderiamos los demas.
    try:
        todos = {(f["modelo_id"], f.get("clasifica_en", ""),
                  f["valor_normalizado"]): dict(f)
                 for f in destino.leer_filas(
                     f'SELECT * FROM "{esquema_sem}"."{TABLA_MAPEO}"')}
    except Exception:  # noqa: BLE001
        todos = {}
    for fila in actual.values():
        todos[(fila["modelo_id"], fila.get("clasifica_en", ""),
               fila["valor_normalizado"])] = fila
    for fila in nuevos.values():
        todos[(fila["modelo_id"], fila.get("clasifica_en", ""),
               fila["valor_normalizado"])] = fila

    nombres = [c for c, _ in COLUMNAS_MAPEO]
    filas = [{c: f.get(c) for c in nombres} for f in todos.values()]
    destino.reconstruir_tabla(esquema_sem, TABLA_MAPEO, COLUMNAS_MAPEO, filas)
    logger.info("mapeo %s.%s: %d entradas (%d nuevas)",
                esquema_sem, TABLA_MAPEO, len(filas), len(nuevos))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def clasificar_cliente(destino, cliente: dict, probar=False,
                       tamano_lote=TAMANO_LOTE, forzar=False) -> dict:
    cid = cliente["cliente_id"]
    metadata = leer_metadata(cliente)
    total = {"pendientes": 0, "clasificados": 0, "sin_resolver": 0,
             "llamadas": 0, "alertas": []}
    if not metadata["modelos"]:
        return total

    from .construir import nombre_esquema_semantico
    esquema_sem = nombre_esquema_semantico(cid)
    for fila in metadata["modelos"]:
        try:
            modelo = Modelo(fila, metadata)
        except RuntimeError as e:
            logger.error("[%s] modelo '%s' invalido: %s",
                         cid, fila.get("modelo_id"), e)
            continue
        if not modelo.valores_a_clasificar():
            continue        # el modelo no clasifica nada; no es un error

        r = clasificar_modelo(destino, esquema_sem, modelo, metadata,
                              probar=probar, tamano_lote=tamano_lote,
                              forzar=forzar)
        for k in ("pendientes", "clasificados", "sin_resolver", "llamadas"):
            total[k] += r[k]
        total["alertas"] += r["alertas"]
    return total


def main():
    ap = argparse.ArgumentParser(
        description="Clasifica valores nuevos con el LLM y llena el mapeo.")
    ap.add_argument("--cliente", help="cliente_id; por defecto, todos")
    ap.add_argument("--probar", action="store_true",
                    help="muestra los lotes sin llamar al LLM ni escribir")
    ap.add_argument("--forzar", action="store_true",
                    help="reclasifica TODO, incluso lo ya mapeado (cuesta "
                         "dinero: usalo cuando cambien las categorias)")
    ap.add_argument("--lote", type=int, default=TAMANO_LOTE,
                    help=f"valores por llamada (por defecto {TAMANO_LOTE})")
    ap.add_argument("--y-construir", action="store_true",
                    help="reconstruye las tablas derivadas al terminar, para "
                         "que el mapeo nuevo se refleje de una vez")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    clientes = [c for c in registry.listar_clientes()
                if not args.cliente or c["cliente_id"] == args.cliente]
    if not clientes:
        logger.error("no hay clientes que coincidan con %s", args.cliente)
        return 1

    alertas = []
    for cliente in clientes:
        destino = crear_destino(config.WAREHOUSE_TIPO,
                                config.dsn_de_cliente(cliente))
        try:
            r = clasificar_cliente(destino, cliente, args.probar,
                                   args.lote, args.forzar)
            alertas += r["alertas"]
            if r["pendientes"]:
                logger.info(
                    "[%s] %d pendiente(s): %d clasificados, %d sin resolver, "
                    "%d llamada(s) al LLM",
                    cliente["cliente_id"], r["pendientes"], r["clasificados"],
                    r["sin_resolver"], r["llamadas"])

            if args.y_construir and r["clasificados"] and not args.probar:
                from .construir import construir_cliente
                logger.info("[%s] reconstruyendo con el mapeo actualizado...",
                            cliente["cliente_id"])
                construir_cliente(destino, cliente)
        finally:
            destino.cerrar()

    if alertas:
        logger.warning("--- %d ALERTA(S) ---", len(alertas))
        for a in alertas:
            logger.warning("  %s", a)
        # Una alerta significa que al menos una parte de la clasificacion no
        # se completo (por ejemplo, fallo el SDK/API o faltan categorias).
        # Devolver exito hacia que Render mostrara "finished successfully"
        # aunque quedaran valores pendientes sin procesar.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
