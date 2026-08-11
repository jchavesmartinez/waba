"""
Construccion de las tablas derivadas: lee raw, aplica los modelos, escribe en
semantic_<cliente>.

    python construir_modelos.py                      todos los clientes
    python construir_modelos.py --cliente cliente_a
    python construir_modelos.py --probar             no escribe nada

POR QUE DROP + CREATE Y NO UPSERT. La tabla derivada es una FUNCION PURA de sus
entradas (raw + mapeo + overrides + metadata). Reconstruirla entera es la
operacion natural y ademas la mas barata de razonar: no hay estado incremental,
no hay migraciones y cambiar una regla corrige TODO el historico con solo
volver a correr. Es lo contrario de la ingesta, donde el UPSERT es obligatorio
porque el origen (Meta, el correo) no se puede volver a leer entero.

Y no es caro: el LLM NO participa de esta ruta. El modelo de lenguaje llena la
tabla de mapeo en un job aparte, y esa tabla es una ENTRADA mas. Reconstruir
100.000 filas son cero llamadas al LLM y unos segundos de SQL.

LA REGLA QUE SOSTIENE TODO: este modulo NUNCA escribe en raw_<cliente>. Si
alguna vez hace falta guardar algo que no viene de una entrada (una correccion
a mano), la salida correcta es declararlo como entrada nueva —una pestania mas
que se lee— y no editar la tabla derivada. El dia que se edite a mano, la
propiedad se pierde en silencio y con ella todo lo anterior.
"""

import argparse
import logging
import sys

import catalogo_cliente
import config
from .metadata import leer as leer_metadata
from .motor import Modelo
import registry
from warehouse import crear_destino
from warehouse.base import nombre_esquema

logger = logging.getLogger("fachavi.modelo.construir")

SUFIJO_RECHAZOS = "__rechazos"
TABLA_MAPEO = "_mapeo"
FUENTE_METADATA_SEMANTICA = "_modelo_semantico"


def nombre_esquema_semantico(cliente_id: str) -> str:
    """
    semantic_<cliente_id>, hermano de raw_<cliente_id>.

    Esquema aparte y no un prefijo dentro de raw: asi la reconstruccion puede
    hacer DROP con confianza, porque fisicamente NO PUEDE tocar los datos
    ingestados aunque alguien se equivoque escribiendo un nombre de tabla.
    """
    return "semantic_" + nombre_esquema(cliente_id)[len("raw_"):]


def construir_cliente(destino, cliente: dict, probar: bool = False) -> dict:
    """Construye todas las tablas derivadas de un cliente."""
    cid = cliente["cliente_id"]
    metadata = leer_metadata(cliente)
    esquema_raw = nombre_esquema(cid)
    esquema_sem = nombre_esquema_semantico(cid)
    if not probar:
        destino.asegurar_esquema(esquema_sem)

    total = {"modelos": 0, "filas": 0, "rechazos": 0, "alertas": []}
    tablas_construidas = set()

    if not metadata["modelos"]:
        logger.info("[%s] sin modelos declarados; se limpia su metadata semantica.", cid)

    for fila in metadata["modelos"]:
        try:
            modelo = Modelo(fila, metadata)
        except RuntimeError as e:
            # Un modelo mal declarado no puede tumbar a los demas del mismo
            # cliente: cada uno es independiente y el error dice cual es.
            msg = f"[{cid}] modelo '{fila.get('modelo_id')}' invalido: {e}"
            logger.error(msg)
            total["alertas"].append(msg)
            continue

        try:
            resultado = _construir_modelo(
                destino, cid, esquema_raw, esquema_sem, modelo, probar)
        except Exception as e:  # noqa: BLE001
            msg = (f"[{cid}] fallo la construccion de "
                   f"'{modelo.tabla_destino}': {type(e).__name__}: {e}")
            logger.exception(msg)
            total["alertas"].append(msg)
            continue

        total["modelos"] += 1
        total["filas"] += resultado["filas"]
        total["rechazos"] += resultado["rechazos"]
        total["alertas"] += resultado["alertas"]
        tablas_construidas.add(modelo.tabla_destino)

    _publicar_metadata_semantica(
        destino, cliente, esquema_raw, esquema_sem, tablas_construidas, probar)

    return total


def _publicar_metadata_semantica(destino, cliente: dict, esquema_raw: str,
                                 esquema_sem: str, tablas_reales: set,
                                 probar: bool = False) -> None:
    """Publica catalogo/KPIs de modelos sin copiar sus datos al esquema raw.

    El bot lee el catalogo consolidado desde ``raw_<cliente>._catalogo``, pero
    resuelve cada nombre logico contra las tablas reales de AMBOS esquemas.
    Por eso solo se escriben aqui las filas pequenas de metadata; la tabla de
    negocio sigue viviendo exclusivamente en ``semantic_<cliente>``.
    """
    catalogo, kpis = catalogo_cliente.leer(cliente)
    nombres = _nombres_logicos(tablas_reales)

    def corresponde(valor_tabla) -> bool:
        mencionadas = [t.lower() for t in catalogo_cliente.tablas_de(valor_tabla)]
        return bool(mencionadas) and all(t in nombres for t in mencionadas)

    catalogo_sem = [f for f in catalogo if corresponde(f.get("tabla"))]
    kpis_sem = [f for f in kpis if corresponde(f.get("tabla"))]

    if probar:
        logger.info(
            "[PRUEBA] metadata semantica de '%s': %d filas de catalogo, %d KPIs",
            cliente.get("cliente_id"), len(catalogo_sem), len(kpis_sem))
        return

    # Estas funciones borran solamente las filas de FUENTE_METADATA_SEMANTICA;
    # nunca pisan las publicadas por fachavi-ingesta bajo otras fuentes.
    destino.escribir_catalogo(
        esquema_raw, FUENTE_METADATA_SEMANTICA, catalogo_sem)
    destino.escribir_kpis(esquema_raw, FUENTE_METADATA_SEMANTICA, kpis_sem)

    aplicar = getattr(destino, "aplicar_comentarios", None)
    if aplicar and catalogo_sem:
        mapa = {}
        for real in tablas_reales:
            logico = real.rsplit("__", 1)[-1].lower()
            mapa[logico] = real
            mapa[real.lower()] = real
        aplicar(esquema_sem, mapa, catalogo_sem)

    logger.info(
        "[%s] metadata semantica publicada: %d filas de catalogo, %d KPIs",
        cliente.get("cliente_id"), len(catalogo_sem), len(kpis_sem))


def _nombres_logicos(tablas_reales: set) -> set:
    """Acepta en el Sheet tanto ``transacciones`` como el nombre fisico."""
    nombres = set()
    for tabla in tablas_reales:
        real = str(tabla or "").strip().lower()
        if not real:
            continue
        nombres.add(real)
        nombres.add(real.rsplit("__", 1)[-1])
    return nombres


def _construir_modelo(destino, cid, esquema_raw, esquema_sem, modelo,
                      probar) -> dict:
    alertas = []
    crudas = _leer_origen(destino, esquema_raw, modelo)
    mapeo = {} if probar else _leer_mapeo(destino, esquema_sem, modelo)
    filas, rechazos = modelo.procesar(crudas, mapeo)

    # --- overrides huerfanos ---
    # Un override apunta a una clave. Si esa clave ya no existe —porque cambio
    # el formato del origen, porque se borro el correo o por un typo— la
    # correccion deja de aplicarse SIN QUE NADIE SE ENTERE, y los numeros
    # vuelven a estar mal en silencio. Es el modo de falla mas traicionero de
    # todo este modulo.
    claves = {f["_clave"] for f in filas}
    huerfanos = modelo.claves_de_override() - claves
    if huerfanos:
        msg = (f"[{cid}] el modelo '{modelo.modelo_id}' tiene "
               f"{len(huerfanos)} override(s) que no corresponden a ninguna "
               f"fila y por lo tanto NO se aplicaron: "
               f"{', '.join(sorted(huerfanos)[:5])}"
               f"{' ...' if len(huerfanos) > 5 else ''}")
        logger.warning(msg)
        alertas.append(msg)

    # --- sin clasificar ---
    sin_clasificar = _contar_sin_clasificar(filas, modelo)
    if sin_clasificar:
        msg = (f"[{cid}] '{modelo.tabla_destino}': {sin_clasificar} fila(s) sin "
               "clasificar. Corre el job de clasificacion o agrega reglas en "
               "'_clasificacion'.")
        logger.info(msg)

    if rechazos:
        msg = (f"[{cid}] '{modelo.tabla_destino}': {len(rechazos)} fila(s) "
               f"RECHAZADAS. Revisa {modelo.tabla_destino}{SUFIJO_RECHAZOS} "
               "para ver el motivo de cada una.")
        logger.warning(msg)
        alertas.append(msg)

    if probar:
        logger.info(
            "[PRUEBA] %s.%s -> %d filas, %d rechazos, columnas: %s",
            esquema_sem, modelo.tabla_destino, len(filas), len(rechazos),
            ", ".join(c for c, _ in modelo.columnas()),
        )
        return {"filas": len(filas), "rechazos": len(rechazos),
                "alertas": alertas}

    _escribir(destino, esquema_sem, modelo.tabla_destino,
              modelo.columnas(), filas)
    _escribir(destino, esquema_sem, modelo.tabla_destino + SUFIJO_RECHAZOS,
              modelo.columnas_rechazos(), rechazos)

    logger.info("[%s] %s.%s: %d filas, %d rechazos, %d sin clasificar",
                cid, esquema_sem, modelo.tabla_destino, len(filas),
                len(rechazos), sin_clasificar)
    return {"filas": len(filas), "rechazos": len(rechazos), "alertas": alertas}


def _leer_origen(destino, esquema_raw, modelo) -> list:
    """
    Lee las filas del origen aplicando el filtro declarado.

    Si la tabla de origen no existe todavia se devuelve vacio con una
    excepcion explicita: pasa cuando se declara un modelo antes de conectar su
    fuente, y el mensaje tiene que decir que tabla falta.
    """
    sql = modelo.sql_origen(esquema_raw)
    try:
        return destino.leer_filas(sql)
    except Exception as e:  # noqa: BLE001
        # El SQL entra en el mensaje a proposito: el filtro lo escribio una
        # persona en el Sheet y el error mas comun es justamente ese texto.
        raise RuntimeError(f"no se pudo leer el origen. SQL: {sql} -- {e}") from e


def _leer_mapeo(destino, esquema_sem, modelo) -> dict:
    """
    Lee {valor_normalizado: valor_asignado} que llena el job de clasificacion.

    Ausente = diccionario vacio, sin error: la primera corrida siempre pasa por
    aca antes de que el job haya escrito nada, y eso no es un problema (las
    filas salen 'sin_clasificar', que es la respuesta honesta).
    """
    sql = (f'SELECT valor_normalizado, valor_asignado FROM "{esquema_sem}"."'
           f'{TABLA_MAPEO}" WHERE modelo_id = :m')
    try:
        filas = destino.leer_filas(sql, {"m": modelo.modelo_id})
        return {f["valor_normalizado"]: f["valor_asignado"] for f in filas
                if f.get("valor_normalizado")}
    except Exception:  # noqa: BLE001
        logger.debug("todavia no hay tabla de mapeo en %s", esquema_sem)
        return {}


def _escribir(destino, esquema, tabla, columnas, filas):
    """
    Delega en el destino, que es quien sabe su dialecto.

    El SQL no vive aca a proposito: DuckDB expone una conexion nativa y
    Postgres un engine de SQLAlchemy, y mezclar los dos en el orquestador fue
    exactamente el error que hubo que corregir la primera vez.
    """
    destino.reconstruir_tabla(esquema, tabla, columnas, filas)


def _contar_sin_clasificar(filas, modelo) -> int:
    from modelo.motor import SIN_CLASIFICAR
    destinos = {c.get("clasifica_en") for c in modelo.campos
                if c.get("clasifica_en")}
    return sum(1 for f in filas
               for d in destinos if f.get(d) == SIN_CLASIFICAR)


def main():
    ap = argparse.ArgumentParser(
        description="Construye las tablas derivadas (capa semantica).")
    ap.add_argument("--cliente", help="cliente_id; por defecto, todos")
    ap.add_argument("--probar", action="store_true",
                    help="extrae y muestra, pero NO escribe en la base")
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
        # Mismo tipo de destino que usa la ingesta: la capa semantica escribe
        # en el MISMO warehouse del cliente, solo que en otro esquema.
        destino = crear_destino(config.WAREHOUSE_TIPO,
                                config.dsn_de_cliente(cliente))
        try:
            r = construir_cliente(destino, cliente, args.probar)
            alertas += r.get("alertas", [])
            logger.info("[%s] %d modelo(s), %d filas, %d rechazos",
                        cliente["cliente_id"], r["modelos"], r["filas"],
                        r["rechazos"])
        finally:
            destino.cerrar()

    if alertas:
        logger.warning("--- %d ALERTA(S) ---", len(alertas))
        for a in alertas:
            logger.warning("  %s", a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
