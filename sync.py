"""
FASE 2 — Job de ingesta (corre FUERA del request de WhatsApp).

Para cada cliente del registro, para cada fuente ACTIVA:
    1. Revisa frescura: si la ultima corrida OK es mas reciente que
       'frescura_minutos', se OMITE (no se gasta llamada de API).
    2. Carga la fuente con el connector existente (sources/) en una DuckDB
       temporal en memoria.
    3. Drena esas tablas hacia el warehouse, agregando columnas de linaje.
    4. Registra la corrida en _meta.sync_corridas (exito, bloqueo o error).

Aislamiento de fallos: si una fuente truena, se registra el error y se sigue
con las demas. Un Sheet mal compartido no debe frenar la ingesta del resto.

Uso:
    python sync.py                      # todos los clientes, respeta frescura
    python sync.py --cliente ferre_a    # solo un cliente
    python sync.py --forzar             # ignora frescura, recarga todo
    python sync.py --estado             # muestra el estado de frescura y sale
"""

import argparse
import logging
import os
import re
import sys
import uuid
from datetime import timedelta

import duckdb

import catalogo_cliente
import config
import registry
from sources import crear_fuente
from warehouse import (
    crear_destino,
    Corrida,
    nombre_esquema,
    nombre_tabla,
    agregar_trazabilidad,
)
from warehouse.base import ahora_utc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("fachavi.sync")


def _esta_fresca(destino, cliente_id, fuente) -> bool:
    """True si la fuente se sincronizo hace menos de 'frescura_minutos'."""
    minutos = int(fuente.get("frescura_minutos", 0) or 0)
    if minutos <= 0:
        return False  # sin politica de frescura -> siempre se sincroniza
    ultima = destino.ultima_corrida_ok(cliente_id, fuente["fuente_id"])
    if not ultima:
        return False
    return (ahora_utc() - ultima) < timedelta(minutes=minutos)


def _revisar_calidad(previo: dict, tabla: str, columnas: list, filas: int):
    """
    Compara la carga actual contra la anterior. Devuelve (alertas, bloquear).

    Dos niveles, a proposito:
      - ALERTA (se escribe igual): drift de esquema, caida fuerte de filas.
        Son cambios que pueden ser legitimos (el cliente agrego una columna).
      - BLOQUEO (NO se escribe): la tabla llego VACIA y antes tenia datos.
        Como la carga es full refresh, escribir 0 filas encima BORRARIA la
        unica copia buena. Mejor quedarse con el dato viejo y gritar.
    """
    alertas, bloquear = [], False
    antes = previo.get(tabla)
    if not antes:
        return alertas, bloquear  # primera vez: no hay con que comparar

    cols_antes = set(antes.get("columnas", []))
    cols_ahora = set(columnas)
    if cols_antes != cols_ahora:
        nuevas = sorted(cols_ahora - cols_antes)
        faltan = sorted(cols_antes - cols_ahora)
        partes = []
        if nuevas:
            partes.append(f"columnas nuevas: {', '.join(nuevas)}")
        if faltan:
            partes.append(f"columnas que desaparecieron: {', '.join(faltan)}")
        alertas.append(f"[{tabla}] schema drift -> " + "; ".join(partes))

    filas_antes = int(antes.get("filas", 0))
    if filas_antes > 0:
        if filas == 0:
            alertas.append(
                f"[{tabla}] llego VACIA y antes tenia {filas_antes} filas -> NO se escribe"
            )
            bloquear = True
        elif filas < filas_antes * 0.5:
            alertas.append(
                f"[{tabla}] las filas cayeron de {filas_antes} a {filas} (mas del 50%)"
            )
    return alertas, bloquear


def sincronizar_fuente(destino, cliente: dict, fuente: dict, forzar=False, probar=False) -> Corrida:
    """
    Sincroniza UNA fuente de UN cliente hacia el warehouse.

    Ya NO escribe catalogo ni KPIs (ver B-40 mas abajo, y sincronizar_todo).
    Un catalogo o un KPI pueden cruzar tablas de VARIAS fuentes del mismo
    cliente (p.ej. runway_inventario junta 'ventas' e 'inventario', que
    podrian venir de fuentes distintas). Escribirlos aca, fuente por fuente,
    hacia imposible que esa fila alguna vez encontrara sus dos tablas juntas
    -- cada fuente solo ve SUS propias tablas en 'frag.tablas'. Por eso el
    catalogo/KPIs se escriben una sola vez por CLIENTE, en sincronizar_todo(),
    despues de que todas sus fuentes activas ya cargaron.
    """
    cid = cliente["cliente_id"]
    corrida = Corrida(
        corrida_id=uuid.uuid4().hex[:12],
        cliente_id=cid,
        fuente_id=fuente["fuente_id"],
        tipo=fuente.get("tipo", ""),
        inicio=ahora_utc(),
    )

    # La frescura controla si se RE-ESCRIBEN las TABLAS DE DATOS (el costo real:
    # full refresh atomico, tablas temporales, logs). Pero el catalogo y los KPIs
    # se reescriben SIEMPRE: son 9 filas, cuestan nada, y son lo que el cliente
    # ajusta con mas frecuencia. Antes, "fresca" hacia un return inmediato que
    # salteaba todo — incluido el catalogo y los KPIs. El cliente editaba un KPI
    # y no pasaba nada hasta que la frescura venciera.
    fresca = (not forzar and not probar
              and _esta_fresca(destino, cid, fuente))

    tmp = duckdb.connect(database=":memory:")
    try:
        obj = crear_fuente(fuente["tipo"], fuente["fuente_id"], fuente.get("config", {}))
        frag = obj.cargar(tmp)

        esquema = nombre_esquema(cid)
        previo = {} if probar else destino.ultimo_detalle(cid, fuente["fuente_id"])

        # Alertas que reporta la propia fuente (paginacion truncada, tipos
        # descartados en la inferencia, etc.). Antes se perdian en el log.
        corrida.alertas += list(getattr(frag, "alertas", []))

        bloqueadas = []
        for tabla in frag.tablas:
            df = tmp.execute(f"SELECT * FROM {tabla}").df()
            columnas = list(df.columns)
            destino_tabla = nombre_tabla(fuente["fuente_id"], tabla)

            alertas, bloquear = _revisar_calidad(previo, destino_tabla, columnas, len(df))
            corrida.alertas += alertas
            for a in alertas:
                logger.warning("[%s/%s] %s", cid, fuente["fuente_id"], a)

            if bloquear:
                # C-01 — EL ARREGLO MAS IMPORTANTE DEL REPO.
                # Antes este 'continue' salteaba la linea que llena
                # corrida.detalle[destino_tabla]. Resultado: la corrida
                # siguiente no encontraba con que comparar, la guarda no se
                # disparaba y escribia 0 filas encima de los datos buenos. La
                # guarda protegia 15 minutos y despues se desarmaba sola.
                # Copiar el detalle anterior la sostiene indefinidamente.
                corrida.detalle[destino_tabla] = previo[destino_tabla]
                bloqueadas.append(destino_tabla)
                continue  # se conserva la tabla anterior; no se pisa con vacio

            corrida.detalle[destino_tabla] = {"columnas": columnas, "filas": len(df)}

            if fresca:
                # Datos frescos: no se reescribe la tabla, pero se registra que
                # la vimos para que el detalle de la corrida quede completo.
                pass
            elif probar:
                logger.info(
                    "[PRUEBA] %s.%s -> %d filas, columnas: %s",
                    esquema, destino_tabla, len(df), ", ".join(columnas),
                )
            else:
                destino.escribir_tabla(
                    esquema, destino_tabla, agregar_trazabilidad(df, corrida)
                )
            corrida.tablas.append(f"{esquema}.{destino_tabla}")
            corrida.tablas_logicas.add(tabla)
            corrida.filas += len(df)

        # Estado propio cuando la guarda bloqueo alguna escritura. Sigue
        # empezando en "ok" a proposito: las consultas de frescura y de detalle
        # filtran por estado LIKE 'ok%', y la corrida SI fue exitosa (la tabla
        # vieja, buena, quedo intacta). El valor propio es para poder auditar:
        #   SELECT * FROM _meta.sync_corridas WHERE estado = 'ok_con_bloqueo';
        if fresca:
            corrida.estado = "omitido"
            logger.info(
                "[%s/%s] datos frescos (se omite reescritura); catalogo y KPIs "
                "actualizados", cid, fuente["fuente_id"],
            )
        elif bloqueadas:
            corrida.estado = "ok_con_bloqueo"
            logger.error(
                "[%s/%s] BLOQUEO de escritura en %d tabla(s): %s. Se conservaron "
                "los datos anteriores. Revisa la fuente (pestania renombrada, "
                "filtro puesto, hoja borrada).",
                cid, fuente["fuente_id"], len(bloqueadas), ", ".join(bloqueadas),
            )
        else:
            corrida.estado = "ok_con_alertas" if corrida.alertas else "ok"

        logger.info(
            "[%s/%s] %s: %d tablas, %d filas",
            cid, fuente["fuente_id"], corrida.estado.upper(),
            len(corrida.tablas), corrida.filas,
        )
    except Exception as e:  # noqa: BLE001
        corrida.estado = "error"
        corrida.error = f"{type(e).__name__}: {e}"
        logger.exception("[%s/%s] FALLO: %s", cid, fuente["fuente_id"], e)
    finally:
        tmp.close()
        corrida.fin = ahora_utc()

    return corrida


class _Destinos:
    """
    Abre un destino por DSN y lo reutiliza. Si varios clientes comparten el
    mismo proyecto de Neon, comparten conexion; si cada uno tiene el suyo,
    se abre una por proyecto y se cierran todas al final.
    """

    def __init__(self, tipo):
        self.tipo = tipo
        self._abiertos = {}

    def para(self, cliente: dict):
        dsn = config.dsn_de_cliente(cliente)
        if dsn not in self._abiertos:
            d = crear_destino(self.tipo, dsn)
            d.conectar()
            self._abiertos[dsn] = d
            logger.info("[%s] warehouse: %s", cliente.get("cliente_id"), _oculta(dsn))
        return self._abiertos[dsn]

    def abiertos(self):
        return list(self._abiertos.values())

    def cerrar(self):
        for d in self._abiertos.values():
            try:
                d.cerrar()
            except Exception:  # noqa: BLE001
                pass
        self._abiertos.clear()


def _oculta(dsn: str) -> str:
    """Enmascara la clave del DSN para poder loguearlo sin filtrar el secreto."""
    return re.sub(r"//[^:]+:[^@]+@", "//***:***@", dsn)


# Identificador reservado para el catalogo/KPIs CONSOLIDADOS de un cliente
# (ya no pertenecen a una fuente individual). No puede colisionar con un
# fuente_id real porque los fuente_id vienen de la pestania 'fuentes' del
# Sheet maestro, y esta constante usa un caracter ('_') que ese registro no
# deberia tener al inicio -- misma convencion que las pestanias reservadas.
_FUENTE_ID_CATALOGO_CLIENTE = "_cliente"


def _escribir_catalogo_del_cliente(destino, cliente_id: str, catalogo_filas: list,
                                   kpis_filas: list, tablas_del_cliente: set,
                                   probar: bool) -> None:
    """
    Escribe el catalogo y los KPIs consolidados de UN cliente, UNA vez, contra
    la union de tablas de TODAS sus fuentes (ver B-40 en sincronizar_todo).

    Filtra por tabla, no por fuente: una fila de catalogo/KPI se conserva si
    CADA tabla que menciona (separadas por ';' en la columna 'tabla', como ya
    hace 'runway_inventario' en este mismo proyecto) fue cargada por ALGUNA
    fuente del cliente. Asi un KPI que cruza 'ventas' e 'inventario' se
    escribe si las dos existen, sin importar si vinieron de la misma fuente o
    de dos distintas.

    El 'fuente_id' opcional de una fila (retrocompatible) ya NO se usa para
    filtrar que fuente la escribe -- eso dejo de existir, el catalogo es del
    cliente -- pero se conserva en la fila tal cual viene, por si sirve como
    dato informativo de donde salio originalmente esa documentacion.
    """
    if not tablas_del_cliente:
        return
    cargadas = {t.strip().lower() for t in tablas_del_cliente}

    def tabla_disponible(valor_tabla: str) -> bool:
        nombres = [n.lower() for n in catalogo_cliente.tablas_de(valor_tabla)]
        if not nombres:
            return True  # fila sin tabla especifica (rara) no se descarta por esto
        return all(n in cargadas for n in nombres)

    catalogo_ok = [f for f in (catalogo_filas or []) if tabla_disponible(f.get("tabla"))]
    kpis_ok = [f for f in (kpis_filas or []) if tabla_disponible(f.get("tabla"))]

    esquema = nombre_esquema(cliente_id)

    if catalogo_ok and not probar:
        try:
            destino.escribir_catalogo(esquema, _FUENTE_ID_CATALOGO_CLIENTE, catalogo_ok)
            aplicar = getattr(destino, "aplicar_comentarios", None)
            if aplicar:
                # mapa: nombre logico de la tabla en el catalogo -> tabla real
                # en el warehouse. Con varias fuentes por cliente, el nombre
                # real de cada tabla ya viene con su prefijo fuente__tabla en
                # 'tablas_del_cliente'; se busca cual empieza con el nombre
                # logico de cada fila del catalogo.
                mapa = {}
                for f in catalogo_ok:
                    logica = str(f.get("tabla", "")).strip().lower()
                    for real in tablas_del_cliente:
                        if real.lower() == logica or real.lower().endswith(f"__{logica}"):
                            mapa[logica] = real
                            break
                if mapa:
                    aplicar(esquema, mapa, catalogo_ok)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] no se pudo guardar el catalogo consolidado: %s",
                           cliente_id, e)
    elif catalogo_ok and probar:
        logger.info("[PRUEBA] catalogo consolidado de '%s': %d filas (no se escribe)",
                    cliente_id, len(catalogo_ok))

    if kpis_ok and not probar:
        try:
            destino.escribir_kpis(esquema, _FUENTE_ID_CATALOGO_CLIENTE, kpis_ok)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] no se pudieron guardar los KPIs consolidados: %s",
                           cliente_id, e)
    elif kpis_ok and probar:
        logger.info("[PRUEBA] KPIs consolidados de '%s': %d filas (no se escribe)",
                    cliente_id, len(kpis_ok))

    logger.info("[%s] catalogo/KPIs consolidados: %d/%d filas escritas (de %d/%d totales)",
               cliente_id, len(catalogo_ok), len(kpis_ok),
               len(catalogo_filas or []), len(kpis_filas or []))


def sincronizar_todo(cliente_filtro=None, forzar=False, probar=False):
    """Recorre el registro completo y sincroniza lo que corresponda."""
    destinos = _Destinos(config.WAREHOUSE_TIPO)
    resumen = {"ok": 0, "ok_con_alertas": 0, "ok_con_bloqueo": 0,
               "error": 0, "omitido": 0, "filas": 0, "alertas": []}

    try:
        for cliente in registry.listar_clientes():
            cid = cliente["cliente_id"]
            if cliente_filtro and cid != cliente_filtro:
                continue
            activas = [f for f in cliente.get("fuentes", []) if f.get("activo", True)]
            if not activas:
                logger.warning("[%s] sin fuentes activas", cid)
                continue

            try:
                destino = destinos.para(cliente)
            except Exception as e:  # noqa: BLE001
                logger.exception("[%s] no se pudo abrir su warehouse: %s", cid, e)
                resumen["error"] += 1
                continue

            # Catalogo y KPIs del cliente, leidos UNA VEZ (no una vez por
            # fuente). Un Sheet central por cliente documenta todas sus
            # fuentes juntas. Un fallo al leerlo NO frena la sincronizacion de
            # datos: las tablas se cargan igual, simplemente quedan sin
            # catalogo (y por lo tanto bloqueadas para el bot, fail-closed)
            # hasta la proxima corrida.
            try:
                catalogo_filas, kpis_filas = catalogo_cliente.leer(cliente)
            except Exception as e:  # noqa: BLE001
                logger.exception("[%s] no se pudo leer su catalogo/KPIs centrales: %s",
                                 cid, e)
                catalogo_filas, kpis_filas = [], []

            # B-40 — POR QUE EL CATALOGO SE ESCRIBE DESPUES DE TODAS LAS
            # FUENTES, Y NO POR FUENTE COMO ANTES.
            #
            # Un KPI o una fila de catalogo pueden cruzar tablas de VARIAS
            # fuentes del mismo cliente. El propio 'runway_inventario' de este
            # proyecto es el ejemplo real: su formula_sql hace JOIN entre
            # 'ventas' e 'inventario', que perfectamente pueden venir de dos
            # fuentes distintas (un ERP y un Excel de bodega, por ejemplo).
            #
            # Filtrar el catalogo DENTRO de sincronizar_fuente() -- fuente por
            # fuente -- hacia estructuralmente IMPOSIBLE que esa fila
            # encontrara sus dos tablas juntas: cada fuente solo conoce SUS
            # propias tablas en el momento en que corre. El resultado no era
            # un bug esporadico, era garantizado: CUALQUIER catalogo o KPI que
            # mencionara mas de una tabla se descartaba siempre, sin
            # excepcion, sin importar que tan bien estuviera escrito el Sheet.
            #
            # La solucion es acumular las tablas que TODAS las fuentes de este
            # cliente cargaron en esta corrida, y recien con esa union
            # completa filtrar y escribir el catalogo, UNA vez, al final.
            tablas_del_cliente = set()

            for fuente in activas:
                corrida = sincronizar_fuente(destino, cliente, fuente,
                                             forzar=forzar, probar=probar)
                if not probar:
                    destino.registrar_corrida(corrida)
                resumen["alertas"] += corrida.alertas
                resumen[corrida.estado] = resumen.get(corrida.estado, 0) + 1
                resumen["filas"] += corrida.filas
                # tablas_logicas trae el nombre TAL COMO lo declara el
                # catalogo (sin el prefijo fuente_id__ que usa el warehouse
                # fisicamente) -- ver Corrida.tablas_logicas y B-40.
                tablas_del_cliente.update(corrida.tablas_logicas)
                # Si la fuente estaba fresca (se omitio, no volvio a cargar),
                # sus tablas de la corrida anterior siguen existiendo en el
                # warehouse y el catalogo las tiene que seguir cubriendo. El
                # detalle previo usa CLAVES FISICAS (fuente__tabla); se les
                # quita el prefijo de esta fuente para recuperar el nombre
                # logico y que la comparacion sea consistente.
                if corrida.estado == "omitido":
                    try:
                        previo = destino.ultimo_detalle(cid, fuente["fuente_id"])
                        prefijo = f'{limpiar_fuente_id(fuente["fuente_id"])}__'
                        for clave_fisica in previo:
                            if clave_fisica.startswith(prefijo):
                                tablas_del_cliente.add(clave_fisica[len(prefijo):])
                    except Exception:  # noqa: BLE001
                        pass

            _escribir_catalogo_del_cliente(
                destino, cid, catalogo_filas, kpis_filas, tablas_del_cliente, probar
            )
        # A-04: la bitacora crecia sin limite. Se purga al final de la corrida
        # completa (barato: una sola sentencia, y solo si hay retencion puesta).
        if not probar:
            for d in destinos.abiertos():
                try:
                    purgar = getattr(d, "purgar_corridas", None)
                    if purgar:
                        purgar()
                except Exception as e:  # noqa: BLE001
                    logger.warning("no se pudo purgar la bitacora: %s", e)
    finally:
        destinos.cerrar()

    logger.info(
        "RESUMEN: %d ok, %d con alertas, %d con BLOQUEO, %d error, %d omitidas, %d filas",
        resumen["ok"], resumen["ok_con_alertas"], resumen["ok_con_bloqueo"],
        resumen["error"], resumen["omitido"], resumen["filas"],
    )
    if resumen["ok_con_bloqueo"]:
        logger.error(
            "--- %d FUENTE(S) CON ESCRITURA BLOQUEADA: llegaron vacias y se "
            "conservo el dato anterior. Hay que revisarlas HOY. ---",
            resumen["ok_con_bloqueo"],
        )
    if resumen["alertas"]:
        logger.warning("--- %d ALERTAS DE CALIDAD ---", len(resumen["alertas"]))
        for a in resumen["alertas"]:
            logger.warning("  %s", a)
    return resumen


def mostrar_estado():
    """Imprime cuando se sincronizo por ultima vez cada fuente."""
    destinos = _Destinos(config.WAREHOUSE_TIPO)
    try:
        print(f"\n{'cliente':<16}{'fuente':<20}{'frescura':<12}{'ultima carga OK':<26}")
        print("-" * 74)
        for cliente in registry.listar_clientes():
            try:
                destino = destinos.para(cliente)
            except Exception as e:  # noqa: BLE001
                print(f"{cliente['cliente_id']:<16}ERROR de warehouse: {e}")
                continue
            for f in cliente.get("fuentes", []):
                if not f.get("activo", True):
                    continue
                ultima = destino.ultima_corrida_ok(cliente["cliente_id"], f["fuente_id"])
                mins = f.get("frescura_minutos", 0) or 0
                print(
                    f"{cliente['cliente_id']:<16}{f['fuente_id']:<20}"
                    f"{(str(mins) + ' min') if mins else 'siempre':<12}"
                    f"{str(ultima) if ultima else 'NUNCA':<26}"
                )
        print()
    finally:
        destinos.cerrar()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Job de ingesta FACHAVI (Fase 2)")
    ap.add_argument("--cliente", help="sincronizar solo este cliente_id")
    ap.add_argument("--forzar", action="store_true", help="ignorar frescura")
    ap.add_argument("--estado", action="store_true", help="ver frescura y salir")
    ap.add_argument("--probar", action="store_true",
                    help="modo prueba: extrae y muestra, pero NO escribe nada")

    # Sin argumentos en la linea de comandos, se leen de la variable SYNC_ARGS.
    # Asi se cambia el modo desde el dashboard de Render sin tocar el codigo
    # ni depender de que el shell expanda variables.
    argv = sys.argv[1:]
    if not argv:
        desde_env = os.environ.get("SYNC_ARGS", "").split()
        if desde_env:
            logger.info("Argumentos tomados de SYNC_ARGS: %s", " ".join(desde_env))
            argv = desde_env

    args = ap.parse_args(argv)

    if args.estado:
        mostrar_estado()
    else:
        if args.probar:
            logger.info("=== MODO PRUEBA: no se escribe nada en el warehouse ===")
        sincronizar_todo(cliente_filtro=args.cliente, forzar=args.forzar,
                         probar=args.probar)
