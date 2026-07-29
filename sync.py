"""
FASE 2 — Job de ingesta (corre FUERA del request de WhatsApp).

Para cada cliente del registro, para cada fuente ACTIVA:
    1. Revisa frescura: si la ultima corrida OK es mas reciente que
       'frescura_minutos', se OMITE (no se gasta llamada de API).
    2. Carga la fuente con el connector existente (sources/) en una DuckDB
       temporal en memoria.
    3. Drena esas tablas hacia el warehouse, agregando columnas de linaje.
    4. Registra la corrida en _meta.sync_corridas (exito o error).

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
    """Sincroniza UNA fuente de UN cliente hacia el warehouse."""
    cid = cliente["cliente_id"]
    corrida = Corrida(
        corrida_id=uuid.uuid4().hex[:12],
        cliente_id=cid,
        fuente_id=fuente["fuente_id"],
        tipo=fuente.get("tipo", ""),
        inicio=ahora_utc(),
    )

    if not forzar and not probar and _esta_fresca(destino, cid, fuente):
        corrida.estado = "omitido"
        corrida.fin = ahora_utc()
        logger.info("[%s/%s] fresca todavia, se omite", cid, fuente["fuente_id"])
        return corrida

    tmp = duckdb.connect(database=":memory:")
    try:
        obj = crear_fuente(fuente["tipo"], fuente["fuente_id"], fuente.get("config", {}))
        frag = obj.cargar(tmp)

        esquema = nombre_esquema(cid)
        previo = {} if probar else destino.ultimo_detalle(cid, fuente["fuente_id"])

        for tabla in frag.tablas:
            df = tmp.execute(f"SELECT * FROM {tabla}").df()
            columnas = list(df.columns)
            destino_tabla = nombre_tabla(fuente["fuente_id"], tabla)

            alertas, bloquear = _revisar_calidad(previo, destino_tabla, columnas, len(df))
            corrida.alertas += alertas
            for a in alertas:
                logger.warning("[%s/%s] %s", cid, fuente["fuente_id"], a)

            if bloquear:
                continue  # se conserva la tabla anterior; no se pisa con vacio

            corrida.detalle[destino_tabla] = {"columnas": columnas, "filas": len(df)}

            if probar:
                logger.info(
                    "[PRUEBA] %s.%s -> %d filas, columnas: %s",
                    esquema, destino_tabla, len(df), ", ".join(columnas),
                )
            else:
                destino.escribir_tabla(
                    esquema, destino_tabla, agregar_trazabilidad(df, corrida)
                )
            corrida.tablas.append(f"{esquema}.{destino_tabla}")
            corrida.filas += len(df)

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


def sincronizar_todo(cliente_filtro=None, forzar=False, probar=False):
    """Recorre el registro completo y sincroniza lo que corresponda."""
    destinos = _Destinos(config.WAREHOUSE_TIPO)
    resumen = {"ok": 0, "ok_con_alertas": 0, "error": 0, "omitido": 0, "filas": 0, "alertas": []}

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

            for fuente in activas:
                corrida = sincronizar_fuente(destino, cliente, fuente,
                                             forzar=forzar, probar=probar)
                if not probar:
                    destino.registrar_corrida(corrida)
                resumen["alertas"] += corrida.alertas
                resumen[corrida.estado] = resumen.get(corrida.estado, 0) + 1
                resumen["filas"] += corrida.filas
    finally:
        destinos.cerrar()

    logger.info(
        "RESUMEN: %d ok, %d con alertas, %d error, %d omitidas, %d filas",
        resumen["ok"], resumen["ok_con_alertas"], resumen["error"],
        resumen["omitido"], resumen["filas"],
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
