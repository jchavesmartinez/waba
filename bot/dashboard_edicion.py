"""Reclasificación confirmada desde el dashboard.

Una corrección puntual se guarda como un override en la metadata del cliente.
Una decisión reutilizable se guarda como regla en ``_clasificacion``: comercio
exacto hacia línea presupuestaria. Ambas sobreviven nuevas ingestas y quedan
auditables en Google Sheets; la tabla canónica nunca se edita directamente.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import date

import config
import registry
from bot import catalogo, dashboard, edicion, escritura_google_sheets, warehouse_ro
from gclient import abrir_libro_escritura
from modelo import metadata
from modelo.construir import construir_cliente
from modelo.motor import Modelo
from warehouse import crear_destino
from sqlalchemy import text
import sync

logger = logging.getLogger("fachavi.bot.dashboard_edicion")


class ErrorReclasificacion(ValueError):
    """Error seguro que se puede mostrar al usuario del dashboard."""


_VALOR_SEGURO = re.compile(r"^[\w.:-]{1,300}$", re.UNICODE)
_MEDIO_PAGO_MAXIMO = 120
_ESQUEMA_JOBS = "_bot"
_TABLA_JOBS = "dashboard_edicion_jobs"
_LOCKS_PROCESAMIENTO: dict[str, threading.Lock] = {}
_LOCKS_PROCESAMIENTO_GUARDIA = threading.Lock()


def _identificador(valor: str) -> str:
    return '"' + str(valor).replace('"', '""') + '"'


def _movimiento(cliente: dict, clave: str) -> tuple[dict, object]:
    ctx = catalogo.construir_contexto(cliente)
    tabla = next((t for t in ctx.permitidas
                  if str(t.tabla_logica).strip().lower() == "movimientos"), None)
    # El modelo canónico es una tabla derivada y puede no aparecer en
    # ``_catalogo`` junto con las fuentes ingestadas. Resuélvelo desde la
    # metadata del modelo para que la edición no dependa de duplicar esa fila.
    if not tabla:
        tabla = dashboard.tabla_movimientos_canonicos(cliente, ctx)
    if not tabla:
        raise ErrorReclasificacion("este dashboard no tiene movimientos canónicos editables")
    columnas = {str(c).lower() for c in tabla.columnas_config}
    requeridas = {"_clave", "_modelo_id", "fuente", "clave_origen", "linea_presupuesto_id"}
    if not requeridas.issubset(columnas):
        raise ErrorReclasificacion("este origen todavía no admite reclasificación desde el dashboard")
    columna_medio_pago = (
        "medio_pago" if "medio_pago" in columnas else "NULL::text AS medio_pago"
    )
    extras = []
    for columna in ("descripcion", "concepto", "categoria"):
        if columna in columnas:
            extras.append(f'CAST("{columna}" AS text) AS "{columna}"')
        else:
            extras.append(f"NULL::text AS \"{columna}\"")
    extras_sql = ", " + ", ".join(extras)
    filas = warehouse_ro.leer_interno(
        cliente,
        f"SELECT \"_clave\", \"_modelo_id\", fuente, clave_origen, linea_presupuesto_id, "
        f"{columna_medio_pago}{extras_sql} "
        f"FROM {_identificador(tabla.tabla_real)} WHERE \"_clave\" = :clave LIMIT 1",
        {"clave": clave},
    )
    if not filas:
        raise ErrorReclasificacion("no encontré ese movimiento o ya cambió")
    return filas[0], ctx


def _valor_si(valor: object) -> bool:
    """Interpreta flags de metadata sin obligar a migrar hojas antiguas."""
    return str(valor or "").strip().casefold() in {"1", "true", "si", "sí", "yes"}


def _periodo_token(payload: dict) -> dict:
    """Normaliza el período firmado para validar pagos manuales."""
    return {
        "inicio": str(payload.get("inicio") or ""),
        "fin_exclusivo": str(payload.get("fin") or payload.get("fin_exclusivo") or ""),
    }


def _fecha_en_periodo(valor: object, periodo: dict) -> str:
    """Exige que un pago pertenezca al período que el usuario está viendo."""
    try:
        fecha = date.fromisoformat(str(valor or "").strip())
        inicio = date.fromisoformat(str(periodo.get("inicio") or ""))
        fin = date.fromisoformat(str(periodo.get("fin_exclusivo") or ""))
    except ValueError as exc:
        raise ErrorReclasificacion("indique una fecha de pago válida") from exc
    if not inicio <= fecha < fin:
        raise ErrorReclasificacion("la fecha de pago debe estar dentro del período de este dashboard")
    return fecha.isoformat()


def _validar_linea(cliente: dict, ctx, linea_id: str,
                   periodo: dict | None = None) -> dict:
    presupuesto = next((t for t in ctx.permitidas
                        if str(t.tabla_logica).strip().lower() == "presupuesto"), None)
    if not presupuesto:
        raise ErrorReclasificacion("no encontré el presupuesto para validar la clasificación")
    columnas = {str(c).lower() for c in presupuesto.columnas_config}
    if not {"linea_id", "categoria", "concepto"}.issubset(columnas):
        raise ErrorReclasificacion("el presupuesto no tiene las columnas requeridas")
    tipo = ""
    if "tipo" in columnas:
        tipo = " AND LOWER(COALESCE(CAST(tipo AS text), 'gasto')) = 'gasto'"
    vigencia = ""
    if periodo and {"vigencia_desde", "vigencia_hasta"}.issubset(columnas):
        try:
            date.fromisoformat(str(periodo.get("inicio") or ""))
        except ValueError as exc:
            raise ErrorReclasificacion("el período del dashboard no es válido") from exc
        vigencia = (
            " AND vigencia_desde <= CAST(:fecha_periodo AS DATE)"
            " AND (vigencia_hasta IS NULL OR vigencia_hasta >= CAST(:fecha_periodo AS DATE))"
        )
    pagable = "CAST(pagable AS text) AS pagable" if "pagable" in columnas else "'' AS pagable"
    filas = warehouse_ro.leer_interno(
        cliente,
        f"SELECT CAST(linea_id AS text) AS linea_id, CAST(categoria AS text) AS categoria, "
        f"CAST(concepto AS text) AS concepto, {pagable} FROM {_identificador(presupuesto.tabla_real)} "
        "WHERE CAST(linea_id AS text) = :linea" + tipo + vigencia + " LIMIT 1",
        {"linea": linea_id, "fecha_periodo": str((periodo or {}).get("inicio") or "")},
    )
    if not filas:
        raise ErrorReclasificacion("esa línea presupuestaria no es válida")
    salida = dict(filas[0])
    salida["pagable"] = _valor_si(salida.get("pagable"))
    return salida


def _modelo_origen(datos: dict, movimiento: dict) -> tuple[str, dict]:
    """Resuelve el modelo derivado que produjo una fuente canónica.

    El vínculo viene de `_movimientos_canonicos`; no hay nombres de clientes ni
    de tablas codificados en el backend.
    """
    modelo_canonico = str(movimiento.get("_modelo_id", "")).strip()
    fuente = str(movimiento.get("fuente", "")).strip().casefold()
    cfg = next((fila for fila in datos.get("movimientos_canonicos", [])
                if str(fila.get("modelo_id", "")).strip() == modelo_canonico
                and str(fila.get("fuente", "")).strip().casefold() == fuente), None)
    if not cfg:
        raise ErrorReclasificacion("no encontré la configuración de origen de este movimiento")
    capa = str(cfg.get("capa_origen", "raw")).strip().lower() or "raw"
    if capa == "raw":
        return capa, cfg
    if capa != "semantic":
        raise ErrorReclasificacion("el origen del movimiento no es compatible con esta edición")
    tabla_origen = str(cfg.get("tabla_origen", "")).strip().casefold()
    modelo = next((fila for fila in datos.get("modelos", [])
                   if str(fila.get("tabla_destino", "")).strip().casefold() == tabla_origen), None)
    if not modelo:
        raise ErrorReclasificacion("no encontré el modelo que originó este movimiento")
    return capa, modelo


def _actualizar_movimiento_manual(cliente: dict, cfg: dict, clave: str,
                                  linea_id: str, medio_pago: str) -> str:
    """Actualiza un gasto manual en su hoja fuente, nunca en Neon.

    La pestaña y sus columnas se resuelven desde `_movimientos_canonicos` y la
    fuente registrada. Esto permite el mismo flujo para clientes con otras
    hojas manuales, sin una ruta especial por cliente.
    """
    tabla = str(cfg.get("tabla_origen", "")).strip()
    hoja_nombre = tabla.rsplit("__", 1)[-1]
    clave_columna = str(cfg.get("clave", "")).strip()
    if not hoja_nombre or not clave_columna:
        raise ErrorReclasificacion("la fuente manual no define su hoja o identificador")
    fuente = next((f for f in cliente.get("fuentes", [])
                   if f.get("tipo") == "google_sheets"
                   and hoja_nombre in (f.get("config", {}).get("hojas", []) or [])), None)
    if not fuente:
        raise ErrorReclasificacion("no encontré la hoja fuente de este movimiento manual")
    try:
        hoja = abrir_libro_escritura(fuente["config"]["spreadsheet_id"]).worksheet(hoja_nombre)
    except Exception as exc:  # gspread no expone una excepción estable
        raise ErrorReclasificacion("no pude abrir la hoja del gasto manual") from exc
    encabezados = [str(v).strip() for v in hoja.row_values(1)]
    columna_linea = str(cfg.get("linea_presupuesto_id", "linea_presupuesto_id")).strip()
    columna_medio_pago = str(cfg.get("medio_pago", "medio_pago")).strip()
    requeridas = {clave_columna, columna_linea, columna_medio_pago}
    if not clave_columna or not requeridas.issubset(encabezados):
        raise ErrorReclasificacion("la hoja manual no tiene las columnas requeridas")
    indice_clave = encabezados.index(clave_columna)
    for numero, fila in enumerate(hoja.get_all_values()[1:], start=2):
        if len(fila) > indice_clave and str(fila[indice_clave]).strip() == clave:
            hoja.update_cell(numero, encabezados.index(columna_linea) + 1, linea_id)
            hoja.update_cell(numero, encabezados.index(columna_medio_pago) + 1, medio_pago)
            return str(fuente.get("fuente_id", "")).strip()
    raise ErrorReclasificacion("no encontré el registro manual que desea reclasificar")


def _sincronizar_fuente_manual(cliente: dict, fuente_id: str,
                               entidad: str = "movimiento") -> None:
    """Sincroniza todo el cliente, validando solamente la fuente editada.

    El catálogo se reconstruye de forma consolidada y por ello una sincronía
    completa sigue siendo necesaria. Sin embargo, una falla en otra fuente
    (por ejemplo, correo) no invalida una escritura que Google Sheets ya
    aceptó y cuya propia fuente se cargó correctamente.
    """
    resumen = sync.sincronizar_todo(cliente_filtro=cliente["cliente_id"], forzar=True)
    fuente = str(fuente_id or "").strip()
    corridas = [c for c in resumen.get("fuentes", [])
                if str(c.get("fuente_id", "")).strip() == fuente]
    # Las versiones previas no devolvían el desglose. En ese caso conserva el
    # comportamiento seguro, en vez de asumir que el origen fue sincronizado.
    if not corridas:
        raise ErrorReclasificacion(
            f"guardé {entidad} en su fuente, pero no pude verificar su sincronización"
        )
    estado = str(corridas[-1].get("estado", "error"))
    if estado in {"error", "ok_con_bloqueo"}:
        raise ErrorReclasificacion(
            f"guardé {entidad} en su fuente, pero esa fuente no pudo sincronizarse todavía"
        )


def _guardar_override(cliente: dict, modelo_id: str, clave: str, columna: str,
                      valor_nuevo: str, nota: str) -> None:
    spreadsheet_id = str(cliente.get("catalogo_spreadsheet_id", "")).strip()
    if not spreadsheet_id:
        raise ErrorReclasificacion("este cliente no tiene metadata editable configurada")
    try:
        hoja = abrir_libro_escritura(spreadsheet_id).worksheet(metadata.OVERRIDES_SHEET)
    except Exception as exc:  # gspread no ofrece una jerarquía estable aquí
        raise ErrorReclasificacion("no pude abrir la metadata de clasificaciones") from exc
    encabezados = [str(v).strip() for v in hoja.row_values(1)]
    requeridas = ["modelo_id", "clave", "columna", "valor", "nota"]
    if any(columna not in encabezados for columna in requeridas):
        raise ErrorReclasificacion("la hoja de overrides no tiene el formato requerido")
    filas = hoja.get_all_values()
    indice = {columna: encabezados.index(columna) for columna in requeridas}
    for numero, fila in enumerate(filas[1:], start=2):
        valor = lambda columna: str(fila[indice[columna]]).strip() if len(fila) > indice[columna] else ""
        if (valor("modelo_id") == modelo_id and valor("clave") == clave
                and valor("columna") == columna):
            hoja.update_cell(numero, indice["valor"] + 1, valor_nuevo)
            hoja.update_cell(numero, indice["nota"] + 1, nota)
            return
    hoja.append_row(
        [modelo_id, clave, columna, valor_nuevo, nota],
        value_input_option="USER_ENTERED",
    )


def _campo_medio_pago(datos: dict, movimiento: dict) -> str:
    """Devuelve la columna de origen que metadata expone como medio de pago.

    El contrato canónico siempre se llama ``medio_pago``, pero una fuente puede
    llamarlo ``tarjeta`` u otro nombre. Un override debe escribirse sobre ese
    campo de origen para que sobreviva la siguiente reconstrucción.
    """
    modelo = str(movimiento.get("_modelo_id", "")).strip()
    fuente = str(movimiento.get("fuente", "")).strip().casefold()
    cfg = next((fila for fila in datos.get("movimientos_canonicos", [])
                if str(fila.get("modelo_id", "")).strip() == modelo
                and str(fila.get("fuente", "")).strip().casefold() == fuente), None)
    campo = str((cfg or {}).get("medio_pago", "")).strip()
    if not campo:
        raise ErrorReclasificacion("este origen no tiene un método de pago editable configurado")
    if not _VALOR_SEGURO.fullmatch(campo):
        raise ErrorReclasificacion("el campo de método de pago configurado no es válido")
    return campo


def _fuente_canonica(datos: dict, movimiento: dict) -> dict:
    """Resuelve la fila de ``_movimientos_canonicos`` para una fuente."""
    modelo = str(movimiento.get("_modelo_id", "")).strip()
    fuente = str(movimiento.get("fuente", "")).strip().casefold()
    return next((fila for fila in datos.get("movimientos_canonicos", [])
                 if str(fila.get("modelo_id", "")).strip() == modelo
                 and str(fila.get("fuente", "")).strip().casefold() == fuente), {})


def _regla_por_comercio(cliente: dict, datos: dict, movimiento: dict,
                        origen: dict) -> tuple[str, str, str]:
    """Resuelve una regla semántica ``comercio -> línea presupuestaria``.

    El comercio es la condición estable; la línea elegida por el usuario es
    el concepto presupuestario de destino. No se infiere por nombres: el
    campo que clasifica en ``linea_presupuesto_id`` viene de ``_campos`` y el
    valor exacto se lee de la fila semántica que originó el movimiento.
    """
    try:
        modelo = Modelo(origen, datos)
    except Exception as exc:
        raise ErrorReclasificacion(
            "no pude resolver la metadata de clasificación de este origen"
        ) from exc
    campos = [campo for campo in modelo.campos
              if str(campo.get("clasifica_en", "")).strip() == "linea_presupuesto_id"]
    if len(campos) != 1:
        raise ErrorReclasificacion(
            "este origen no declara una clasificación única por comercio"
        )
    campo = str(campos[0].get("columna", "")).strip()
    fuente = _fuente_canonica(datos, movimiento)
    clave_fuente = str(fuente.get("clave", "")).strip()
    tabla_origen = str(origen.get("tabla_destino", "")).strip()
    # La fuente semántica puede no estar publicada en ``_catalogo`` para el
    # chat: es una entrada interna del modelo canónico, no una tabla que el
    # LLM deba poder consultar. Aquí se autoriza por la relación declarada en
    # ``_movimientos_canonicos`` + ``_modelos`` y por las columnas que Modelo
    # ya validó, no por el catálogo público.
    columnas = {str(nombre).lower() for nombre, _ in modelo.columnas()}
    if (not _VALOR_SEGURO.fullmatch(tabla_origen)
            or not _VALOR_SEGURO.fullmatch(campo)
            or not _VALOR_SEGURO.fullmatch(clave_fuente)
            or campo.lower() not in columnas or clave_fuente.lower() not in columnas):
        raise ErrorReclasificacion(
            "este origen no expone un comercio editable para crear una regla"
        )
    filas = warehouse_ro.leer_interno(
        cliente,
        f"SELECT CAST({_identificador(campo)} AS text) AS comercio "
        f"FROM {_identificador(tabla_origen)} "
        f"WHERE CAST({_identificador(clave_fuente)} AS text) = :clave LIMIT 1",
        {"clave": str(movimiento.get("clave_origen", "")).strip()},
    )
    comercio = str((filas[0] if filas else {}).get("comercio") or "").strip()
    if not comercio:
        raise ErrorReclasificacion(
            "no encontré el comercio de este movimiento para crear la regla"
        )
    return modelo.modelo_id, campo, comercio


def _guardar_regla_clasificacion(cliente: dict, modelo_id: str, campo: str,
                                 comercio: str, linea_id: str) -> None:
    """Crea o actualiza una regla general en la metadata existente.

    ``_clasificacion`` es la fuente de verdad para decisiones reutilizables.
    La regla exacta se evalúa antes del mapeo de IA y al reconstruir corrige
    tanto el historial como compras futuras del mismo comercio.
    """
    spreadsheet_id = str(cliente.get("catalogo_spreadsheet_id", "")).strip()
    if not spreadsheet_id:
        raise ErrorReclasificacion("este cliente no tiene metadata editable configurada")
    try:
        hoja = abrir_libro_escritura(spreadsheet_id).worksheet(metadata.CLASIFICACION_SHEET)
    except Exception as exc:
        raise ErrorReclasificacion("no pude abrir la metadata de clasificación") from exc
    encabezados = [str(valor).strip() for valor in hoja.row_values(1)]
    requeridas = ["modelo_id", "columnas", "patron", "valor", "prioridad", "clasifica_en"]
    if any(columna not in encabezados for columna in requeridas):
        raise ErrorReclasificacion("la hoja de clasificación no tiene el formato requerido")
    indice = {columna: encabezados.index(columna) for columna in requeridas}
    for numero, fila in enumerate(hoja.get_all_values()[1:], start=2):
        obtener = lambda columna: str(fila[indice[columna]]).strip() if len(fila) > indice[columna] else ""
        if (obtener("modelo_id") == modelo_id and obtener("columnas") == campo
                and obtener("patron") == comercio
                and obtener("clasifica_en") == "linea_presupuesto_id"):
            hoja.update_cell(numero, indice["valor"] + 1, linea_id)
            return
    nueva = [""] * len(encabezados)
    nueva[indice["modelo_id"]] = modelo_id
    nueva[indice["columnas"]] = campo
    nueva[indice["patron"]] = comercio
    nueva[indice["valor"]] = linea_id
    # Es una coincidencia exacta elegida explícitamente por la persona. Debe
    # prevalecer sobre reglas generales como ``%SUPERMERCADO%`` (que suelen
    # usar prioridades mayores), sin desplazar un override puntual.
    nueva[indice["prioridad"]] = "0"
    nueva[indice["clasifica_en"]] = "linea_presupuesto_id"
    hoja.append_row(nueva, value_input_option="USER_ENTERED")


def _validar_medio_pago(valor: object) -> str:
    medio = str(valor or "").strip()
    if not medio:
        raise ErrorReclasificacion("indique un método de pago")
    if len(medio) > _MEDIO_PAGO_MAXIMO or any(ord(caracter) < 32 for caracter in medio):
        raise ErrorReclasificacion("el método de pago no es válido")
    return medio


def _reconstruir(cliente: dict) -> None:
    destino = crear_destino(config.WAREHOUSE_TIPO, config.dsn_de_cliente(cliente))
    try:
        resultado = construir_cliente(destino, cliente)
    finally:
        destino.cerrar()
    if not resultado.get("modelos") or resultado.get("alertas"):
        detalle = "; ".join(str(a) for a in resultado.get("alertas", [])[:2])
        raise ErrorReclasificacion(
            "guardé la clasificación, pero no pude reconstruir los movimientos"
            + (f": {detalle}" if detalle else "")
        )


def _motor_jobs(cliente: dict):
    """Abre el mismo Neon del cliente para la cola durable de ediciones."""
    destino = crear_destino(config.WAREHOUSE_TIPO, config.dsn_de_cliente(cliente))
    if not hasattr(destino, "conectar"):
        destino.cerrar()
        raise ErrorReclasificacion(
            "las ediciones en segundo plano requieren un warehouse PostgreSQL"
        )
    return destino, destino.conectar()


def _asegurar_jobs(cx) -> None:
    cx.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{_ESQUEMA_JOBS}"'))
    cx.execute(text(
        f'''CREATE TABLE IF NOT EXISTS "{_ESQUEMA_JOBS}"."{_TABLA_JOBS}" (
            cliente_id TEXT NOT NULL,
            movimiento_clave TEXT NOT NULL,
            version BIGINT NOT NULL DEFAULT 1,
            fuente_id TEXT NOT NULL DEFAULT '',
            estado TEXT NOT NULL DEFAULT 'pendiente',
            creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            actualizado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            bloqueado_en TIMESTAMPTZ,
            ultimo_error TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (cliente_id, movimiento_clave)
        )'''
    ))


def _encolar_reconstruccion(cliente: dict, movimiento_clave: str,
                            fuente_id: str = "") -> int:
    """Persiste una edición pendiente y devuelve su versión creciente.

    La clave única es el movimiento canónico. Dos guardados consecutivos no
    crean dos trabajos que puedan pelear: el segundo incrementa ``version`` y
    reemplaza el trabajo pendiente. El worker solo marca listo la versión que
    efectivamente reconstruyó.
    """
    destino, motor = _motor_jobs(cliente)
    try:
        with motor.begin() as cx:
            _asegurar_jobs(cx)
            fila = cx.execute(text(f'''
                INSERT INTO "{_ESQUEMA_JOBS}"."{_TABLA_JOBS}"
                    (cliente_id, movimiento_clave, fuente_id)
                VALUES (:cliente_id, :movimiento_clave, :fuente_id)
                ON CONFLICT (cliente_id, movimiento_clave) DO UPDATE SET
                    version = "{_TABLA_JOBS}".version + 1,
                    fuente_id = EXCLUDED.fuente_id,
                    estado = 'pendiente',
                    actualizado_en = CURRENT_TIMESTAMP,
                    bloqueado_en = NULL,
                    ultimo_error = ''
                RETURNING version
            '''), {
                "cliente_id": str(cliente.get("cliente_id", "")),
                "movimiento_clave": movimiento_clave,
                "fuente_id": fuente_id,
            }).scalar_one()
            return int(fila)
    finally:
        destino.cerrar()


def _tomar_reconstruccion(cliente: dict) -> dict | None:
    destino, motor = _motor_jobs(cliente)
    try:
        with motor.begin() as cx:
            _asegurar_jobs(cx)
            fila = cx.execute(text(f'''
                WITH siguiente AS (
                    SELECT cliente_id, movimiento_clave
                    FROM "{_ESQUEMA_JOBS}"."{_TABLA_JOBS}"
                    WHERE cliente_id = :cliente_id AND (
                        estado = 'pendiente' OR
                        (estado = 'procesando' AND bloqueado_en < CURRENT_TIMESTAMP - INTERVAL '10 minutes')
                    )
                    ORDER BY actualizado_en
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE "{_ESQUEMA_JOBS}"."{_TABLA_JOBS}" trabajo
                SET estado = 'procesando', bloqueado_en = CURRENT_TIMESTAMP
                FROM siguiente
                WHERE trabajo.cliente_id = siguiente.cliente_id
                  AND trabajo.movimiento_clave = siguiente.movimiento_clave
                RETURNING trabajo.movimiento_clave, trabajo.version, trabajo.fuente_id
            '''), {"cliente_id": str(cliente.get("cliente_id", ""))}).mappings().first()
            return dict(fila) if fila else None
    finally:
        destino.cerrar()


def _terminar_reconstruccion(cliente: dict, trabajo: dict) -> bool:
    """Marca listo únicamente si nadie guardó una versión posterior."""
    destino, motor = _motor_jobs(cliente)
    try:
        with motor.begin() as cx:
            actualizadas = cx.execute(text(f'''
                UPDATE "{_ESQUEMA_JOBS}"."{_TABLA_JOBS}"
                SET estado = 'listo', actualizado_en = CURRENT_TIMESTAMP,
                    bloqueado_en = NULL, ultimo_error = ''
                WHERE cliente_id = :cliente_id AND movimiento_clave = :movimiento_clave
                  AND version = :version AND estado = 'procesando'
            '''), {
                "cliente_id": str(cliente.get("cliente_id", "")),
                "movimiento_clave": trabajo["movimiento_clave"],
                "version": trabajo["version"],
            }).rowcount
            return bool(actualizadas)
    finally:
        destino.cerrar()


def _fallar_reconstruccion(cliente: dict, trabajo: dict, error: Exception) -> None:
    destino, motor = _motor_jobs(cliente)
    try:
        with motor.begin() as cx:
            cx.execute(text(f'''
                UPDATE "{_ESQUEMA_JOBS}"."{_TABLA_JOBS}"
                SET estado = 'error', actualizado_en = CURRENT_TIMESTAMP,
                    bloqueado_en = NULL, ultimo_error = :error
                WHERE cliente_id = :cliente_id AND movimiento_clave = :movimiento_clave
                  AND version = :version AND estado = 'procesando'
            '''), {
                "cliente_id": str(cliente.get("cliente_id", "")),
                "movimiento_clave": trabajo["movimiento_clave"],
                "version": trabajo["version"], "error": str(error)[:500],
            })
    finally:
        destino.cerrar()


def procesar_reconstrucciones(cliente: dict, maximo: int = 20) -> int:
    """Materializa trabajos durables; es seguro invocarlo varias veces."""
    cliente_id = str(cliente.get("cliente_id", ""))
    with _LOCKS_PROCESAMIENTO_GUARDIA:
        candado = _LOCKS_PROCESAMIENTO.setdefault(cliente_id, threading.Lock())
    # La segunda edición queda en Neon como pendiente. No iniciamos otra
    # reconstrucción en paralelo: el trabajo que ya está activo la recogerá
    # al terminar su versión actual, conservando el orden de versiones.
    if not candado.acquire(blocking=False):
        return 0
    procesadas = 0
    try:
        while procesadas < maximo:
            trabajo = _tomar_reconstruccion(cliente)
            if not trabajo:
                return procesadas
            try:
                if str(trabajo.get("fuente_id", "")).strip():
                    _sincronizar_fuente_manual(cliente, str(trabajo["fuente_id"]), "el movimiento")
                _reconstruir(cliente)
            except Exception as exc:  # el cambio ya quedó guardado; el estado queda visible y reintentable
                logger.exception("[%s] no se pudo materializar edición %s", cliente.get("cliente_id"), trabajo.get("movimiento_clave"))
                _fallar_reconstruccion(cliente, trabajo, exc)
                return procesadas
            if _terminar_reconstruccion(cliente, trabajo):
                dashboard.invalidar_cache(cliente_id)
            procesadas += 1
        return procesadas
    finally:
        candado.release()


def procesar_reconstrucciones_cliente(cliente_id: str) -> int:
    cliente = next((c for c in registry.listar_clientes()
                    if str(c.get("cliente_id", "")) == str(cliente_id)), None)
    return procesar_reconstrucciones(cliente) if cliente else 0


def recuperar_reconstrucciones_pendientes() -> None:
    """Al iniciar el web procesa ediciones que sobrevivieron un reinicio."""
    try:
        clientes = registry.listar_clientes()
    except Exception:
        logger.exception("no se pudo leer el registro para recuperar ediciones pendientes")
        return
    for cliente in clientes:
        try:
            procesar_reconstrucciones(cliente)
        except Exception:  # un cliente no debe bloquear la recuperación de los demás
            logger.exception("[%s] no se pudo recuperar ediciones pendientes", cliente.get("cliente_id"))


def estado_reconstruccion(token: str, movimiento_clave: object) -> dict:
    _, cliente = dashboard.validar_enlace(token)
    clave = str(movimiento_clave or "").strip()
    if not _VALOR_SEGURO.fullmatch(clave):
        raise ErrorReclasificacion("la solicitud de edición no es válida")
    destino, motor = _motor_jobs(cliente)
    try:
        with motor.begin() as cx:
            _asegurar_jobs(cx)
            fila = cx.execute(text(f'''
                SELECT estado, version, actualizado_en, ultimo_error
                FROM "{_ESQUEMA_JOBS}"."{_TABLA_JOBS}"
                WHERE cliente_id = :cliente_id AND movimiento_clave = :movimiento_clave
            '''), {"cliente_id": str(cliente.get("cliente_id", "")), "movimiento_clave": clave}).mappings().first()
            if not fila:
                return {"ok": True, "estado": "listo"}
            return {"ok": True, "estado": str(fila["estado"]), "version": int(fila["version"])}
    finally:
        destino.cerrar()


def reclasificar(token: str, movimiento_clave: object, linea_id: object,
                 medio_pago: object = None, alcance: object = "individual",
                 ) -> dict:
    """Aplica un override puntual o una regla general de metadata."""
    _, cliente = dashboard.validar_enlace(token)
    clave = str(movimiento_clave or "").strip()
    linea = str(linea_id or "").strip()
    if not _VALOR_SEGURO.fullmatch(clave) or not _VALOR_SEGURO.fullmatch(linea):
        raise ErrorReclasificacion("la solicitud de edición no es válida")
    movimiento, ctx = _movimiento(cliente, clave)
    destino = _validar_linea(cliente, ctx, linea)
    alcance = str(alcance or "individual").strip().lower()
    # ``grupo`` se conserva como alias durante el despliegue de la nueva UI.
    if alcance == "grupo":
        alcance = "regla"
    if alcance not in {"individual", "regla"}:
        raise ErrorReclasificacion("el alcance de la reclasificación no es válido")
    medio = _validar_medio_pago(
        movimiento.get("medio_pago") if medio_pago is None else medio_pago
    )
    datos = metadata.leer(cliente)
    capa, origen = _modelo_origen(datos, movimiento)
    columna_medio_pago = _campo_medio_pago(datos, movimiento)
    clave_origen = str(movimiento.get("clave_origen", "")).strip()
    if not _VALOR_SEGURO.fullmatch(clave_origen):
        raise ErrorReclasificacion("el movimiento no tiene una identidad estable para corregirse")
    regla = None
    if alcance == "regla":
        if capa != "semantic":
            raise ErrorReclasificacion(
                "este origen no admite reglas generales; reclasifíquelo solo de forma puntual"
            )
        regla = _regla_por_comercio(cliente, datos, movimiento, origen)
    fuente_id = ""
    if capa == "semantic":
        if alcance == "individual":
            _guardar_override(
                cliente, str(origen.get("modelo_id", "")).strip(), clave_origen,
                "linea_presupuesto_id", linea,
                f"Reclasificado desde dashboard: {destino['categoria']} > {destino['concepto']}",
            )
        if medio != str(movimiento.get("medio_pago") or "").strip():
            _guardar_override(
                cliente, str(origen.get("modelo_id", "")).strip(), clave_origen,
                columna_medio_pago, medio,
                "Método de pago actualizado desde dashboard",
            )
    else:
        fuente_id = _actualizar_movimiento_manual(cliente, origen, clave_origen, linea, medio)
    if regla:
        modelo_regla, campo_regla, comercio = regla
        _guardar_regla_clasificacion(
            cliente, modelo_regla, campo_regla, comercio, linea,
        )
    version = _encolar_reconstruccion(
        cliente, clave, fuente_id,
    )
    resultado = {
        "ok": True,
        "linea_id": destino["linea_id"],
        "categoria": destino["categoria"],
        "concepto": destino["concepto"],
        "medio_pago": medio,
        "estado": "pendiente",
        "version": version,
    }
    if alcance == "regla":
        resultado["alcance"] = alcance
        resultado["comercio"] = regla[2]
    return resultado


def _politica_creacion_manual(cliente: dict) -> edicion.PoliticaEdicion:
    politica = edicion.politica_para(cliente, "gastos_manuales")
    if not politica or "crear" not in politica.acciones:
        raise ErrorReclasificacion("este dashboard no tiene gastos manuales habilitados para crear")
    if politica.origen_tipo != "google_sheets":
        raise ErrorReclasificacion("el origen de gastos manuales no admite creación desde el dashboard")
    return politica


def crear_movimiento(token: str, valores: object, *, periodo: dict | None = None) -> dict:
    """Crea un gasto manual desde el dashboard, en su fuente y no en Neon."""
    _, cliente = dashboard.validar_enlace(token)
    if not isinstance(valores, dict):
        raise ErrorReclasificacion("la solicitud de creación no es válida")
    politica = _politica_creacion_manual(cliente)
    entrada = {str(k): v for k, v in valores.items() if isinstance(k, str)}

    # La relación presupuesto es siempre validada en el servidor. El navegador
    # sólo elige entre líneas que ya recibió; no puede inventar una categoría
    # ni un concepto, y la categoría queda coherente con la línea elegida.
    campo_linea = next((c for c in politica.campos.values()
                         if c.generador == "concepto_a_linea_id"), None)
    destino = None
    if campo_linea:
        linea = str(entrada.get(campo_linea.nombre, "")).strip()
        if not _VALOR_SEGURO.fullmatch(linea):
            raise ErrorReclasificacion("seleccione un concepto presupuestario válido")
        ctx = catalogo.construir_contexto(cliente)
        destino = _validar_linea(cliente, ctx, linea, periodo)
        entrada[campo_linea.nombre] = destino["linea_id"]
        if "categoria" in politica.campos:
            entrada["categoria"] = destino["categoria"]

    borrador = edicion.validar_borrador(politica, "crear", entrada)
    if borrador.errores:
        raise ErrorReclasificacion("; ".join(borrador.errores))
    if borrador.faltantes:
        raise ErrorReclasificacion("faltan: " + ", ".join(borrador.faltantes))
    try:
        guardado = escritura_google_sheets.aplicar_confirmado(
            cliente, politica, "crear", borrador.valores)
    except escritura_google_sheets.ErrorEscritura as exc:
        raise ErrorReclasificacion(str(exc)) from exc
    _sincronizar_fuente_manual(cliente, politica.origen_fuente_id, "el movimiento")
    _reconstruir(cliente)
    dashboard.invalidar_cache(str(cliente.get("cliente_id", "")))
    return {
        "ok": True,
        "movimiento_id": guardado.get("clave", ""),
        "categoria": destino.get("categoria", "") if destino else "",
        "concepto": destino.get("concepto", "") if destino else "",
    }


def registrar_pago(token: str, linea_id: object, monto: object, fecha_pago: object) -> dict:
    """Registra un pago como movimiento manual ordinario.

    No existe tabla ni estado de pagos: el gasto y el saldo continúan siendo la
    suma de movimientos asociados a la línea del presupuesto. Esta función solo
    prepara los valores seguros y delega la escritura al flujo normal de
    ``crear_movimiento``.
    """
    payload, cliente = dashboard.validar_enlace(token)
    periodo = _periodo_token(payload)
    fecha = _fecha_en_periodo(fecha_pago, periodo)
    linea = str(linea_id or "").strip()
    if not _VALOR_SEGURO.fullmatch(linea):
        raise ErrorReclasificacion("seleccione un concepto presupuestario válido")
    ctx = catalogo.construir_contexto(cliente)
    destino = _validar_linea(cliente, ctx, linea, periodo)
    if not destino.get("pagable"):
        raise ErrorReclasificacion("este concepto no está habilitado para registrar pagos")

    politica = _politica_creacion_manual(cliente)
    campo_linea = next((campo for campo in politica.campos.values()
                         if campo.generador == "concepto_a_linea_id"), None)
    campo_fecha = next((campo for campo in politica.campos.values()
                        if campo.tipo == "fecha_iso"), None)
    campo_monto = next((campo for campo in politica.campos.values()
                        if campo.tipo == "monto_positivo"), None)
    if not campo_linea or not campo_fecha or not campo_monto:
        raise ErrorReclasificacion("la creación manual no define fecha, monto y concepto")

    valores = {
        campo_linea.nombre: destino["linea_id"],
        campo_fecha.nombre: fecha,
        campo_monto.nombre: monto,
    }
    descripcion = next((campo for campo in politica.campos.values()
                        if campo.nombre.casefold() in {"descripcion", "descripción"}
                        and not campo.calculado), None)
    if descripcion:
        valores[descripcion.nombre] = f"Pago - {destino['concepto']}"
    # ``crear_movimiento`` conserva todas las validaciones, defaults de moneda,
    # escritura en Google Sheets, sincronización y reconstrucción existentes.
    return crear_movimiento(token, valores, periodo=periodo)
