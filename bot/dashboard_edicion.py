"""Reclasificación confirmada desde el dashboard.

El dashboard nunca altera la tabla canónica: para un movimiento derivado crea
un override en la metadata del cliente y reconstruye las tablas semánticas.
Así la decisión sobrevive nuevas ingestas y queda auditable en Google Sheets.
"""

from __future__ import annotations

import logging
import re

import config
from bot import catalogo, dashboard, edicion, escritura_google_sheets, warehouse_ro
from gclient import abrir_libro_escritura
from modelo import metadata
from modelo.construir import construir_cliente
from warehouse import crear_destino
import sync

logger = logging.getLogger("fachavi.bot.dashboard_edicion")


class ErrorReclasificacion(ValueError):
    """Error seguro que se puede mostrar al usuario del dashboard."""


_VALOR_SEGURO = re.compile(r"^[\w.:-]{1,300}$", re.UNICODE)


def _identificador(valor: str) -> str:
    return '"' + str(valor).replace('"', '""') + '"'


def _movimiento(cliente: dict, clave: str) -> tuple[dict, object]:
    ctx = catalogo.construir_contexto(cliente)
    tabla = next((t for t in ctx.permitidas
                  if str(t.tabla_logica).strip().lower() == "movimientos"), None)
    if not tabla:
        raise ErrorReclasificacion("este dashboard no tiene movimientos canónicos editables")
    columnas = {str(c).lower() for c in tabla.columnas_config}
    requeridas = {"_clave", "_modelo_id", "fuente", "clave_origen", "linea_presupuesto_id"}
    if not requeridas.issubset(columnas):
        raise ErrorReclasificacion("este origen todavía no admite reclasificación desde el dashboard")
    filas = warehouse_ro.leer_interno(
        cliente,
        f"SELECT \"_clave\", \"_modelo_id\", fuente, clave_origen, linea_presupuesto_id "
        f"FROM {_identificador(tabla.tabla_real)} WHERE \"_clave\" = :clave LIMIT 1",
        {"clave": clave},
    )
    if not filas:
        raise ErrorReclasificacion("no encontré ese movimiento o ya cambió")
    return filas[0], ctx


def _validar_linea(cliente: dict, ctx, linea_id: str) -> dict:
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
    filas = warehouse_ro.leer_interno(
        cliente,
        f"SELECT CAST(linea_id AS text) AS linea_id, CAST(categoria AS text) AS categoria, "
        f"CAST(concepto AS text) AS concepto FROM {_identificador(presupuesto.tabla_real)} "
        "WHERE CAST(linea_id AS text) = :linea" + tipo + " LIMIT 1",
        {"linea": linea_id},
    )
    if not filas:
        raise ErrorReclasificacion("esa línea presupuestaria no es válida")
    return filas[0]


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
                                  linea_id: str) -> None:
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
    if clave_columna not in encabezados or "linea_presupuesto_id" not in encabezados:
        raise ErrorReclasificacion("la hoja manual no tiene las columnas requeridas")
    indice_clave = encabezados.index(clave_columna)
    for numero, fila in enumerate(hoja.get_all_values()[1:], start=2):
        if len(fila) > indice_clave and str(fila[indice_clave]).strip() == clave:
            hoja.update_cell(numero, encabezados.index("linea_presupuesto_id") + 1, linea_id)
            return
    raise ErrorReclasificacion("no encontré el registro manual que desea reclasificar")


def _sincronizar_fuente_manual(cliente: dict) -> None:
    resumen = sync.sincronizar_todo(cliente_filtro=cliente["cliente_id"], forzar=True)
    if resumen.get("error") or resumen.get("ok_con_bloqueo"):
        raise ErrorReclasificacion(
            "guardé la clasificación manual, pero no pude sincronizarla todavía"
        )


def _guardar_override(cliente: dict, modelo_id: str, clave: str, linea_id: str,
                      nota: str) -> None:
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
                and valor("columna") == "linea_presupuesto_id"):
            hoja.update_cell(numero, indice["valor"] + 1, linea_id)
            hoja.update_cell(numero, indice["nota"] + 1, nota)
            return
    hoja.append_row(
        [modelo_id, clave, "linea_presupuesto_id", linea_id, nota],
        value_input_option="USER_ENTERED",
    )


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


def reclasificar(token: str, movimiento_clave: object, linea_id: object) -> dict:
    """Valida, persiste y reconstruye una reclasificación puntual."""
    _, cliente = dashboard.validar_enlace(token)
    clave = str(movimiento_clave or "").strip()
    linea = str(linea_id or "").strip()
    if not _VALOR_SEGURO.fullmatch(clave) or not _VALOR_SEGURO.fullmatch(linea):
        raise ErrorReclasificacion("la solicitud de edición no es válida")
    movimiento, ctx = _movimiento(cliente, clave)
    destino = _validar_linea(cliente, ctx, linea)
    datos = metadata.leer(cliente)
    capa, origen = _modelo_origen(datos, movimiento)
    clave_origen = str(movimiento.get("clave_origen", "")).strip()
    if not _VALOR_SEGURO.fullmatch(clave_origen):
        raise ErrorReclasificacion("el movimiento no tiene una identidad estable para corregirse")
    if capa == "semantic":
        _guardar_override(
            cliente, str(origen.get("modelo_id", "")).strip(), clave_origen, linea,
            f"Reclasificado desde dashboard: {destino['categoria']} > {destino['concepto']}",
        )
    else:
        _actualizar_movimiento_manual(cliente, origen, clave_origen, linea)
        _sincronizar_fuente_manual(cliente)
    _reconstruir(cliente)
    dashboard.invalidar_cache(str(cliente.get("cliente_id", "")))
    return {
        "ok": True,
        "linea_id": destino["linea_id"],
        "categoria": destino["categoria"],
        "concepto": destino["concepto"],
    }


def _politica_creacion_manual(cliente: dict) -> edicion.PoliticaEdicion:
    politica = edicion.politica_para(cliente, "gastos_manuales")
    if not politica or "crear" not in politica.acciones:
        raise ErrorReclasificacion("este dashboard no tiene gastos manuales habilitados para crear")
    if politica.origen_tipo != "google_sheets":
        raise ErrorReclasificacion("el origen de gastos manuales no admite creación desde el dashboard")
    return politica


def crear_movimiento(token: str, valores: object) -> dict:
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
        destino = _validar_linea(cliente, ctx, linea)
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
    _sincronizar_fuente_manual(cliente)
    _reconstruir(cliente)
    dashboard.invalidar_cache(str(cliente.get("cliente_id", "")))
    return {
        "ok": True,
        "movimiento_id": guardado.get("clave", ""),
        "categoria": destino.get("categoria", "") if destino else "",
        "concepto": destino.get("concepto", "") if destino else "",
    }
