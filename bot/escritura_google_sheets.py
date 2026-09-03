"""Adaptador genérico, confirmado y auditable para tablas de Google Sheets.

No conoce clientes, nombres de tablas ni columnas de negocio. Esas decisiones
vienen de ``PoliticaEdicion`` y de la fuente declarada en la metadata.
"""

from __future__ import annotations

from datetime import date
import secrets

from gclient import abrir_libro_escritura
from bot.edicion import PoliticaEdicion


class ErrorEscritura(ValueError):
    """Error visible y seguro para el flujo de confirmación."""


def _fuente(cliente: dict, politica: PoliticaEdicion) -> dict:
    if politica.origen_tipo != "google_sheets":
        raise ErrorEscritura("el origen de esta tabla no es Google Sheets")
    fuente = next((f for f in cliente.get("fuentes", [])
                   if f.get("fuente_id") == politica.origen_fuente_id), None)
    if not fuente or fuente.get("tipo") != "google_sheets":
        raise ErrorEscritura("no encontré la fuente Google Sheets configurada")
    if not fuente.get("activo"):
        raise ErrorEscritura("la fuente de edición está inactiva")
    if not str((fuente.get("config") or {}).get("spreadsheet_id", "")).strip():
        raise ErrorEscritura("la fuente no tiene spreadsheet_id")
    return fuente


def _id_generado(campo, usados: set[str]) -> str:
    if campo.generador != "id_aleatorio_fecha":
        raise ErrorEscritura(f"no conozco el generador seguro '{campo.generador}'")
    prefijo = f"MAN-{date.today():%Y%m%d}-"
    for _ in range(20):
        candidato = prefijo + secrets.token_hex(4).upper()
        if candidato not in usados:
            return candidato
    raise ErrorEscritura("no pude reservar un identificador único; inténtelo otra vez")


def _encabezados(hoja) -> list[str]:
    encabezados = [str(v).strip() for v in hoja.row_values(1)]
    if not encabezados:
        raise ErrorEscritura("la hoja de destino no tiene encabezados")
    return encabezados


def aplicar_confirmado(cliente: dict, politica: PoliticaEdicion, accion: str,
                        valores: dict[str, object]) -> dict:
    """Ejecuta una creación/modificación/anulación ya validada y confirmada.

    Esta es la única función que abre la credencial de escritura. Nunca acepta
    SQL ni un nombre de hoja escrito por el usuario; ambos vienen de metadata.
    """
    fuente = _fuente(cliente, politica)
    libro = abrir_libro_escritura(fuente["config"]["spreadsheet_id"])
    try:
        hoja = libro.worksheet(politica.hoja_origen)
    except Exception as exc:  # gspread no expone una excepción estable en mocks
        raise ErrorEscritura("no encontré la hoja de destino configurada") from exc
    encabezados = _encabezados(hoja)
    faltan = [c for c in politica.campos if c not in encabezados]
    if faltan:
        raise ErrorEscritura("faltan columnas declaradas en la hoja: " + ", ".join(faltan))

    if accion == "crear":
        salida = dict(valores)
        for nombre, campo in politica.campos.items():
            if campo.calculado:
                indice = encabezados.index(nombre) + 1
                usados = {str(v).strip() for v in hoja.col_values(indice) if str(v).strip()}
                salida[nombre] = _id_generado(campo, usados)
        hoja.append_row([salida.get(c, "") for c in encabezados], value_input_option="USER_ENTERED")
        return {"accion": accion, "clave": salida.get(politica.clave_primaria, ""),
                "anterior": {}, "valores": salida}

    clave = str(valores.get(politica.clave_primaria, "")).strip()
    if not clave:
        raise ErrorEscritura("indique el identificador del registro a modificar")
    try:
        celda = hoja.find(clave, in_column=encabezados.index(politica.clave_primaria) + 1)
    except Exception as exc:
        raise ErrorEscritura("no encontré un registro con ese identificador") from exc
    cambios = {k: v for k, v in valores.items() if k in encabezados and k != politica.clave_primaria}
    fila_anterior = hoja.row_values(celda.row)
    anterior = {nombre: fila_anterior[i] if i < len(fila_anterior) else ""
                for i, nombre in enumerate(encabezados)}
    if accion == "anular":
        if not politica.anulacion_campo:
            raise ErrorEscritura("la tabla no define cómo anular registros")
        cambios = {politica.anulacion_campo: "no"}
    for columna, valor in cambios.items():
        hoja.update_cell(celda.row, encabezados.index(columna) + 1, valor)
    return {"accion": accion, "clave": clave, "anterior": anterior, "valores": cambios}
