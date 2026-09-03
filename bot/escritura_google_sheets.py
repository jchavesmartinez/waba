"""Adaptador genérico, confirmado y auditable para tablas de Google Sheets.

No conoce clientes, nombres de tablas ni columnas de negocio. Esas decisiones
vienen de ``PoliticaEdicion`` y de la fuente declarada en la metadata.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import unicodedata
import secrets

from gclient import abrir_libro, abrir_libro_escritura
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


def _comparable(valor: object) -> str:
    return " ".join(unicodedata.normalize("NFKD", str(valor or "")).encode("ascii", "ignore").decode().casefold().split())


def _coincide(real: object, buscado: object, campo: str) -> bool:
    a, b = str(real or "").strip(), str(buscado or "").strip()
    if not a or not b:
        return False
    if campo in {"monto", "monto_moneda"}:
        try:
            return Decimal(a.replace(".", "").replace(",", ".")) == Decimal(b)
        except (InvalidOperation, ValueError):
            pass
    if campo in {"fecha", "fecha_transaccion"}:
        return a[:10] == b[:10]
    normal_a, normal_b = _comparable(a), _comparable(b)
    return normal_a == normal_b or normal_b in normal_a


def _resolver_referencias(libro, politica: PoliticaEdicion,
                          valores: dict[str, object]) -> dict[str, object]:
    """Convierte referencias legibles declaradas por metadata a sus llaves.

    Un campo con generador ``concepto_a_linea_id`` acepta el concepto que ve el
    cliente y guarda el ``linea_id`` estable del origen presupuestario. La hoja
    de referencia se descubre por sus encabezados, sin asumir un nombre de
    pestaña ni posiciones fijas.
    """
    salida = dict(valores)
    for nombre, campo in politica.campos.items():
        if campo.generador != "concepto_a_linea_id" or not salida.get(nombre):
            continue
        candidatos = []
        for hoja_ref in getattr(libro, "worksheets", lambda: [])():
            try:
                encabezados = _encabezados(hoja_ref)
            except Exception:
                continue
            normalizados = {_comparable(h): h for h in encabezados}
            col_concepto = normalizados.get("concepto")
            col_linea = normalizados.get("linea id") or normalizados.get("linea_id")
            if not col_concepto or not col_linea:
                continue
            filas = hoja_ref.get_all_values()
            i_concepto, i_linea = encabezados.index(col_concepto), encabezados.index(col_linea)
            for fila in filas[1:]:
                concepto = fila[i_concepto] if i_concepto < len(fila) else ""
                linea = fila[i_linea] if i_linea < len(fila) else ""
                if concepto and linea and _coincide(concepto, salida[nombre], "texto"):
                    candidatos.append(str(linea).strip())
        candidatos = list(dict.fromkeys(candidatos))
        if len(candidatos) == 1:
            salida[nombre] = candidatos[0]
        elif not candidatos:
            raise ErrorEscritura(
                f"no encontré el concepto presupuestario '{salida[nombre]}'")
        else:
            raise ErrorEscritura(
                f"el concepto presupuestario '{salida[nombre]}' es ambiguo")
    return salida


def buscar_registros(cliente: dict, politica: PoliticaEdicion,
                     criterios: dict[str, object]) -> list[dict[str, str]]:
    """Busca candidatos en el origen sin exponer la llave técnica al cliente."""
    fuente = _fuente(cliente, politica)
    libro = abrir_libro(fuente["config"]["spreadsheet_id"])
    try:
        hoja = libro.worksheet(politica.hoja_origen)
    except Exception as exc:
        raise ErrorEscritura("no encontré la hoja de origen configurada") from exc
    encabezados = _encabezados(hoja)
    filas = hoja.get_all_values()
    resultados = []
    for fila in filas[1:]:
        registro = {nombre: fila[i] if i < len(fila) else ""
                    for i, nombre in enumerate(encabezados)}
        criterios_validos = {c: v for c, v in criterios.items() if c in registro}
        if criterios_validos and all(_coincide(registro.get(c), v, c)
                                     for c, v in criterios_validos.items()):
            resultados.append(registro)
    return resultados


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
        salida = _resolver_referencias(libro, politica, dict(valores))
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
