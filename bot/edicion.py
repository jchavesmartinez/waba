"""Politicas y validacion deterministica para editar datos desde WhatsApp.

El LLM puede interpretar el texto del usuario, pero nunca decide por si solo que
campos son obligatorios, que formato es valido o si una tabla se puede editar.
Todo eso viene de ``_catalogo`` y se valida antes de mostrar una confirmacion.

Este modulo no escribe en Neon: Neon es el warehouse de consulta. Un adaptador
de escritura debe apuntar al origen declarado en ``origen_edicion`` y solo se
invoca despues de una confirmacion explicita del usuario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import json
import logging
import re

import config
from bot import catalogo
from bot.salida import Respuesta
from bot.tiempo import fecha_local
import llm


_SI = {"si", "sí", "true", "1", "yes"}
_ACCIONES = {"crear", "modificar", "anular"}
logger = logging.getLogger("fachavi.bot.edicion")


def _si(valor: object) -> bool:
    return str(valor or "").strip().casefold() in _SI


def _lista(valor: object) -> list[str]:
    return [v.strip() for v in str(valor or "").split(",") if v.strip()]


def _normalizar_fecha(valor: object) -> str:
    texto = str(valor or "").strip()
    if not texto:
        return ""
    # Solo se acepta ISO o una fecha escrita sin ambiguedad. 03/04/2026 puede
    # ser 3 de abril o 4 de marzo: el bot debe preguntarlo, no adivinarlo.
    try:
        return date.fromisoformat(texto).isoformat()
    except ValueError:
        pass
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", texto):
        raise ValueError("la fecha es ambigua; use AAAA-MM-DD")
    for patron in ("%d de %B de %Y", "%d de %b de %Y"):
        try:
            return datetime.strptime(texto.casefold(), patron).date().isoformat()
        except ValueError:
            continue
    raise ValueError("use el formato AAAA-MM-DD")


def _decimal_local(valor: object) -> Decimal:
    texto = str(valor).strip().replace("₡", "").replace("$", "").replace(" ", "")
    # 1.234,56 (formato CR) y 1234.56 son validos. Nunca se adivina cuando
    # ambos separadores aparecen en un orden imposible.
    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        texto = texto.replace(",", ".")
    return Decimal(texto)


@dataclass(frozen=True)
class CampoEdicion:
    nombre: str
    etiqueta: str
    requerido: bool = False
    editable: bool = True
    tipo: str = "texto"
    valores: tuple[str, ...] = ()
    defecto: str = ""
    calculado: bool = False
    generador: str = ""
    ejemplo: str = ""


@dataclass(frozen=True)
class PoliticaEdicion:
    tabla: str
    origen: str
    clave_primaria: str
    anulacion_campo: str
    origen_tipo: str
    origen_fuente_id: str
    hoja_origen: str
    acciones: tuple[str, ...]
    campos: dict[str, CampoEdicion] = field(default_factory=dict)


@dataclass(frozen=True)
class BorradorValidado:
    valores: dict[str, object]
    faltantes: tuple[str, ...] = ()
    errores: tuple[str, ...] = ()

    @property
    def listo_para_confirmar(self) -> bool:
        return not self.faltantes and not self.errores


def politica_desde_tabla(tabla) -> PoliticaEdicion | None:
    """Construye la politica desde una TablaPermitida ya leida del catalogo."""
    if tabla is None or not _si(tabla.configuracion.get("editable")):
        return None

    acciones = tuple(a for a in _lista(tabla.configuracion.get("acciones_permitidas"))
                      if a.casefold() in _ACCIONES)
    if not acciones:
        return None
    campos = {}
    for nombre, fila in tabla.columnas_config.items():
        campos[nombre] = CampoEdicion(
            nombre=nombre,
            etiqueta=str(fila.get("etiqueta_usuario") or nombre).strip(),
            requerido=_si(fila.get("requerido")),
            editable=not str(fila.get("editable_campo", "")).strip() or _si(fila.get("editable_campo")),
            tipo=str(fila.get("tipo_validacion") or "texto").strip().casefold(),
            valores=tuple(_lista(fila.get("valores_permitidos"))),
            defecto=str(fila.get("valor_por_defecto") or "").strip(),
            calculado=_si(fila.get("calculado_por_sistema")),
            generador=str(fila.get("generador") or "").strip().casefold(),
            ejemplo=str(fila.get("ejemplo") or "").strip(),
        )
    return PoliticaEdicion(
        tabla=tabla.tabla_logica,
        origen=str(tabla.configuracion.get("origen_edicion") or "").strip(),
        clave_primaria=str(tabla.configuracion.get("clave_primaria") or "").strip(),
        anulacion_campo=str(tabla.configuracion.get("anulacion_campo") or "").strip(),
        origen_tipo=str(tabla.configuracion.get("origen_tipo") or "").strip().casefold(),
        origen_fuente_id=str(tabla.configuracion.get("origen_fuente_id") or "").strip(),
        hoja_origen=str(tabla.configuracion.get("hoja_origen") or "").strip(),
        acciones=acciones,
        campos=campos,
    )


def politica_para(cliente: dict, tabla_logica: str) -> PoliticaEdicion | None:
    """Obtiene una politica solo si el cliente la habilito explicitamente."""
    ctx = catalogo.construir_contexto(cliente)
    tabla = next((t for t in ctx.permitidas if t.tabla_logica == tabla_logica), None)
    return politica_desde_tabla(tabla)


def validar_borrador(politica: PoliticaEdicion, accion: str,
                      valores: dict[str, object]) -> BorradorValidado:
    """Normaliza y valida un borrador sin escribir ni modificar datos."""
    accion = str(accion or "").casefold().strip()
    if accion not in politica.acciones:
        return BorradorValidado({}, errores=("esa acción no está permitida para esta tabla",))

    salida: dict[str, object] = {}
    faltantes: list[str] = []
    errores: list[str] = []
    for nombre, campo in politica.campos.items():
        # Los valores por defecto pertenecen a una creación. En una modificación
        # no deben sobrescribir campos que el usuario no mencionó.
        valor = valores.get(nombre, campo.defecto if accion == "crear" else "")
        if campo.calculado:
            # Por seguridad, los ids y marcas calculadas no se aceptan desde el
            # mensaje al crear. Para modificar/anular, la llave calculada sí se
            # requiere únicamente para localizar el registro existente.
            if accion != "crear" and nombre == politica.clave_primaria:
                clave = str(valores.get(nombre, "")).strip()
                if clave:
                    salida[nombre] = clave
                else:
                    faltantes.append(campo.etiqueta)
            continue
        if valor in (None, ""):
            if accion == "crear" and campo.requerido:
                faltantes.append(campo.etiqueta)
            continue
        try:
            if campo.tipo == "fecha_iso":
                valor = _normalizar_fecha(valor)
            elif campo.tipo == "monto_positivo":
                numero = _decimal_local(valor)
                if numero <= 0:
                    raise ValueError("debe ser mayor que cero")
                valor = str(numero)
            elif campo.tipo == "lista":
                opciones = {v.casefold(): v for v in campo.valores}
                elegido = opciones.get(str(valor).strip().casefold())
                if not elegido:
                    raise ValueError("elija una de: " + ", ".join(campo.valores))
                valor = elegido
        except (InvalidOperation, ValueError) as e:
            errores.append(f"{campo.etiqueta}: {e}")
            continue
        salida[nombre] = valor
    return BorradorValidado(salida, tuple(faltantes), tuple(errores))


def resumen_inicio(politica: PoliticaEdicion) -> str:
    acciones = ", ".join(politica.acciones)
    requeridos = [c for c in politica.campos.values() if c.requerido and not c.calculado]
    etiquetas = ", ".join(c.etiqueta for c in requeridos)
    return (
        f"La edición de *{' '.join(politica.tabla.split('_'))}* quedó configurada "
        f"para: *{acciones}*. Los campos obligatorios serán: {etiquetas}. "
        "Escriba primero la acción que desea realizar: crear, modificar o anular. "
        "Luego puede enviar los datos como `Campo: valor`, uno por línea. "
        "Nada se guarda sin una vista previa y su confirmación explícita."
    )


def _normalizar(texto: object) -> str:
    return re.sub(r"\s+", " ", str(texto or "").strip().casefold().replace("_", " "))


def _estado(historial: list) -> dict | None:
    """Recupera solo el último flujo de edición pendiente de esta conversación."""
    for turno in reversed(historial):
        if turno.get("rol") != "assistant":
            continue
        estado = turno.get("estado") or {}
        edicion = estado.get("edicion")
        if isinstance(edicion, dict):
            return edicion or None
        menu = estado.get("menu") or {}
        if menu.get("accion") == "editar" and menu.get("tabla"):
            return {"tabla": menu["tabla"], "paso": "accion", "valores": {}}
    return None


def _accion(texto: str) -> str:
    valor = _normalizar(texto)
    return valor if valor in _ACCIONES else ""


def _campos_desde_texto(politica: PoliticaEdicion, texto: str) -> dict[str, str]:
    """Reconoce `Etiqueta: valor` usando solo los nombres declarados en metadata."""
    aliases = {}
    for nombre, campo in politica.campos.items():
        aliases[_normalizar(nombre)] = nombre
        aliases[_normalizar(campo.etiqueta)] = nombre
    salida = {}
    for linea in str(texto or "").splitlines():
        if ":" not in linea:
            continue
        etiqueta, valor = linea.split(":", 1)
        nombre = aliases.get(_normalizar(etiqueta))
        if nombre:
            salida[nombre] = valor.strip()
    return salida


def _extraer_natural(politica: PoliticaEdicion, texto: str,
                     accion: str = "crear") -> tuple[dict[str, str], dict[str, str]]:
    """Extrae campos claros de una frase natural; omite lo ambiguo o ausente."""
    campos = {n: c for n, c in politica.campos.items() if not c.calculado and c.editable}
    if not campos or len(_normalizar(texto).split()) <= 1:
        return {}, {}
    propiedades = {n: {"type": "string"} for n in campos}
    aliases = "; ".join(f"{n} = {c.etiqueta}" for n, c in campos.items())
    sistema = (
        "Extraes datos para un registro de una tabla. Responde SOLO JSON. "
        "Incluye una propiedad únicamente cuando el mensaje la expresa con claridad; "
        "no inventes, no completes con suposiciones y no incluyas propiedades vacías. "
        "Convierte fechas relativas (hoy, ayer) a AAAA-MM-DD usando la fecha actual. "
        "Conserva el importe como texto y usa exactamente los nombres de propiedad. "
        "Para modificar, separa los datos que identifican el registro (criterios) "
        "de los valores que se desean cambiar (cambios). Para crear, pon todo en cambios."
    )
    contenido = (f"Acción: {accion}\nFecha actual en la zona del negocio: {fecha_local().isoformat()}\n"
                 f"Campos disponibles y sus etiquetas: {aliases}\n"
                 f"Mensaje del usuario: {texto}")
    try:
        respuesta = llm.generar_texto(
            config.BOT_MODELO_KPIS, contenido, system=sistema,
            max_tokens=300, thinking_level="minimal",
            response_schema={"type": "object", "properties": {
                "criterios": {"type": "object", "properties": propiedades,
                               "additionalProperties": False},
                "cambios": {"type": "object", "properties": propiedades,
                             "additionalProperties": False},
            }, "additionalProperties": False},
        )
        bruto = re.search(r"\{.*\}", respuesta.texto or "", re.DOTALL)
        if not bruto:
            return {}, {}
        datos = json.loads(bruto.group(0))
        if accion == "crear":
            cambios = datos.get("cambios") or datos
            return ({n: str(v).strip() for n, v in cambios.items()
                     if n in campos and str(v).strip()}, {})
        criterios = datos.get("criterios") or {}
        cambios = datos.get("cambios") or {}
        return ({n: str(v).strip() for n, v in criterios.items()
                 if n in campos and str(v).strip()},
                {n: str(v).strip() for n, v in cambios.items()
                 if n in campos and str(v).strip()})
    except Exception as exc:  # noqa: BLE001
        logger.info("extracción natural de edición no disponible: %s", exc)
        return {}, {}


def _texto_previa(politica: PoliticaEdicion, accion: str, valores: dict) -> str:
    lineas = []
    for nombre, valor in valores.items():
        campo = politica.campos.get(nombre)
        if campo and nombre != politica.clave_primaria:
            lineas.append(f"• *{campo.etiqueta}:* {valor}")
    verbo = {"crear": "crear", "modificar": "modificar", "anular": "anular"}[accion]
    return (f"Revisá este cambio para *{' '.join(politica.tabla.split('_'))}*: "
            f"se va a *{verbo}* este registro:\n\n" + "\n".join(lineas) +
            "\n\nResponda *Confirmar* para aplicarlo o *Cancelar* para descartarlo.")


def procesar_mensaje(cliente: dict, numero: str, pregunta: str, historial: list) -> Respuesta | None:
    """Avanza un flujo de edición; devuelve None cuando no hay edición en curso."""
    estado = _estado(historial)
    if not estado:
        return None
    politica = politica_para(cliente, str(estado.get("tabla") or ""))
    if not politica:
        return Respuesta("La tabla que estaba editando ya no está habilitada.",
                         estado={"edicion": {}})

    texto = _normalizar(pregunta)
    if texto in {"cancelar", "cancela", "cancel"}:
        return Respuesta("Listo, descarté la edición. No se modificó ningún dato.",
                         estado={"edicion": {}})
    if estado.get("paso") == "confirmar":
        if texto not in {"confirmar", "confirmo", "sí", "si"}:
            return Respuesta("La edición sigue pendiente. Responda *Confirmar* para aplicarla "
                             "o *Cancelar* para descartarla.", estado={"edicion": estado})
        from bot import escritura_google_sheets, memoria
        try:
            resultado = escritura_google_sheets.aplicar_confirmado(
                cliente, politica, estado["accion"], estado["valores"])
        except escritura_google_sheets.ErrorEscritura as exc:
            return Respuesta(f"No pude aplicar el cambio: {exc}. No se modificó ningún dato.",
                             estado={"edicion": estado})
        memoria.registrar_edicion(cliente, numero, politica.tabla, resultado)
        verbo = {"crear": "creado", "modificar": "modificado", "anular": "anulado"}[resultado["accion"]]
        return Respuesta(f"Listo. El registro fue {verbo} correctamente.",
                         estado={"edicion": {}})

    accion = str(estado.get("accion") or "")
    if not accion:
        accion = _accion(pregunta)
        if not accion or accion not in politica.acciones:
            permitidas = ", ".join(politica.acciones)
            return Respuesta(f"¿Qué desea hacer con *{' '.join(politica.tabla.split('_'))}*? "
                             f"Escriba una acción: {permitidas}.",
                             estado={"edicion": {"tabla": politica.tabla, "paso": "accion", "valores": {}}})
        estado = {"tabla": politica.tabla, "accion": accion, "paso": "campos", "valores": {}}
        if accion == "anular":
            return Respuesta("Describa el registro que desea anular, por ejemplo: “Anula el gasto de Walmart del 22 de agosto”.",
                             estado={"edicion": estado})
        if accion == "modificar":
            return Respuesta("Describa el cambio en lenguaje natural, por ejemplo: “Cambia el gasto de Walmart del 22 de agosto a ₡220.000”.",
                             estado={"edicion": estado})

    valores = dict(estado.get("valores") or {})
    valores.update(_campos_desde_texto(politica, pregunta))
    # Cuando el usuario escribe una frase natural, el extractor entiende solo
    # lo explícito y deja que la validación pregunte por lo que falte.
    criterios_naturales = {}
    cambios_naturales = {}
    if ":" not in str(pregunta):
        criterios_naturales, cambios_naturales = _extraer_natural(politica, pregunta, accion)
        valores.update(cambios_naturales if accion == "modificar" else criterios_naturales)
    # Para modificar/anular, la llave primaria se resuelve internamente. El
    # cliente identifica el registro con lenguaje natural; si hay ambigüedad,
    # solicitamos un dato adicional sin revelar el ID técnico.
    if accion in {"modificar", "anular"} and not valores.get(politica.clave_primaria):
        from bot import escritura_google_sheets
        criterios = dict(criterios_naturales or valores)
        # En una modificación, el monto suele ser el nuevo valor; no lo uses
        # para localizar el importe anterior del registro.
        if accion == "modificar" and len(criterios) > 1:
            criterios.pop("monto", None)
        try:
            candidatos = escritura_google_sheets.buscar_registros(cliente, politica, criterios)
        except escritura_google_sheets.ErrorEscritura as exc:
            return Respuesta(f"No pude buscar el registro: {exc}.",
                             estado={"edicion": {"tabla": politica.tabla, "accion": accion,
                                                  "paso": "campos", "valores": valores}})
        if len(candidatos) == 1:
            valores[politica.clave_primaria] = candidatos[0].get(politica.clave_primaria, "")
        elif len(candidatos) > 1:
            return Respuesta("Encontré varios registros que coinciden. Indique un dato adicional, "
                             "como la fecha exacta, el comercio o el monto original.",
                             estado={"edicion": {"tabla": politica.tabla, "accion": accion,
                                                  "paso": "campos", "valores": valores}})
        elif criterios:
            return Respuesta("No encontré un registro con esos datos. Indique una descripción "
                             "más precisa o la fecha del movimiento.",
                             estado={"edicion": {"tabla": politica.tabla, "accion": accion,
                                                  "paso": "campos", "valores": valores}})
    validado = validar_borrador(politica, accion, valores)
    if not validado.listo_para_confirmar:
        detalles = []
        if validado.faltantes:
            detalles.append("Faltan: " + ", ".join(validado.faltantes) + ".")
        if validado.errores:
            detalles.extend(validado.errores)
        return Respuesta("Todavía no puedo mostrar la vista previa. " + "\n".join(detalles) +
                         "\nEnvíelos como `Campo: valor`, uno por línea.",
                         estado={"edicion": {"tabla": politica.tabla, "accion": accion,
                                              "paso": "campos", "valores": valores}})
    if accion == "modificar" and set(validado.valores) == {politica.clave_primaria}:
        return Respuesta("Indique al menos un campo adicional para modificar; el identificador "
                         "solo sirve para encontrar el registro.",
                         estado={"edicion": {"tabla": politica.tabla, "accion": accion,
                                              "paso": "campos", "valores": valores}})
    nuevo = {"tabla": politica.tabla, "accion": accion, "paso": "confirmar",
             "valores": validado.valores}
    return Respuesta(_texto_previa(politica, accion, validado.valores),
                     estado={"edicion": nuevo})
