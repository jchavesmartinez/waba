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
import re

from bot import catalogo


_SI = {"si", "sí", "true", "1", "yes"}
_ACCIONES = {"crear", "modificar", "anular"}


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
    ejemplo: str = ""


@dataclass(frozen=True)
class PoliticaEdicion:
    tabla: str
    origen: str
    clave_primaria: str
    anulacion_campo: str
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
            ejemplo=str(fila.get("ejemplo") or "").strip(),
        )
    return PoliticaEdicion(
        tabla=tabla.tabla_logica,
        origen=str(tabla.configuracion.get("origen_edicion") or "").strip(),
        clave_primaria=str(tabla.configuracion.get("clave_primaria") or "").strip(),
        anulacion_campo=str(tabla.configuracion.get("anulacion_campo") or "").strip(),
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
        valor = valores.get(nombre, campo.defecto)
        if campo.calculado:
            # Por seguridad, los ids y marcas calculadas no se aceptan desde el
            # mensaje. El adaptador de origen los asigna al confirmar.
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
        "La captura y confirmación se activarán al conectar el origen de escritura; "
        "por ahora no se modificará ningún dato."
    )
