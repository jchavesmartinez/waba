"""Modelo derivado y configurable de movimientos financieros consolidados.

Las fuentes reales (correo bancario, POS, una hoja manual, un ERP) no tienen
por qué nombrar igual sus columnas. Este módulo las proyecta a un contrato
común a partir de ``_movimientos_canonicos`` en el Sheet de metadata. Así el
bot y los KPIs consultan una sola tabla y no tienen que volver a unir fuentes
ni decidir cuál columna significa fecha, comercio o importe en cada consulta.

No modifica ninguna fuente. Como el resto de la capa semántica, se reconstruye
completa a partir de raw/semantic + metadata y conserva la clave de origen.
"""

from __future__ import annotations

import re

from .metadata import movimientos_canonicos_de
from .tipos import convertir


_IDENTIFICADOR = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SI = {"1", "si", "sí", "true", "yes", "x"}
_REVERSOS = {"REVERSO", "ANULACION", "ANULACIÓN"}

COLUMNAS = [
    ("_clave", "texto"),
    ("_origen", "texto"),
    ("_modelo_id", "texto"),
    ("fuente", "texto"),
    ("clave_origen", "texto"),
    ("fecha", "fecha_iso"),
    ("descripcion", "texto"),
    ("categoria", "texto"),
    ("linea_presupuesto_id", "texto"),
    ("concepto", "texto"),
    ("moneda", "texto"),
    ("monto", "decimal"),
    ("monto_neto", "decimal"),
    ("tipo_movimiento", "texto"),
    ("titular", "texto"),
    ("medio_pago", "texto"),
]

COLUMNAS_RECHAZOS = COLUMNAS + [("motivo", "texto")]


def construir(destino, cliente_id: str, esquema_raw: str, esquema_sem: str,
              fila_modelo: dict, metadata: dict, probar: bool = False) -> dict:
    """Reconstruye una tabla de movimientos desde fuentes declaradas.

    ``fila_modelo`` solo elige la tabla destino. Las filas fuente (y sus
    mapeos) están en metadata para que agregar otra entidad no requiera código.
    """
    modelo_id = str(fila_modelo.get("modelo_id", "")).strip()
    destino_tabla = str(fila_modelo.get("tabla_destino") or modelo_id).strip()
    _validar_identificador(destino_tabla, "tabla_destino")
    fuentes = movimientos_canonicos_de(metadata, modelo_id)
    if not fuentes:
        raise RuntimeError(
            f"el modelo '{modelo_id}' usa extractor movimientos_canonicos "
            "pero no tiene filas en '_movimientos_canonicos'."
        )

    filas, rechazos = [], []
    for config in fuentes:
        _validar_fuente(config, modelo_id)
        esquema = _esquema_de(config.get("capa_origen"), esquema_raw, esquema_sem)
        origen = config["tabla_origen"].strip()
        crudas = _leer(destino, esquema, origen, config.get("filtro", ""))
        referencias = _referencias(destino, config, esquema_raw, esquema_sem)
        for cruda in crudas:
            resultado, motivo = _proyectar(cruda, config, modelo_id, referencias)
            if motivo:
                rechazo = _rechazo(cruda, config, modelo_id, motivo)
                rechazos.append(rechazo)
            elif resultado:
                filas.append(resultado)

    # Una clave es estable dentro de su fuente. El prefijo evita que dos
    # sistemas independientes con IDs parecidos colisionen al consolidarlos.
    claves = set()
    duplicadas = []
    unicas = []
    for movimiento in filas:
        if movimiento["_clave"] in claves:
            duplicadas.append({**movimiento, "motivo": "clave duplicada en la fuente"})
        else:
            claves.add(movimiento["_clave"])
            unicas.append(movimiento)
    filas = sorted(unicas, key=lambda f: (str(f.get("fecha") or ""), f["_clave"]))
    rechazos.extend(duplicadas)

    if not probar:
        destino.reconstruir_tabla(esquema_sem, destino_tabla, COLUMNAS, filas)
        destino.reconstruir_tabla(
            esquema_sem, destino_tabla + "__rechazos", COLUMNAS_RECHAZOS, rechazos)

    alertas = []
    if rechazos:
        alertas.append(
            f"[{cliente_id}] '{destino_tabla}': {len(rechazos)} fila(s) rechazadas; "
            f"revisa {destino_tabla}__rechazos."
        )
    return {"filas": len(filas), "rechazos": len(rechazos), "alertas": alertas}


def _proyectar(cruda: dict, cfg: dict, modelo_id: str,
               referencias: dict) -> tuple[dict | None, str]:
    if not _incluida(cruda, cfg):
        return None, ""

    def valor(nombre: str):
        columna = str(cfg.get(nombre, "")).strip()
        return cruda.get(columna) if columna else None

    fecha = _fecha(valor("fecha"))
    descripcion = _texto(valor("descripcion"))
    moneda = _texto(valor("moneda")).upper()
    monto = convertir("decimal", valor("monto"))[""]
    clave = _texto(valor("clave"))
    faltantes = [n for n, v in (("fecha", fecha), ("descripcion", descripcion),
                                 ("moneda", moneda), ("monto", monto), ("clave", clave))
                 if v in (None, "")]
    if faltantes:
        return None, "faltan o son inválidos: " + ", ".join(faltantes)

    linea = _texto(valor("linea_presupuesto_id"))
    llave_referencia = str(cfg.get("llave_referencia_origen", "")).strip()
    clave_referencia = _texto(cruda.get(llave_referencia)) if llave_referencia else linea
    referencia = referencias.get(clave_referencia, {}) if clave_referencia else {}
    categoria = _texto(valor("categoria")) or _texto(referencia.get("categoria"))
    concepto = _texto(valor("concepto")) or _texto(referencia.get("concepto"))
    tipo = _texto(valor("tipo_movimiento"))
    signo = -1 if (
        str(cfg.get("signo", "")).strip().lower() == "reversos_negativos"
        and tipo.upper() in _REVERSOS
    ) else 1
    fuente = _texto(cfg.get("fuente"))
    if not fuente:
        return None, "la fuente no declara una etiqueta"
    return {
        "_clave": f"{fuente}:{clave}",
        "_origen": clave,
        "_modelo_id": modelo_id,
        "fuente": fuente,
        "clave_origen": clave,
        "fecha": fecha,
        "descripcion": descripcion,
        "categoria": categoria,
        "linea_presupuesto_id": linea,
        "concepto": concepto,
        "moneda": moneda,
        "monto": monto,
        "monto_neto": monto * signo,
        "tipo_movimiento": tipo,
        "titular": _texto(valor("titular")),
        "medio_pago": _texto(valor("medio_pago")),
    }, ""


def _incluida(fila: dict, cfg: dict) -> bool:
    """Aplica las banderas configuradas sin suponer que toda fuente las tiene."""
    for nombre in ("activo", "incluir_en_gasto"):
        columna = str(cfg.get(nombre, "")).strip()
        if columna and str(fila.get(columna, "")).strip().lower() not in _SI:
            return False
    return True


def _referencias(destino, cfg: dict, esquema_raw: str, esquema_sem: str) -> dict:
    tabla = str(cfg.get("tabla_referencia", "")).strip()
    origen = str(cfg.get("llave_referencia_origen", "")).strip()
    llave = str(cfg.get("llave_referencia", "")).strip()
    if not any((tabla, origen, llave)):
        return {}
    if not all((tabla, origen, llave)):
        raise RuntimeError("la referencia debe declarar tabla y ambas llaves")
    for nombre in (tabla, origen, llave):
        _validar_identificador(nombre, "referencia")
    esquema = _esquema_de(cfg.get("capa_referencia"), esquema_raw, esquema_sem)
    filas = _leer(destino, esquema, tabla, "")
    categoria = str(cfg.get("categoria_referencia", "")).strip()
    concepto = str(cfg.get("concepto_referencia", "")).strip()
    for columna in (categoria, concepto):
        if columna:
            _validar_identificador(columna, "columna de referencia")
    salida = {}
    for fila in filas:
        clave = _texto(fila.get(llave))
        if clave:
            salida[clave] = {
                "categoria": fila.get(categoria) if categoria else None,
                "concepto": fila.get(concepto) if concepto else None,
            }
    return salida


def _rechazo(cruda: dict, cfg: dict, modelo_id: str, motivo: str) -> dict:
    clave_col = str(cfg.get("clave", "")).strip()
    clave = _texto(cruda.get(clave_col)) if clave_col else ""
    fuente = _texto(cfg.get("fuente"))
    return {
        "_clave": f"{fuente}:{clave}" if clave else "",
        "_origen": clave,
        "_modelo_id": modelo_id,
        "fuente": fuente,
        "clave_origen": clave,
        "motivo": motivo,
    }


def _leer(destino, esquema: str, tabla: str, filtro: str) -> list:
    _validar_identificador(tabla, "tabla_origen")
    sql = f'SELECT * FROM "{esquema}"."{tabla}"'
    if str(filtro or "").strip():
        sql += " WHERE " + str(filtro).strip()
    return destino.leer_filas(sql)


def _validar_fuente(cfg: dict, modelo_id: str) -> None:
    for campo in ("fuente", "tabla_origen", "fecha", "descripcion", "moneda",
                  "monto", "clave"):
        if not str(cfg.get(campo, "")).strip():
            raise RuntimeError(
                f"el movimiento canónico '{modelo_id}' requiere '{campo}' en "
                "'_movimientos_canonicos'."
            )
    _validar_identificador(str(cfg["tabla_origen"]).strip(), "tabla_origen")
    for campo in ("fecha", "descripcion", "categoria", "linea_presupuesto_id",
                  "concepto", "moneda", "monto", "tipo_movimiento", "titular",
                  "medio_pago", "clave", "activo", "incluir_en_gasto"):
        nombre = str(cfg.get(campo, "")).strip()
        if nombre:
            _validar_identificador(nombre, campo)


def _validar_identificador(nombre: str, etiqueta: str) -> None:
    if not _IDENTIFICADOR.fullmatch(nombre):
        raise RuntimeError(f"{etiqueta} inválido: '{nombre}'")


def _esquema_de(capa: str, esquema_raw: str, esquema_sem: str) -> str:
    valor = str(capa or "raw").strip().lower()
    if valor in {"", "raw"}:
        return esquema_raw
    if valor == "semantic":
        return esquema_sem
    raise RuntimeError("capa debe ser raw o semantic")


def _texto(valor) -> str:
    return "" if valor is None else str(valor).strip()


def _fecha(valor):
    if hasattr(valor, "year") and hasattr(valor, "month"):
        return valor
    return convertir("fecha_iso", valor)[""]
