"""Saldos de cuentas derivados de un corte y de movimientos reales.

La pestaña existente ``tarjetas`` aporta únicamente la configuración inicial.
Después del primer uso, las cuentas, operaciones e ingresos recurrentes viven
en el esquema privado del cliente. Los gastos siguen saliendo de la tabla
canónica; una edición de gasto no crea una segunda transacción financiera.
"""

from __future__ import annotations

import calendar
import json
import logging
import re
import uuid
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from sqlalchemy import text

import config
from bot import catalogo, dashboard, dashboard_edicion, warehouse_ro
from bot.tiempo import fecha_local
from gclient import abrir_libro
from warehouse import crear_destino

logger = logging.getLogger("fachavi.bot.cuentas")
_ESQUEMA = "_bot"
_ID = re.compile(r"^[a-zA-Z0-9_:-]{1,100}$")
_MONEDAS = {"CRC", "USD"}
_TIPOS = {"banco", "ahorro", "credito"}
_DESCRIPCION_PAGO = re.compile(
    r"\b(?:PAGO|ABONO)\b.{0,40}\b(?:TARJETA|TC|TDC|CREDITO|CRÉDITO|AMEX|MCARD|VISA|MASTERCARD)\b",
    re.IGNORECASE,
)


class ErrorCuentas(ValueError):
    """Error de entrada o configuración que puede mostrarse en la app."""


def _decimal(valor: object, *, positivo: bool = False) -> Decimal:
    try:
        numero = Decimal(str(valor).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ErrorCuentas("indique un monto válido") from exc
    if not numero.is_finite() or (positivo and numero <= 0):
        raise ErrorCuentas("el monto debe ser mayor que cero")
    if abs(numero) >= Decimal("1000000000000") or abs(numero.as_tuple().exponent) > 6:
        raise ErrorCuentas("el monto está fuera del rango admitido")
    return numero


def _fecha(valor: object, *, permitir_futura: bool = False) -> date:
    try:
        resultado = date.fromisoformat(str(valor or "")[:10])
    except ValueError as exc:
        raise ErrorCuentas("indique una fecha válida") from exc
    if not permitir_futura and resultado > fecha_local():
        raise ErrorCuentas("la fecha todavía no ha ocurrido")
    return resultado


def _id(valor: object) -> str:
    resultado = str(valor or "").strip()
    if not _ID.fullmatch(resultado):
        raise ErrorCuentas("el identificador no es válido")
    return resultado


def _identificador(valor: str) -> str:
    return '"' + str(valor).replace('"', '""') + '"'


def _motor(cliente: dict):
    destino = crear_destino(config.WAREHOUSE_TIPO, config.dsn_de_cliente(cliente))
    if destino.tipo != "postgres":
        destino.cerrar()
        raise ErrorCuentas("los saldos requieren PostgreSQL")
    return destino, destino.conectar()


def _asegurar(cx) -> None:
    cx.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{_ESQUEMA}"'))
    cx.execute(text(f'''CREATE TABLE IF NOT EXISTS "{_ESQUEMA}".cuentas (
        cliente_id TEXT NOT NULL,
        cuenta_id TEXT NOT NULL,
        nombre TEXT NOT NULL,
        tipo TEXT NOT NULL,
        ultimos4 TEXT NOT NULL DEFAULT '',
        moneda TEXT NOT NULL DEFAULT 'CRC',
        saldo_inicial_crc NUMERIC(20,6) NOT NULL DEFAULT 0,
        saldo_inicial_usd NUMERIC(20,6) NOT NULL DEFAULT 0,
        fecha_corte DATE NOT NULL,
        cuenta_pago_default TEXT NOT NULL DEFAULT '',
        activo BOOLEAN NOT NULL DEFAULT TRUE,
        PRIMARY KEY (cliente_id, cuenta_id)
    )'''))
    cx.execute(text(f'''CREATE TABLE IF NOT EXISTS "{_ESQUEMA}".operaciones_cuenta (
        cliente_id TEXT NOT NULL,
        operacion_id TEXT NOT NULL,
        tipo TEXT NOT NULL,
        fecha DATE NOT NULL,
        descripcion TEXT NOT NULL,
        cuenta_origen TEXT NOT NULL DEFAULT '',
        cuenta_destino TEXT NOT NULL,
        monto_origen NUMERIC(20,6) NOT NULL DEFAULT 0,
        moneda_origen TEXT NOT NULL DEFAULT 'CRC',
        monto_destino NUMERIC(20,6) NOT NULL,
        moneda_destino TEXT NOT NULL DEFAULT 'CRC',
        regla_id TEXT NOT NULL DEFAULT '',
        anulado BOOLEAN NOT NULL DEFAULT FALSE,
        creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (cliente_id, operacion_id)
    )'''))
    cx.execute(text(f'''CREATE TABLE IF NOT EXISTS "{_ESQUEMA}".reglas_ingreso (
        cliente_id TEXT NOT NULL,
        regla_id TEXT NOT NULL,
        nombre TEXT NOT NULL,
        cuenta_id TEXT NOT NULL,
        monto NUMERIC(20,6) NOT NULL,
        moneda TEXT NOT NULL DEFAULT 'CRC',
        frecuencia TEXT NOT NULL,
        dias_mes JSONB NOT NULL DEFAULT '[]'::jsonb,
        dia_semana SMALLINT,
        desde DATE NOT NULL,
        hasta DATE,
        activo BOOLEAN NOT NULL DEFAULT TRUE,
        creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (cliente_id, regla_id)
    )'''))
    cx.execute(text(f'''CREATE INDEX IF NOT EXISTS operaciones_cuenta_fecha_idx
        ON "{_ESQUEMA}".operaciones_cuenta (cliente_id, fecha DESC)'''))


def _semillas_hoja(cliente: dict) -> list[dict]:
    fuente = next((f for f in cliente.get("fuentes", [])
                   if f.get("tipo") == "google_sheets"
                   and "tarjetas" in (f.get("config", {}).get("hojas", []) or [])), None)
    if not fuente:
        return []
    libro = abrir_libro(fuente["config"]["spreadsheet_id"])
    filas = libro.worksheet("tarjetas").get_all_records()
    salida = []
    for fila in filas:
        cuenta_id = str(fila.get("cuenta_id") or "").strip()
        if not cuenta_id:
            continue
        tipo = str(fila.get("tipo_cuenta") or "").strip().lower()
        if tipo not in _TIPOS:
            raise ErrorCuentas(f"tipo de cuenta inválido en tarjetas: {cuenta_id}")
        corte = _fecha(fila.get("fecha_corte"))
        salida.append({
            "cuenta_id": _id(cuenta_id),
            "nombre": str(fila.get("nombre_cuenta") or cuenta_id).strip()[:120],
            "tipo": tipo,
            "ultimos4": str(fila.get("ultimos4") or "").strip(),
            "moneda": str(fila.get("moneda_cuenta") or "CRC").strip().upper(),
            "saldo_crc": _decimal(fila.get("saldo_corte_crc") or 0),
            "saldo_usd": _decimal(fila.get("saldo_corte_usd") or 0),
            "corte": corte,
            "default": str(fila.get("cuenta_pago_default") or "").strip(),
        })
    return salida


def _inicializar(cliente: dict) -> None:
    """Importa una vez la configuración del mismo Sheet que ya contiene tarjetas."""
    cid = str(cliente["cliente_id"])
    destino, motor = _motor(cliente)
    try:
        with motor.begin() as cx:
            _asegurar(cx)
            existe = cx.execute(text(f'SELECT 1 FROM "{_ESQUEMA}".cuentas '
                                     'WHERE cliente_id = :cid LIMIT 1'), {"cid": cid}).first()
        if existe:
            return
        semillas = _semillas_hoja(cliente)
        if not semillas:
            return
        with motor.begin() as cx:
            for cuenta in semillas:
                cx.execute(text(f'''INSERT INTO "{_ESQUEMA}".cuentas
                    (cliente_id, cuenta_id, nombre, tipo, ultimos4, moneda,
                     saldo_inicial_crc, saldo_inicial_usd, fecha_corte,
                     cuenta_pago_default)
                    VALUES (:cid, :cuenta_id, :nombre, :tipo, :ultimos4, :moneda,
                            :saldo_crc, :saldo_usd, :corte, :default)
                    ON CONFLICT (cliente_id, cuenta_id) DO NOTHING'''),
                    {"cid": cid, **cuenta})
    finally:
        destino.cerrar()


def _leer_config(cliente: dict) -> tuple[list[dict], list[dict], list[dict]]:
    _inicializar(cliente)
    cid = str(cliente["cliente_id"])
    destino, motor = _motor(cliente)
    try:
        with motor.begin() as cx:
            _asegurar(cx)
            cuentas = [dict(f) for f in cx.execute(text(f'''SELECT * FROM "{_ESQUEMA}".cuentas
                WHERE cliente_id = :cid AND activo ORDER BY tipo, nombre'''), {"cid": cid}).mappings()]
            reglas = [dict(f) for f in cx.execute(text(f'''SELECT * FROM "{_ESQUEMA}".reglas_ingreso
                WHERE cliente_id = :cid ORDER BY nombre'''), {"cid": cid}).mappings()]
            operaciones = [dict(f) for f in cx.execute(text(f'''SELECT * FROM "{_ESQUEMA}".operaciones_cuenta
                WHERE cliente_id = :cid AND NOT anulado ORDER BY fecha DESC, creado_en DESC'''),
                {"cid": cid}).mappings()]
        return cuentas, reglas, operaciones
    finally:
        destino.cerrar()


def _fechas_regla(regla: dict, hasta: date):
    desde = regla["desde"]
    if isinstance(desde, str):
        desde = date.fromisoformat(desde)
    limite = regla.get("hasta")
    if isinstance(limite, str):
        limite = date.fromisoformat(limite)
    fin = min(hasta, limite) if limite else hasta
    if fin < desde:
        return
    if regla["frecuencia"] == "semanal":
        dia = int(regla["dia_semana"])
        actual = desde + timedelta(days=(dia - desde.weekday()) % 7)
        while actual <= fin:
            yield actual
            actual += timedelta(days=7)
        return
    dias = regla["dias_mes"]
    if isinstance(dias, str):
        dias = json.loads(dias)
    anio, mes = desde.year, desde.month
    while (anio, mes) <= (fin.year, fin.month):
        ultimo = calendar.monthrange(anio, mes)[1]
        for dia in sorted(set(min(int(d), ultimo) for d in dias)):
            actual = date(anio, mes, dia)
            if desde <= actual <= fin:
                yield actual
        mes += 1
        if mes == 13:
            anio, mes = anio + 1, 1


def _materializar_ingresos(cliente: dict, reglas: list[dict]) -> None:
    cid = str(cliente["cliente_id"])
    hoy = fecha_local()
    destino, motor = _motor(cliente)
    try:
        with motor.begin() as cx:
            for regla in reglas:
                if not regla["activo"]:
                    continue
                for dia in _fechas_regla(regla, hoy):
                    cx.execute(text(f'''INSERT INTO "{_ESQUEMA}".operaciones_cuenta
                        (cliente_id, operacion_id, tipo, fecha, descripcion,
                         cuenta_destino, monto_destino, moneda_destino, regla_id)
                        VALUES (:cid, :oid, 'ingreso', :fecha, :nombre,
                                :cuenta, :monto, :moneda, :regla)
                        ON CONFLICT (cliente_id, operacion_id) DO NOTHING'''), {
                        "cid": cid, "oid": f"recurrente:{regla['regla_id']}:{dia.isoformat()}",
                        "fecha": dia, "nombre": regla["nombre"],
                        "cuenta": regla["cuenta_id"], "monto": regla["monto"],
                        "moneda": regla["moneda"], "regla": regla["regla_id"],
                    })
    finally:
        destino.cerrar()


def _canonicos(cliente: dict, corte: date, hasta: date) -> list[dict]:
    ctx = catalogo.construir_contexto(cliente)
    tabla = dashboard.tabla_movimientos_canonicos(cliente, ctx)
    if not tabla:
        raise ErrorCuentas("no encontré los movimientos canónicos para calcular saldos")
    columnas = {str(c).lower() for c in tabla.columnas_config}
    requeridas = {"_clave", "fecha", "medio_pago", "monto_neto", "moneda_original", "monto_original"}
    if not requeridas.issubset(columnas):
        raise ErrorCuentas("los movimientos canónicos no tienen moneda y método de pago completos")
    tipo = "tipo_movimiento" if "tipo_movimiento" in columnas else "'' AS tipo_movimiento"
    descripcion = "descripcion" if "descripcion" in columnas else "'' AS descripcion"
    return warehouse_ro.leer_interno(cliente, f'''SELECT _clave, fecha, {descripcion},
        medio_pago, monto_neto, monto_original, moneda_original, {tipo}
        FROM {_identificador(tabla.tabla_real)}
        WHERE fecha > :corte AND fecha <= :hasta ORDER BY fecha, _clave''',
        {"corte": corte, "hasta": hasta})


def _aplicar_pendientes(cliente: dict, movimientos: list[dict]) -> list[dict]:
    """Mantiene los saldos coherentes con la edición rápida del dashboard."""
    por_clave = {str(m["_clave"]): dict(m) for m in movimientos}
    for pendiente in dashboard_edicion.proyecciones_pendientes(cliente):
        clave = str(pendiente.get("movimiento_clave") or "")
        if pendiente.get("tipo") == "editar" and clave in por_clave:
            fila = por_clave[clave]
            anterior = _decimal(fila["monto_neto"])
            if "monto" in pendiente:
                nuevo = _decimal(pendiente["monto"])
                signo = -1 if anterior < 0 else 1
                fila["monto_neto"] = nuevo * signo
                if anterior:
                    fila["monto_original"] = _decimal(fila["monto_original"]) * nuevo / abs(anterior)
            if "medio_pago" in pendiente:
                fila["medio_pago"] = pendiente["medio_pago"]
        elif pendiente.get("tipo") == "crear" and clave not in por_clave:
            m = pendiente.get("movimiento") or {}
            if m.get("fecha"):
                por_clave[clave] = {
                    "_clave": clave, "fecha": m["fecha"],
                    "descripcion": m.get("descripcion", "Movimiento manual"),
                    "medio_pago": m.get("medio_pago", ""),
                    "monto_neto": m.get("monto", 0),
                    "monto_original": m.get("monto", 0),
                    "moneda_original": m.get("moneda", "CRC"),
                    "tipo_movimiento": "GASTO",
                }
    return list(por_clave.values())


def _cuenta_de_medio(medio: object, por_ultimos4: dict[str, dict],
                     por_id: dict[str, dict]) -> dict | None:
    valor = str(medio or "").strip()
    if valor.startswith("cuenta:"):
        return por_id.get(valor.removeprefix("cuenta:"))
    grupos = re.findall(r"(?<!\d)\d{4}(?!\d)", valor)
    return por_ultimos4.get(grupos[-1]) if grupos else None


def proyectar_saldos(cuentas: list[dict], movimientos: list[dict],
                     operaciones: list[dict], hasta: date) -> dict:
    """Calcula saldos sin guardar un contador mutable independiente."""
    por_id = {}
    por_ultimos4 = {}
    for cuenta in cuentas:
        id_cuenta = str(cuenta["cuenta_id"])
        por_id[id_cuenta] = {
            "cuenta_id": id_cuenta, "nombre": cuenta["nombre"],
            "tipo": cuenta["tipo"], "ultimos4": cuenta.get("ultimos4") or "",
            "moneda": cuenta.get("moneda") or "CRC",
            "fecha_corte": str(cuenta["fecha_corte"])[:10],
            "cuenta_pago_default": cuenta.get("cuenta_pago_default") or "",
            "saldo_crc": _decimal(cuenta["saldo_inicial_crc"]),
            "saldo_usd": _decimal(cuenta["saldo_inicial_usd"]),
            "movimientos": [],
        }
        ultimos4 = str(cuenta.get("ultimos4") or "").strip()
        if ultimos4:
            # Una colisión queda sin resolver: atribuirla sería peor que avisar.
            por_ultimos4[ultimos4] = None if ultimos4 in por_ultimos4 else por_id[id_cuenta]
    pagos_registrados = [
        (str(o.get("cuenta_origen") or ""), str(o["fecha"])[:10],
         _decimal(o["monto_origen"]))
        for o in operaciones if o.get("tipo") == "pago_tarjeta"
    ]
    sin_vincular = []
    ingresos_importados: list[tuple[str, str, Decimal]] = []
    for movimiento in movimientos:
        tipo = str(movimiento.get("tipo_movimiento") or "").strip().upper()
        if tipo in {"TRANSFERENCIA", "PAGO_TARJETA", "PAGO TARJETA"}:
            continue
        cuenta = _cuenta_de_medio(movimiento.get("medio_pago"), por_ultimos4, por_id)
        fecha = str(movimiento.get("fecha") or "")[:10]
        if not cuenta:
            if not any(str(c["fecha_corte"]) < fecha <= hasta.isoformat() for c in por_id.values()):
                continue
            sin_vincular.append({"fecha": fecha, "descripcion": movimiento.get("descripcion") or "",
                                 "medio_pago": movimiento.get("medio_pago") or ""})
            continue
        if not (cuenta["fecha_corte"] < fecha <= hasta.isoformat()):
            continue
        neto = _decimal(movimiento.get("monto_neto") or 0)
        es_ingreso = tipo == "INGRESO"
        if (not es_ingreso and cuenta["tipo"] != "credito"
                and _DESCRIPCION_PAGO.search(str(movimiento.get("descripcion") or ""))):
            firma_pago = (cuenta["cuenta_id"], fecha, abs(neto))
            if firma_pago in pagos_registrados:
                pagos_registrados.remove(firma_pago)
                continue
        if cuenta["tipo"] == "credito":
            moneda = str(movimiento.get("moneda_original") or "CRC").upper()
            if moneda == "USD":
                original = _decimal(movimiento.get("monto_original") or 0)
                cambio = abs(original) * (-1 if neto < 0 or es_ingreso else 1)
                cuenta["saldo_usd"] += cambio
            else:
                moneda = "CRC"
                cambio = -abs(neto) if es_ingreso else neto
                cuenta["saldo_crc"] += cambio
        else:
            moneda = "CRC"
            cambio = abs(neto) if es_ingreso else -neto
            cuenta["saldo_crc"] += cambio
            if es_ingreso:
                ingresos_importados.append((cuenta["cuenta_id"], fecha, cambio))
        cuenta["movimientos"].append({
            "id": str(movimiento.get("_clave") or ""), "fecha": fecha,
            "descripcion": movimiento.get("descripcion") or "Movimiento",
            "monto": cambio, "moneda": moneda, "origen": "canónico",
        })
    for operacion in operaciones:
        fecha = str(operacion["fecha"])[:10]
        if fecha > hasta.isoformat():
            continue
        origen = por_id.get(operacion.get("cuenta_origen"))
        destino = por_id.get(operacion.get("cuenta_destino"))
        # La recurrencia se concilia con un abono bancario idéntico, uno a uno.
        if operacion.get("regla_id") and operacion.get("tipo") == "ingreso":
            firma = (str(operacion["cuenta_destino"]), fecha,
                     _decimal(operacion["monto_destino"]))
            if firma in ingresos_importados:
                ingresos_importados.remove(firma)
                continue
        if origen and origen["fecha_corte"] <= fecha:
            moneda = str(operacion["moneda_origen"]).upper()
            monto = _decimal(operacion["monto_origen"])
            origen[f"saldo_{moneda.lower()}"] -= monto
            origen["movimientos"].append({
                "id": operacion["operacion_id"], "fecha": fecha,
                "descripcion": operacion["descripcion"], "monto": -monto,
                "moneda": moneda, "origen": "operación",
            })
        if destino and destino["fecha_corte"] <= fecha:
            moneda = str(operacion["moneda_destino"]).upper()
            monto = _decimal(operacion["monto_destino"])
            # En una tarjeta de crédito, el pago reduce la deuda.
            efecto = -monto if destino["tipo"] == "credito" else monto
            destino[f"saldo_{moneda.lower()}"] += efecto
            destino["movimientos"].append({
                "id": operacion["operacion_id"], "fecha": fecha,
                "descripcion": operacion["descripcion"], "monto": efecto,
                "moneda": moneda, "origen": "operación",
            })
    for cuenta in por_id.values():
        cuenta["movimientos"] = sorted(
            cuenta["movimientos"], key=lambda fila: (fila["fecha"], fila["id"]), reverse=True,
        )[:50]
        for campo in ("saldo_crc", "saldo_usd"):
            cuenta[campo] = str(cuenta[campo].quantize(Decimal("0.01")))
        for m in cuenta["movimientos"]:
            m["monto"] = str(m["monto"].quantize(Decimal("0.01")))
    return {"cuentas": list(por_id.values()), "sin_vincular": sin_vincular[-20:]}


def obtener(cliente: dict) -> dict:
    cuentas, reglas, operaciones = _leer_config(cliente)
    if not cuentas:
        return {"ok": True, "configurado": False, "cuentas": [], "reglas": []}
    _materializar_ingresos(cliente, reglas)
    # Los ingresos recién generados deben estar presentes en la misma respuesta.
    cuentas, reglas, operaciones = _leer_config(cliente)
    hoy = fecha_local()
    corte = min(_fecha(c["fecha_corte"]) for c in cuentas)
    movimientos = _aplicar_pendientes(cliente, _canonicos(cliente, corte, hoy))
    proyeccion = proyectar_saldos(cuentas, movimientos, operaciones, hoy)
    proyeccion.update({
        "ok": True, "configurado": True, "fecha": hoy.isoformat(),
        "reglas": [{
            "regla_id": r["regla_id"], "nombre": r["nombre"],
            "cuenta_id": r["cuenta_id"], "monto": str(r["monto"]),
            "moneda": r["moneda"], "frecuencia": r["frecuencia"],
            "dias_mes": r["dias_mes"], "dia_semana": r["dia_semana"],
            "desde": str(r["desde"]), "hasta": str(r["hasta"]) if r["hasta"] else "",
            "activo": bool(r["activo"]),
        } for r in reglas],
    })
    return proyeccion


def registrar_operacion(cliente: dict, datos: dict) -> dict:
    if not isinstance(datos, dict):
        raise ErrorCuentas("la operación no es válida")
    _inicializar(cliente)
    cid = str(cliente["cliente_id"])
    tipo = str(datos.get("tipo") or "").strip()
    if tipo not in {"ingreso", "transferencia", "pago_tarjeta"}:
        raise ErrorCuentas("seleccione un tipo de operación válido")
    operacion_id = str(datos.get("operacion_id") or uuid.uuid4())
    try:
        uuid.UUID(operacion_id)
    except ValueError as exc:
        raise ErrorCuentas("el identificador de operación no es válido") from exc
    fecha = _fecha(datos.get("fecha"))
    descripcion = str(datos.get("descripcion") or "").strip()
    if not descripcion or len(descripcion) > 180:
        raise ErrorCuentas("indique una descripción breve")
    origen_id = str(datos.get("cuenta_origen") or "").strip()
    destino_id = _id(datos.get("cuenta_destino"))
    monto_destino = _decimal(datos.get("monto_destino"), positivo=True)
    moneda_destino = str(datos.get("moneda_destino") or "CRC").upper()
    if moneda_destino not in _MONEDAS:
        raise ErrorCuentas("la moneda no es válida")
    monto_origen = (_decimal(datos.get("monto_origen"), positivo=True)
                    if tipo != "ingreso" else Decimal("0"))
    moneda_origen = str(datos.get("moneda_origen") or "CRC").upper()
    if moneda_origen not in _MONEDAS:
        raise ErrorCuentas("la moneda de origen no es válida")
    destino, motor = _motor(cliente)
    try:
        with motor.begin() as cx:
            _asegurar(cx)
            cuentas = {f["cuenta_id"]: dict(f) for f in cx.execute(text(
                f'SELECT * FROM "{_ESQUEMA}".cuentas WHERE cliente_id = :cid AND activo'),
                {"cid": cid}).mappings()}
            cuenta_destino = cuentas.get(destino_id)
            cuenta_origen = cuentas.get(origen_id) if origen_id else None
            if not cuenta_destino or (tipo != "ingreso" and not cuenta_origen):
                raise ErrorCuentas("seleccione cuentas activas")
            if tipo == "ingreso" and cuenta_destino["tipo"] == "credito":
                raise ErrorCuentas("un ingreso debe entrar a una cuenta bancaria o de ahorro")
            if tipo == "pago_tarjeta" and (cuenta_destino["tipo"] != "credito"
                                             or cuenta_origen["tipo"] == "credito"):
                raise ErrorCuentas("el pago requiere una tarjeta de crédito y una cuenta de origen")
            if tipo == "transferencia" and (cuenta_destino["tipo"] == "credito"
                                              or cuenta_origen["tipo"] == "credito"):
                raise ErrorCuentas("use Pagar tarjeta para transferir hacia una tarjeta de crédito")
            if cuenta_origen and origen_id == destino_id:
                raise ErrorCuentas("origen y destino deben ser distintos")
            if fecha < cuenta_destino["fecha_corte"] or (cuenta_origen and fecha < cuenta_origen["fecha_corte"]):
                raise ErrorCuentas("la operación no puede ser anterior al saldo inicial de las cuentas")
            if moneda_destino == "USD" and cuenta_destino["tipo"] != "credito":
                raise ErrorCuentas("esta cuenta no admite ingresos en dólares")
            if cuenta_origen and moneda_origen != cuenta_origen["moneda"]:
                raise ErrorCuentas("la moneda de origen no coincide con la cuenta")
            if tipo == "transferencia" and (moneda_origen != moneda_destino
                                             or monto_origen != monto_destino):
                raise ErrorCuentas("la transferencia entre cuentas debe conservar monto y moneda")
            fila = cx.execute(text(f'''INSERT INTO "{_ESQUEMA}".operaciones_cuenta
                (cliente_id, operacion_id, tipo, fecha, descripcion,
                 cuenta_origen, cuenta_destino, monto_origen, moneda_origen,
                 monto_destino, moneda_destino)
                VALUES (:cid, :oid, :tipo, :fecha, :descripcion,
                        :origen, :destino, :monto_origen, :moneda_origen,
                        :monto_destino, :moneda_destino)
                ON CONFLICT (cliente_id, operacion_id) DO NOTHING
                RETURNING operacion_id'''), {
                "cid": cid, "oid": operacion_id, "tipo": tipo, "fecha": fecha,
                "descripcion": descripcion, "origen": origen_id,
                "destino": destino_id, "monto_origen": monto_origen,
                "moneda_origen": moneda_origen, "monto_destino": monto_destino,
                "moneda_destino": moneda_destino,
            }).scalar_one_or_none()
            if fila is None:
                existente = cx.execute(text(f'''SELECT tipo, fecha, descripcion, cuenta_origen,
                    cuenta_destino, monto_origen, moneda_origen, monto_destino, moneda_destino, anulado
                    FROM "{_ESQUEMA}".operaciones_cuenta
                    WHERE cliente_id = :cid AND operacion_id = :oid'''),
                    {"cid": cid, "oid": operacion_id}).mappings().one()
                if (existente["anulado"] or str(existente["tipo"]) != tipo
                        or existente["fecha"] != fecha
                        or existente["cuenta_origen"] != origen_id
                        or existente["cuenta_destino"] != destino_id
                        or _decimal(existente["monto_origen"]) != monto_origen
                        or _decimal(existente["monto_destino"]) != monto_destino
                        or existente["moneda_origen"] != moneda_origen
                        or existente["moneda_destino"] != moneda_destino
                        or existente["descripcion"] != descripcion):
                    raise ErrorCuentas("esta operación ya se usó con otros datos")
    finally:
        destino.cerrar()
    return {"ok": True, "operacion_id": operacion_id}


def anular_operacion(cliente: dict, operacion_id: str) -> dict:
    cid = str(cliente["cliente_id"])
    oid = _id(operacion_id)
    destino, motor = _motor(cliente)
    try:
        with motor.begin() as cx:
            _asegurar(cx)
            fila = cx.execute(text(f'''UPDATE "{_ESQUEMA}".operaciones_cuenta
                SET anulado = TRUE WHERE cliente_id = :cid AND operacion_id = :oid
                RETURNING operacion_id'''), {"cid": cid, "oid": oid}).first()
            if not fila:
                raise ErrorCuentas("no encontré esa operación")
    finally:
        destino.cerrar()
    return {"ok": True}


def guardar_regla(cliente: dict, datos: dict) -> dict:
    if not isinstance(datos, dict):
        raise ErrorCuentas("la regla no es válida")
    _inicializar(cliente)
    cid = str(cliente["cliente_id"])
    regla_id = str(datos.get("regla_id") or uuid.uuid4())
    try:
        uuid.UUID(regla_id)
    except ValueError as exc:
        raise ErrorCuentas("el identificador de regla no es válido") from exc
    nombre = str(datos.get("nombre") or "").strip()
    if not nombre or len(nombre) > 120:
        raise ErrorCuentas("indique el nombre del ingreso")
    cuenta_id = _id(datos.get("cuenta_id"))
    monto = _decimal(datos.get("monto"), positivo=True)
    moneda = str(datos.get("moneda") or "CRC").upper()
    frecuencia = str(datos.get("frecuencia") or "").lower()
    if moneda != "CRC" or frecuencia not in {"mensual", "semanal"}:
        raise ErrorCuentas("seleccione una moneda y frecuencia válidas")
    desde = _fecha(datos.get("desde"), permitir_futura=True)
    hasta = _fecha(datos.get("hasta"), permitir_futura=True) if datos.get("hasta") else None
    if hasta and hasta < desde:
        raise ErrorCuentas("la fecha final precede a la inicial")
    dias = datos.get("dias_mes") if frecuencia == "mensual" else []
    if frecuencia == "mensual":
        if not isinstance(dias, list) or not dias or len(dias) > 5:
            raise ErrorCuentas("seleccione de uno a cinco días del mes")
        try:
            dias = sorted(set(int(d) for d in dias))
        except (TypeError, ValueError) as exc:
            raise ErrorCuentas("los días del mes no son válidos") from exc
        if any(d < 1 or d > 31 for d in dias):
            raise ErrorCuentas("los días del mes deben estar entre 1 y 31")
    dia_semana = datos.get("dia_semana") if frecuencia == "semanal" else None
    if frecuencia == "semanal":
        try:
            dia_semana = int(dia_semana)
        except (TypeError, ValueError) as exc:
            raise ErrorCuentas("seleccione un día de la semana") from exc
        if dia_semana not in range(7):
            raise ErrorCuentas("seleccione un día de la semana")
    destino, motor = _motor(cliente)
    try:
        with motor.begin() as cx:
            _asegurar(cx)
            cuenta = cx.execute(text(f'''SELECT tipo, fecha_corte FROM "{_ESQUEMA}".cuentas
                WHERE cliente_id = :cid AND cuenta_id = :cuenta AND activo'''),
                {"cid": cid, "cuenta": cuenta_id}).mappings().first()
            if not cuenta or cuenta["tipo"] == "credito":
                raise ErrorCuentas("seleccione una cuenta bancaria o de ahorro")
            if desde <= cuenta["fecha_corte"]:
                raise ErrorCuentas("la recurrencia debe comenzar después del saldo inicial")
            creada = cx.execute(text(f'''INSERT INTO "{_ESQUEMA}".reglas_ingreso
                (cliente_id, regla_id, nombre, cuenta_id, monto, moneda,
                 frecuencia, dias_mes, dia_semana, desde, hasta)
                VALUES (:cid, :regla, :nombre, :cuenta, :monto, :moneda,
                        :frecuencia, CAST(:dias AS jsonb), :dia_semana, :desde, :hasta)
                ON CONFLICT (cliente_id, regla_id) DO NOTHING
                RETURNING regla_id'''), {
                "cid": cid, "regla": regla_id, "nombre": nombre, "cuenta": cuenta_id,
                "monto": monto, "moneda": moneda, "frecuencia": frecuencia,
                "dias": json.dumps(dias), "dia_semana": dia_semana,
                "desde": desde, "hasta": hasta,
            }).scalar_one_or_none()
            if creada is None:
                existente = cx.execute(text(f'''SELECT nombre, cuenta_id, monto, moneda,
                    frecuencia, dias_mes, dia_semana, desde, hasta, activo
                    FROM "{_ESQUEMA}".reglas_ingreso
                    WHERE cliente_id = :cid AND regla_id = :regla'''),
                    {"cid": cid, "regla": regla_id}).mappings().one()
                if (not existente["activo"] or existente["nombre"] != nombre
                        or existente["cuenta_id"] != cuenta_id
                        or _decimal(existente["monto"]) != monto
                        or existente["moneda"] != moneda
                        or existente["frecuencia"] != frecuencia
                        or list(existente["dias_mes"]) != dias
                        or existente["dia_semana"] != dia_semana
                        or existente["desde"] != desde
                        or existente["hasta"] != hasta):
                    raise ErrorCuentas("esta recurrencia ya se usó con otros datos")
    finally:
        destino.cerrar()
    return {"ok": True, "regla_id": regla_id}


def desactivar_regla(cliente: dict, regla_id: str) -> dict:
    cid = str(cliente["cliente_id"])
    regla = _id(regla_id)
    destino, motor = _motor(cliente)
    try:
        with motor.begin() as cx:
            _asegurar(cx)
            fila = cx.execute(text(f'''UPDATE "{_ESQUEMA}".reglas_ingreso SET activo = FALSE
                WHERE cliente_id = :cid AND regla_id = :regla RETURNING regla_id'''),
                {"cid": cid, "regla": regla}).first()
            if not fila:
                raise ErrorCuentas("no encontré esa recurrencia")
    finally:
        destino.cerrar()
    return {"ok": True}
