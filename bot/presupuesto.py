"""Editor del plan en su Google Sheet original, con vigencias y auditoría.

Nunca escribe movimientos, reglas de clasificación ni saldos. Una modificación
divide la vigencia conservando el linea_id; Sheets aplica todo el lote de forma
atómica. La revisión evita sobrescribir una hoja cambiada desde otra pestaña.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import logging
import re
import unicodedata
import uuid

from sqlalchemy import text

from bot import catalogo, dashboard_edicion, edicion
from bot.tiempo import fecha_local
from gclient import abrir_libro, abrir_libro_escritura

logger = logging.getLogger("fachavi.bot.presupuesto")
REQUERIDAS = {"linea_id", "tipo", "categoria", "concepto", "monto_mensual",
              "monto_quincenal", "vigencia_desde", "vigencia_hasta"}
TABLA = '"_bot"."presupuesto_ediciones"'


class ErrorPresupuesto(ValueError):
    """Mensaje seguro para el editor."""


def _normal(valor):
    return " ".join(unicodedata.normalize("NFKD", str(valor or "")).encode(
        "ascii", "ignore").decode().casefold().split())


def _mes(valor):
    try:
        fecha = date.fromisoformat(str(valor))
    except ValueError as exc:
        raise ErrorPresupuesto("Seleccione un mes válido.") from exc
    if fecha.day != 1:
        raise ErrorPresupuesto("La vigencia debe comenzar el primer día del mes.")
    return fecha


def _fecha_celda(valor, *, opcional=False):
    if not str(valor).strip() and opcional:
        return None
    try:
        if isinstance(valor, (int, float)) or re.fullmatch(r"\d{5}(?:\.0+)?", str(valor)):
            return date(1899, 12, 30) + timedelta(days=int(float(valor)))
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", str(valor)):
            return datetime.strptime(str(valor), "%d/%m/%Y").date()
        return date.fromisoformat(str(valor)[:10])
    except (ValueError, OverflowError) as exc:
        raise ErrorPresupuesto("El presupuesto contiene una vigencia inválida; revise la hoja.") from exc


def _monto(valor, *, decimales=2):
    try:
        numero = Decimal(edicion.normalizar_monto(valor, permitir_cero=True))
    except (ValueError, InvalidOperation) as exc:
        raise ErrorPresupuesto("Indique un monto válido, cero o mayor.") from exc
    if numero >= Decimal("1000000000000") or numero.as_tuple().exponent < -decimales:
        raise ErrorPresupuesto("El monto admite hasta dos decimales y debe ser menor que un billón.")
    return numero


def _nombre(valor):
    valor = " ".join(str(valor or "").split())
    if not valor or len(valor) > 120 or valor.startswith(("=", "+", "-", "@")):
        raise ErrorPresupuesto("Escriba un nombre de entre 1 y 120 caracteres, sin fórmulas.")
    return valor


def _origen(cliente):
    ctx = catalogo.construir_contexto(cliente)
    tablas = [t for t in ctx.permitidas if t.tabla_logica == "presupuesto"]
    if ctx.error_lectura or len(tablas) != 1:
        raise ErrorPresupuesto("No encontré un presupuesto habilitado para este cliente.")
    tabla = tablas[0]
    # El destino sale exclusivamente del registro y de la tabla permitida.
    # Nunca aceptamos spreadsheet_id, hoja o fuente enviados por el navegador.
    fuentes = [f for f in cliente.get("fuentes", []) if f.get("activo")
               and f.get("tipo") == "google_sheets"
               and tabla.tabla_real == f'{f.get("fuente_id")}__presupuesto'
               and (not f.get("config", {}).get("hojas")
                    or "presupuesto" in f["config"]["hojas"])
               and "presupuesto" not in f.get("config", {}).get("excluir", [])]
    if len(fuentes) != 1 or not fuentes[0].get("config", {}).get("spreadsheet_id"):
        raise ErrorPresupuesto("El presupuesto no tiene un origen Google Sheets editable inequívoco.")
    return fuentes[0]


def _leer(cliente, *, escritura=False):
    fuente = _origen(cliente)
    abrir = abrir_libro_escritura if escritura else abrir_libro
    libro = abrir(fuente["config"]["spreadsheet_id"])
    hoja = libro.worksheet("presupuesto")
    # FORMULA permite conservar fórmulas de columnas ajenas y detectar montos
    # calculados: nunca los sustituimos silenciosamente por un valor fijo.
    valores = hoja.get_all_values(value_render_option="FORMULA")
    if not valores or len(valores) > 10000:
        raise ErrorPresupuesto("La hoja está vacía o supera el tamaño admitido por el editor.")
    encabezados = [str(v).strip() for v in valores[0]]
    if not REQUERIDAS.issubset(encabezados) or any(encabezados.count(c) != 1 for c in REQUERIDAS):
        raise ErrorPresupuesto("La hoja no tiene las columnas únicas requeridas para editar con seguridad.")
    ancho = max(len(encabezados), max(map(len, valores)))
    grilla = [list(f) + [""] * (ancho - len(f)) for f in valores]
    while len(grilla) > 1 and not any(str(v).strip() for v in grilla[-1]):
        grilla.pop()
    for fila in grilla[1:]:
        if not str(fila[encabezados.index("linea_id")]).strip():
            continue
        for col in ("vigencia_desde", "vigencia_hasta"):
            i = encabezados.index(col)
            if str(fila[i]).strip() and not str(fila[i]).startswith("="):
                fila[i] = _fecha_celda(fila[i]).isoformat()
    return fuente, libro, hoja, encabezados, grilla


def _revision(grilla):
    # Canoniza números almacenados por Sheets como texto o número.
    datos = [[str(v) if v is not None else "" for v in f] for f in grilla]
    if datos:
        for campo in ("monto_mensual", "monto_quincenal"):
            if campo not in datos[0]:
                continue
            indice = datos[0].index(campo)
            for fila in datos[1:]:
                try:
                    numero = Decimal(fila[indice])
                    if numero.is_finite():
                        fila[indice] = format(numero.normalize(), "f")
                except (InvalidOperation, IndexError):
                    pass
    return hashlib.sha256(json.dumps(datos, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _registros(encabezados, grilla):
    return [(i, dict(zip(encabezados, fila))) for i, fila in enumerate(grilla[1:], 1)
            if str(fila[encabezados.index("linea_id")]).strip()]


def _vigente(fila, mes):
    inicio = _fecha_celda(fila["vigencia_desde"])
    fin = _fecha_celda(fila["vigencia_hasta"], opcional=True)
    return inicio <= mes and (fin is None or mes <= fin)


def _vista(encabezados, grilla, mes):
    lineas = []
    vistas = set()
    for _, fila in _registros(encabezados, grilla):
        if not _vigente(fila, mes):
            continue
        llave = str(fila["linea_id"])
        if llave in vistas:
            raise ErrorPresupuesto("Hay vigencias superpuestas para una partida. Revise la hoja antes de editar.")
        vistas.add(llave)
        tipo = _normal(fila["tipo"])
        if tipo not in {"ingreso", "gasto"}:
            raise ErrorPresupuesto("El presupuesto contiene un tipo de partida desconocido.")
        lineas.append({c: str(fila.get(c, "")) for c in (
            "linea_id", "tipo", "categoria", "concepto", "vigencia_desde", "vigencia_hasta")}
            | {"monto_mensual": str(_monto(fila["monto_mensual"])),
               "monto_quincenal": str(_monto(fila["monto_quincenal"], decimales=3))})
    return sorted(lineas, key=lambda f: (f["tipo"], f["categoria"], f["concepto"]))


def _asegurar(cx):
    cx.execute(text('CREATE SCHEMA IF NOT EXISTS "_bot"'))
    cx.execute(text(f'''CREATE TABLE IF NOT EXISTS {TABLA} (
        cliente_id TEXT NOT NULL, operacion_id TEXT NOT NULL,
        solicitud_hash TEXT NOT NULL, estado TEXT NOT NULL,
        fuente_id TEXT NOT NULL, revision_despues TEXT NOT NULL,
        detalle JSONB NOT NULL, creado_en TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (cliente_id, operacion_id))'''))


@contextmanager
def _bloqueo(cliente):
    destino, motor = dashboard_edicion._motor_jobs(cliente)
    llave = int.from_bytes(hashlib.sha256(
        ("presupuesto:" + str(cliente["cliente_id"])).encode()).digest()[:8], "big", signed=True)
    try:
        with motor.connect() as guardia:
            _asegurar(guardia)
            guardia.commit()
            # El lock pertenece a una transacción separada. Funciona también
            # detrás de un pooler por transacciones y no se libera al confirmar
            # la auditoría antes de llamar a Sheets.
            tomado = guardia.execute(text("SELECT pg_try_advisory_xact_lock(:llave)"), {"llave": llave}).scalar()
            if not tomado:
                raise ErrorPresupuesto("Ya se está guardando otro cambio. Espere unos segundos e inténtelo otra vez.")
            try:
                with motor.connect() as cx:
                    yield cx
            finally:
                guardia.rollback()
    finally:
        destino.cerrar()


def consultar(cliente, inicio):
    mes = _mes(inicio)
    _, _, _, encabezados, grilla = _leer(cliente)
    return {"ok": True, "revision": _revision(grilla), "inicio": mes.isoformat(),
            "mes_minimo": fecha_local().replace(day=1).isoformat(), "moneda": "CRC",
            "lineas": _vista(encabezados, grilla, mes)}


def _celda(valor, *, numero=False):
    if numero and valor != "" and not str(valor).startswith("="):
        return {"userEnteredValue": {"numberValue": float(Decimal(str(valor)))}}
    if str(valor).startswith("="):
        return {"userEnteredValue": {"formulaValue": str(valor)}}
    return {"userEnteredValue": {"stringValue": str(valor)}} if valor != "" else {}


def planificar(encabezados, grilla, datos, operacion_id):
    """Función pura: celdas exactas a escribir y resumen para auditar."""
    mes = _mes(datos.get("inicio"))
    if not fecha_local().replace(day=1) <= mes <= date(fecha_local().year + 5, 12, 1):
        raise ErrorPresupuesto("Solo puede modificar el mes actual o meses futuros (hasta cinco años).")
    cambios, nuevas = datos.get("cambios", []), datos.get("nuevas", [])
    if not isinstance(cambios, list) or not isinstance(nuevas, list) or not 0 < len(cambios) + len(nuevas) <= 100:
        raise ErrorPresupuesto("Incluya entre 1 y 100 cambios de partidas.")
    _vista(encabezados, grilla, mes)  # falla antes de escribir ante solapamientos
    salida = [list(f) for f in grilla]
    escrituras, detalle, ids = [], [], set()
    indice = {c: i for i, c in enumerate(encabezados) if c}
    registros = _registros(encabezados, grilla)
    for cambio in cambios:
        if not isinstance(cambio, dict) or set(cambio) != {"linea_id", "monto_mensual"}:
            raise ErrorPresupuesto("Solo se permite cambiar el monto de una partida existente.")
        linea = str(cambio["linea_id"])
        if linea in ids:
            raise ErrorPresupuesto("Una partida aparece repetida en los cambios.")
        ids.add(linea)
        candidatas = [(i, f) for i, f in registros if str(f["linea_id"]) == linea and _vigente(f, mes)]
        if len(candidatas) != 1:
            raise ErrorPresupuesto("Una partida ya no está vigente en el mes seleccionado. Recargue el editor.")
        fila_i, anterior = candidatas[0]
        for col in ("monto_mensual", "monto_quincenal", "vigencia_desde", "vigencia_hasta"):
            if str(anterior[col]).startswith("="):
                raise ErrorPresupuesto("Esta partida usa fórmulas; edítela directamente en la hoja.")
        monto = _monto(cambio["monto_mensual"])
        antes = _monto(anterior["monto_mensual"])
        if monto == antes:
            continue
        reemplazo = list(grilla[fila_i])
        reemplazo[indice["monto_mensual"]] = str(monto)
        reemplazo[indice["monto_quincenal"]] = str((monto / 2).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        if _fecha_celda(anterior["vigencia_desde"]) < mes:
            if any(str(v).startswith("=") for v in grilla[fila_i]):
                raise ErrorPresupuesto("Esta partida contiene fórmulas. Cree su nueva vigencia directamente en la hoja.")
            salida[fila_i][indice["vigencia_hasta"]] = (mes - timedelta(days=1)).isoformat()
            escrituras.append((fila_i, indice["vigencia_hasta"], salida[fila_i][indice["vigencia_hasta"]]))
            reemplazo[indice["vigencia_desde"]] = mes.isoformat()
            salida.append(reemplazo)
        else:
            for col in ("monto_mensual", "monto_quincenal"):
                salida[fila_i][indice[col]] = reemplazo[indice[col]]
                escrituras.append((fila_i, indice[col], reemplazo[indice[col]]))
        detalle.append({"linea_id": linea, "categoria": anterior["categoria"], "concepto": anterior["concepto"],
                        "tipo": anterior["tipo"], "antes": str(antes), "despues": str(monto),
                        "hasta": str(anterior["vigencia_hasta"]), "fila_anterior": grilla[fila_i]})
    categorias = {_normal(f["categoria"]): str(f["categoria"]) for _, f in registros}
    existentes = {(_normal(f["tipo"]), _normal(f["categoria"]), _normal(f["concepto"]))
                  for _, f in registros if _vigente(f, mes) or _fecha_celda(f["vigencia_desde"]) > mes}
    for n, nueva in enumerate(nuevas):
        if not isinstance(nueva, dict) or set(nueva) != {"tipo", "categoria", "concepto", "monto_mensual"}:
            raise ErrorPresupuesto("La nueva partida no es válida.")
        tipo = nueva["tipo"]
        if tipo not in {"ingreso", "gasto"}:
            raise ErrorPresupuesto("Seleccione Ingreso o Gasto.")
        categoria = _nombre(nueva["categoria"])
        categoria = categorias.get(_normal(categoria), categoria)
        categorias[_normal(categoria)] = categoria
        concepto = _nombre(nueva["concepto"])
        llave = (tipo, _normal(categoria), _normal(concepto))
        if llave in existentes:
            raise ErrorPresupuesto("Ya existe esa subpartida en la categoría. Cambie su monto en lugar de duplicarla.")
        existentes.add(llave)
        monto = _monto(nueva["monto_mensual"])
        linea = ("gas_" if tipo == "gasto" else "ing_") + uuid.uuid5(uuid.NAMESPACE_URL, operacion_id + ":" + str(n)).hex
        fila = [""] * len(grilla[0])
        valores = {"linea_id": linea, "tipo": tipo, "categoria": categoria, "concepto": concepto,
                   "subcategoria": concepto, "clave": categoria + " > " + concepto,
                   "monto_mensual": str(monto), "monto_quincenal": str((monto / 2).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
                   "vigencia_desde": mes.isoformat(), "vigencia_hasta": "",
                   "rastreable": "si", "medio_pago": "transferencia", "pagable": "si" if tipo == "gasto" else "no"}
        for col, valor in valores.items():
            if col in indice:
                fila[indice[col]] = valor
        salida.append(fila)
        detalle.append({"linea_id": linea, "tipo": tipo, "categoria": categoria, "concepto": concepto,
                        "antes": None, "despues": str(monto), "hasta": ""})
    if not detalle:
        raise ErrorPresupuesto("No hay cambios de montos para guardar.")
    return {"grilla": salida, "escrituras": escrituras, "detalle": detalle, "inicio": mes.isoformat(),
            "nuevas_filas": salida[len(grilla):]}


def _aplicar(libro, hoja, plan):
    peticiones = []
    monetarias = {plan["grilla"][0].index(col) for col in ("monto_mensual", "monto_quincenal")}
    for fila, columna, valor in plan["escrituras"]:
        peticiones.append({"updateCells": {"start": {"sheetId": hoja.id, "rowIndex": fila,
                                                     "columnIndex": columna},
                                             "rows": [{"values": [_celda(valor, numero=columna in monetarias)]}], "fields": "userEnteredValue"}})
    if plan["nuevas_filas"]:
        peticiones.append({"appendCells": {"sheetId": hoja.id,
            "rows": [{"values": [_celda(v, numero=i in monetarias) for i, v in enumerate(fila)]}
                     for fila in plan["nuevas_filas"]],
            "fields": "userEnteredValue"}})
    libro.batch_update({"requests": peticiones})


def guardar(cliente, datos):
    if not isinstance(datos, dict) or datos.get("confirmado") is not True:
        raise ErrorPresupuesto("Revise y confirme los cambios antes de guardar.")
    try:
        operacion_id = str(uuid.UUID(str(datos.get("operacion_id"))))
    except ValueError as exc:
        raise ErrorPresupuesto("La solicitud de guardado no es válida.") from exc
    firma = hashlib.sha256(json.dumps(datos, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    parametros = {"cid": str(cliente["cliente_id"]), "id": operacion_id}
    with _bloqueo(cliente) as cx:
        anterior = cx.execute(text(f"SELECT * FROM {TABLA} WHERE cliente_id=:cid AND operacion_id=:id"), parametros).mappings().first()
        fuente, libro, hoja, encabezados, grilla = _leer(cliente, escritura=True)
        if anterior:
            if anterior["solicitud_hash"] != firma:
                raise ErrorPresupuesto("Esta confirmación ya se usó para otros cambios. Recargue el editor.")
            if anterior["estado"] in {"preparado", "incierto"} and _revision(grilla) != anterior["revision_despues"]:
                raise ErrorPresupuesto("No pude verificar el guardado anterior. Recargue y revise la hoja; no lo repetiré automáticamente.")
            if anterior["estado"] == "encolado":
                return {"ok": True, "guardado": True, "trabajo_clave": "presupuesto." + operacion_id}
        else:
            if datos.get("revision") != _revision(grilla):
                raise ErrorPresupuesto("El presupuesto cambió desde que abrió el editor. Recargue para no sobrescribir cambios.")
            plan = planificar(encabezados, grilla, datos, operacion_id)
            cx.execute(text(f'''INSERT INTO {TABLA}
                (cliente_id,operacion_id,solicitud_hash,estado,fuente_id,revision_despues,detalle)
                VALUES (:cid,:id,:firma,'preparado',:fuente,:revision,CAST(:detalle AS jsonb))'''),
                parametros | {"firma": firma, "fuente": fuente["fuente_id"], "revision": _revision(plan["grilla"]),
                              "detalle": json.dumps({"inicio": plan["inicio"], "cambios": plan["detalle"]}, ensure_ascii=False)})
            cx.commit()  # auditoría durable ANTES de la escritura externa
            try:
                _aplicar(libro, hoja, plan)
            except Exception as exc:
                # Un timeout no prueba que Sheets rechazó el lote. Leer una vez
                # permite reconocer un guardado completo sin duplicarlo.
                _, _, _, _, actuales = _leer(cliente)
                if _revision(actuales) != _revision(plan["grilla"]):
                    cx.execute(text(f"UPDATE {TABLA} SET estado='incierto' WHERE cliente_id=:cid AND operacion_id=:id"), parametros)
                    cx.commit()
                    raise ErrorPresupuesto("No pude confirmar el guardado en Google Sheets. Recargue y revise antes de reintentar.") from exc
        cx.execute(text(f"UPDATE {TABLA} SET estado='guardado' WHERE cliente_id=:cid AND operacion_id=:id"), parametros)
        cx.commit()
        try:
            _encolar(cliente, cx, operacion_id, fuente["fuente_id"])
        except Exception:
            logger.exception("Presupuesto guardado; sincronización pendiente de recuperación")
        return {"ok": True, "guardado": True, "trabajo_clave": "presupuesto." + operacion_id,
                "mensaje": "Presupuesto guardado. Actualizando el dashboard…"}


def _encolar(cliente, cx, operacion_id, fuente_id):
    dashboard_edicion._encolar_reconstruccion(cliente, "presupuesto." + operacion_id, fuente_id)
    cx.execute(text(f"UPDATE {TABLA} SET estado='encolado' WHERE cliente_id=:cid AND operacion_id=:id"),
               {"cid": str(cliente["cliente_id"]), "id": operacion_id})
    cx.commit()


def recuperar_guardados(cliente):
    """Recupera un reinicio entre Sheets y la cola sin repetir escrituras."""
    with _bloqueo(cliente) as cx:
        dashboard_edicion._asegurar_jobs(cx)
        pendientes = cx.execute(text(f'''SELECT p.* FROM {TABLA} p
            LEFT JOIN "_bot"."dashboard_edicion_jobs" j
              ON j.cliente_id=p.cliente_id AND j.movimiento_clave='presupuesto.' || p.operacion_id
            WHERE p.cliente_id=:cid AND (p.estado IN ('guardado','preparado')
                OR (p.estado='encolado' AND j.estado='error'
                    AND j.actualizado_en < CURRENT_TIMESTAMP - INTERVAL '1 minute'))'''),
                                {"cid": str(cliente["cliente_id"])}).mappings().all()
        revision = None
        for fila in pendientes:
            if fila["estado"] == "preparado":
                if revision is None:
                    _, _, _, _, grilla = _leer(cliente)
                    revision = _revision(grilla)
                if revision != fila["revision_despues"]:
                    continue  # nunca ejecuta de nuevo un lote de resultado incierto
            _encolar(cliente, cx, fila["operacion_id"], fila["fuente_id"])


def estado_guardado(cliente, operacion_id):
    try:
        operacion_id = str(uuid.UUID(str(operacion_id)))
    except ValueError as exc:
        raise ErrorPresupuesto("La operación no es válida.") from exc
    destino, motor = dashboard_edicion._motor_jobs(cliente)
    try:
        with motor.begin() as cx:
            _asegurar(cx)
            fila = cx.execute(text(f"SELECT estado FROM {TABLA} WHERE cliente_id=:cid AND operacion_id=:id"),
                              {"cid": str(cliente["cliente_id"]), "id": operacion_id}).mappings().first()
        if not fila:
            raise ErrorPresupuesto("No encontré ese guardado.")
        return str(fila["estado"])
    finally:
        destino.cerrar()
