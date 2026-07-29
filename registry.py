"""
Registro de clientes, usuarios y FUENTES, leido del SHEET MAESTRO.

Pestanias del Sheet maestro:

  'clientes':  cliente_id | nombre | [spreadsheet_id]
      (spreadsheet_id es OPCIONAL y solo por retrocompatibilidad: si esta y el
       cliente no tiene filas en 'fuentes', se toma como una fuente Google Sheets.)

  'usuarios':  numero | cliente_id
      (varios numeros pueden apuntar al mismo cliente_id)

  'fuentes':   cliente_id | fuente_id | tipo | activo | config      <-- NUEVO
      - tipo:   google_sheets | csv_url | postgres | ...
      - activo: si / no  (asi se PRENDE/APAGA una fuente, sin tocar codigo)
      - config: JSON con los datos de conexion de esa fuente
                p.ej. google_sheets -> {"spreadsheet_id":"1AbC..."}

resolver(numero) devuelve el dict del cliente (con su lista de 'fuentes'
activas e inactivas), o None si el numero no esta registrado.
"""

import json
import time
import logging

import config
from gclient import abrir_libro

logger = logging.getLogger("fachavi.registry")

# Cache: {"ts", "usuarios": {numero: cliente_id}, "clientes": {cliente_id: {...}}}
_cache = {"ts": 0.0, "usuarios": {}, "clientes": {}}

_ACTIVO_SI = {"si", "sí", "true", "1", "x", "activo", "on", "yes"}
_ACTIVO_NO = {"no", "false", "0", "", "inactivo", "off"}


def _normaliza_numero(n: str) -> str:
    """Deja solo digitos (quita +, espacios, guiones) para comparar parejo."""
    return "".join(ch for ch in str(n) if ch.isdigit())


def _parse_activo(v) -> bool:
    s = str(v).strip().lower()
    if s in _ACTIVO_NO:
        return False
    if s in _ACTIVO_SI:
        return True
    return True  # por defecto, activa


def _parse_entero(v) -> int:
    """
    Convierte a entero; vacio o invalido -> 0.

    Tolera '15.0': Google Sheets y pandas devuelven los numeros como float
    segun el formato de la celda, y un int('15.0') pelado revienta. Ese fallo
    era SILENCIOSO: la frescura caia a 0 y la fuente se resincronizaba en cada
    corrida, gastando llamadas de API sin que nadie lo notara.
    """
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        logger.warning("Valor entero invalido en el registro: %r -> se usa 0", v)
        return 0


def _parse_config(v) -> dict:
    """La columna config es JSON. Vacio -> {}. JSON invalido -> {} + warning."""
    s = str(v).strip()
    if not s:
        return {}
    try:
        cfg = json.loads(s)
        return cfg if isinstance(cfg, dict) else {}
    except json.JSONDecodeError as e:
        logger.warning("config no es JSON valido (%s): %.60s", e, s)
        return {}


def _cargar():
    """Lee 'clientes', 'usuarios' y (si existe) 'fuentes' del Sheet maestro."""
    if not config.MASTER_SPREADSHEET_ID:
        raise RuntimeError("Falta MASTER_SPREADSHEET_ID (Sheet maestro).")

    libro = abrir_libro(config.MASTER_SPREADSHEET_ID)

    # --- clientes ---
    ws_cli = libro.worksheet("clientes")
    clientes = {}
    for f in ws_cli.get_all_records():
        cid = str(f.get("cliente_id", "")).strip()
        if not cid:
            continue
        clientes[cid] = {
            "cliente_id": cid,
            "nombre": str(f.get("nombre", "")).strip(),
            # legado: spreadsheet_id directo en la fila de cliente (opcional)
            "_spreadsheet_id_legado": str(f.get("spreadsheet_id", "")).strip(),
            "fuentes": [],
        }

    # --- usuarios (solo lo usa el BOT para resolver numero -> cliente) ---
    # Opcional: la ingesta no lo necesita. Asi se puede arrancar la Fase 2 con
    # un Sheet maestro que solo tenga 'clientes' y 'fuentes'.
    usuarios = {}
    try:
        ws_usr = libro.worksheet("usuarios")
        filas_usr = ws_usr.get_all_records()
    except Exception:  # noqa: BLE001
        filas_usr = []
        logger.info("Sin pestania 'usuarios' (solo hace falta para el bot).")

    for f in filas_usr:
        num = _normaliza_numero(f.get("numero", ""))
        cid = str(f.get("cliente_id", "")).strip()
        if num and cid:
            usuarios[num] = cid

    # --- fuentes (pestania nueva; opcional) ---
    try:
        ws_src = libro.worksheet("fuentes")
        filas_src = ws_src.get_all_records()
    except Exception:  # noqa: BLE001
        filas_src = []
        logger.info("Sin pestania 'fuentes'; se usara retrocompat de spreadsheet_id.")

    for f in filas_src:
        cid = str(f.get("cliente_id", "")).strip()
        if cid not in clientes:
            continue
        fuente = {
            "fuente_id": str(f.get("fuente_id", "")).strip() or "fuente",
            "tipo": str(f.get("tipo", "")).strip().lower(),
            "activo": _parse_activo(f.get("activo", "si")),
            # Cada cuantos minutos vale la pena re-sincronizar esta fuente.
            # 0 / vacio = sin politica (se sincroniza en cada corrida del job).
            "frescura_minutos": _parse_entero(f.get("frescura_minutos", 0)),
            "config": _parse_config(f.get("config", "")),
        }
        clientes[cid]["fuentes"].append(fuente)

    # --- retrocompatibilidad ---
    # Cliente sin filas en 'fuentes' pero con spreadsheet_id -> 1 fuente Sheets.
    for cid, c in clientes.items():
        if not c["fuentes"] and c["_spreadsheet_id_legado"]:
            c["fuentes"].append({
                "fuente_id": "sheet_principal",
                "tipo": "google_sheets",
                "activo": True,
                "config": {"spreadsheet_id": c["_spreadsheet_id_legado"]},
            })
            logger.info("[%s] retrocompat: spreadsheet_id -> fuente google_sheets", cid)
        c.pop("_spreadsheet_id_legado", None)

    total_fuentes = sum(len(c["fuentes"]) for c in clientes.values())
    logger.info(
        "Registro cargado: %d clientes, %d usuarios, %d fuentes",
        len(clientes), len(usuarios), total_fuentes,
    )
    return usuarios, clientes


def _asegura_cache():
    ahora = time.time()
    if _cache["clientes"] and (ahora - _cache["ts"]) < config.REGISTRY_CACHE_TTL:
        return
    usuarios, clientes = _cargar()
    _cache.update({"ts": ahora, "usuarios": usuarios, "clientes": clientes})


def resolver(numero: str):
    """
    Dado el numero de quien escribe, devuelve el dict del cliente
    {cliente_id, nombre, fuentes:[...]} o None si no esta registrado.
    """
    _asegura_cache()
    num = _normaliza_numero(numero)
    cid = _cache["usuarios"].get(num)
    if not cid:
        return None
    return _cache["clientes"].get(cid)


def listar_clientes() -> list:
    """
    Devuelve todos los clientes del registro con sus fuentes.
    Lo usa el job de ingesta (sync.py) para recorrer el universo completo.
    """
    _asegura_cache()
    return list(_cache["clientes"].values())


def recargar():
    """Fuerza relectura del registro en la proxima resolucion."""
    _cache["ts"] = 0.0
