"""
Registro de clientes y usuarios, leido del SHEET MAESTRO.

El Sheet maestro tiene dos pestanias:
  - 'clientes':  columnas  cliente_id | nombre | spreadsheet_id
  - 'usuarios':  columnas  numero | cliente_id

Un mismo cliente puede tener varios usuarios (varios numeros -> mismo cliente_id).

resolver(numero) devuelve el dict del cliente correspondiente, o None si el
numero no esta registrado.
"""

import time
import logging

import config
from gclient import abrir_libro

logger = logging.getLogger("fachavi.registry")

# Cache: {"ts": epoch, "usuarios": {numero: cliente_id}, "clientes": {cliente_id: {...}}}
_cache = {"ts": 0.0, "usuarios": {}, "clientes": {}}


def _normaliza_numero(n: str) -> str:
    """Deja solo digitos (quita +, espacios, guiones) para comparar parejo."""
    return "".join(ch for ch in str(n) if ch.isdigit())


def _cargar():
    """Lee las pestanias 'clientes' y 'usuarios' del Sheet maestro."""
    if not config.MASTER_SPREADSHEET_ID:
        raise RuntimeError("Falta MASTER_SPREADSHEET_ID (Sheet maestro).")

    libro = abrir_libro(config.MASTER_SPREADSHEET_ID)

    # --- clientes ---
    ws_cli = libro.worksheet("clientes")
    filas_cli = ws_cli.get_all_records()  # usa la 1a fila como encabezado
    clientes = {}
    for f in filas_cli:
        cid = str(f.get("cliente_id", "")).strip()
        if not cid:
            continue
        clientes[cid] = {
            "cliente_id": cid,
            "nombre": str(f.get("nombre", "")).strip(),
            "spreadsheet_id": str(f.get("spreadsheet_id", "")).strip(),
        }

    # --- usuarios ---
    ws_usr = libro.worksheet("usuarios")
    filas_usr = ws_usr.get_all_records()
    usuarios = {}
    for f in filas_usr:
        num = _normaliza_numero(f.get("numero", ""))
        cid = str(f.get("cliente_id", "")).strip()
        if num and cid:
            usuarios[num] = cid

    logger.info("Registro cargado: %d clientes, %d usuarios", len(clientes), len(usuarios))
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
    {cliente_id, nombre, spreadsheet_id} o None si no esta registrado.
    """
    _asegura_cache()
    num = _normaliza_numero(numero)
    cid = _cache["usuarios"].get(num)
    if not cid:
        return None
    return _cache["clientes"].get(cid)
