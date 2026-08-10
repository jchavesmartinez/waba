"""
Propone filas de metadata a partir de ejemplos reales.

Le das unas filas de una tabla raw y devuelve el 'filtro', el extractor y los
'_campos' listos para revisar y pegar en el Sheet. Es el paso que hoy toma
media hora por cada tipo de documento nuevo: abrir varios correos, ver que
etiquetas se repiten, notar que la fecha viene en espanol, que el monto trae la
moneda pegada, que la etiqueta de la tarjeta cambia entre MASTER y AMEX.

CORRE BAJO DEMANDA, no en cada corrida. Y esa es toda la diferencia.

Lo que queda escrito en el Sheet es texto DECLARATIVO ('etiqueta: Comercio'),
que el motor ejecuta despues con reglas fijas, siempre igual. El LLM escribe la
receta; el motor cocina. Si mañana se apaga la API, el pipeline sigue
funcionando identico.

La alternativa —que un modelo interprete "extrae el gasto de los correos del
BAC" en cada corrida— parece mas simple y es peor en todo lo que importa: el
mismo correo puede dar dos resultados distintos, el costo crece con cada fila,
un error sale directo a produccion y no hay donde mirar cuando un numero no
cuadra.

NUNCA ESCRIBE EN EL SHEET. Devuelve texto para pegar. El paso de revision no es
burocracia: es la diferencia entre "una IA decidio esto" y "esto esta definido
y alguien lo aprobo". Cuando un numero no cuadre —y va a pasar— vas a poder
abrir el Sheet, ver la regla y arreglarla.

ADEMAS PRUEBA LA PROPUESTA. Antes de devolverla, corre el motor con esa
metadata contra las mismas filas de ejemplo y reporta que extrajo y que
rechazo. Una propuesta que se ve razonable pero no extrae nada es facil de
producir; que la revises ya probada cambia por completo cuanto te podes fiar.
"""

import argparse
import json
import logging
import sys

import config
import registry
from warehouse import crear_destino
from warehouse.base import nombre_esquema
from . import extractores
from .motor import Modelo
from .tipos import tipos_disponibles

logger = logging.getLogger("fachavi.modelo.proponer")

MAX_TOKENS = 3000
MAX_CHARS_MUESTRA = 2500
MUESTRAS_POR_DEFECTO = 5

_cliente = None


def _anthropic():
    global _cliente
    if _cliente is None:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("falta ANTHROPIC_API_KEY")
        _cliente = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _cliente


def proponer(muestras: list, tabla_origen: str, columna_texto: str = "cuerpo",
             modelo_id: str = "mi_modelo") -> dict:
    """
    Devuelve {"propuesta": {...}, "preset": str|None, "prueba": {...},
              "avisos": [...]}.

    `muestras` son dicts de la tabla raw.
    """
    avisos = []
    textos = [str(m.get(columna_texto) or "")[:MAX_CHARS_MUESTRA]
              for m in muestras if m.get(columna_texto)]
    if not textos:
        raise RuntimeError(
            f"ninguna de las {len(muestras)} filas tiene contenido en la "
            f"columna '{columna_texto}'.")

    # Antes de pedirle nada al modelo: si ya existe un extractor CON NOMBRE que
    # lee este documento, usarlo es mejor que inventar campos nuevos. Un preset
    # ya viene probado, con sus tipos y sus casos raros resueltos.
    preset = _preset_que_calza(textos)
    if preset:
        avisos.append(
            f"Estos documentos los lee el extractor '{preset}', que ya existe. "
            "La propuesta usa ese preset: no hace falta escribir '_campos' "
            "salvo para pisar algo puntual.")
        propuesta = {
            "modelo_id": modelo_id,
            "tabla_origen": tabla_origen,
            "tabla_destino": f"{modelo_id}__tabla",
            "extractor": preset,
            "columna_texto": columna_texto,
            "filtro": extractores.crear_extractor(preset).filtro_sugerido,
            "campos": [],
        }
    else:
        propuesta = _preguntar(textos, tabla_origen, columna_texto, modelo_id)
        propuesta, mas = _validar(propuesta, tabla_origen, columna_texto,
                                  modelo_id)
        avisos += mas

    prueba = _probar(propuesta, muestras)
    return {"propuesta": propuesta, "preset": preset, "prueba": prueba,
            "avisos": avisos}


def _preset_que_calza(textos: list) -> str:
    """
    Busca un extractor con nombre cuyo filtro sugerido calce con las muestras.

    Es una heuristica deliberadamente simple: se mira si el texto del filtro
    (por ejemplo '%baccredomatic%') aparece en las muestras. Alcanza para
    evitar el caso caro, que es proponerle a alguien campos nuevos para un
    documento que el sistema ya sabe leer.
    """
    for tipo in extractores.tipos_disponibles():
        ex = extractores.crear_extractor(tipo)
        filtro = (ex.filtro_sugerido or "").strip()
        if not filtro or not ex.campos_por_defecto:
            continue
        marcas = [p.strip("%' ") for p in filtro.split("'") if "%" in p]
        for marca in marcas:
            if marca and all(marca.lower() in t.lower() for t in textos):
                return tipo
    return ""


def _preguntar(textos: list, tabla_origen: str, columna_texto: str,
               modelo_id: str) -> dict:
    bloques = "\n\n".join(
        f"--- EJEMPLO {i + 1} ---\n{t}" for i, t in enumerate(textos))

    prompt = f"""Analiza estos documentos y proponé la metadata para extraerlos
a una tabla.

{bloques}

EXTRACTORES DISPONIBLES:
- etiqueta_valor: el documento tiene pares "Etiqueta: valor", uno por linea.
- regex: no esta etiquetado; cada campo necesita su patron con un grupo de
  captura.
- json_ruta: el contenido es JSON; el patron es una ruta tipo "a.b.c".

TIPOS DISPONIBLES: {", ".join(tipos_disponibles())}
- monto_con_moneda: para valores como "CRC 3,320.00". Produce DOS columnas
  (monto y monto_moneda).
- etiqueta_de_lista: cuando la ETIQUETA misma es un dato porque cambia entre
  documentos. El patron es la lista de etiquetas posibles separadas por comas.
  Produce DOS columnas (valor y valor_tipo).
- fecha_hora_es: fechas en espanol tipo "Ago 9, 2026 , 16:22".

REGLAS:
- Identificadores como numeros de autorizacion o referencia van como "texto",
  NUNCA como "entero": pierden los ceros de la izquierda y a veces traen
  letras.
- Marca requerido="si" SOLO en los campos sin los cuales la fila no sirve de
  nada. Un campo que a veces viene vacio en los ejemplos NO puede ser
  requerido.
- El filtro es SQL que va despues de un WHERE sobre la columna
  '{columna_texto}'. Tiene que reconocer estos documentos y NO otros. Preferi
  buscar en el contenido y no en el remitente: un documento reenviado a mano
  tiene el remitente de quien lo reenvio.
- Si un campo conviene clasificar despues (un comercio, un proveedor, una
  categoria), poné en 'clasifica_en' el nombre de la columna donde caeria esa
  clasificacion.
- Nombres de columna en minusculas, sin acentos ni espacios.

Responde SOLO este JSON, sin texto alrededor ni bloques de codigo:
{{"filtro": "...",
  "extractor": "etiqueta_valor",
  "tabla_destino": "...",
  "campos": [
    {{"columna": "...", "tipo": "...", "patron": "...", "requerido": "si",
      "clasifica_en": "", "descripcion": "..."}}
  ]}}"""

    resp = _anthropic().messages.create(
        model=config.BOT_MODELO_KPIS,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if getattr(resp, "stop_reason", "") == "max_tokens":
        raise RuntimeError(
            "la respuesta se corto por longitud; probá con menos ejemplos o "
            "con documentos mas cortos.")

    texto = "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    texto = texto.removeprefix("```json").removeprefix("```").removesuffix("```")
    datos = json.loads(texto.strip())
    datos["modelo_id"] = modelo_id
    datos["tabla_origen"] = tabla_origen
    datos["columna_texto"] = columna_texto
    return datos


def _validar(propuesta: dict, tabla_origen, columna_texto, modelo_id) -> tuple:
    """
    Corrige lo que se puede y avisa lo que no.

    El modelo puede proponer un extractor o un tipo que no existe. Dejarlo
    pasar significaria que el error aparece recien cuando alguien pega las
    filas en el Sheet y corre el motor, con un mensaje que no menciona de donde
    salio.
    """
    avisos = []

    extractor = (propuesta.get("extractor") or "").strip()
    if extractor not in extractores.tipos_disponibles():
        avisos.append(
            f"proponia el extractor '{extractor}', que no existe; se cambia a "
            "'etiqueta_valor'. Revisalo.")
        extractor = "etiqueta_valor"
    propuesta["extractor"] = extractor

    validos = set(tipos_disponibles())
    campos = []
    for campo in propuesta.get("campos", []):
        columna = str(campo.get("columna", "")).strip()
        if not columna:
            continue
        tipo = str(campo.get("tipo", "texto")).strip()
        if tipo not in validos:
            avisos.append(
                f"columna '{columna}': el tipo '{tipo}' no existe; se usa "
                "'texto'. Revisalo.")
            tipo = "texto"
        # Un identificador guardado como entero pierde los ceros de la
        # izquierda ('000382' -> 382) y revienta si trae letras. Es un error
        # silencioso y facil de cometer, asi que se corrige de oficio.
        if tipo == "entero" and any(
                p in columna for p in (
                    "id", "autoriza", "referencia", "codigo", "telefono",
                    "cedula", "consecutivo", "clave", "numero", "num_",
                    "factura", "cuenta", "tarjeta", "serie", "folio")):
            avisos.append(
                f"columna '{columna}': venia como 'entero' y se cambia a "
                "'texto'. Los identificadores pierden los ceros de la "
                "izquierda si se guardan como numero.")
            tipo = "texto"
        campos.append({
            "columna": columna,
            "tipo": tipo,
            "patron": str(campo.get("patron", "")).strip(),
            "requerido": str(campo.get("requerido", "")).strip().lower(),
            "clasifica_en": str(campo.get("clasifica_en", "")).strip(),
            "descripcion": str(campo.get("descripcion", "")).strip(),
        })

    if not campos:
        raise RuntimeError("la propuesta no trae ningun campo utilizable.")

    propuesta["campos"] = campos
    propuesta["modelo_id"] = modelo_id
    propuesta["tabla_origen"] = tabla_origen
    propuesta["columna_texto"] = columna_texto
    propuesta.setdefault("tabla_destino", f"{modelo_id}__tabla")
    propuesta.setdefault("filtro", "")
    return propuesta, avisos


def _probar(propuesta: dict, muestras: list) -> dict:
    """
    Corre el motor con la metadata propuesta contra las mismas muestras.

    Es la parte que hace util a todo esto. Una propuesta que se ve razonable
    pero no extrae nada es facil de producir; verla ya probada es la diferencia
    entre revisar texto y revisar un resultado.
    """
    metadata = {
        "campos": [{**c, "modelo_id": propuesta["modelo_id"]}
                   for c in propuesta.get("campos", [])],
        "clasificacion": [], "categorias": [], "overrides": [],
    }
    try:
        modelo = Modelo(propuesta, metadata)
        filas, rechazos = modelo.procesar(muestras)
    except RuntimeError as e:
        return {"error": str(e), "filas": [], "rechazos": [], "columnas": []}

    vacias = {}
    for columna, _ in modelo.columnas():
        if columna.startswith("_"):
            continue
        n = sum(1 for f in filas if f.get(columna) in (None, ""))
        if filas and n == len(filas):
            vacias[columna] = n

    return {"filas": filas, "rechazos": rechazos,
            "columnas": [c for c, _ in modelo.columnas()],
            "siempre_vacias": sorted(vacias)}


# --------------------------------------------------------------------------
# Salida para pegar en el Sheet
# --------------------------------------------------------------------------

_COLS_MODELOS = ("modelo_id", "tabla_origen", "tabla_destino", "extractor",
                 "columna_texto", "filtro", "activo")
_COLS_CAMPOS = ("modelo_id", "columna", "tipo", "patron", "requerido",
                "clasifica_en", "descripcion")


def a_tsv(propuesta: dict) -> str:
    """
    TSV listo para pegar en Google Sheets.

    Separado por tabuladores y no por comas: al pegar, Sheets reparte los
    tabuladores en columnas solo, y las descripciones suelen traer comas.
    """
    fila_modelo = [propuesta.get(c, "") for c in _COLS_MODELOS[:-1]] + ["si"]
    lineas = [
        "# Pegá esta fila en la pestaña _modelos:",
        "\t".join(_COLS_MODELOS),
        "\t".join(str(v) for v in fila_modelo),
    ]
    if propuesta.get("campos"):
        lineas += ["", "# Pegá estas filas en la pestaña _campos:",
                   "\t".join(_COLS_CAMPOS)]
        for campo in propuesta["campos"]:
            lineas.append("\t".join(
                str(campo.get(c, "") if c != "modelo_id"
                    else propuesta["modelo_id"]) for c in _COLS_CAMPOS))
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Propone filas de _modelos y _campos desde ejemplos reales.")
    ap.add_argument("--cliente", required=True)
    ap.add_argument("--tabla", required=True,
                    help="tabla raw de la que salen los ejemplos")
    ap.add_argument("--columna", default="cuerpo",
                    help="columna con el texto a analizar (por defecto 'cuerpo')")
    ap.add_argument("--filtro", default="",
                    help="SQL opcional para elegir los ejemplos")
    ap.add_argument("--muestras", type=int, default=5,
                    help="cuantas filas mirar (por defecto 5)")
    ap.add_argument("--modelo-id", default="mi_modelo")
    ap.add_argument("--salida", help="archivo .tsv donde guardar la propuesta")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cliente = next((c for c in registry.listar_clientes()
                    if c["cliente_id"] == args.cliente), None)
    if not cliente:
        logger.error("no existe el cliente '%s'", args.cliente)
        return 1

    destino = crear_destino(config.WAREHOUSE_TIPO,
                            config.dsn_de_cliente(cliente))
    try:
        esquema = nombre_esquema(args.cliente)
        sql = f'SELECT * FROM "{esquema}"."{args.tabla}"'
        if args.filtro:
            sql += f" WHERE {args.filtro}"
        sql += f" LIMIT {max(1, args.muestras)}"
        muestras = destino.leer_filas(sql)
    finally:
        destino.cerrar()

    if not muestras:
        logger.error("la consulta no devolvio filas. SQL: %s", sql)
        return 1

    logger.info("analizando %d fila(s) de %s.%s ...",
                len(muestras), esquema, args.tabla)
    r = proponer(muestras, args.tabla, args.columna, args.modelo_id)
    propuesta, prueba = r["propuesta"], r["prueba"]

    print()
    if r["preset"]:
        print(f"YA EXISTE UN EXTRACTOR PARA ESTO: {r['preset']}")
        print()

    for aviso in r["avisos"]:
        print(f"  aviso: {aviso}")
    if r["avisos"]:
        print()

    print("--- PROPUESTA (revisala antes de pegarla) ---")
    print(a_tsv(propuesta))
    print()

    print("--- RESULTADO DE PROBARLA CON ESTOS MISMOS EJEMPLOS ---")
    if prueba.get("error"):
        print(f"  la propuesta NO es valida: {prueba['error']}")
        return 1

    print(f"  {len(prueba['filas'])} fila(s) extraidas, "
          f"{len(prueba['rechazos'])} rechazada(s)")
    for fila in prueba["filas"][:3]:
        visibles = {k: v for k, v in fila.items() if not k.startswith("_")}
        print("   ", json.dumps(visibles, ensure_ascii=False, default=str)[:300])
    for rechazo in prueba["rechazos"][:3]:
        print(f"    RECHAZADA: {rechazo['motivo']}")

    if prueba.get("siempre_vacias"):
        # Casi siempre significa que la etiqueta propuesta no coincide con la
        # real. Es el error mas comun y el mas facil de pasar por alto, porque
        # la tabla se crea igual y la columna simplemente queda en blanco.
        print(f"\n  OJO: estas columnas quedaron VACIAS en todos los "
              f"ejemplos: {', '.join(prueba['siempre_vacias'])}")
        print("  Revisá que el 'patron' coincida con la etiqueta real del "
              "documento.")

    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            f.write(a_tsv(propuesta))
        print(f"\nGuardado en {args.salida}")

    print("\nSiguiente paso: pegá las filas en el Sheet del cliente y corré")
    print("  python construir_modelos.py --cliente "
          f"{args.cliente} --probar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
