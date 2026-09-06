"""Evaluación manual reproducible de seguimientos contra el bot real.

No usa webhook ni escribe en Neon: llama al mismo ``responder`` que usa
WhatsApp y sustituye únicamente la memoria por una memoria efímera por número.
Ejemplo:
  python scripts/evaluar_followups.py --env-file /ruta/waba.env
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv


CONVERSACIONES = [
    ("alimentos_periodos", [
        "¿Cuánto gasté en alimentación en agosto de 2026?",
        "¿Y en septiembre?",
        "¿Cuál de los dos meses fue mayor?",
    ]),
    ("walmart_periodos", [
        "¿Cuánto gasté en Walmart en agosto de 2026?",
        "¿Y en septiembre?",
        "¿Cuál de los dos meses fue mayor?",
    ]),
    ("conteo_periodos", [
        "¿Cuántos gastos hubo en agosto de 2026?",
        "¿Y en septiembre?",
        "¿Cuál de los dos meses tuvo más movimientos?",
    ]),
    ("presupuesto_a_ejecucion", [
        "¿Cuál fue el presupuesto de alimentación en agosto de 2026?",
        "¿Cuánto se gastó?",
        "¿Cuánto queda disponible?",
    ]),
    ("conceptos_presupuesto", [
        "¿Cuánto gasté en comedera en agosto de 2026?",
        "¿Y en comidas afuera?",
        "¿Cuál tiene mayor presupuesto?",
    ]),
    ("categorias_ordinal", [
        "¿Cuánto gasté por categoría en septiembre de 2026?",
        "¿Cuál fue la segunda?",
        "¿Cuánto gasté en esa?",
    ]),
    ("detalle_suma_maximo", [
        "¿Qué gastos hubo el 5 de septiembre de 2026?",
        "¿Cuánto suman?",
        "¿Cuál fue el más alto?",
    ]),
    ("transporte_periodos", [
        "¿Cuánto gasté en transporte en agosto de 2026?",
        "¿Y en septiembre?",
        "¿Cuál fue mayor?",
    ]),
    ("categorias_comparacion", [
        "¿Cuánto gasté en vivienda en agosto de 2026?",
        "¿Y en deudas?",
        "¿Cuál fue mayor?",
    ]),
    ("excesos", [
        "¿Qué categorías excedieron el presupuesto en agosto de 2026?",
        "¿Cuál tuvo el mayor exceso?",
        "¿Cuánto fue el exceso?",
    ]),
    ("concepto_disponible", [
        "¿Cuánto gasté en comidas afuera en agosto de 2026?",
        "¿Cuánto queda disponible?",
        "¿Qué porcentaje del presupuesto se consumió?",
    ]),
    ("comercio_detalle", [
        "¿Cuánto gasté en Walmart en agosto de 2026?",
        "¿Qué transacciones forman ese total?",
        "¿Cuál fue la más alta?",
    ]),
    ("roga_clasificacion", [
        "¿Cuál es la clasificación de la transacción de Roga del 4 de septiembre de 2026?",
        "¿Cuál es su categoría y concepto?",
        "¿Cuánto fue el gasto?",
    ]),
    ("detalles_fecha", [
        "¿Qué gastos hubo el 4 de septiembre de 2026?",
        "¿Cuántos fueron?",
        "¿Cuál fue el de mayor monto?",
    ]),
    ("top_conceptos", [
        "¿Cuáles fueron los dos conceptos con mayor gasto en CRC en agosto de 2026?",
        "¿Cuál fue el primero?",
        "¿Cuánto gasté en ese concepto?",
    ]),
    ("otros_periodos", [
        "¿Cuánto gasté en otros en agosto de 2026?",
        "¿Y en septiembre?",
        "¿Cuánto varió?",
    ]),
    ("presupuesto_categoria", [
        "¿Cuál fue el presupuesto de transporte en agosto de 2026?",
        "¿Y cuánto se gastó?",
        "¿Quedó disponible?",
    ]),
    ("comidas_detalle", [
        "¿Qué transacciones forman el gasto de comidas afuera en agosto de 2026?",
        "¿Cuánto suman?",
        "¿Cuál fue la más alta?",
    ]),
    ("categorias_mes_actual", [
        "¿Cuánto gasté por categoría este mes?",
        "¿Cuál fue la mayor?",
        "¿Cuánto gasté en esa?",
    ]),
    ("linea_presupuesto", [
        "¿Cuál es el presupuesto de ChatGPT en agosto de 2026?",
        "¿Cuánto se gastó?",
        "¿Se excedió?",
    ]),
]

# Esta matriz es una evaluación de regresión, no lógica del producto. Los
# fragmentos se comparan tras normalizar puntuación y formato numérico para
# tolerar que el redactor diga "80.953" o "80953", sin aceptar un error de
# filtro, período o magnitud.
EXPECTATIVAS = {
    "alimentos_periodos": (("alimentacion", "763007"), ("alimentacion", "80953"), ("agosto", "682054")),
    "walmart_periodos": (("walmart", "218920"), ("walmart", "28498"), ("agosto", "190422")),
    "conteo_periodos": (("conteo", "162"), ("conteo", "42"), ("disminuyo", "120")),
    "presupuesto_a_ejecucion": (("620000",), ("763007", "143007"), ("disponible", "143007")),
    "conceptos_presupuesto": (("comedera", "457737"), ("comidasafuera", "305270"), ("comedera", "180000")),
    "categorias_ordinal": (("vivienda", "1648992"), ("deudas", "504568"), ("504568",)),
    "detalle_suma_maximo": (("escritorioajustable",), ("154123",), ("escritorioajustable", "109500")),
    "transporte_periodos": (("transporte", "328909"), ("transporte", "254000"), ("agosto", "74909")),
    "categorias_comparacion": (("vivienda", "1783262"), ("deudas", "436977"), ("vivienda", "1346285")),
    "excesos": (("alimentacion", "143007"), ("alimentacion", "143007"), ("exceso", "143007")),
    "concepto_disponible": (("comidasafuera", "305270"), ("disponible", "85270"), ("porcentaje", "1388")),
    "comercio_detalle": (("walmart", "218920"), ("walmart", "208620"), ("walmart", "208620")),
    "roga_clasificacion": (("roga", "saludimprevistos"), ("otros", "saludimprevistos"), ("68000",)),
    "detalles_fecha": (("roga", "subway"), ("2", "73450"), ("roga", "68000")),
    "top_conceptos": (("hipotecatdc", "780075"), ("hipotecatdc",), ("hipotecatdc", "780075")),
    "otros_periodos": (("otros", "537492"), ("otros", "188100"), ("disminuyo", "349392")),
    "presupuesto_categoria": (("492000",), ("328909",), ("disponible", "163090")),
    "comidas_detalle": (("comidasafuera",), ("305270",), ("artisan", "35760")),
    "categorias_mes_actual": (("vivienda", "1648992"), ("vivienda", "1648992"), ("1648992",)),
    "linea_presupuesto": (("10000",), ("10900",), ("exceso", "900")),
}


def _normalizar(valor: object) -> str:
    texto = unicodedata.normalize("NFKD", str(valor or ""))
    return "".join(c for c in texto.lower() if c.isalnum())


def _validar(nombre, salidas):
    esperadas = EXPECTATIVAS[nombre]
    errores = []
    for indice, (salida, fragmentos) in enumerate(zip(salidas, esperadas), 1):
        evidencia = _normalizar(salida.get("texto", "") + json.dumps(
            {"columnas": salida.get("columnas"), "filas": salida.get("filas")},
            ensure_ascii=False, default=str,
        ))
        faltan = [fragmento for fragmento in fragmentos if _normalizar(fragmento) not in evidencia]
        if faltan:
            errores.append({"turno": indice, "faltan": faltan})
    return errores


def _compactar(respuesta):
    estado = respuesta.estado or {}
    return {
        "texto": respuesta.texto,
        "contrato": estado.get("contrato", {}),
        "columnas": estado.get("columnas", []),
        "filas": estado.get("filas", [])[:12],
        "filas_totales": estado.get("filas_totales"),
        "kpi": estado.get("kpi", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    load_dotenv(args.env_file)

    import registry
    from bot import memoria, responder

    _, clientes = registry._cargar()
    cliente = clientes["cliente_a"]
    registry.resolver = lambda _: cliente
    responder.registry.resolver = registry.resolver
    historial = {}
    candado = threading.Lock()

    def cargar(_, numero):
        with candado:
            return list(historial.get(numero, []))

    def guardar(_, numero, pregunta, respuesta, sql="", estado=None):
        with candado:
            historial.setdefault(numero, []).extend([
                {"rol": "user", "contenido": pregunta, "sql": "", "estado": {}},
                {"rol": "assistant", "contenido": respuesta, "sql": sql or "",
                 "estado": estado or {}},
            ])

    memoria.cargar_historial = cargar
    memoria.guardar_intercambio = guardar

    def ejecutar(indice_nombre):
        indice, (nombre, preguntas) = indice_nombre
        numero = f"+5068809{indice:04d}"
        salidas = []
        for pregunta in preguntas:
            salidas.append(_compactar(responder.responder(numero, pregunta)))
        return nombre, preguntas, salidas, _validar(nombre, salidas)

    resultados = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futuros = [pool.submit(ejecutar, item) for item in enumerate(CONVERSACIONES, 1)]
        for futuro in as_completed(futuros):
            resultados.append(futuro.result())
    resultados.sort(key=lambda r: [n for n, _ in CONVERSACIONES].index(r[0]))
    contenido = json.dumps([
        {"caso": nombre, "preguntas": preguntas, "respuestas": salidas,
         "errores": errores, "correcto": not errores}
        for nombre, preguntas, salidas, errores in resultados
    ], ensure_ascii=False, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as archivo:
            archivo.write(contenido)
    else:
        print(contenido)
    if any(errores for _, _, _, errores in resultados):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
