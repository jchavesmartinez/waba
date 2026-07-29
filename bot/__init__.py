"""
FACHAVI — Bot de WhatsApp (capa de LECTURA).

Este paquete es la contraparte de la ingesta: NO toca los sistemas fuente ni
Google Sheets de datos en tiempo de request. Lee del warehouse (Neon) lo que la
ingesta ya aterrizo, y responde preguntas del cliente en lenguaje natural.

Flujo (bot/responder.py):

    numero de WhatsApp
        |  registry.resolver()  (Sheet maestro, pestania 'usuarios')
        v
    cliente_id  ->  esquema raw_<cliente_id> en Neon
        |  bot/catalogo.py: lee raw_<cliente_id>._catalogo y decide
        |  QUE tablas puede leer el bot segun la columna 'instruccion'
        v
    schema de SOLO las tablas permitidas
        |  bot/nl2sql.py: Claude genera un SELECT, se VALIDA con sqlglot
        v
    bot/warehouse_ro.py: ejecuta el SELECT en una transaccion READ ONLY
        |
        v
    bot/nl2sql.py: redacta la respuesta en español a partir de las filas
"""
