# FACHAVI SQL Bot

Bot de WhatsApp que responde preguntas en lenguaje natural sobre tus datos.
El usuario pregunta ("¿cuánto se vendió ayer?"), el bot traduce a SQL con Claude,
consulta las tablas y responde en español. Arranca con una tabla de ventas de
Google Sheets y está diseñado para escalar a **varias tablas**: cada pestaña del
libro se vuelve una tabla consultable automáticamente.

## Cómo funciona (pipeline)

```
Cliente escribe en WhatsApp
   -> Meta llama a POST /webhook   (se responde 200 al instante)
   -> en segundo plano:
        1. Se leen las tablas de Google Sheets -> DuckDB en memoria (con caché 60s)
        2. Claude Haiku genera SQL a partir de la pregunta + el esquema
        3. Se valida que el SQL sea SOLO lectura (guardas de seguridad)
        4. Se ejecuta en DuckDB
        5. Claude redacta la respuesta en lenguaje natural
        6. Se envía por WhatsApp
```

## Archivos

| Archivo | Rol |
|---|---|
| `main.py` | Webhook FastAPI (verificación + recepción, ack rápido) |
| `sheets.py` | Google Sheets -> DuckDB, inferencia de tipos, caché por TTL |
| `nl2sql.py` | Motor text-to-SQL: genera, valida, ejecuta y redacta |
| `memory.py` | Memoria de conversación (SQLite) para preguntas de seguimiento |
| `whatsapp.py` | Envío de mensajes por Graph API |
| `config.py` | Configuración central desde variables de entorno |

## Diseño pensado para "múltiples tablas"

Cada **pestaña (worksheet)** del libro de Google Sheets se convierte en una tabla
de DuckDB con el nombre de la pestaña. Hoy tenés "ventas"; mañana agregás una
pestaña "clientes" o "inventario" y el bot ya puede cruzarlas con JOINs, sin tocar
código. El esquema se le describe a Claude automáticamente en cada consulta.

---

## Configuración de Google Sheets (service account)

1. Andá a **Google Cloud Console** > creá un proyecto (o usá uno).
2. Habilitá la **Google Sheets API**.
3. **Credenciales** > **Crear credenciales** > **Cuenta de servicio**.
4. Creada la cuenta, entrá > **Claves** > **Agregar clave** > **JSON**. Se
   descarga un archivo JSON.
5. Abrí tu Google Sheet > **Compartir** > pegá el `client_email` de ese JSON
   (algo tipo `...@....iam.gserviceaccount.com`) con permiso de **Lector**.
6. El contenido completo de ese JSON va en la variable `GOOGLE_CREDENTIALS_JSON`.
7. El `SPREADSHEET_ID` es lo que va entre `/d/` y `/edit` en la URL del Sheet.

> La primera fila de cada pestaña debe ser el **encabezado** (nombres de columna).

---

## Variables de entorno

| Variable | Qué es |
|---|---|
| `VERIFY_TOKEN` | String que inventás vos; el mismo va en Meta |
| `WHATSAPP_TOKEN` | Token de WhatsApp (Meta > app > WhatsApp > Config API) |
| `PHONE_NUMBER_ID` | ID del número (misma pantalla) |
| `ANTHROPIC_API_KEY` | Tu API key de Anthropic (console.anthropic.com) |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` |
| `GOOGLE_CREDENTIALS_JSON` | El JSON completo del service account |
| `SPREADSHEET_ID` | ID del libro de Google Sheets |
| `CACHE_TTL_SECONDS` | Frescura de datos (60 = relee la hoja máx. cada 60s) |
| `MAX_RESULT_ROWS` | Tope de filas por consulta (default 100) |

---

## Deploy en Render

1. Subí todo a un repo de GitHub (el `.gitignore` protege tus secretos).
2. Render > **New +** > **Web Service** > conectá el repo.
3. Confirmá: build `pip install -r requirements.txt`, start
   `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Cargá **todas** las variables de entorno de la tabla de arriba.
5. Create. Render te da una URL fija. Verificá que viva abriéndola: `{"status":"ok"...}`.

## Conectar el webhook en Meta

1. App de Meta > **WhatsApp** > **Configuración** > **Webhooks**.
2. Callback URL: `https://TU-URL.onrender.com/webhook` · Verify token: tu `VERIFY_TOKEN`.
3. Verificar y guardar. Suscribite al campo **messages**.

## Probar

Escribile al número de prueba desde tu WhatsApp:
- "¿Cuánto se vendió en total?"
- "¿Quién vendió más este mes?"
- "Ventas por producto"

---

## Notas de seguridad y costos

- **Solo lectura:** el validador rechaza cualquier SQL que no sea SELECT/WITH,
  bloquea `;` (inyección) y comandos administrativos. Además DuckDB corre en
  memoria sobre una copia, así que la hoja original nunca se toca.
- **Token de WhatsApp:** el temporal dura 24h. Para producción, generá un
  System User token permanente.
- **Spin-down (Render Free):** el servicio duerme a los 15 min. Para producción,
  subí a Starter (~$7/mes).
- **Costo de Claude:** Haiku es barato. Cada consulta usa ~2 llamadas (generar
  SQL + redactar), ambas con pocos tokens. Ideal para alto volumen.

## Próximas mejoras posibles

- Agregar más pestañas al Sheet (clientes, inventario) → JOINs automáticos.
- Portar el **debounce** (juntar mensajes fragmentados) de tu bot anterior.
- Cachear esquemas y respuestas frecuentes para bajar aún más el costo.
