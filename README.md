# FACHAVI SQL Bot — Multi-cliente

Bot de WhatsApp que responde preguntas en lenguaje natural sobre los datos de
**varios clientes**, todos escribiendo al **mismo número**. Según quién escribe,
el bot resuelve a qué cliente pertenece y consulta **su** Google Sheet de datos.

## Cómo distingue a los clientes

Un solo número de WhatsApp (el de Fachavi) atiende a todos. Cuando llega un
mensaje, el bot mira el número de quien escribe y lo busca en un **Sheet maestro**
(el registro). Un mismo cliente puede tener varios usuarios (varios números).

```
Cliente escribe -> POST /webhook (200 al instante)
  -> resolver(número) en el Sheet maestro:
        número -> cliente_id -> spreadsheet_id (Sheet de datos del cliente)
  -> si NO está registrado: mensaje pidiendo contactar a FACHAVI
  -> si está: cargar SU Sheet en DuckDB
  -> Claude genera SQL -> ejecutar -> Claude redacta -> responder
```

## El Sheet maestro (registro)

Es un Google Sheet **tuyo**, aparte de los Sheets de datos. Tiene dos pestañas:

**Pestaña `clientes`:**

| cliente_id | nombre | spreadsheet_id |
|---|---|---|
| ferreteria_a | Ferretería A | 1AbC... (ID del Sheet de datos de A) |
| farmacia_b | Farmacia B | 1XyZ... (ID del Sheet de datos de B) |

**Pestaña `usuarios`:**

| numero | cliente_id |
|---|---|
| 50688889999 | ferreteria_a |
| 50611112222 | ferreteria_a |
| 50677778888 | farmacia_b |

Para dar de alta un cliente nuevo: agregás una fila en `clientes` y sus números
en `usuarios`. **Sin tocar código ni redeploy.** El número puede ir con o sin
`+`, con espacios o guiones — el bot lo normaliza.

## Un solo service account para todo

El **mismo** service account de Google lee el Sheet maestro Y los Sheets de datos
de todos los clientes. Cada cliente solo comparte su hoja con el `client_email`
del service account (permiso Lector). No necesitás credenciales por cliente.

## Archivos

| Archivo | Rol |
|---|---|
| `main.py` | Webhook; resuelve cliente por número y orquesta |
| `registry.py` | Lee el Sheet maestro (clientes/usuarios), resuelve número->cliente |
| `gclient.py` | Autenticación única del service account de Google |
| `sheets.py` | Sheet de datos -> DuckDB, caché por cliente, multi-tabla |
| `nl2sql.py` | Motor text-to-SQL: genera, valida, ejecuta, redacta |
| `memory.py` | Memoria de conversación (separada por cliente:usuario) |
| `whatsapp.py` | Envío por Graph API |
| `config.py` | Config desde variables de entorno |

## Variables de entorno (en Render)

| Variable | Qué es |
|---|---|
| `VERIFY_TOKEN` | String que inventás; el mismo va en Meta |
| `WHATSAPP_TOKEN` | Token de WhatsApp (Meta) |
| `PHONE_NUMBER_ID` | ID del número (Meta) |
| `ANTHROPIC_API_KEY` | Tu key de Anthropic |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` |
| `GOOGLE_CREDENTIALS_JSON` | JSON del service account (uno solo) |
| `MASTER_SPREADSHEET_ID` | ID del **Sheet maestro** (registro), no de datos |
| `DATA_CACHE_TTL` | Frescura de datos (60s) |
| `REGISTRY_CACHE_TTL` | Cada cuánto relee el registro (300s) |

> Importante: el registro y los datos viven en Google Sheets, así que el disco
> efímero de Render NO es problema. La única SQLite local es la memoria de chat,
> que no pasa nada si se pierde en un redeploy.

## Setup de Google (service account)

1. Google Cloud Console > proyecto > habilitar **Google Sheets API**.
2. Crear **cuenta de servicio** > Claves > JSON (se descarga).
3. Compartir el **Sheet maestro** y **cada Sheet de datos** con el `client_email`
   del JSON, como **Lector**.
4. El JSON completo va en `GOOGLE_CREDENTIALS_JSON`; el ID del maestro en
   `MASTER_SPREADSHEET_ID`.

## Deploy en Render + webhook en Meta

1. Subí a GitHub (el `.gitignore` protege los secretos).
2. Render > New Web Service > conectá el repo > cargá las variables.
3. Callback URL en Meta: `https://TU-URL.onrender.com/webhook` + tu `VERIFY_TOKEN`;
   suscribite a **messages**.

## Probar

1. En el Sheet maestro, agregá tu propio número (el que usás para probar) en
   `usuarios`, apuntando a un `cliente_id` que tenga un Sheet de datos de prueba.
2. Escribile al número de prueba: "¿cuánto se vendió en total?".
3. Con un número que NO esté en el registro, deberías recibir el mensaje de
   contactar a FACHAVI.

## Seguridad

- Números no registrados no acceden a ningún dato.
- SQL de solo lectura (rechaza escrituras, `;`, comandos administrativos).
- DuckDB en memoria sobre una copia; los Sheets nunca se modifican.
- Cada cliente solo ve su propio Sheet (aislamiento por spreadsheet_id).

## Notas

- Token de WhatsApp temporal dura 24h; para producción generá el permanente.
- Render Free duerme a los 15 min; para producción, Starter (~$7/mes).

---

## Catálogo de datos (governance) — OBLIGATORIO

Cada Sheet de datos debe tener una pestaña llamada **`_catalogo`** (con guion bajo).
El bot NO la trata como tabla consultable; la usa para responder preguntas de
governance. Si falta, el bot avisa que el catálogo no está documentado.

Cualquier pestaña que empiece con `_` se excluye de las tablas consultables.

**Estructura de la pestaña `_catalogo`** (primera fila = encabezados exactos):

| tabla | columna | descripcion | sistema_origen | frecuencia | dueño |
|---|---|---|---|---|---|
| ventas | * | Registro de ventas diarias | POS Toshiba | Diaria | Gerencia comercial |
| ventas | monto | Monto de la venta en colones | POS Toshiba | Diaria | Gerencia comercial |
| ventas | vendedor | Vendedor que cerró la venta | CRM interno | Diaria | RRHH |

- Una fila con `columna = *` describe la tabla completa.
- Las demás filas describen cada columna.

**Qué puede responder el bot ahora:**
- Datos: "¿cuánto se vendió?", "¿quién vendió más?"
- Governance: "¿de qué sistema vienen estos datos?", "¿qué significa monto?",
  "¿quién es el dueño de esta data?", "¿cada cuánto se actualiza?"
- Saludos: responde con una bienvenida y ejemplos de lo que puede hacer.
