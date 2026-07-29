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
        número -> cliente_id -> lista de FUENTES del cliente
  -> si NO está registrado: mensaje pidiendo contactar a FACHAVI
  -> si está: cargar TODAS sus fuentes ACTIVAS en UNA sola DuckDB
        (Google Sheets, CSV, Postgres, ... combinadas y consultables juntas)
  -> Claude genera SQL -> ejecutar -> Claude redacta -> responder
```

## Arquitectura modular de fuentes (encendé/apagá por cliente)

Cada cliente puede tener **varias fuentes de datos**, y cada una se **prende o
apaga** desde el registro sin tocar código ni redeploy. Todas las fuentes
activas de un cliente se cargan en **la misma** base DuckDB en memoria, así se
pueden **cruzar datos entre fuentes** en una sola consulta (p.ej. ventas de
Google Sheets con inventario de un Postgres).

El pipeline (`nl2sql.py`) no sabe de dónde salen los datos: solo ve una conexión
DuckDB, un texto de *schema* y un texto de *catálogo*. Por eso **agregar una
fuente nueva = escribir una subclase**; no se toca el pipeline.

Tipos de fuente incluidos: `google_sheets`, `csv_url`, `postgres` (ejemplo listo
para activar).

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

**Pestaña `fuentes`** (la parte modular — una fila por fuente de cada cliente):

| cliente_id | fuente_id | tipo | activo | config |
|---|---|---|---|---|
| ferreteria_a | ventas_sheet | google_sheets | si | `{"spreadsheet_id":"1AbC..."}` |
| ferreteria_a | inv_csv | csv_url | no | `{"url":"https://.../inv.csv","tabla":"inventario"}` |
| farmacia_b | erp | postgres | si | `{"dsn":"postgresql://...","tablas":["ventas"]}` |

- `activo`: `si` / `no` → así se **prende o apaga** una fuente. Cambiás la celda
  y listo (efectivo en el próximo refresco del registro, `REGISTRY_CACHE_TTL`).
- `config`: **JSON** con los datos de conexión de esa fuente (ver cada tipo abajo).

Para dar de alta un cliente nuevo: agregás una fila en `clientes`, sus números
en `usuarios`, y una o más filas en `fuentes`. **Sin tocar código ni redeploy.**
El número puede ir con o sin `+`, con espacios o guiones — el bot lo normaliza.

> **Retrocompatibilidad:** si un cliente **no** tiene filas en `fuentes` pero sí
> un `spreadsheet_id` en la pestaña `clientes`, el bot lo trata como una única
> fuente `google_sheets` activa. Tu configuración actual sigue funcionando sin
> cambios; podés migrar a `fuentes` cuando quieras.

### Config por tipo de fuente

- **`google_sheets`** → `{"spreadsheet_id": "1AbC..."}`
  Cada pestaña que no empiece con `_` se carga como una tabla; `_catalogo` aporta
  governance. Un Sheet con pestañas `ventas` e `inventario` produce **dos tablas**
  con una sola fila en `fuentes`.
  Filtros opcionales de pestañas:
  - `{"spreadsheet_id":"1AbC...","hojas":["ventas"]}` → **solo** esas pestañas
  - `{"spreadsheet_id":"1AbC...","excluir":["notas"]}` → todas menos esas

  El filtro `hojas` sirve para partir **un mismo Sheet en varias fuentes** y darle
  a cada una su propia `frescura_minutos` (ver abajo).
- **`csv_url`** → `{"url": "https://.../datos.csv", "tabla": "ventas"}`
  o varias: `{"tablas":[{"url":"...","tabla":"ventas"},{"url":"...","tabla":"inv"}]}`.
  Catálogo opcional inline: `{"url":"...","tabla":"ventas","catalogo":[{...}]}`.
- **`postgres`** → `{"dsn":"postgresql://user:pass@host:5432/db","tablas":["ventas"],"esquema":"public"}`
  Trae una copia de solo lectura de cada tabla; la base del cliente no se toca.

## Un solo service account para todo

El **mismo** service account de Google lee el Sheet maestro Y los Sheets de datos
de todos los clientes. Cada cliente solo comparte su hoja con el `client_email`
del service account (permiso Lector). No necesitás credenciales por cliente.

## Archivos

| Archivo | Rol |
|---|---|
| `main.py` | Webhook; resuelve cliente por número y orquesta |
| `registry.py` | Lee el Sheet maestro (clientes/usuarios/**fuentes**), resuelve número->cliente |
| `gclient.py` | Autenticación única del service account de Google |
| `loader.py` | **Orquestador**: combina las fuentes activas del cliente en una DuckDB; caché por cliente |
| `sources/base.py` | Contrato `Source` + `Fragmento` + helpers compartidos |
| `sources/__init__.py` | Registro de tipos de fuente + `crear_fuente()` (factory) |
| `sources/google_sheets.py` | Fuente Google Sheets (datos + `_catalogo`) |
| `sources/csv_url.py` | Fuente CSV por URL (una o varias tablas) |
| `sources/postgres.py` | Fuente PostgreSQL (ejemplo listo para activar) |
| `nl2sql.py` | Motor text-to-SQL: genera, valida, ejecuta, redacta |
| `memory.py` | Memoria de conversación (separada por cliente:usuario) |
| `whatsapp.py` | Envío por Graph API |
| `config.py` | Config desde variables de entorno |

## Fase 2 — Job de ingesta (`sync.py`)

La ingesta corre **fuera** del request de WhatsApp. El bot ya no lee las fuentes:
lee el warehouse. Esto desacopla la disponibilidad (si Google se cae, el bot
sigue respondiendo con la última carga) y baja la latencia a una consulta SQL.

```bash
python sync.py --probar           # extrae y muestra, NO escribe (usar al dar de alta)
python sync.py                    # todos los clientes, respeta frescura
python sync.py --cliente ferre_a  # solo un cliente
python sync.py --forzar           # ignora frescura, recarga todo
python sync.py --estado           # ¿cuándo se cargó cada fuente por última vez?
```

### Controles de calidad (semilla de la Fase 6)

**Normalización determinista de encabezados.** Los Sheets reales traen columnas
duplicadas y encabezados vacíos. Si no los normalizamos nosotros, DuckDB inventa
`C3`/`monto_1` y Postgres se comporta distinto — la misma hoja daría nombres
distintos según el warehouse. Regla: encabezado vacío → `columna_N` (por
posición); duplicado → `monto`, `monto_2`, `monto_3`.

**Schema drift.** Cada corrida guarda las columnas y el conteo de filas de cada
tabla. En la siguiente se compara: si aparecen o desaparecen columnas, la corrida
queda en estado `ok_con_alertas` con el detalle, pero **sí escribe** (agregar una
columna suele ser legítimo).

**Guarda de borrado.** Si una tabla llega **vacía** y antes tenía filas, la
escritura se **bloquea**. Como la carga es full refresh, escribir cero filas
encima borraría la única copia buena. Se conserva el dato anterior y se alerta.
Una caída de más del 50% de filas alerta pero no bloquea.

**Modo prueba (`--probar`).** Extrae de las fuentes y muestra qué tablas y
columnas produciría, sin escribir ni registrar nada. Es lo que hay que correr al
dar de alta la fuente de un cliente nuevo, antes de automatizar.

**Frescura por fuente, no global.** En la pestaña `fuentes` del Sheet maestro se
agrega la columna `frescura_minutos`. El job omite las fuentes que todavía están
frescas, así no gasta llamadas de API de gusto:

| cliente_id | fuente_id | tipo | activo | frescura_minutos | config |
|---|---|---|---|---|---|
| ferreteria_a | sheet_ventas | google_sheets | si | 15 | `{"spreadsheet_id":"1AbC..."}` |
| ferreteria_a | sheet_inv | google_sheets | si | 1440 | `{"spreadsheet_id":"1XyZ..."}` |

`0` o vacío = sin política (se sincroniza en cada corrida).

**Cómo queda el warehouse.** Un esquema por cliente (aislamiento), una tabla por
fuente y pestaña:

```
raw_ferreteria_a.sheet_ventas__ventas
raw_ferreteria_a.sheet_inv__inventario
_meta.sync_corridas                     <- bitácora de todas las corridas
```

**Linaje (Fase 0).** Cada fila escrita lleva `_corrida_id`, `_fuente_id` e
`_ingestado_en`. Con eso podés rastrear cualquier número hasta la corrida que lo
trajo, y responder "¿de cuándo es este dato?".

**Aislamiento de fallos.** Si una fuente truena (Sheet no compartido, token
vencido), se registra el error en la bitácora y el job sigue con las demás. El
error queda **auditado, no silenciado**: `--estado` lo muestra como `NUNCA` o con
la fecha vieja.

### Un warehouse por cliente (aislamiento fisico)

Cada cliente puede aterrizar en **su propio proyecto de Neon**, no solo en un
esquema distinto del mismo proyecto. Con el free tier de Neon (0.5 GB por
proyecto, hasta 100 proyectos) eso le da a cada cliente su propio espacio.

El DSN **nunca** va en el Sheet maestro — llevaria la contrasenia en una hoja de
calculo. Se resuelve por variable de entorno, en este orden:

1. La variable nombrada en la columna opcional `dsn_env` de la pestania `clientes`
2. Por convencion: `WAREHOUSE_DSN_<CLIENTE_ID en mayusculas>`
3. `WAREHOUSE_DSN` global (todos comparten proyecto, separados por esquema)

Ejemplo: para darle a `ferreteria_a` su propio proyecto, basta con agregar en
Render la variable `WAREHOUSE_DSN_FERRETERIA_A`. Sin tocar codigo ni el Sheet.
Asi se puede empezar con un proyecto compartido y migrar clientes de a uno.

Nota: con proyectos separados, la bitacora `_meta.sync_corridas` vive en cada
proyecto. `--estado` recorre todos los clientes y consulta el warehouse de cada
uno, asi que sigue mostrando el panorama completo.

### Elegir warehouse

| Opción | `WAREHOUSE_TIPO` | `WAREHOUSE_DSN` |
|---|---|---|
| DuckDB local (desarrollo) | `duckdb` | `fachavi.duckdb` |
| MotherDuck (producción) | `duckdb` | `md:fachavi?motherduck_token=XXX` |
| Postgres / Supabase / Neon | `postgres` | `postgresql://user:pass@host:5432/db` |

DuckDB local y MotherDuck usan **el mismo código**: solo cambia el DSN. Por eso
arrancar local no es trabajo tirado.

---


1. Creá `sources/mi_fuente.py` con una subclase de `Source`:
   ```python
   from .base import Source, Fragmento, registrar_df, describir_tabla, inferir_tipos

   class MiFuenteSource(Source):
       tipo = "mi_fuente"  # este string va en la columna 'tipo' del registro

       def cargar(self, con) -> Fragmento:
           df = ...  # traé tus datos como DataFrame (API, DB, archivo, etc.)
           df = inferir_tipos(df)
           tabla = registrar_df(con, df, "mi_tabla", self.fuente_id)
           return Fragmento(schema=describir_tabla(con, tabla),
                            catalogo="", tablas=[tabla])
   ```
2. Registrala en `sources/__init__.py`: `registrar(MiFuenteSource)`.
3. En el Sheet maestro, agregá filas con `tipo = mi_fuente`. Listo — el pipeline
   no se toca.

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

## Catálogo de datos (governance) — recomendado

Cada fuente puede aportar **catálogo** (metadata/governance). En Google Sheets se
documenta en una pestaña **`_catalogo`** (con guion bajo); en `csv_url`/`postgres`
va inline en el `config` (clave `catalogo`, mismo formato de columnas). El bot usa
ese catálogo para responder preguntas de governance.

> **Cambio de comportamiento:** el catálogo ya **no** es obligatorio para que el
> bot funcione. Antes, un cliente sin `_catalogo` quedaba inutilizable. Ahora las
> consultas de **datos** funcionan aunque no haya catálogo; solo las preguntas de
> **governance** responden "todavía no está documentado" si ninguna fuente activa
> aporta catálogo. Se combina el catálogo de **todas** las fuentes del cliente.

En Google Sheets, cualquier pestaña que empiece con `_` se excluye de las tablas
consultables.

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
