# FACHAVI — waba

Un sistema que permite que el dueño de un negocio pregunte por WhatsApp
«¿cuánto vendimos ayer?» y reciba la respuesta correcta en segundos.

El repo tiene **dos mitades**, deliberadamente separadas:

| | Qué hace | Cuándo corre | Servicio en Render |
|---|---|---|---|
| **Ingesta** (`sync.py`, `sources/`, `warehouse/`) | Extrae de los sistemas fuente del cliente y aterriza los datos en el warehouse | Cron, cada 15 min, **fuera de cualquier request** | `fachavi-ingesta` |
| **Bot** (`bot/`) | Responde preguntas por WhatsApp leyendo del warehouse, con text-to-SQL | Servicio web, en el momento del mensaje | `waba` |

El bot **nunca** toca los sistemas fuente: lee lo que la ingesta ya dejó
limpio. Esa separación es lo que hace que una consulta responda en segundos y
que un Sheet mal compartido no rompa el chat.

```
Sheets / CSV / API / Postgres          (sistemas fuente del cliente)
        |
        v   job programado, respeta frescura por fuente
    sync.py
        |
        v
   raw_<cliente>.<fuente>__<tabla>     (datos, con columnas de linaje)
   raw_<cliente>._catalogo             (governance)
   _meta.sync_corridas                 (bitácora de todas las corridas)
```

## Cómo correr

```bash
python sync.py --probar           # extrae y muestra, NO escribe (al dar de alta)
python sync.py                    # corrida normal, respeta frescura
python sync.py --cliente demo     # solo un cliente
python sync.py --forzar           # ignora frescura, recarga todo
python sync.py --estado           # ¿cuándo se cargó cada fuente por última vez?
```

En Render el modo se controla con la variable **`SYNC_ARGS`**, sin tocar código.

## El Sheet maestro (registro)

Un Google Sheet tuyo, aparte de los de datos. Dos pestañas obligatorias:

**`clientes`** — `cliente_id | nombre | [dsn_env]`

| cliente_id | nombre |
|---|---|
| cliente_a | Cliente A |

`cliente_id` da nombre al esquema en el warehouse (`cliente_a` → `raw_cliente_a`).
Usar minúsculas, sin espacios ni tildes. La columna `dsn_env` es opcional (ver
"un warehouse por cliente").

**`fuentes`** — una fila por fuente de datos de cada cliente

| cliente_id | fuente_id | tipo | activo | frescura_minutos | config |
|---|---|---|---|---|---|
| cliente_a | sheet_ventas | google_sheets | si | 15 | `{"spreadsheet_id":"1AbC...","hojas":["ventas"]}` |
| cliente_a | sheet_inventario | google_sheets | si | 1440 | `{"spreadsheet_id":"1AbC...","hojas":["inventario"]}` |

- **`fuente_id` debe ser único por cliente.** Si se repite, las dos filas escriben
  las mismas tablas y se pisan el catálogo entre sí. El registro ignora la
  repetida y lo avisa en el log.
- **`activo`**: `si` / `no` → así se prende y apaga una fuente, sin tocar código.
- **`frescura_minutos`**: cada cuánto vale la pena re-sincronizar. `0` o vacío =
  en cada corrida. Un solo cron, N frecuencias efectivas.
- **`config`**: JSON con los datos de conexión (ver abajo por tipo).

Una pestaña `usuarios` puede existir para el futuro bot; la ingesta la ignora.

## Config por tipo de fuente

- **`google_sheets`** → `{"spreadsheet_id": "1AbC..."}`
  Cada pestaña que no empiece con `_` se carga como tabla. Un Sheet con pestañas
  `ventas` e `inventario` produce **dos tablas** con una sola fila en `fuentes`.
  Filtros opcionales:
  - `{"spreadsheet_id":"...","hojas":["ventas"]}` → **solo** esas pestañas
  - `{"spreadsheet_id":"...","excluir":["notas"]}` → todas menos esas

  `hojas` sirve para partir **un mismo Sheet en varias fuentes** y darle a cada
  una su propia `frescura_minutos`.
- **`csv_url`** → `{"url":"https://.../datos.csv","tabla":"ventas"}`
  o varias: `{"tablas":[{"url":"...","tabla":"ventas"},{"url":"...","tabla":"inv"}]}`
- **`api_rest`** → `{"url":"https://api...","tabla":"ventas","ruta_datos":"data.items"}`
  Admite `headers_env` (auth), `params` y `paginacion`. El JSON anidado se aplana solo.
- **`postgres`** → `{"dsn_env":"PG_CLIENTE_A","tablas":["ventas"],"esquema":"public"}`
  Trae una copia de solo lectura; la base del cliente no se toca. Acepta
  `{"where":{"ventas":"fecha >= current_date - 90"}}` para no copiar la tabla
  entera en cada corrida.
- **`zoho_imap`** →
  `{"usuario":"central@fachavi.com","password_env":"ZOHO_PASSWORD_CLIENTE_A","carpeta":"INBOX/Cliente A","dias":30,"tabla":"correos"}`
  Lee el buzon en modo solo lectura. `dias` es una ventana de consulta, **no de
  retencion**: la primera corrida crea la tabla y las siguientes hacen UPSERT,
  agregando correos nuevos sin borrar los que ya superaron los 30 dias. Releer
  la misma ventana no duplica mensajes. Si dos cuentas Gmail reenvian a esa
  carpeta de Zoho, ambas quedan acumuladas en la misma tabla y la columna
  `destinatarios` permite distinguir el Gmail original cuando viene en `To`.
  El secreto vive en la variable indicada por `password_env`.

> **Los secretos de los conectores tampoco van en el Sheet.** La regla es la
> misma que para el warehouse: la celda `config` lleva el **nombre de la
> variable de entorno** (`dsn_env`, `headers_env`), nunca la contraseña ni el
> token. Cualquiera con acceso de lectura a la hoja los vería, y el historial de
> versiones de Google los guarda para siempre aunque después se borre la celda.
> Los campos `dsn` y `headers` literales siguen funcionando por
> retrocompatibilidad, pero dejan una advertencia en el log.

## Catálogo de datos (governance)

**No se configura como fuente: viaja solo.** Cada Sheet puede tener una pestaña
`_catalogo` y la ingesta la persiste en `raw_<cliente>._catalogo`, filtrada a las
tablas que esa fuente cargó.

| tabla | columna | descripcion | sistema_origen | frecuencia | dueño |
|---|---|---|---|---|---|
| ventas | * | Registro de ventas | POS | Diaria | Gerencia comercial |
| ventas | monto_total | Monto en colones | POS | Diaria | Gerencia comercial |

Una fila con `columna = *` describe la tabla completa.

En Postgres, además de la tabla, las descripciones se escriben como **`COMMENT ON`
nativo**, así Metabase, DBeaver o dbt las muestran sin conocer este catálogo.

### La columna `instruccion` (qué puede leer el bot)

El catálogo tiene una columna **`instruccion`** de texto libre que gobierna qué
tablas puede consultar el bot de WhatsApp. Es **dato, no código**: se edita en el
Sheet del cliente y viaja a `raw_<cliente>._catalogo` en cada corrida.

| tabla | columna | instruccion |
|---|---|---|
| ventas | * | esta tabla puede ser usada por el bot |
| nomina | * | uso interno, no exponer al bot |

El bot lee ese texto (`bot/catalogo.py`) y decide **tabla por tabla**: si la
instrucción habilita → entra al set consultable; si prohíbe o está vacía → queda
fuera y **el modelo ni sabe que existe**. La política ante instrucción vacía se
controla con `BOT_PERMITIR_SIN_INSTRUCCION` (por defecto `no` = fail-closed).

> **Importante:** re-corré la ingesta (`python sync.py --forzar`) después de
> agregar la columna `instruccion` al Sheet, para que aparezca en Neon. El
> catálogo viejo se migra solo (la escritura hace `ADD COLUMN IF NOT EXISTS`).

## El bot de WhatsApp (capa de lectura)

Paquete `bot/`. **Solo lee del warehouse**, nunca de los sistemas fuente.

```
WhatsApp (Meta Cloud API)
   |  bot/app.py         webhook FastAPI (JSON) + envio por Graph API
   v
numero -> cliente        registry.resolver() sobre la pestaña 'usuarios'
   |
   v
raw_<cliente>._catalogo  bot/catalogo.py: filtra tablas por 'instruccion'
   |                     y arma el schema de SOLO las permitidas
   v
text-to-SQL              bot/nl2sql.py: Claude genera un SELECT; se VALIDA con
   |                     sqlglot (1 sentencia, solo SELECT, solo tablas de la
   |                     lista blanca, sin esquema ajeno) antes de correrlo
   v
bot/warehouse_ro.py      ejecuta en transacción READ ONLY (search_path al
   |                     esquema del cliente, statement_timeout, tope de filas)
   v
respuesta en español     bot/nl2sql.redactar_respuesta()
```

Correr local: `uvicorn bot.app:app --reload --port 8000`, exponer con
ngrok/cloudflared y en el panel de Meta (**WhatsApp > Configuration**) pegar la
URL `.../webhook` y el mismo `WHATSAPP_VERIFY_TOKEN`. Meta hace primero un `GET`
de verificación (devuelve `hub.challenge`) y luego manda los mensajes por `POST`
en JSON; la respuesta al usuario sale por una llamada al Graph API
(`bot/whatsapp.py`), no en el mismo response. Dependencias:
`pip install -r requirements-bot.txt`.

Dos barreras de seguridad, no una: (1) el prompt solo contiene el schema de las
tablas permitidas; (2) el SQL se valida antes de ejecutarse y corre en una
transacción de solo lectura. Aunque algo se colara, Postgres aborta cualquier
escritura.

## Controles de calidad

- **Normalización determinista de encabezados.** Encabezado vacío → `columna_N`;
  duplicado → `monto`, `monto_2`. Si no lo hiciéramos nosotros, cada motor
  inventaría nombres distintos.
- **Schema drift.** Cada corrida guarda columnas y conteo. Si aparecen o
  desaparecen columnas, la corrida queda `ok_con_alertas` con el detalle, pero
  escribe igual.
- **Guarda de borrado.** Si una tabla llega **vacía** y antes tenía filas, la
  escritura se **bloquea**: como la carga es full refresh, escribir cero filas
  encima borraría la única copia buena. Caída de más del 50% alerta, no bloquea.
  La corrida queda en estado `ok_con_bloqueo`, que se puede auditar de una:
  ```sql
  SELECT * FROM _meta.sync_corridas WHERE estado = 'ok_con_bloqueo'
  ORDER BY fin DESC;
  ```
- **Aislamiento de fallos.** Si una fuente truena, se registra el error en la
  bitácora y el job sigue con las demás.

## Un warehouse por cliente (opcional)

Cada cliente puede aterrizar en **su propio proyecto de Neon**. El DSN **nunca**
va en el Sheet — llevaría la contraseña en una hoja de cálculo. Se resuelve por
variable de entorno, en este orden:

1. La variable nombrada en la columna `dsn_env` de `clientes`
2. Por convención: `WAREHOUSE_DSN_<CLIENTE_ID en mayúsculas>`
3. `WAREHOUSE_DSN` global (todos comparten proyecto, separados por esquema)

Para darle a `cliente_a` su propio proyecto basta con agregar
`WAREHOUSE_DSN_CLIENTE_A` en Render. Sin tocar código ni el Sheet.

## Archivos

| Archivo | Rol |
|---|---|
| `sync.py` | Job de ingesta: frescura, calidad, linaje, bitácora |
| `registry.py` | Lee el Sheet maestro (clientes/fuentes) |
| `gclient.py` | Autenticación única del service account de Google |
| `config.py` | Config desde variables de entorno + resolución de DSN por cliente |
| `sources/base.py` | Contrato `Source` + helpers compartidos |
| `sources/__init__.py` | Registro de tipos de fuente + factory |
| `sources/google_sheets.py`, `csv_url.py`, `api_rest.py`, `postgres.py`, `zoho_imap.py` | Connectors |
| `warehouse/base.py` | Contrato `Destino` |
| `warehouse/duckdb_dest.py` | DuckDB local / MotherDuck |
| `warehouse/postgres_dest.py` | Neon / Supabase / Postgres |
| `bot/app.py` | Webhook de la Meta Cloud API + validación de firma y topes |
| `bot/responder.py` | Orquestador: registro → memoria → catálogo → SQL |
| `bot/nl2sql.py` | Genera el SELECT y lo valida con sqlglot |
| `bot/catalogo.py` | Gobernanza: qué tablas puede leer el bot |
| `bot/kpis.py`, `intencion.py`, `memoria.py`, `warehouse_ro.py`, `whatsapp.py` | Capa semántica, ruteo, memoria, lectura RO y envío |
| `render.yaml` | Blueprint de los dos servicios |
| `tests/` | Contrato de destinos y guarda de vaciado |

## Agregar una fuente nueva (3 pasos)

1. Creá `sources/mi_fuente.py` con una subclase de `Source`:
   ```python
   from .base import Source, Fragmento, registrar_df, describir_tabla, inferir_tipos

   class MiFuenteSource(Source):
       tipo = "mi_fuente"

       def cargar(self, con) -> Fragmento:
           df = ...                      # traé tus datos como DataFrame
           df = inferir_tipos(df)
           tabla = registrar_df(con, df, "mi_tabla", self.fuente_id)
           return Fragmento(schema=describir_tabla(con, tabla), tablas=[tabla])
   ```
2. Registrala en `sources/__init__.py`: `registrar(MiFuenteSource)`
3. Agregá filas con `tipo = mi_fuente` en el Sheet maestro

Frescura, linaje, drift, guardas y modo prueba se heredan gratis: viven en
`sync.py`, no en el connector.

## Despliegue

`render.yaml` declara **dos servicios**: el cron `fachavi-ingesta` (corre
`python sync.py` cada 15 minutos) y el web `waba` (el bot). Ver `env.example`
para la lista completa de variables.

> **`PYTHON_VERSION` importa**: si Render usa 3.14, pandas y duckdb no tienen
> wheels y pip intenta compilarlos desde fuente → el build falla.

### Antes de desplegar, tres cosas que hay que verificar en el panel

1. **`SYNC_ARGS` en vacío.** Con `--forzar` todo el sistema de frescura queda
   inerte y cada fuente se recarga 96 veces al día.
2. **`WHATSAPP_APP_SECRET` configurado.** Sin ese valor el bot no puede validar
   que el POST venga de Meta y **rechaza todo el tráfico entrante**. (Para
   desarrollo local: `BOT_PERMITIR_SIN_FIRMA=si`.)
3. **`WHATSAPP_TOKEN` permanente**, no el temporal del panel de Meta: ese vence
   en 24 h y el síntoma es que el bot deja de contestar al día siguiente.

El bot revisa las tres al arrancar y deja advertencias de nivel alto en el log;
también se ven en `GET /salud`.

## Pruebas

```bash
pip install pytest
pytest -q
```

Dos pruebas, y no son decorativas: cubren exactamente los dos errores que solo
se encontraban leyendo el código con cuidado.

- `tests/test_contrato_destinos.py` — verifica que los dos destinos implementen
  los ocho métodos del contrato. Un decorador mal ubicado había sacado
  `ultimo_detalle` de la lista de métodos obligatorios sin que nada fallara.
- `tests/test_guarda_vaciado.py` — verifica que la guarda de borrado siga
  bloqueando ante **dos** cargas vacías consecutivas, no solo la primera.
