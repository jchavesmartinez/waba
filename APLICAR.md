# Cómo aplicar estos archivos

Los 25 archivos de este paquete reemplazan a los del repo **respetando la misma
ruta**. Descomprimí sobre la raíz de `waba/` y listo — no hay que mover nada.

```bash
# desde la raíz del repo, con el zip descomprimido al lado
cp -r waba-arreglos/* .
git status          # revisá el diff antes de commitear
```

## Además hay que BORRAR tres archivos (A-06)

No se pueden "entregar" borrados, así que va a mano:

```bash
git rm app.py whatsapp.py memoria.py
```

Son duplicados de `bot/app.py`, `bot/whatsapp.py` y `bot/memoria.py`. Dos de
ellos son código muerto verificado (nadie los importa). El riesgo real no es que
ocupen espacio: es que alguien arregle un problema de seguridad en el archivo de
la raíz, despliegue, y crea que lo arregló.

Antes de borrarlos, confirmá que ningún `startCommand` apunte a `app:app`. El
`render.yaml` de este paquete ya usa `bot.app:app`.

---

## ⚠️ Tres cosas que cambian de comportamiento

**1. El bot rechaza los POST sin `WHATSAPP_APP_SECRET` (C-08).**
Antes, sin ese valor, se aceptaba cualquier POST sin avisar. Ahora se rechaza.
Configurá el app secret en el panel de Render **antes** de desplegar. Para
desarrollo local: `BOT_PERMITIR_SIN_FIRMA=si`.

**2. `SYNC_ARGS` pasa a vacío en `render.yaml` (C-02).**
Cambiar el archivo **no cambia lo que hay hoy en el panel de Render**. Hay que
ponerlo en vacío también en el dashboard, en el servicio de ingesta.

**3. La escritura en Postgres usa una tabla intermedia (A-02).**
`escribir_tabla` ahora crea `<tabla>__nueva`, la llena y hace el swap dentro de
una transacción. Si una corrida falla justo en el medio te puede quedar esa
tabla huérfana en Neon; la siguiente corrida la borra sola.

---

## Qué cierra cada archivo

| Archivo | Hallazgos |
|---|---|
| `sync.py` | **C-01**, A-04 (purga), estado `ok_con_bloqueo`, canal de alertas de fuentes |
| `warehouse/base.py` | **C-07**, B-05, `METODOS_CONTRATO` |
| `warehouse/postgres_dest.py` | **C-01**, **C-03**, A-02, A-04, A-05, B-17 |
| `warehouse/duckdb_dest.py` | **C-01** (misma fusión), B-18 (paridad documentada) |
| `config.py` | `secreto_de_env()`, `revisar_arranque_bot()`, B-39, B-24, A-07, A-13, A-14 |
| `registry.py` | B-01, B-02, B-09 |
| `gclient.py` | B-10, C-06 (documentado: es operativo, no de código) |
| `sources/base.py` | A-03, A-07, B-06, B-07 |
| `sources/google_sheets.py` | A-08, B-12, A-03 |
| `sources/csv_url.py` | **A-09** |
| `sources/api_rest.py` | **C-04**, A-10, B-14 |
| `sources/postgres.py` | **C-04**, **C-03**, A-11 |
| `bot/app.py` | **C-08**, **C-05**, A-13, A-12 (lock) |
| `bot/catalogo.py` | A-01, A-17, **C-05**, B-26 |
| `bot/nl2sql.py` | A-15, B-20, B-24 |
| `bot/responder.py` | A-14, A-19, B-21, B-26 |
| `bot/kpis.py` | B-29, B-30, B-31 |
| `bot/warehouse_ro.py` | B-28, A-18 (documentado) |
| `bot/whatsapp.py` | B-19 |
| `render.yaml` | **C-02**, B-38 |
| `env.example`, `README.md` | B-39 |
| `tests/` | Previene la reaparición de **C-01** y **C-07** |

---

## Lo que NO está acá (a propósito)

Estos hallazgos no se arreglan con código:

- **C-06** (una sola credencial de Google para todos los clientes) — es una
  decisión de negocio. Separar en una cuenta por cliente destruye la ventaja
  comercial de «compartí tu hoja con este correo». Las mitigaciones que sí
  corresponden quedaron documentadas arriba de `gclient.py`: rotación con
  periodicidad definida, restringir el acceso al panel de Render con la misma
  seriedad que a la base, y auditoría de acceso de Google Cloud.
- **A-20** (la memoria guarda cifras del negocio en texto plano) — es política.
  Como mínimo, documentárselo al cliente y considerar bajar `BOT_MEMORIA_TTL_DIAS`.
- **A-12** (dedup de mensajes en memoria del proceso) y la separación de roles de
  base de datos (**C-03** en su raíz, B-27, B-35) — son el Bloque 4 del plan:
  con un cliente el beneficio es teórico.
- **B-30**: el default de `runway_inventario` salió del prompt, pero hay que
  moverlo como dato a la columna `supuestos` del tab `_kpis` del cliente:
  `default: por producto (evita el divide-por-cero del total)`.

---

## Verificación

```bash
pip install pytest
pytest -q          # 17 pruebas
```

Las dos pruebas nuevas fueron verificadas contra el código **viejo**: ahí fallan.
`test_dos_vaciados_seguidos_siguen_bloqueando` da directamente
`SE PERDIERON LOS DATOS: se escribió una tabla vacía encima de la buena`, y en el
contrato de destinos `ultimo_detalle` aparece como no-abstracto. O sea que no son
pruebas decorativas: reproducen los dos errores reales.
