(async () => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const API_BASE = "/api/dashboard";
  const parametros = new URLSearchParams(window.location.search);
  const inicioSolicitado = parametros.get("mes") || "";
  const cargarUrl = new URL(`${API_BASE}/datos`, window.location.origin);
  if (inicioSolicitado) cargarUrl.searchParams.set("inicio", `${inicioSolicitado.slice(0, 7)}-01`);
  let data;
  try {
    const response = await fetch(cargarUrl, { headers: { Accept: "application/json" } });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "No pude cargar el dashboard.");
    data = result.dashboard;
    byId("cargando-dashboard").hidden = true;
    const claveRecarga = `dashboard-actualizando:${data.periodo?.inicio || "actual"}`;
    if (result.actualizando) {
      const intentos = Number(sessionStorage.getItem(claveRecarga) || 0);
      document.querySelector(".estado")?.replaceChildren(document.createTextNode("Actualizando datos…"));
      if (intentos < 3) {
        sessionStorage.setItem(claveRecarga, String(intentos + 1));
        window.setTimeout(() => window.location.reload(), 5000);
      }
    } else {
      sessionStorage.removeItem(claveRecarga);
    }
  } catch (reason) {
    const loading = byId("cargando-dashboard");
    loading.innerHTML = `<strong>${reason.message || "No pude cargar el dashboard."}</strong>`;
    return;
  }
  const clean = (s) => String(s ?? "").replaceAll("_", " ");
  const isMoney = (name, unit) => /monto|gasto|presupuesto|disponible|exceso|venta|ingreso|saldo|total/i.test(name) || /colon|crc|usd|moneda/i.test(unit);
  const number = (value) => typeof value === "number" ? value : Number(value);
  const format = (value, name = "", unit = "") => {
    if (value === null || value === undefined || value === "") return "—";
    const n = number(value);
    if (!Number.isNaN(n)) {
      if (/pct|porcentaje/i.test(name)) return new Intl.NumberFormat("es-CR", { maximumFractionDigits: 1 }).format(n) + "%";
      if (isMoney(name, unit)) return new Intl.NumberFormat("es-CR", { style: "currency", currency: /usd/i.test(unit) ? "USD" : "CRC", maximumFractionDigits: 2 }).format(n);
      return new Intl.NumberFormat("es-CR", { maximumFractionDigits: 2 }).format(n);
    }
    return String(value);
  };
  const gastoExcedePresupuesto = (presupuesto, gastado) => {
    const presupuestoNumero = number(presupuesto);
    const gastadoNumero = number(gastado);
    return Number.isFinite(presupuestoNumero) && Number.isFinite(gastadoNumero) && gastadoNumero > presupuestoNumero;
  };
  const agregarMeta = (contenedor, texto, clase = "") => {
    if (!texto) return;
    if (contenedor.childElementCount || contenedor.childNodes.length) contenedor.append(" · ");
    const parte = document.createElement("span");
    if (clase) parte.className = clase;
    parte.textContent = texto;
    contenedor.append(parte);
  };
  const findKpi = (name) => data.kpis.find((k) => k.kpi === name);
  // El dashboard presenta siempre las tres dimensiones mensuales en un orden
  // estable, independientemente del orden en que lleguen desde metadata.
  const presentacion = (kpi) => {
    const id = String(kpi.kpi || "").toLowerCase();
    if (id === "gasto_por_categoria" || id === "ejecucion_presupuesto_mes" || id.includes("presupuesto_categoria")) {
      return { titulo: "Categoría mensual", orden: 0 };
    }
    if (id === "ejecucion_presupuesto_concepto" || id === "gasto_por_concepto" || id.includes("presupuesto_concepto")) {
      return { titulo: "Concepto mensual", orden: 1 };
    }
    if (id === "gasto_por_comercio" || id.includes("presupuesto_comercio")) {
      return { titulo: "Comercio mensual", orden: 2 };
    }
    return { titulo: kpi.nombre, orden: 10 };
  };
  const rowObject = (kpi, row) => Object.fromEntries(kpi.columnas.map((c, i) => [c, row[i]]));
  const keyMatch = (obj, pattern) => Object.keys(obj).find((key) => pattern.test(key));
  const normalized = (value) => String(value ?? "").trim().toLocaleLowerCase("es").normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  const lines = Array.isArray(data.lineas_presupuesto) ? data.lineas_presupuesto : [];
  const linesById = new Map(lines.map((line) => [String(line.linea_id || ""), line]));
  const esPagable = (valor) => [true, "true", "1", "si", "sí", "yes"].includes(
    typeof valor === "string" ? valor.trim().toLocaleLowerCase("es") : valor,
  );
  const creation = data.creacion_manual && Array.isArray(data.creacion_manual.campos)
    ? data.creacion_manual : null;
  const irAMes = (desplazamiento) => {
    const actual = new Date(`${String(data.periodo?.inicio).slice(0, 10)}T12:00:00`);
    actual.setMonth(actual.getMonth() + desplazamiento);
    const mes = `${actual.getFullYear()}-${String(actual.getMonth() + 1).padStart(2, "0")}`;
    const url = new URL(window.location.href); url.searchParams.set("mes", mes);
    window.location.assign(url);
  };
  byId("mes-anterior")?.addEventListener("click", () => irAMes(-1));
  byId("mes-siguiente")?.addEventListener("click", () => irAMes(1));
  byId("mes-actual")?.addEventListener("click", () => window.location.assign("/dashboard"));
  const colasEdicion = new Map();
  const encolarEdicion = (clave, trabajo) => {
    const anterior = colasEdicion.get(clave) || Promise.resolve();
    const actual = anterior.catch(() => {}).then(trabajo);
    colasEdicion.set(clave, actual);
    const limpiar = () => { if (colasEdicion.get(clave) === actual) colasEdicion.delete(clave); };
    actual.then(limpiar, limpiar);
    return actual;
  };
  const mostrarAviso = (mensaje) => {
    document.querySelectorAll(".aviso-dashboard").forEach((aviso) => aviso.remove());
    const aviso = document.createElement("div"); aviso.className = "aviso-dashboard";
    aviso.setAttribute("role", "status"); aviso.setAttribute("aria-live", "polite");
    const icono = document.createElement("span"); icono.setAttribute("aria-hidden", "true"); icono.textContent = "✓";
    const texto = document.createElement("span"); texto.textContent = mensaje;
    aviso.append(icono, texto); document.body.append(aviso);
  };
  const vigilarSincronizacion = (movimientoClave, intento = 0) => {
    if (!movimientoClave || intento >= 20) return;
    window.setTimeout(async () => {
      try {
        const response = await fetch(
          `${API_BASE}/movimientos/${encodeURIComponent(movimientoClave)}/estado`,
          { headers: { Accept: "application/json" } },
        );
        const result = await response.json();
        if (!response.ok || !result.ok) return;
        if (result.estado === "listo") {
          mostrarAviso("Cambios sincronizados correctamente.");
        } else if (result.estado === "error") {
          mostrarAviso("El cambio quedó guardado, pero no pudo sincronizarse todavía.");
        } else {
          vigilarSincronizacion(movimientoClave, intento + 1);
        }
      } catch (_) {
        // La edición sigue guardada; se confirmará al recargar aunque falle un sondeo puntual.
      }
    }, 1500);
  };
  const ajustarKpisMovimiento = (movimiento, factor) => {
    if (!movimiento || !movimiento.linea_id || !Number.isFinite(number(movimiento.monto))) return;
    const monto = number(movimiento.monto) * factor;
    (data.kpis || []).forEach((kpi) => {
      if (!Array.isArray(kpi.columnas) || !Array.isArray(kpi.filas)) return;
      kpi.filas.forEach((fila) => {
        const row = rowObject(kpi, fila);
        const lineaKey = keyMatch(row, /^linea_id$|linea_presupuesto_id/i);
        const categoriaKey = keyMatch(row, /^categoria$|categoría/i);
        const resumenGeneral = ["presupuesto_disponible", "gasto_total"].includes(
          String(kpi.kpi || "").toLocaleLowerCase("es"),
        ) && !lineaKey && !categoriaKey && kpi.filas.length === 1;
        const coincide = resumenGeneral ||
          (lineaKey && String(row[lineaKey]) === String(movimiento.linea_id)) ||
          (!lineaKey && categoriaKey && normalized(row[categoriaKey]) === normalized(movimiento.categoria));
        if (!coincide) return;
        const keys = metricKeys(row);
        const spentKey = keys.spent;
        const indice = spentKey ? kpi.columnas.indexOf(spentKey) : -1;
        if (indice < 0) return;
        fila[indice] = (number(fila[indice]) || 0) + monto;
        const gastado = number(fila[indice]);
        const presupuesto = keys.budget ? number(row[keys.budget]) : NaN;
        if (keys.available && Number.isFinite(presupuesto)) {
          fila[kpi.columnas.indexOf(keys.available)] = presupuesto - gastado;
        }
        if (keys.pct && Number.isFinite(presupuesto) && presupuesto !== 0) {
          fila[kpi.columnas.indexOf(keys.pct)] = gastado / presupuesto * 100;
        }
      });
    });
  };
  const aplicarMovimientoPendiente = (movimiento, agregar = true) => {
    if (!movimiento || !movimiento.linea_id || !Number.isFinite(number(movimiento.monto))) return;
    // La proyección local es sólo UX: la fuente de verdad ya se confirmó en
    // el endpoint y la cola durable la materializa en Neon. Así el usuario no
    // espera la reconstrucción para ver su gasto.
    data.movimientos = Array.isArray(data.movimientos) ? data.movimientos : [];
    if (agregar) data.movimientos.push(movimiento);
    else data.movimientos = data.movimientos.filter((item) => item !== movimiento);
    ajustarKpisMovimiento(movimiento, agregar ? 1 : -1);
    renderVista();
  };
  const iniciarChat = () => {
    const mensajes = byId("chat-mensajes");
    const formulario = byId("chat-formulario");
    const entrada = byId("chat-entrada");
    const enviar = byId("chat-enviar");
    const error = byId("chat-error");
    if (!mensajes || !formulario || !entrada || !enviar || !error) return;

    const desplazarAlFinal = () => { mensajes.scrollTop = mensajes.scrollHeight; };
    const mostrarError = (texto = "") => {
      error.textContent = texto; error.hidden = !texto;
    };
    const adjuntos = (lista) => {
      if (!Array.isArray(lista) || !lista.length) return null;
      const grupo = document.createElement("div"); grupo.className = "chat-adjuntos";
      lista.forEach((adjunto) => {
        if (!adjunto?.contenido_b64 || !adjunto?.nombre) return;
        try {
          const binario = atob(adjunto.contenido_b64);
          const bytes = Uint8Array.from(binario, (caracter) => caracter.charCodeAt(0));
          const enlace = document.createElement("a");
          enlace.className = "chat-adjunto";
          enlace.href = URL.createObjectURL(new Blob([bytes], { type: adjunto.mime || "application/octet-stream" }));
          enlace.download = adjunto.nombre;
          enlace.textContent = `Descargar ${adjunto.nombre}`;
          grupo.append(enlace);
        } catch (_) { /* Un adjunto inválido nunca impide leer la respuesta. */ }
      });
      return grupo.childElementCount ? grupo : null;
    };
    const agregarMensaje = (rol, contenido, botones = [], archivos = []) => {
      const burbuja = document.createElement("article");
      burbuja.className = `chat-mensaje chat-${rol === "user" ? "usuario" : "asistente"}`;
      const texto = document.createElement("p"); texto.textContent = String(contenido || ""); burbuja.append(texto);
      const descargas = adjuntos(archivos); if (descargas) burbuja.append(descargas);
      if (Array.isArray(botones) && botones.length) {
        const acciones = document.createElement("div"); acciones.className = "chat-acciones";
        botones.forEach((boton) => {
          const titulo = String(boton?.title || "").trim(); if (!titulo) return;
          const accion = document.createElement("button"); accion.type = "button"; accion.textContent = titulo;
          accion.addEventListener("click", () => enviarMensaje(titulo)); acciones.append(accion);
        });
        if (acciones.childElementCount) burbuja.append(acciones);
      }
      mensajes.append(burbuja); desplazarAlFinal();
    };
    const respuestaJson = async (respuesta) => {
      try { return await respuesta.json(); } catch (_) { return { ok: false, error: "No recibí una respuesta válida." }; }
    };
    const enviarMensaje = async (textoOriginal) => {
      const texto = String(textoOriginal || "").trim();
      if (!texto || enviar.disabled) return;
      mostrarError(""); agregarMensaje("user", texto); entrada.value = "";
      enviar.disabled = true; entrada.disabled = true; enviar.textContent = "Pensando…";
      try {
        const respuesta = await fetch(`${API_BASE}/chat`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mensaje: texto }),
        });
        const resultado = await respuestaJson(respuesta);
        if (!respuesta.ok || !resultado.ok) throw new Error(resultado.error || "No pude procesar la consulta.");
        agregarMensaje("assistant", resultado.mensaje?.contenido, resultado.botones, resultado.adjuntos);
      } catch (razon) {
        agregarMensaje("assistant", "No pude procesar esa consulta en este momento.");
        mostrarError(razon.message || "Inténtelo nuevamente.");
      } finally {
        enviar.disabled = false; entrada.disabled = false; enviar.textContent = "Enviar"; entrada.focus();
      }
    };
    formulario.addEventListener("submit", (evento) => { evento.preventDefault(); enviarMensaje(entrada.value); });
    entrada.addEventListener("keydown", (evento) => {
      if (evento.key === "Enter" && !evento.shiftKey) { evento.preventDefault(); formulario.requestSubmit(); }
    });
    mensajes.textContent = "Cargando conversación…";
    fetch(`${API_BASE}/chat`, { headers: { Accept: "application/json" } })
      .then(respuestaJson)
      .then((resultado) => {
        mensajes.replaceChildren();
        if (!resultado.ok) throw new Error(resultado.error || "No pude cargar la conversación.");
        if (resultado.mensajes?.length) resultado.mensajes.forEach((mensaje) => agregarMensaje(mensaje.rol, mensaje.contenido));
        else agregarMensaje("assistant", "Hola. Puedes preguntarme lo mismo que por WhatsApp.");
      })
      .catch((razon) => {
        mensajes.replaceChildren(); agregarMensaje("assistant", "No pude cargar el historial todavía.");
        mostrarError(razon.message || "Inténtelo nuevamente.");
      });
  };
  const abrirEditor = (movement) => {
    if (!movement.movimiento_clave || !lines.length) return;
    const dialog = document.createElement("dialog"); dialog.className = "editor-movimiento";
    const form = document.createElement("form"); form.method = "dialog";
    const title = document.createElement("h2"); title.textContent = "Editar movimiento";
    const detail = document.createElement("p"); detail.className = "editor-descripcion";
    detail.textContent = `${movement.descripcion || "Movimiento"} · ${String(movement.fecha || "").slice(0, 10)}`;
    const label = document.createElement("label"); label.textContent = "Concepto presupuestario";
    const select = document.createElement("select"); select.required = true;
    lines.forEach((line) => {
      const option = document.createElement("option"); option.value = line.linea_id;
      option.textContent = `${line.categoria} · ${line.concepto}`;
      option.selected = line.linea_id === movement.linea_id;
      select.append(option);
    });
    label.append(select);
    const paymentLabel = document.createElement("label"); paymentLabel.textContent = "Método de pago";
    const payment = document.createElement("input"); payment.type = "text"; payment.name = "medio_pago";
    payment.autocomplete = "off"; payment.required = true;
    payment.value = String(movement.medio_pago || "").trim() === "Sin método de pago" ? "" : String(movement.medio_pago || "");
    payment.placeholder = "Ej.: Efectivo, SINPE o tarjeta";
    const paymentList = document.createElement("datalist"); const paymentListId = `metodos-pago-${movement.movimiento_clave}`.replace(/[^a-z0-9_-]/gi, "-"); paymentList.id = paymentListId;
    const methods = [...new Set((Array.isArray(data.movimientos) ? data.movimientos : [])
      .map((item) => String(item.medio_pago || "").trim())
      .filter((method) => method && method !== "Sin método de pago"))].sort((a, b) => a.localeCompare(b, "es"));
    methods.forEach((method) => { const option = document.createElement("option"); option.value = method; paymentList.append(option); });
    payment.setAttribute("list", paymentListId); paymentLabel.append(payment, paymentList);
    const amountLabel = document.createElement("label");
    const amountCurrency = String(movement.moneda || "CRC").trim().toUpperCase() || "CRC";
    amountLabel.textContent = `Monto (${amountCurrency})`;
    const amount = document.createElement("input"); amount.type = "number"; amount.name = "monto";
    // Las conversiones históricas pueden producir fracciones de colón. El
    // valor que se muestra y edita es CRC, por lo que el navegador no debe
    // rechazar una corrección válida sólo por tener más de dos decimales.
    amount.min = "0"; amount.step = "0.000001"; amount.inputMode = "decimal"; amount.required = true;
    amount.value = String(movement.monto ?? "");
    amountLabel.append(amount);
    const scopeFieldset = document.createElement("fieldset"); scopeFieldset.className = "editor-alcance";
    const scopeLegend = document.createElement("legend"); scopeLegend.textContent = "Alcance de la reclasificación";
    const scopeOptions = document.createElement("div"); scopeOptions.className = "editor-alcance-opciones";
    const scopeIndividual = document.createElement("label");
    const individualRadio = document.createElement("input"); individualRadio.type = "radio";
    individualRadio.name = `alcance-${movement.movimiento_clave}`; individualRadio.value = "individual"; individualRadio.checked = true;
    scopeIndividual.append(individualRadio, document.createTextNode("Solo este gasto"));
    const scopeGroup = document.createElement("label");
    const groupRadio = document.createElement("input"); groupRadio.type = "radio";
    groupRadio.name = `alcance-${movement.movimiento_clave}`; groupRadio.value = "regla";
    scopeGroup.append(groupRadio, document.createTextNode("Crear una regla para este comercio, pasada y futura"));
    scopeOptions.append(scopeIndividual, scopeGroup);
    scopeFieldset.append(scopeLegend, scopeOptions);
    const note = document.createElement("p"); note.className = "editor-nota";
    note.textContent = "El monto se edita en la misma moneda que muestra el dashboard. El monto y el método de pago solo cambian este gasto; la regla solo clasifica por comercio.";
    const actions = document.createElement("div"); actions.className = "editor-acciones";
    const cancel = document.createElement("button"); cancel.type = "button"; cancel.textContent = "Cancelar";
    cancel.addEventListener("click", () => dialog.close());
    const save = document.createElement("button"); save.type = "submit"; save.textContent = "Guardar cambios";
    const error = document.createElement("p"); error.className = "editor-error"; error.hidden = true;
    actions.append(cancel, save); form.append(title, detail, label, paymentLabel, amountLabel, scopeFieldset, note, error, actions); dialog.append(form);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const lineaNueva = linesById.get(String(select.value));
      const anterior = {
        linea_id: movement.linea_id, categoria: movement.categoria, concepto: movement.concepto,
        medio_pago: movement.medio_pago, monto: movement.monto, moneda: movement.moneda,
      };
      const revision = (number(movement.__revision_local) || 0) + 1;
      movement.__revision_local = revision;
      ajustarKpisMovimiento(anterior, -1);
      movement.linea_id = select.value;
      movement.categoria = lineaNueva?.categoria || movement.categoria;
      movement.concepto = lineaNueva?.concepto || movement.concepto;
      movement.medio_pago = payment.value;
      movement.monto = amount.value;
      ajustarKpisMovimiento(movement, 1);
      dialog.close(); renderVista();
      mostrarAviso("Cambio aplicado. Guardando en segundo plano…");
      const payload = {
        movimiento_clave: movement.movimiento_clave,
        linea_id: select.value,
        medio_pago: payment.value,
        alcance: groupRadio.checked ? "regla" : "individual",
        monto: amount.value, periodo_inicio: data.periodo?.inicio,
      };
      encolarEdicion(movement.movimiento_clave, async () => {
        const response = await fetch(`${API_BASE}/movimientos/reclasificar`, {
          method: "POST", headers: { "Content-Type": "application/json" }, keepalive: true,
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || "No pude guardar la clasificación.");
        if (movement.__revision_local === revision) {
          ajustarKpisMovimiento(movement, -1);
          movement.linea_id = result.linea_id; movement.categoria = result.categoria;
          movement.concepto = result.concepto; movement.medio_pago = result.medio_pago;
          if (result.monto !== undefined) movement.monto = result.monto;
          if (result.moneda) movement.moneda = result.moneda;
          ajustarKpisMovimiento(movement, 1);
          renderVista();
        }
        vigilarSincronizacion(movement.movimiento_clave);
        window.dispatchEvent(new Event("fachavi:movimiento-editado"));
      }).catch((reason) => {
        if (movement.__revision_local === revision) {
          ajustarKpisMovimiento(movement, -1); Object.assign(movement, anterior);
          ajustarKpisMovimiento(movement, 1); renderVista();
        }
        mostrarAviso(reason.message || "No pude guardar el cambio; restauré el valor anterior.");
      });
    });
    document.body.append(dialog); dialog.addEventListener("close", () => dialog.remove()); dialog.showModal();
  };
  const abrirCreador = () => {
    if (!creation) return;
    const dialog = document.createElement("dialog"); dialog.className = "editor-movimiento editor-creacion";
    const form = document.createElement("form");
    const title = document.createElement("h2"); title.textContent = "Agregar movimiento";
    const note = document.createElement("p"); note.className = "editor-nota";
    note.textContent = "Se guardará como gasto manual y aparecerá en el dashboard al actualizarse.";
    const controls = new Map();
    const derived = new Map();
    const updateDerived = () => {
      const lineField = creation.campos.find((field) => field.seleccion_linea);
      const selected = lineField ? lines.find((line) => line.linea_id === controls.get(lineField.nombre)?.value) : null;
      derived.forEach((control, name) => {
        control.value = name === "categoria" ? (selected?.categoria || "") : "";
      });
    };
    creation.campos.forEach((field) => {
      const label = document.createElement("label"); label.textContent = field.etiqueta;
      let control;
      if (field.derivado_de_linea) {
        control = document.createElement("input"); control.type = "text"; control.readOnly = true;
        control.placeholder = "Se completa al elegir el concepto"; derived.set(field.nombre, control);
      } else if (field.seleccion_linea) {
        control = document.createElement("select"); control.required = Boolean(field.requerido);
        const empty = document.createElement("option"); empty.value = ""; empty.textContent = "Seleccione un concepto"; control.append(empty);
        lines.forEach((line) => {
          const option = document.createElement("option"); option.value = line.linea_id;
          option.textContent = `${line.categoria} · ${line.concepto}`; control.append(option);
        });
        control.addEventListener("change", updateDerived);
      } else if (Array.isArray(field.valores) && field.valores.length) {
        control = document.createElement("select");
        if (!field.requerido) { const empty = document.createElement("option"); empty.value = ""; empty.textContent = "Sin especificar"; control.append(empty); }
        field.valores.forEach((value) => {
          const option = document.createElement("option"); option.value = value; option.textContent = value;
          option.selected = value === field.defecto; control.append(option);
        });
      } else {
        control = document.createElement("input");
        if (field.tipo === "fecha_iso") control.type = "date";
        else if (field.tipo === "monto_positivo") { control.type = "number"; control.min = "0.01"; control.step = "0.01"; control.inputMode = "decimal"; }
        else { control.type = "text"; if (field.tipo === "moneda_iso") { control.maxLength = 3; control.pattern = "[A-Za-z]{3}"; control.autocapitalize = "characters"; } }
        control.value = field.defecto || "";
        control.placeholder = field.ejemplo || "";
      }
      control.name = field.nombre; control.required = Boolean(field.requerido); controls.set(field.nombre, control);
      label.append(control); form.append(label);
    });
    const error = document.createElement("p"); error.className = "editor-error"; error.hidden = true;
    const actions = document.createElement("div"); actions.className = "editor-acciones";
    const cancel = document.createElement("button"); cancel.type = "button"; cancel.textContent = "Cancelar";
    cancel.addEventListener("click", () => dialog.close());
    const save = document.createElement("button"); save.type = "submit"; save.textContent = "Guardar movimiento";
    actions.append(cancel, save); form.prepend(title, note); form.append(error, actions); dialog.append(form);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const valores = {};
      creation.campos.filter((field) => !field.derivado_de_linea).forEach((field) => {
        const control = controls.get(field.nombre); if (control) valores[field.nombre] = control.value;
      });
      const campoLinea = creation.campos.find((field) => field.seleccion_linea);
      const campoMonto = creation.campos.find((field) => field.tipo === "monto_positivo");
      const campoFecha = creation.campos.find((field) => field.tipo === "fecha_iso");
      const campoMoneda = creation.campos.find((field) => field.tipo === "moneda_iso");
      const campoDescripcion = creation.campos.find((field) => /descripci/i.test(field.nombre));
      const linea = linesById.get(String(valores[campoLinea?.nombre] || ""));
      const pendiente = {
        linea_id: linea?.linea_id, categoria: linea?.categoria, concepto: linea?.concepto,
        fecha: valores[campoFecha?.nombre] || "", descripcion: valores[campoDescripcion?.nombre] || "Movimiento manual",
        monto: valores[campoMonto?.nombre] || "0", moneda: valores[campoMoneda?.nombre] || "CRC",
        medio_pago: valores.medio_pago || "Sin método de pago",
        movimiento_clave: `pendiente:${Date.now()}`, pendiente_sincronizacion: true,
      };
      dialog.close(); aplicarMovimientoPendiente(pendiente);
      mostrarAviso("Movimiento agregado. Guardando en segundo plano…");
      (async () => {
        const response = await fetch(`${API_BASE}/movimientos/crear`, {
          method: "POST", headers: { "Content-Type": "application/json" }, keepalive: true,
          body: JSON.stringify({ valores, periodo_inicio: data.periodo?.inicio }),
        });
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || "No pude guardar el movimiento.");
        Object.assign(pendiente, result.movimiento);
        mostrarAviso("Movimiento guardado. Sincronizando en segundo plano…");
        vigilarSincronizacion(result.movimiento_clave);
        window.dispatchEvent(new Event("fachavi:movimiento-editado"));
      })().catch((reason) => {
        aplicarMovimientoPendiente(pendiente, false);
        mostrarAviso(reason.message || "No pude guardar el movimiento; lo retiré de la vista.");
      });
    });
    document.body.append(dialog); dialog.addEventListener("close", () => dialog.remove()); dialog.showModal();
  };
  const fechaPredeterminadaPago = () => {
    const inicio = String(data.periodo?.inicio || "").slice(0, 10);
    const fin = String(data.periodo?.fin_exclusivo || "").slice(0, 10);
    const ahora = new Date();
    const hoy = [ahora.getFullYear(), String(ahora.getMonth() + 1).padStart(2, "0"), String(ahora.getDate()).padStart(2, "0")].join("-");
    return inicio && fin && hoy >= inicio && hoy < fin ? hoy : inicio;
  };
  const estadoPago = (presupuesto, gastado) => {
    const budget = number(presupuesto);
    const spent = number(gastado);
    const monto = Number.isFinite(budget) ? budget : 0;
    const pagado = Number.isFinite(spent) ? spent : 0;
    if (pagado <= 0) return { etiqueta: "Pendiente", accion: "Pagar", sugerido: monto > 0 ? monto : "" };
    if (pagado < monto) return { etiqueta: "Pago parcial", accion: "Completar pago", sugerido: monto - pagado };
    if (pagado === monto) return { etiqueta: "Pagado ✓", accion: "Registrar pago adicional", sugerido: monto > 0 ? monto : "" };
    return { etiqueta: `Pagado · excedido ${format(pagado - monto, "monto", "CRC")}`, accion: "Registrar pago adicional", sugerido: monto > 0 ? monto : "" };
  };
  const abrirPago = async (line, row, budgetKey, spentKey) => {
    if (!line?.linea_id || !esPagable(line.pagable)) return;
    let cuentasPago = window.fachaviCuentasDisponibles?.() || [];
    if (!cuentasPago.length && window.fachaviCargarCuentas) {
      await window.fachaviCargarCuentas();
      cuentasPago = window.fachaviCuentasDisponibles?.() || [];
    }
    const presupuesto = number(row[budgetKey]);
    const gastado = number(row[spentKey]);
    const estado = estadoPago(presupuesto, gastado);
    const dialog = document.createElement("dialog"); dialog.className = "editor-movimiento editor-pago";
    const form = document.createElement("form");
    const title = document.createElement("h2"); title.textContent = "Pagar";
    const detail = document.createElement("p"); detail.className = "editor-descripcion";
    detail.textContent = `${line.categoria} · ${line.concepto}`;
    const resumen = document.createElement("dl"); resumen.className = "resumen-pago";
    [["Monto presupuestado", presupuesto], ["Monto ya gastado", gastado], ["Saldo presupuestario", Math.max((presupuesto || 0) - (gastado || 0), 0)]]
      .forEach(([etiqueta, valor]) => {
        const term = document.createElement("dt"); term.textContent = etiqueta;
        const definition = document.createElement("dd"); definition.textContent = format(valor, "monto", row.moneda || "CRC");
        resumen.append(term, definition);
      });
    const montoLabel = document.createElement("label"); montoLabel.textContent = "Monto a pagar";
    const monto = document.createElement("input"); monto.type = "number"; monto.name = "monto";
    monto.min = "0.01"; monto.step = "0.01"; monto.inputMode = "decimal"; monto.required = true;
    monto.value = estado.sugerido === "" ? "" : String(estado.sugerido);
    montoLabel.append(monto);
    const fechaLabel = document.createElement("label"); fechaLabel.textContent = "Fecha del pago";
    const fecha = document.createElement("input"); fecha.type = "date"; fecha.name = "fecha"; fecha.required = true;
    fecha.value = fechaPredeterminadaPago();
    fecha.min = String(data.periodo?.inicio || "").slice(0, 10);
    const fin = String(data.periodo?.fin_exclusivo || "").slice(0, 10);
    if (fin) {
      const ultimo = new Date(`${fin}T12:00:00`); ultimo.setDate(ultimo.getDate() - 1);
      fecha.max = [ultimo.getFullYear(), String(ultimo.getMonth() + 1).padStart(2, "0"), String(ultimo.getDate()).padStart(2, "0")].join("-");
    }
    fechaLabel.append(fecha);
    const metodoLabel = document.createElement("label"); metodoLabel.textContent = "Pagar desde";
    const metodo = document.createElement("select"); metodo.required = true;
    if (cuentasPago.length) {
      const cuentaPredeterminada = window.fachaviCuentaPagoPredeterminada?.() || cuentasPago[0].cuenta_id;
      cuentasPago.forEach((cuenta) => {
        const option = document.createElement("option");
        option.value = cuenta.ultimos4 || `cuenta:${cuenta.cuenta_id}`;
        option.textContent = cuenta.nombre;
        option.dataset.cuentaOrigen = cuenta.nombre;
        option.selected = cuenta.cuenta_id === cuentaPredeterminada;
        metodo.append(option);
      });
    } else {
      const option = document.createElement("option"); option.value = "Sin método de pago";
      option.textContent = "Sin cuenta vinculada"; metodo.append(option);
    }
    metodoLabel.append(metodo);
    const note = document.createElement("p"); note.className = "editor-nota";
    note.textContent = "Se guardará como un gasto manual normal asociado a este concepto.";
    const error = document.createElement("p"); error.className = "editor-error"; error.hidden = true;
    const actions = document.createElement("div"); actions.className = "editor-acciones";
    const cancel = document.createElement("button"); cancel.type = "button"; cancel.textContent = "Cancelar";
    cancel.addEventListener("click", () => dialog.close());
    const save = document.createElement("button"); save.type = "submit"; save.textContent = "Confirmar pago";
    actions.append(cancel, save); form.append(title, detail, resumen, montoLabel, fechaLabel, metodoLabel, note, error, actions); dialog.append(form);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const pendiente = {
        linea_id: line.linea_id, categoria: line.categoria, concepto: line.concepto,
        fecha: fecha.value, descripcion: `Pago - ${line.concepto}`,
        monto: monto.value, moneda: row.moneda || "CRC", medio_pago: metodo.value,
        movimiento_clave: `pendiente:${Date.now()}`, pendiente_sincronizacion: true,
      };
      const cuentaOrigen = metodo.selectedOptions[0]?.dataset.cuentaOrigen || "";
      dialog.close(); aplicarMovimientoPendiente(pendiente);
      mostrarAviso("Pago aplicado. Guardando en segundo plano…");
      (async () => {
        const response = await fetch(`${API_BASE}/conceptos/${encodeURIComponent(line.linea_id)}/pagar`, {
          method: "POST", headers: { "Content-Type": "application/json" }, keepalive: true,
          body: JSON.stringify({ monto: monto.value, fecha: fecha.value,
            medio_pago: metodo.value, cuenta_origen: cuentaOrigen,
            periodo_inicio: data.periodo?.inicio }),
        });
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || "No pude guardar el pago.");
        Object.assign(pendiente, result.movimiento);
        mostrarAviso("Pago guardado. Sincronizando en segundo plano…");
        vigilarSincronizacion(result.movimiento_clave);
        window.dispatchEvent(new Event("fachavi:movimiento-editado"));
      })().catch((reason) => {
        aplicarMovimientoPendiente(pendiente, false);
        mostrarAviso(reason.message || "No pude guardar el pago; lo retiré de la vista.");
      });
    });
    document.body.append(dialog); dialog.addEventListener("close", () => dialog.remove()); dialog.showModal();
  };
  const dimensionFor = (kpi, row) => {
    const texto = `${kpi.kpi || ""} ${kpi.nombre || ""} ${kpi.descripcion || ""}`.toLowerCase();
    // Prefer the exact dimension named by the KPI. A concept KPI can also
    // return `categoria` for context, but that must not replace `concepto` as
    // the chart label merely because it appears first in the SQL result.
    const exact = texto.includes("comercio")
      ? /^comercio$/i
      : texto.includes("concepto")
        ? /^concepto$/i
        : texto.includes("categoria")
          ? /^categoria$/i
          : null;
    if (exact) {
      const exactKey = keyMatch(row, exact);
      if (exactKey) return exactKey;
    }
    const fallback = texto.includes("comercio")
      ? /descripcion|comercio|concepto|nombre|categoria/i
      : texto.includes("concepto")
        ? /concepto|descripcion|comercio|nombre/i
        : /categoria|comercio|concepto|descripcion|nombre/i;
    return keyMatch(row, fallback);
  };

  const metricKeys = (row) => ({
    budget: keyMatch(row, /^presupuesto$|monto_presupuestado|presupuesto_mensual|^mensual$/i),
    spent: keyMatch(row, /^gastado$|^gasto$|gasto_neto|^monto$|total_gastado/i),
    available: keyMatch(row, /^disponible$|saldo/i),
    pct: keyMatch(row, /pct|porcentaje/i),
  });

  const appendMetricBars = (container, row) => {
    const keys = metricKeys(row);
    const numeric = [keys.budget, keys.spent].map((key) => key && Math.abs(number(row[key]))).filter(Number.isFinite);
    if (!numeric.length) return;
    const max = Math.max(...numeric, 1);
    [[keys.budget, "Presupuesto", "bar-budget"], [keys.spent, "Gastado", "bar-spent"]]
      .filter(([key]) => key)
      .forEach(([key, label, cls]) => {
        const line = document.createElement("div"); line.className = "bar-line";
        const labelEl = document.createElement("span"); labelEl.textContent = label;
        const track = document.createElement("span"); track.className = "bar-track";
        const fill = document.createElement("span"); fill.className = `bar-fill ${cls}`;
        fill.style.width = `${Math.min(100, Math.abs(number(row[key])) / max * 100)}%`;
        track.append(fill);
        const value = document.createElement("strong"); value.textContent = format(row[key], key, row.moneda);
        line.append(labelEl, track, value); container.append(line);
      });
  };

  const renderHierarchy = (target) => {
    const byIdInsensitive = (ids) => data.kpis.find((k) => ids.includes(String(k.kpi || "").toLowerCase()));
    const conceptKpi = byIdInsensitive([
      "ejecucion_presupuesto_concepto", "gasto_por_concepto", "plan_presupuesto_concepto",
    ]);
    const categoryKpi = byIdInsensitive([
      "gasto_por_categoria", "ejecucion_presupuesto_mes", "plan_presupuesto",
    ]);
    const commerceKpi = byIdInsensitive(["gasto_por_comercio"]);
    if (!conceptKpi && !categoryKpi) return new Set();

    const concepts = (conceptKpi || categoryKpi).filas.map((row) => rowObject(conceptKpi || categoryKpi, row));
    const categories = new Map();
    const conceptRows = [];
    concepts.forEach((row) => {
      const categoryKey = keyMatch(row, /^categoria$|categoría/i);
      const conceptKey = keyMatch(row, /^concepto$|rubro/i);
      const category = row[categoryKey] || "Sin categoría";
      const concept = conceptKey ? row[conceptKey] : null;
      const bucket = categories.get(normalized(category)) || { name: category, rows: [], totals: {} };
      bucket.rows.push(row);
      categories.set(normalized(category), bucket);
      if (concept) conceptRows.push({ row, category: normalized(category), name: concept });
    });

    // Si el KPI de conceptos ya trae presupuesto/gasto por linea, sumar esos
    // valores produce el encabezado de categoría sin otra consulta.
    categories.forEach((bucket) => {
      const sample = bucket.rows[0];
      const keys = metricKeys(sample);
      bucket.totals = { ...sample };
      if (keys.budget) bucket.totals[keys.budget] = bucket.rows.reduce((sum, row) => sum + (number(row[keys.budget]) || 0), 0);
      if (keys.spent) bucket.totals[keys.spent] = bucket.rows.reduce((sum, row) => sum + (number(row[keys.spent]) || 0), 0);
      if (keys.available) bucket.totals[keys.available] = (number(bucket.totals[keys.budget]) || 0) - (number(bucket.totals[keys.spent]) || 0);
    });

    const movements = Array.isArray(data.movimientos) && data.movimientos.length
      ? data.movimientos
      : (commerceKpi ? commerceKpi.filas.map((row) => rowObject(commerceKpi, row)) : []);
    const movementName = (row) => {
      const comercio = keyMatch(row, /^comercio$/i);
      const descripcion = keyMatch(row, /^descripcion$|^descripción$/i);
      const nombre = keyMatch(row, /^nombre$/i);
      const concepto = keyMatch(row, /^concepto$/i);
      return row[comercio || descripcion || nombre || concepto] || "Movimiento";
    };
    const movementLine = (row) => normalized(row[keyMatch(row, /^linea_id$|linea_presupuesto_id/i)]);
    const movementCategory = (row) => normalized(row[keyMatch(row, /^categoria$|categoría/i)]);
    const movementConcept = (row) => normalized(row[keyMatch(row, /^concepto$|rubro/i)]);

    // Los KPIs del presupuesto sólo contienen líneas presupuestadas. Un
    // movimiento sin línea no debe ocultarse por ello: llega desde el backend
    // como "Sin clasificar / Gastos sin identificar" y se agrega como un nodo
    // de presupuesto cero. No se le asigna una categoría de negocio inventada.
    const unclassified = new Map();
    movements.forEach((movement) => {
      const categoryKey = keyMatch(movement, /^categoria$|categoría/i);
      const category = movement[categoryKey];
      const categoryName = category || "Sin clasificar";
      const categoryId = normalized(categoryName);
      if (categories.has(categoryId)) return;
      const conceptKey = keyMatch(movement, /^concepto$|rubro/i);
      const conceptName = movement[conceptKey] || "Gastos sin identificar";
      const currencyKey = keyMatch(movement, /^moneda$/i);
      const moneda = movement[currencyKey] || "CRC";
      const conceptId = `${categoryId}|${normalized(conceptName)}|${normalized(moneda)}`;
      const entry = unclassified.get(conceptId) || {
        categoria: categoryName, concepto: conceptName, moneda, gastado: 0,
      };
      entry.gastado += number(movement[keyMatch(movement, /^monto$|gasto.?neto|gastado/i)]);
      unclassified.set(conceptId, entry);
    });
    unclassified.forEach((entry) => {
      const categoryId = normalized(entry.categoria);
      const row = {
        categoria: entry.categoria,
        concepto: `${entry.concepto} · ${entry.moneda}`,
        origen_concepto: entry.concepto,
        moneda: entry.moneda,
        presupuesto: 0,
        gastado: entry.gastado,
      };
      const bucket = categories.get(categoryId) || {
        name: entry.categoria, rows: [], totals: {}, monedas: new Set(),
      };
      bucket.rows.push(row);
      if (bucket.monedas) bucket.monedas.add(entry.moneda);
      bucket.totals.presupuesto = (number(bucket.totals.presupuesto) || 0) + row.presupuesto;
      bucket.totals.gastado = (number(bucket.totals.gastado) || 0) + row.gastado;
      categories.set(categoryId, bucket);
    });

    // Algunas metadata solo publica el plan (``mensual``) y no un KPI de
    // ejecución. En ese contrato el gasto real se obtiene de los movimientos
    // canónicos, agrupados por línea presupuestaria, sin usar subtotales como
    // si fueran gastos.
    const gastoPorLinea = new Map();
    const gastoPorConcepto = new Map();
    movements.forEach((movement) => {
      const linea = movementLine(movement);
      const amountKey = keyMatch(movement, /^monto$|gasto.?neto|^gastado$|total_gastado/i);
      if (!linea || !amountKey) return;
      gastoPorLinea.set(linea, (gastoPorLinea.get(linea) || 0) + (number(movement[amountKey]) || 0));
      const categoriaKey = keyMatch(movement, /^categoria$|categoría/i);
      const conceptoKey = keyMatch(movement, /^concepto$|rubro/i);
      const claveConcepto = `${normalized(movement[categoriaKey])}|${normalized(movement[conceptoKey])}`;
      if (conceptoKey && movement[conceptoKey]) {
        gastoPorConcepto.set(claveConcepto, (gastoPorConcepto.get(claveConcepto) || 0) + (number(movement[amountKey]) || 0));
      }
    });
    categories.forEach((bucket) => {
      bucket.rows.forEach((row) => {
        const keys = metricKeys(row);
        if (keys.spent) return;
        const lineKey = keyMatch(row, /^linea_id$|linea_presupuesto_id/i);
        if (lineKey && row[lineKey]) {
          row.gastado = gastoPorLinea.get(normalized(row[lineKey])) || 0;
        } else {
          const conceptKey = keyMatch(row, /^concepto$|rubro/i);
          const categoriaKey = keyMatch(row, /^categoria$|categoría/i);
          const claveConcepto = `${normalized(row[categoriaKey])}|${normalized(row[conceptKey])}`;
          row.gastado = gastoPorConcepto.get(claveConcepto) || 0;
        }
      });
      const sample = bucket.rows[0];
      const keys = metricKeys(sample);
      bucket.totals = { ...sample };
      if (keys.budget) bucket.totals[keys.budget] = bucket.rows.reduce((sum, row) => sum + (number(row[keys.budget]) || 0), 0);
      if (keys.spent) bucket.totals[keys.spent] = bucket.rows.reduce((sum, row) => sum + (number(row[keys.spent]) || 0), 0);
      else bucket.totals.gastado = bucket.rows.reduce((sum, row) => sum + (number(row.gastado) || 0), 0);
      if (keys.available) bucket.totals[keys.available] = (number(bucket.totals[keys.budget]) || 0) - (number(bucket.totals[keys.spent]) || 0);
    });
    const usedMovements = new Set();

    const panel = document.createElement("article"); panel.className = "panel panel-jerarquia";
    const title = document.createElement("h2"); title.textContent = "Gasto mensual por categoría"; panel.append(title);
    const description = document.createElement("p"); description.className = "descripcion";
    description.textContent = "Expande una categoría para ver sus conceptos y cada movimiento asociado."; panel.append(description);
    const tree = document.createElement("div"); tree.className = "jerarquia";

    categories.forEach((bucket) => {
      const categoryDetails = document.createElement("details"); categoryDetails.className = "nivel nivel-categoria";
      const categorySummary = document.createElement("summary");
      const categoryHeading = document.createElement("span"); categoryHeading.className = "nivel-titulo"; categoryHeading.textContent = bucket.name;
      const categoryMeta = document.createElement("span"); categoryMeta.className = "nivel-meta";
      const categoryKeys = metricKeys(bucket.totals);
      const multipleCurrencies = bucket.monedas && bucket.monedas.size > 1;
      if (multipleCurrencies) {
        categoryMeta.textContent = "Gastos en varias monedas";
      } else {
        const currency = bucket.monedas ? [...bucket.monedas][0] : "";
        agregarMeta(categoryMeta, categoryKeys.budget && `Presupuesto: ${format(bucket.totals[categoryKeys.budget], categoryKeys.budget, currency)}`);
        agregarMeta(
          categoryMeta,
          categoryKeys.spent && `Gastado: ${format(bucket.totals[categoryKeys.spent], categoryKeys.spent, currency)}`,
          gastoExcedePresupuesto(bucket.totals[categoryKeys.budget], bucket.totals[categoryKeys.spent]) ? "meta-gastado-excedido" : "",
        );
      }
      categorySummary.append(categoryHeading, categoryMeta); categoryDetails.append(categorySummary);
      const conceptsWrap = document.createElement("div"); conceptsWrap.className = "nivel-hijos";
      const rows = bucket.rows.filter((row) => keyMatch(row, /^concepto$|rubro/i));
      rows.forEach((row) => {
        const conceptKey = keyMatch(row, /^concepto$|rubro/i);
        const conceptLineKey = keyMatch(row, /^linea_id$|linea_presupuesto_id/i);
        const linea = conceptLineKey ? linesById.get(String(row[conceptLineKey] || "")) : null;
        const conceptDetails = document.createElement("details"); conceptDetails.className = "nivel nivel-concepto";
        const conceptSummary = document.createElement("summary");
        const conceptHeading = document.createElement("span"); conceptHeading.className = "nivel-titulo"; conceptHeading.textContent = row[conceptKey];
        const conceptKeys = metricKeys(row); const conceptMeta = document.createElement("span"); conceptMeta.className = "nivel-meta";
        agregarMeta(conceptMeta, conceptKeys.budget && `Presupuesto: ${format(row[conceptKeys.budget], conceptKeys.budget, row.moneda)}`);
        agregarMeta(
          conceptMeta,
          conceptKeys.spent && `Gastado: ${format(row[conceptKeys.spent], conceptKeys.spent, row.moneda)}`,
          gastoExcedePresupuesto(row[conceptKeys.budget], row[conceptKeys.spent]) ? "meta-gastado-excedido" : "",
        );
        conceptSummary.append(conceptHeading, conceptMeta);
        let accionesPago = null;
        if (linea && esPagable(linea.pagable) && conceptKeys.budget && conceptKeys.spent) {
          const pago = estadoPago(row[conceptKeys.budget], row[conceptKeys.spent]);
          accionesPago = document.createElement("div"); accionesPago.className = "pago-acciones";
          const estado = document.createElement("span"); estado.className = "pago-estado"; estado.textContent = pago.etiqueta;
          const boton = document.createElement("button"); boton.type = "button"; boton.className = "boton-pagar"; boton.textContent = pago.accion;
          boton.addEventListener("click", (event) => {
            event.preventDefault(); event.stopPropagation();
            abrirPago(linea, row, conceptKeys.budget, conceptKeys.spent);
          });
          accionesPago.append(estado, boton);
        }
        conceptDetails.append(conceptSummary);
        if (accionesPago) conceptDetails.append(accionesPago);
        const conceptBody = document.createElement("div"); conceptBody.className = "nivel-detalle"; appendMetricBars(conceptBody, row);
        const matches = movements.filter((movement, index) => {
          const originalConcept = row.origen_concepto || row[conceptKey];
          const sameLine = conceptLineKey && movementLine(movement) && movementLine(movement) === normalized(row[conceptLineKey]);
          const sameConcept = movementConcept(movement) === normalized(originalConcept);
          const sameCategory = movementCategory(movement) && movementCategory(movement) === normalized(bucket.name);
          const sameName = normalized(movementName(movement)) === normalized(originalConcept);
          const movementCurrency = movement[keyMatch(movement, /^moneda$/i)];
          const sameCurrency = !row.moneda || !movementCurrency || normalized(movementCurrency) === normalized(row.moneda);
          if ((sameLine || sameConcept || sameName) && (!movementCategory(movement) || sameCategory) && sameCurrency) { usedMovements.add(index); return true; }
          return false;
        });
        if (matches.length) {
          const movementList = document.createElement("ul"); movementList.className = "movimientos";
          matches.forEach((movement) => {
            const item = document.createElement("li");
            const detail = document.createElement("span");
            const name = document.createElement("strong"); name.textContent = movementName(movement);
            const fechaKey = keyMatch(movement, /^fecha$|fecha_transaccion/i);
            const date = document.createElement("small"); date.textContent = fechaKey ? String(movement[fechaKey]).slice(0, 10) : "";
            detail.append(name, date);
            const keys = metricKeys(movement); const value = document.createElement("strong");
            value.textContent = keys.spent ? format(movement[keys.spent], keys.spent, movement.moneda || commerceKpi?.unidad) : "";
            item.append(detail, value);
            if (movement.movimiento_clave && !movement.pendiente_sincronizacion && lines.length) {
              const edit = document.createElement("button"); edit.type = "button"; edit.className = "editar-movimiento";
              edit.setAttribute("aria-label", `Reclasificar ${movementName(movement)}`); edit.title = "Reclasificar"; edit.textContent = "✎";
              edit.addEventListener("click", () => abrirEditor(movement)); item.append(edit);
            }
            movementList.append(item);
          });
          conceptBody.append(movementList);
        }
        conceptDetails.append(conceptBody); conceptsWrap.append(conceptDetails);
      });
      categoryDetails.append(conceptsWrap); tree.append(categoryDetails);
    });
    panel.append(tree); target.append(panel);
    return new Set([conceptKpi, categoryKpi, commerceKpi].filter(Boolean));
  };

  const renderPaymentHierarchy = (target) => {
    const movements = Array.isArray(data.movimientos) ? data.movimientos : [];
    if (!movements.length) return false;
    const groups = new Map();
    movements.forEach((movement) => {
      const methodKey = keyMatch(movement, /^medio_pago$|metodo_pago|método de pago/i);
      const currencyKey = keyMatch(movement, /^moneda$/i);
      const amountKey = keyMatch(movement, /^monto$|gasto.?neto|gastado/i);
      const method = String(movement[methodKey] || "Sin método de pago").trim() || "Sin método de pago";
      const currency = String(movement[currencyKey] || "CRC").trim() || "CRC";
      const groupKey = `${normalized(method)}|${normalized(currency)}`;
      const group = groups.get(groupKey) || { method, currency, spent: 0, movements: [] };
      group.spent += number(movement[amountKey]) || 0;
      group.movements.push(movement); groups.set(groupKey, group);
    });

    const movementName = (movement) => {
      const nameKey = keyMatch(movement, /^descripcion$|^descripción$|^comercio$|^nombre$/i);
      return movement[nameKey] || "Movimiento";
    };
    const paymentLabel = (method) => {
      // Los datos bancarios almacenan la tarjeta enmascarada. Mostramos solo
      // los últimos cuatro dígitos para que el encabezado sea claro sin
      // exponer más información de la necesaria.
      const maskedCard = String(method).match(/^\*+(\d{4})$/);
      return maskedCard ? `Tarjeta · ****${maskedCard[1]}` : method;
    };
    const movementDate = (movement) => {
      const dateKey = keyMatch(movement, /^fecha$|fecha_transaccion/i);
      return dateKey ? String(movement[dateKey] || "") : "";
    };
    const panel = document.createElement("article"); panel.className = "panel panel-jerarquia";
    const title = document.createElement("h2"); title.textContent = "Gasto mensual por método de pago"; panel.append(title);
    const description = document.createElement("p"); description.className = "descripcion";
    description.textContent = "Expande una tarjeta o método de pago para ver los movimientos que lo componen."; panel.append(description);
    const tree = document.createElement("div"); tree.className = "jerarquia";
    [...groups.values()].sort((a, b) => b.spent - a.spent).forEach((group) => {
      const details = document.createElement("details"); details.className = "nivel nivel-categoria";
      const summary = document.createElement("summary");
      const heading = document.createElement("span"); heading.className = "nivel-titulo"; heading.textContent = paymentLabel(group.method);
      const meta = document.createElement("span"); meta.className = "nivel-meta";
      meta.textContent = `Gastado: ${format(group.spent, "monto", group.currency)} · ${group.movements.length} movimiento${group.movements.length === 1 ? "" : "s"}`;
      summary.append(heading, meta); details.append(summary);
      const body = document.createElement("div"); body.className = "nivel-detalle";
      const bar = document.createElement("div"); bar.className = "bar-line";
      const label = document.createElement("span"); label.textContent = "Gastado";
      const track = document.createElement("span"); track.className = "bar-track";
      const fill = document.createElement("span"); fill.className = "bar-fill bar-spent"; fill.style.width = "100%"; track.append(fill);
      const value = document.createElement("strong"); value.textContent = format(group.spent, "monto", group.currency);
      bar.append(label, track, value); body.append(bar);
      const list = document.createElement("ul"); list.className = "movimientos";
      group.movements.sort((a, b) => movementDate(b).localeCompare(movementDate(a))).forEach((movement) => {
        const item = document.createElement("li");
        const detail = document.createElement("span");
        const name = document.createElement("strong"); name.textContent = movementName(movement);
        const date = document.createElement("small"); date.textContent = movementDate(movement).slice(0, 10);
        detail.append(name, date);
        const amountKey = keyMatch(movement, /^monto$|gasto.?neto|gastado/i);
        const amount = document.createElement("strong"); amount.textContent = amountKey ? format(movement[amountKey], amountKey, group.currency) : "";
        item.append(detail, amount);
        if (movement.movimiento_clave && !movement.pendiente_sincronizacion && lines.length) {
          const edit = document.createElement("button"); edit.type = "button"; edit.className = "editar-movimiento";
          edit.setAttribute("aria-label", `Reclasificar ${movementName(movement)}`); edit.title = "Reclasificar"; edit.textContent = "✎";
          edit.addEventListener("click", () => abrirEditor(movement)); item.append(edit);
        }
        list.append(item);
      });
      body.append(list); details.append(body); tree.append(details);
    });
    panel.append(tree); target.append(panel);
    return true;
  };

  byId("cliente").textContent = data.cliente.nombre;
  byId("periodo").textContent = data.periodo.etiqueta;
  byId("actualizado").textContent = "Actualizado: " + new Date(data.actualizado_en).toLocaleString("es-CR");
  const addMovement = byId("agregar-movimiento");
  if (creation) { addMovement.hidden = false; addMovement.addEventListener("click", abrirCreador); }

  const summary = findKpi("presupuesto_disponible") || data.kpis.find((k) => k.filas.length === 1 && k.columnas.some((c) => /presupuesto/i.test(c)));
  const summaryRow = summary?.filas?.[0] ? rowObject(summary, summary.filas[0]) : null;
  const descargarReporte = async () => {
    const boton = byId("descargar-reporte");
    if (!boton || boton.disabled) return;
    boton.disabled = true;
    const textoOriginal = boton.textContent;
    boton.textContent = "Preparando descargas…";
    try {
      const inicio = String(data.periodo?.inicio || "").slice(0, 10);
      const respuesta = await fetch(
        `${API_BASE}/cuentas?inicio=${encodeURIComponent(inicio)}`,
        { headers: { Accept: "application/json" } },
      );
      const saldos = await respuesta.json();
      if (!respuesta.ok || !saldos.ok) throw new Error(saldos.error || "No pude cargar los saldos del período.");

      // Un CSV con BOM y punto y coma se abre directamente en Excel con la
      // configuración regional de Costa Rica, sin alterar montos ni fechas.
      const filas = [];
      const fila = (valores) => filas.push(valores.map((valor) =>
        `"${String(valor ?? "").replaceAll('"', '""')}"`).join(";"));
      const separador = () => filas.push("");
      const tabla = (titulo, columnas, registros) => {
        fila([titulo]);
        fila(columnas);
        registros.forEach((registro) => fila(columnas.map((columna) => registro[columna])));
        separador();
      };
      const valorKpi = (valor, columna, unidad = "") => format(valor, columna, unidad);

      fila(["Reporte financiero detallado"]);
      fila(["Cliente", data.cliente?.nombre || ""]);
      fila(["Período", data.periodo?.etiqueta || inicio]);
      fila(["Generado", new Date().toLocaleString("es-CR")]);
      fila(["Saldos calculados al", saldos.fecha || ""]);
      separador();

      if (summaryRow) {
        tabla("Resumen mensual", ["Concepto", "Valor"], Object.keys(summaryRow).map((clave) => ({
          Concepto: clean(clave), Valor: valorKpi(summaryRow[clave], clave, summary?.unidad),
        })));
      }

      tabla("Saldos por cuenta", ["Cuenta", "Tipo", "Moneda", "Saldo CRC", "Saldo USD", "Corte inicial"],
        (saldos.cuentas || []).map((cuenta) => ({
          Cuenta: cuenta.nombre, Tipo: cuenta.tipo, Moneda: cuenta.moneda,
          "Saldo CRC": format(cuenta.saldo_crc, "saldo", "CRC"),
          "Saldo USD": cuenta.tipo === "credito" ? format(cuenta.saldo_usd, "saldo", "USD") : "",
          "Corte inicial": cuenta.fecha_corte,
        })));

      const movimientos = Array.isArray(data.movimientos) ? data.movimientos : [];
      tabla("Gastos y movimientos del mes", ["Fecha", "Categoría", "Concepto", "Descripción", "Método de pago", "Monto", "Moneda", "Monto original", "Moneda original", "Estado"],
        movimientos.map((movimiento) => ({
          Fecha: String(movimiento.fecha || "").slice(0, 10),
          "Categoría": movimiento.categoria || "Sin clasificar",
          Concepto: movimiento.concepto || "Gastos sin identificar",
          "Descripción": movimiento.descripcion || "Movimiento",
          "Método de pago": movimiento.medio_pago || "Sin método de pago",
          Monto: valorKpi(movimiento.monto, "monto", movimiento.moneda),
          Moneda: movimiento.moneda || "CRC",
          "Monto original": movimiento.monto_original ?? "",
          "Moneda original": movimiento.moneda_original || "",
          Estado: movimiento.pendiente_sincronizacion ? "Pendiente de sincronización" : "Verificado",
        })));

      (data.kpis || []).filter((kpi) => kpi !== summary).forEach((kpi) => {
        const columnas = kpi.columnas || [];
        tabla(kpi.nombre || clean(kpi.kpi || "Detalle presupuestario"), columnas,
          (kpi.filas || []).map((valores) => Object.fromEntries(columnas.map((columna, indice) => [
            columna, valorKpi(valores[indice], columna, kpi.unidad),
          ]))));
      });

      const movimientosCuenta = (saldos.cuentas || []).flatMap((cuenta) =>
        (cuenta.movimientos || []).map((movimiento) => ({
          Cuenta: cuenta.nombre, Fecha: movimiento.fecha, Descripción: movimiento.descripcion,
          Monto: format(movimiento.monto, "monto", movimiento.moneda),
          Moneda: movimiento.moneda, Origen: movimiento.origen,
        })));
      tabla("Movimientos incluidos en los saldos", ["Cuenta", "Fecha", "Descripción", "Monto", "Moneda", "Origen"], movimientosCuenta);

      if (saldos.advertencias?.length || saldos.sin_vincular?.length) {
        tabla("Notas y movimientos sin vincular", ["Tipo", "Detalle"], [
          ...(saldos.advertencias || []).map((detalle) => ({ Tipo: "Advertencia", Detalle: detalle })),
          ...(saldos.sin_vincular || []).map((movimiento) => ({
            Tipo: "Sin cuenta vinculada", Detalle: `${movimiento.fecha} · ${movimiento.descripcion} · ${movimiento.medio_pago}`,
          })),
        ]);
      }

      const periodoArchivo = String(data.periodo?.inicio || "mes").slice(0, 7);
      const csv = new Blob([`\ufeff${filas.join("\r\n")}`], { type: "text/csv;charset=utf-8" });
      const pdfRespuesta = await fetch(
        `${API_BASE}/reporte.pdf?inicio=${encodeURIComponent(inicio)}`,
        { headers: { Accept: "application/pdf" } },
      );
      if (!pdfRespuesta.ok) throw new Error("No pude preparar el reporte PDF.");
      const pdf = await pdfRespuesta.blob();
      const descargar = (archivo, nombre) => {
        const enlace = document.createElement("a");
        enlace.href = URL.createObjectURL(archivo); enlace.download = nombre; enlace.click();
        window.setTimeout(() => URL.revokeObjectURL(enlace.href), 1000);
      };
      descargar(csv, `reporte-financiero-${periodoArchivo}.csv`);
      descargar(pdf, `reporte-financiero-${periodoArchivo}.pdf`);
      mostrarAviso("CSV y PDF descargados.");
    } catch (razon) {
      mostrarAviso(razon.message || "No pude preparar el reporte.");
    } finally {
      boton.disabled = false;
      boton.textContent = textoOriginal;
    }
  };
  byId("descargar-reporte")?.addEventListener("click", descargarReporte);
  const summaryKeys = summaryRow ? Object.keys(summaryRow).filter((k) => !/pct|gasto.?neto/i.test(k)).slice(0, 4) : [];
  const cards = byId("resumen");
  summaryKeys.forEach((key) => {
    const card = document.createElement("article");
    card.className = "card";
    const label = document.createElement("div"); label.className = "label"; label.textContent = clean(key);
    const value = document.createElement("div"); value.className = "valor"; value.textContent = format(summaryRow[key], key, summary.unidad);
    card.append(label, value); cards.append(card);
  });

  if (summaryRow) {
    const budgetKey = Object.keys(summaryRow).find((k) => /^presupuesto$/i.test(k));
    const spentKey = Object.keys(summaryRow).find((k) => /^gastado$/i.test(k));
    const availableKey = Object.keys(summaryRow).find((k) => /^disponible$/i.test(k));
    const pctKey = Object.keys(summaryRow).find((k) => /pct|porcentaje/i.test(k));
    if (budgetKey && spentKey) {
      const pct = Math.max(0, Math.min(100, number(summaryRow[pctKey]) || number(summaryRow[spentKey]) / number(summaryRow[budgetKey]) * 100));
      const panel = byId("presupuesto"); panel.hidden = false;
      const donut = document.createElement("div"); donut.className = "donut";
      const donutPct = Math.min(100, Math.max(0, pct));
      const donutColor = pct > 100 ? "var(--red)" : "var(--green)";
      donut.style.background = `conic-gradient(${donutColor} 0 ${donutPct}%, var(--lime) ${donutPct}% 100%)`;
      const strong = document.createElement("strong"); strong.textContent = format(pct, "pct"); donut.append(strong);
      const legend = document.createElement("div"); legend.className = "leyenda";
      [["Gastado", spentKey], ["Disponible", availableKey], ["Presupuesto", budgetKey]].filter(([, k]) => k).forEach(([label, key]) => {
        const line = document.createElement("div");
        const l = document.createElement("span"); l.textContent = label;
        const v = document.createElement("strong"); v.textContent = format(summaryRow[key], key, summary.unidad);
        line.append(l, v); legend.append(line);
      });
      panel.append(donut, legend);
    }
  }

  const target = byId("indicadores");
  const renderVistaCategoria = () => {
    const hierarchyKpis = renderHierarchy(target);
    // La jerarquía es la vista financiera principal: reúne categorías,
    // conceptos y movimientos en el mismo árbol. Cuando está disponible no
    // repetimos los KPI auxiliares debajo. Para clientes que todavía no poseen
    // relaciones presupuesto-movimientos se conserva el fallback genérico.
    const visible = hierarchyKpis.size ? [] : data.kpis
      .filter((k) => k !== summary)
      .filter((k) => !hierarchyKpis.has(k))
      .filter((k) => !/^(gasto_total|gasto_neto)$/i.test(String(k.kpi || "")))
      .sort((a, b) => presentacion(a).orden - presentacion(b).orden);
    visible.forEach((kpi) => {
    const panel = document.createElement("article"); panel.className = "panel";
    const title = document.createElement("h2"); title.textContent = presentacion(kpi).titulo;
    panel.append(title);
    if (kpi.descripcion) { const p = document.createElement("p"); p.className = "descripcion"; p.textContent = kpi.descripcion; panel.append(p); }
    if (!kpi.filas.length) { const p = document.createElement("p"); p.className = "vacio"; p.textContent = "Sin registros para este período."; panel.append(p); target.append(panel); return; }

    // Para resultados por categoría/comercio, mostramos primero una lectura
    // visual y dejamos la tabla completa como detalle opcional.
    const rows = kpi.filas.map((row) => rowObject(kpi, row));
    const dimensionKey = dimensionFor(kpi, rows[0]);
    const budgetKey = keyMatch(rows[0], /^presupuesto$/i);
    const spentKey = keyMatch(rows[0], /^gastado$|gasto|monto|total/i);
    const availableKey = keyMatch(rows[0], /^disponible$|saldo/i);
    const pctKey = keyMatch(rows[0], /pct|porcentaje/i);
    if (dimensionKey && (budgetKey || spentKey || availableKey) && rows.length > 1) {
      const chart = document.createElement("div"); chart.className = "bar-chart";
      const numericValues = rows.flatMap((row) => [budgetKey, spentKey, availableKey].map((key) => Math.abs(number(row[key]))).filter(Number.isFinite));
      const globalMax = Math.max(...numericValues, 1);
      rows.forEach((row) => {
        const item = document.createElement("div"); item.className = "bar-item";
        const heading = document.createElement("div"); heading.className = "bar-heading";
        const name = document.createElement("strong"); name.textContent = row[dimensionKey] ?? "Sin nombre";
        const pct = pctKey ? number(row[pctKey]) : (budgetKey && spentKey ? number(row[spentKey]) / number(row[budgetKey]) * 100 : null);
        const alert = (Number.isFinite(pct) && pct > 100) || (availableKey && number(row[availableKey]) < 0);
        const flag = document.createElement("span"); flag.className = alert ? "flag flag-danger" : "flag flag-ok"; flag.textContent = alert ? "Excedido" : "En rango";
        heading.append(name, flag); item.append(heading);
        // Cuando hay presupuesto y gasto, cada fila usa su propio máximo.
        // Así se ve claramente si el gasto casi llena su presupuesto. Para
        // gráficos de gasto sin presupuesto conservamos la escala global.
        const rowMax = budgetKey && spentKey
          ? Math.max(Math.abs(number(row[budgetKey])) || 0, Math.abs(number(row[spentKey])) || 0, 1)
          : globalMax;
        [[budgetKey, "Presupuesto", "bar-budget"], [spentKey, "Gastado", "bar-spent"]].filter(([key]) => key).forEach(([key, label, cls]) => {
          const line = document.createElement("div"); line.className = "bar-line";
          const labelEl = document.createElement("span"); labelEl.textContent = label;
          const track = document.createElement("span"); track.className = "bar-track";
          const fill = document.createElement("span"); fill.className = `bar-fill ${cls}`; fill.style.width = `${Math.min(100, Math.abs(number(row[key])) / rowMax * 100)}%`; track.append(fill);
          const value = document.createElement("strong"); value.textContent = format(row[key], key, kpi.unidad);
          if (key === spentKey && gastoExcedePresupuesto(row[budgetKey], row[spentKey])) value.className = "valor-gastado-excedido";
          line.append(labelEl, track, value); item.append(line);
        });
        chart.append(item);
      });
      panel.append(chart);
    }
    const wrap = document.createElement("div"); wrap.className = "tabla-wrap";
    const table = document.createElement("table");
    const head = document.createElement("thead"); const hr = document.createElement("tr");
    kpi.columnas.forEach((c) => { const th = document.createElement("th"); th.textContent = clean(c); hr.append(th); });
    head.append(hr); table.append(head);
    const body = document.createElement("tbody");
    kpi.filas.forEach((row) => { const tr = document.createElement("tr"); row.forEach((v, i) => { const td = document.createElement("td"); td.dataset.label = clean(kpi.columnas[i]); td.textContent = format(v, kpi.columnas[i], kpi.unidad); tr.append(td); }); body.append(tr); });
    table.append(body); wrap.append(table);
    const details = document.createElement("details"); details.className = "detalle";
    const summaryDetails = document.createElement("summary"); summaryDetails.textContent = "Ver datos detallados";
    details.append(summaryDetails, wrap); panel.append(details); target.append(panel);
    });
    if (!data.kpis.length) target.innerHTML = '<article class="panel vacio">No hay KPIs habilitados para mostrar.</article>';
  };
  const selectorVista = byId("agrupar-gastos");
  const renderVista = () => {
    target.replaceChildren();
    if (selectorVista?.value === "medio_pago") {
      if (!renderPaymentHierarchy(target)) {
        target.innerHTML = '<article class="panel vacio">No hay movimientos para agrupar por método de pago en este período.</article>';
      }
      return;
    }
    renderVistaCategoria();
  };
  if (selectorVista) selectorVista.addEventListener("change", renderVista);
  renderVista();
  iniciarChat();
})();
