(() => {
  "use strict";
  const abrir = document.getElementById("administrar-presupuesto");
  if (!abrir) return;
  const base = "/api/dashboard/presupuesto";
  const moneda = (n) => new Intl.NumberFormat("es-CR", { style: "currency", currency: "CRC" }).format(n);
  const hoyCR = () => {
    const p = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
      timeZone: "America/Costa_Rica", year: "numeric", month: "2-digit", day: "2-digit",
    }).formatToParts(new Date()).map((parte) => [parte.type, parte.value]));
    return `${p.year}-${p.month}`;
  };
  const nodo = (tag, texto, clase) => {
    const n = document.createElement(tag);
    if (texto !== undefined) n.textContent = texto;
    if (clase) n.className = clase;
    return n;
  };
  const boton = (texto, accion, clase = "") => {
    const b = nodo("button", texto, clase); b.type = "button";
    b.addEventListener("click", accion); return b;
  };
  const pedir = async (url, datos) => {
    const r = await fetch(url, { method: datos ? "POST" : "GET", headers: {
      Accept: "application/json", ...(datos ? { "Content-Type": "application/json" } : {}),
    }, ...(datos ? { body: JSON.stringify(datos) } : {}) });
    const json = await r.json();
    if (!r.ok || !json.ok) throw new Error(json.error || "No pude procesar el presupuesto.");
    return json;
  };
  const control = (padre, texto, input) => {
    const label = nodo("label", texto); label.append(input); padre.append(label); return input;
  };
  const campo = (tipo, valor = "") => {
    const input = document.createElement("input"); input.type = tipo; input.value = valor;
    if (tipo === "number") { input.min = "0"; input.max = "999999999999.99"; input.step = "0.01"; input.inputMode = "decimal"; }
    return input;
  };
  let dialogo = null;
  abrir.addEventListener("click", () => {
    if (dialogo) return;
    const dialog = nodo("dialog", undefined, "editor-movimiento presupuesto-dialogo"); dialogo = dialog;
    const titulo = nodo("h2", "Administrar presupuesto"); titulo.id = "presupuesto-editor-titulo";
    dialog.setAttribute("aria-labelledby", titulo.id);
    const cerrar = boton("Cerrar", () => {
      if (ocupado) return;
      if (modificado() && !window.confirm("Hay cambios sin guardar. ¿Desea descartarlos?")) return;
      dialog.close();
    });
    const cabecera = nodo("div", undefined, "presupuesto-editor-cabecera"); cabecera.append(titulo, cerrar);
    const ayuda = nodo("p", "Edita el plan, no los pagos ni los saldos. Los meses anteriores quedan intactos. Los cambios de monto conservan la fecha final de la partida; las nuevas partidas siguen vigentes hasta que se cambien.", "descripcion");
    const mes = campo("month", [hoyCR(), new URLSearchParams(location.search).get("mes")?.slice(0, 7) || ""].sort().at(-1));
    mes.min = hoyCR(); mes.max = `${Number(hoyCR().slice(0, 4)) + 5}-12`;
    const controles = nodo("div", undefined, "presupuesto-editor-controles"); control(controles, "Aplicar desde el mes", mes);
    const recargar = boton("Recargar presupuesto", () => cargar(true)); controles.append(recargar);
    const error = nodo("p", undefined, "editor-error"); error.hidden = true; error.setAttribute("role", "alert");
    const aviso = nodo("p", undefined, "presupuesto-editor-aviso"); aviso.setAttribute("role", "status");
    const totales = nodo("div", undefined, "presupuesto-editor-totales");
    const lista = nodo("div", undefined, "presupuesto-editor-lista");
    const alta = nodo("details", undefined, "presupuesto-editor-alta"); alta.append(nodo("summary", "＋ Nueva categoría o subpartida"));
    const altaForm = nodo("form", undefined, "presupuesto-alta-form");
    const tipo = document.createElement("select");
    for (const [valor, texto] of [["gasto", "Gasto"], ["ingreso", "Ingreso"]]) {
      const o = nodo("option", texto); o.value = valor; tipo.append(o);
    }
    control(altaForm, "Tipo de presupuesto", tipo);
    const categoria = control(altaForm, "Categoría (existente o nueva)", campo("text")); categoria.required = true; categoria.maxLength = 120;
    const opciones = nodo("datalist"); opciones.id = "presupuesto-categorias"; categoria.setAttribute("list", opciones.id); altaForm.append(opciones);
    const concepto = control(altaForm, "Subpartida / concepto", campo("text")); concepto.required = true; concepto.maxLength = 120;
    const monto = control(altaForm, "Monto mensual (CRC)", campo("number", "0")); monto.required = true;
    altaForm.append(nodo("p", "Una categoría se crea con su primera subpartida. También puedes dejar su monto en cero."));
    const agregar = nodo("button", "Añadir al borrador"); agregar.type = "submit"; altaForm.append(agregar); alta.append(altaForm);
    const revisar = boton("Revisar cambios", () => revisarCambios(), "presupuesto-primario"); revisar.disabled = true;
    const preview = nodo("section", undefined, "presupuesto-editor-preview"); preview.hidden = true;
    dialog.append(cabecera, ayuda, controles, error, aviso, totales, lista, alta, revisar, preview);
    document.body.append(dialog); dialog.showModal();
    let datos = null, nuevas = [], importes = new Map(), ocupado = false, solicitud = null;
    const mostrarError = (mensaje = "") => {
      error.textContent = mensaje; error.hidden = !mensaje;
      if (mensaje) error.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };
    const cambios = () => (datos?.lineas || []).filter((l) => Number(importes.get(l.linea_id)?.value) !== Number(l.monto_mensual));
    const modificado = () => cambios().length > 0 || nuevas.length > 0;
    const bloquear = (valor) => {
      ocupado = valor;
      dialog.querySelectorAll("input, select, button").forEach((n) => { n.disabled = valor; });
      revisar.disabled = valor || !datos || !modificado();
    };
    const actualizar = () => {
      preview.hidden = true; solicitud = null;
      const suma = { gasto: 0, ingreso: 0 }, anterior = { gasto: 0, ingreso: 0 };
      (datos?.lineas || []).forEach((l) => {
        anterior[l.tipo] += Number(l.monto_mensual);
        suma[l.tipo] += Number(importes.get(l.linea_id)?.value || 0);
      });
      nuevas.forEach((l) => { suma[l.tipo] += Number(l.monto_mensual); });
      totales.replaceChildren();
      for (const [nombre, nuevo, previo] of [["Ingresos", suma.ingreso, anterior.ingreso], ["Gastos", suma.gasto, anterior.gasto],
        ["Balance planeado", suma.ingreso - suma.gasto, anterior.ingreso - anterior.gasto]]) {
        const c = nodo("div"); c.append(nodo("span", nombre), nodo("strong", moneda(nuevo)), nodo("small", `Antes: ${moneda(previo)}`)); totales.append(c);
      }
      revisar.disabled = ocupado || !datos || !modificado();
    };
    const pintar = () => {
      lista.replaceChildren(); importes = new Map(); opciones.replaceChildren();
      [...new Set(datos.lineas.map((l) => l.categoria))].sort().forEach((c) => { const o = nodo("option"); o.value = c; opciones.append(o); });
      const grupos = new Map();
      for (const l of [...datos.lineas, ...nuevas]) {
        const clave = `${l.tipo}: ${l.categoria}`;
        if (!grupos.has(clave)) {
          const g = nodo("details", undefined, "presupuesto-editor-grupo"); g.open = true;
          g.append(nodo("summary", `${l.tipo === "ingreso" ? "Ingresos" : "Gastos"} · ${l.categoria}`)); grupos.set(clave, g); lista.append(g);
        }
        const fila = nodo("div", undefined, "presupuesto-editor-partida");
        fila.append(nodo("strong", l.concepto));
        if (l.linea_id) {
          const input = campo("number", l.monto_mensual); input.required = true;
          const detalle = nodo("small", `Quincenal actual: ${moneda(Number(l.monto_quincenal))} · Vigente hasta: ${l.vigencia_hasta?.slice(0, 10) || "sin fecha final"}`);
          importes.set(l.linea_id, input); control(fila, "Mensual (CRC)", input); fila.append(detalle);
          input.addEventListener("input", () => { l.borrador = input.value; actualizar(); });
          if (l.borrador !== undefined) input.value = l.borrador;
        } else {
          fila.append(nodo("span", `${moneda(Number(l.monto_mensual))} mensuales · Nueva subpartida`));
          fila.append(boton("Quitar del borrador", () => { nuevas = nuevas.filter((n) => n !== l); pintar(); }));
        }
        grupos.get(clave).append(fila);
      }
      if (!datos.lineas.length) lista.append(nodo("p", "No hay partidas vigentes en este mes. No se trasladarán automáticamente presupuestos vencidos."));
      actualizar();
    };
    const cargar = async (confirmar = false) => {
      if (ocupado) return;
      if (confirmar && modificado() && !window.confirm("Recargar descartará su borrador. ¿Continuar?")) return;
      bloquear(true); mostrarError(); aviso.textContent = "Leyendo presupuesto de Google Sheets…";
      datos = null; nuevas = []; importes = new Map(); lista.replaceChildren(); totales.replaceChildren(); preview.hidden = true;
      try {
        datos = await pedir(`${base}?inicio=${encodeURIComponent(mes.value + "-01")}`);
        mes.min = datos.mes_minimo.slice(0, 7); nuevas = []; pintar(); aviso.textContent = "Los cambios no se guardan hasta que los revise y confirme.";
      } catch (e) { mostrarError(e.message); aviso.textContent = "No se pudo cargar el editor. No se modificó el presupuesto."; }
      finally { bloquear(false); agregar.disabled = !datos; }
    };
    mes.addEventListener("change", () => {
      if (!mes.checkValidity()) { mostrarError("Seleccione el mes actual o uno futuro."); return; }
      if (modificado() && !window.confirm("Cambiar el mes descartará los cambios sin guardar. ¿Continuar?")) {
        mes.value = datos.inicio.slice(0, 7); return;
      }
      cargar();
    });
    altaForm.addEventListener("submit", (e) => {
      e.preventDefault(); if (ocupado || !datos || !altaForm.reportValidity()) return;
      const normal = (s) => s.trim().normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("es");
      const cat = categoria.value.trim(), con = concepto.value.trim();
      if ([...datos.lineas, ...nuevas].some((l) => l.tipo === tipo.value && normal(l.categoria) === normal(cat) && normal(l.concepto) === normal(con))) {
        mostrarError("Esa subpartida ya existe. Edite su monto en la lista."); return;
      }
      nuevas.push({ tipo: tipo.value, categoria: cat, concepto: con, monto_mensual: monto.value });
      categoria.value = ""; concepto.value = ""; monto.value = "0"; mostrarError(); pintar();
    });
    const revisarCambios = () => {
      if (ocupado || !modificado()) return;
      if ([...importes.values()].some((i) => !i.reportValidity())) return;
      mostrarError(); preview.replaceChildren(nodo("h3", "Confirmar cambios del presupuesto"));
      preview.append(nodo("p", `Aplicar desde ${mes.value}. El pasado, los movimientos registrados y los saldos bancarios no se modifican.`));
      const ul = nodo("ul");
      cambios().forEach((l) => ul.append(nodo("li", `${l.categoria} · ${l.concepto}: ${moneda(Number(l.monto_mensual))} → ${moneda(Number(importes.get(l.linea_id).value))}. Hasta ${l.vigencia_hasta?.slice(0, 10) || "sin fecha final"}.`)));
      nuevas.forEach((l) => ul.append(nodo("li", `Nueva ${l.tipo === "ingreso" ? "partida de ingresos" : "partida de gastos"}: ${l.categoria} · ${l.concepto}, ${moneda(Number(l.monto_mensual))} mensuales, sin fecha final.`)));
      preview.append(ul, nodo("p", "Para los montos modificados o nuevos, el quincenal será la mitad del mensual, redondeado a céntimos. Las demás partidas conservarán sus valores originales."));
      solicitud = { inicio: mes.value + "-01", revision: datos.revision, confirmado: true,
        operacion_id: crypto.randomUUID(), cambios: cambios().map((l) => ({ linea_id: l.linea_id, monto_mensual: importes.get(l.linea_id).value })),
        nuevas: nuevas.map((l) => ({ ...l })) };
      preview.append(boton("Volver a editar", () => { preview.hidden = true; }), boton("Confirmar y guardar", guardar, "presupuesto-primario"));
      preview.hidden = false; preview.scrollIntoView({ behavior: "smooth", block: "nearest" });
    };
    const guardar = async () => {
      if (ocupado || !solicitud) return;
      bloquear(true); mostrarError(); aviso.textContent = "Guardando presupuesto… No cierre esta ventana.";
      try {
        const resultado = await pedir(base, solicitud);
        aviso.textContent = resultado.mensaje || "Presupuesto guardado. Actualizando el dashboard…";
        preview.hidden = true;
        try { sessionStorage.setItem("presupuesto-sincronizando", solicitud.operacion_id); } catch (_) { /* almacenamiento opcional */ }
        nuevas = []; datos.lineas.forEach((l) => { l.monto_mensual = importes.get(l.linea_id).value; });
        esperar(solicitud.operacion_id, aviso);
        cerrar.disabled = false;
        cerrar.onclick = () => dialog.close();
      } catch (e) { mostrarError(e.message); aviso.textContent = "No se confirmó el resultado. Puede recargar para revisar o reintentar la misma confirmación."; bloquear(false); }
    };
    dialog.addEventListener("cancel", (e) => { e.preventDefault(); cerrar.click(); });
    dialog.addEventListener("close", () => { dialog.remove(); dialogo = null; });
    cargar();
  });
  const esperar = async (id, aviso, intento = 0) => {
    try {
      const resultado = await pedir(`${base}/${encodeURIComponent(id)}/estado`);
      if (resultado.estado === "listo") {
        try { sessionStorage.removeItem("presupuesto-sincronizando"); } catch (_) { /* almacenamiento opcional */ }
        location.reload(); return;
      }
      if (resultado.estado === "error") {
        if (aviso) aviso.textContent = "El presupuesto está guardado, pero su sincronización falló. Se reintentará en segundo plano; no vuelva a crear las partidas.";
        return;
      }
    } catch (_) { /* la escritura ya se confirmó; nunca se vuelve a enviar */ }
    if (intento < 40) window.setTimeout(() => esperar(id, aviso, intento + 1), 3000);
    else if (aviso) aviso.textContent = "El presupuesto está guardado. La actualización sigue pendiente; puede cerrar y recargar más tarde.";
  };
  let pendiente;
  try { pendiente = sessionStorage.getItem("presupuesto-sincronizando"); } catch (_) { /* almacenamiento opcional */ }
  if (pendiente) esperar(pendiente);
})();
