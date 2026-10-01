(() => {
  "use strict";
  const base = "/api/dashboard/cuentas";
  const mesSolicitado = new URLSearchParams(window.location.search).get("mes") || "";
  const urlSaldos = mesSolicitado
    ? `${base}?inicio=${encodeURIComponent(`${mesSolicitado.slice(0, 7)}-01`)}`
    : base;
  const contenido = document.getElementById("cuentas-contenido");
  const acciones = document.getElementById("cuentas-acciones");
  const recurrencias = document.getElementById("cuentas-recurrencias");
  const error = document.getElementById("cuentas-error");
  const fecha = document.getElementById("cuentas-fecha");
  if (!contenido || !acciones || !recurrencias || !error || !fecha) return;

  let estado = null;
  let confirmado = null;
  const pendientes = new Map();
  let ultimaCarga = 0;
  const dinero = (valor, moneda = "CRC") => new Intl.NumberFormat("es-CR", {
    style: "currency", currency: moneda, minimumFractionDigits: 2, maximumFractionDigits: 2,
  }).format(Number(valor) || 0);
  const fechaLocal = () => {
    const hoy = new Date();
    return [hoy.getFullYear(), String(hoy.getMonth() + 1).padStart(2, "0"),
      String(hoy.getDate()).padStart(2, "0")].join("-");
  };
  const mostrarError = (mensaje = "") => {
    error.textContent = mensaje;
    error.hidden = !mensaje;
  };
  const aviso = (mensaje) => {
    const elemento = document.createElement("div");
    elemento.className = "aviso-dashboard";
    elemento.setAttribute("role", "status");
    elemento.textContent = mensaje;
    document.body.append(elemento);
    window.setTimeout(() => elemento.remove(), 4500);
  };
  const peticion = async (url, method = "GET", body) => {
    const response = await fetch(url, {
      method, headers: { "Content-Type": "application/json", Accept: "application/json" },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "No pude guardar el cambio.");
    return result;
  };
  const boton = (texto, accion, clase = "") => {
    const el = document.createElement("button");
    el.type = "button"; el.textContent = texto;
    if (clase) el.className = clase;
    el.addEventListener("click", accion);
    return el;
  };
  const cuentasDinero = () => (estado?.cuentas || []).filter((c) => c.tipo !== "credito");
  const tarjetasCredito = () => (estado?.cuentas || []).filter((c) => c.tipo === "credito");
  const seleccionar = (opciones, valor = "") => {
    const select = document.createElement("select");
    opciones.forEach((c) => {
      const option = document.createElement("option");
      option.value = c.cuenta_id; option.textContent = c.nombre;
      select.append(option);
    });
    if (valor) select.value = valor;
    return select;
  };
  const control = (form, titulo, input) => {
    const label = document.createElement("label"); label.textContent = titulo;
    label.append(input); form.append(label); return input;
  };
  const campo = (tipo, nombre, valor = "") => {
    const input = document.createElement("input");
    input.type = tipo; input.name = nombre; input.value = valor; input.required = true;
    if (tipo === "number") {
      input.min = "0.01"; input.step = "0.01"; input.inputMode = "decimal";
    }
    return input;
  };
  const abrirDialogo = (titulo) => {
    const dialog = document.createElement("dialog"); dialog.className = "editor-movimiento cuenta-dialogo";
    const form = document.createElement("form");
    const heading = document.createElement("h2"); heading.textContent = titulo; form.append(heading);
    const errorForm = document.createElement("p"); errorForm.className = "editor-error"; errorForm.hidden = true;
    const accionesForm = document.createElement("div"); accionesForm.className = "editor-acciones";
    const cancelar = boton("Cancelar", () => dialog.close());
    const guardar = document.createElement("button"); guardar.type = "submit"; guardar.textContent = "Guardar";
    accionesForm.append(cancelar, guardar);
    dialog.append(form); document.body.append(dialog);
    dialog.addEventListener("close", () => dialog.remove());
    return { dialog, form, errorForm, accionesForm, guardar };
  };
  const efectoLocal = (datos) => {
    const origen = estado.cuentas.find((c) => c.cuenta_id === datos.cuenta_origen);
    const destino = estado.cuentas.find((c) => c.cuenta_id === datos.cuenta_destino);
    if (origen) {
      const llave = `saldo_${datos.moneda_origen.toLowerCase()}`;
      origen[llave] = String(Number(origen[llave]) - Number(datos.monto_origen));
    }
    if (destino) {
      const llave = `saldo_${datos.moneda_destino.toLowerCase()}`;
      const signo = destino.tipo === "credito" ? -1 : 1;
      destino[llave] = String(Number(destino[llave]) + signo * Number(datos.monto_destino));
    }
  };
  const recomponer = () => {
    if (!confirmado) return;
    estado = structuredClone(confirmado);
    pendientes.forEach(({ datos }) => efectoLocal(datos));
    render();
  };
  const guardarOperacion = async (datos) => {
    pendientes.set(datos.operacion_id, { datos, guardado: false });
    recomponer();
    aviso("Cambio aplicado. Guardando…");
    try {
      await peticion(`${base}/operaciones`, "POST", datos);
      pendientes.get(datos.operacion_id).guardado = true;
      if (await cargar()) aviso("Operación guardada.");
      else aviso("Operación guardada; el saldo se actualizará al reconectar.");
    } catch (reason) {
      pendientes.delete(datos.operacion_id); recomponer();
      mostrarError(reason.message || "No pude guardar la operación.");
    }
  };

  const abrirOperacion = (tipo) => {
    const titulos = { ingreso: "Agregar ingreso", transferencia: "Transferir entre cuentas",
      pago_tarjeta: "Pagar tarjeta" };
    const { dialog, form, errorForm, accionesForm } = abrirDialogo(titulos[tipo]);
    const descripcion = control(form, "Descripción", campo("text", "descripcion",
      tipo === "pago_tarjeta" ? "Pago de tarjeta" : ""));
    descripcion.maxLength = 180;
    const fechaPago = control(form, "Fecha", campo("date", "fecha", fechaLocal()));
    fechaPago.max = fechaLocal();
    const origen = tipo === "ingreso" ? null : control(form, "Desde la cuenta",
      seleccionar(cuentasDinero()));
    const destino = control(form, tipo === "pago_tarjeta" ? "Tarjeta a pagar" : "Cuenta de destino",
      seleccionar(tipo === "pago_tarjeta" ? tarjetasCredito() : cuentasDinero()));
    const cuentaPredeterminada = tarjetasCredito().find((c) => c.cuenta_pago_default)?.cuenta_pago_default;
    const bancoPrincipal = cuentasDinero().find((c) => c.cuenta_id === cuentaPredeterminada);
    if (tipo === "ingreso" && bancoPrincipal) destino.value = bancoPrincipal.cuenta_id;
    if (tipo === "transferencia" && bancoPrincipal) {
      origen.value = bancoPrincipal.cuenta_id;
      destino.value = cuentasDinero().find((c) => c.cuenta_id !== origen.value)?.cuenta_id || "";
    }
    const moneda = tipo === "pago_tarjeta" ? document.createElement("select") : null;
    if (moneda) {
      ["CRC", "USD"].forEach((valor) => {
        const option = document.createElement("option"); option.value = valor;
        option.textContent = valor; moneda.append(option);
      });
      control(form, "Moneda de la deuda que vas a pagar", moneda);
      const seleccionarDefault = () => {
        const tarjeta = tarjetasCredito().find((c) => c.cuenta_id === destino.value);
        if (tarjeta?.cuenta_pago_default) origen.value = tarjeta.cuenta_pago_default;
      };
      destino.addEventListener("change", seleccionarDefault); seleccionarDefault();
    }
    const monto = control(form, tipo === "pago_tarjeta" ? "Monto aplicado a la tarjeta" : "Monto",
      campo("number", "monto"));
    const debito = tipo === "pago_tarjeta" ? control(form, "Monto debitado de la cuenta (CRC)",
      campo("number", "debito")) : null;
    if (debito) {
      monto.addEventListener("input", () => { if (moneda.value === "CRC") debito.value = monto.value; });
    }
    if (tipo === "pago_tarjeta") {
      const nota = document.createElement("p"); nota.className = "editor-nota";
      nota.textContent = "Si pagas una deuda en USD desde una cuenta en colones, indica también los colones debitados por el banco.";
      form.append(nota);
    }
    form.append(errorForm, accionesForm);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      if (tipo === "transferencia" && origen.value === destino.value) {
        errorForm.textContent = "Seleccione dos cuentas distintas."; errorForm.hidden = false; return;
      }
      const datos = {
        operacion_id: crypto.randomUUID(), tipo, fecha: fechaPago.value,
        descripcion: descripcion.value.trim(), cuenta_origen: origen?.value || "",
        cuenta_destino: destino.value, monto_origen: tipo === "ingreso" ? "0" :
          tipo === "pago_tarjeta" ? debito.value : monto.value,
        moneda_origen: "CRC", monto_destino: monto.value,
        moneda_destino: tipo === "pago_tarjeta" ? moneda.value : "CRC",
      };
      dialog.close(); void guardarOperacion(datos);
    });
    dialog.showModal();
  };

  const abrirRecurrencia = () => {
    const { dialog, form, errorForm, accionesForm } = abrirDialogo("Programar ingreso recurrente");
    const nombre = control(form, "Ingreso", campo("text", "nombre")); nombre.maxLength = 120;
    const cuenta = control(form, "Cuenta donde ingresa", seleccionar(cuentasDinero()));
    const monto = control(form, "Monto (CRC)", campo("number", "monto"));
    const frecuencia = document.createElement("select");
    [["mensual", "Mensual"], ["semanal", "Semanal"]].forEach(([valor, etiqueta]) => {
      const option = document.createElement("option"); option.value = valor;
      option.textContent = etiqueta; frecuencia.append(option);
    });
    control(form, "Frecuencia", frecuencia);
    const diasMes = control(form, "Días del mes (ej.: 14, 28)", campo("text", "dias_mes", "14, 28"));
    const diaSemana = document.createElement("select");
    ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
      .forEach((dia, numero) => {
        const option = document.createElement("option"); option.value = String(numero);
        option.textContent = dia; diaSemana.append(option);
      });
    const semanaLabel = control(form, "Día de la semana", diaSemana);
    const corte = cuentasDinero().map((c) => c.fecha_corte).sort().at(-1);
    const minimo = corte ? new Date(`${corte}T12:00:00`) : null;
    if (minimo) minimo.setDate(minimo.getDate() + 1);
    const primerDia = minimo ? [minimo.getFullYear(), String(minimo.getMonth() + 1).padStart(2, "0"),
      String(minimo.getDate()).padStart(2, "0")].join("-") : fechaLocal();
    const inicio = control(form, "Desde", campo("date", "desde",
      primerDia > fechaLocal() ? primerDia : fechaLocal()));
    inicio.min = primerDia;
    const fin = control(form, "Hasta (opcional)", campo("date", "hasta")); fin.required = false;
    const cambiarFrecuencia = () => {
      diasMes.parentElement.hidden = frecuencia.value !== "mensual";
      semanaLabel.parentElement.hidden = frecuencia.value !== "semanal";
      diasMes.required = frecuencia.value === "mensual";
    };
    frecuencia.addEventListener("change", cambiarFrecuencia); cambiarFrecuencia();
    const nota = document.createElement("p"); nota.className = "editor-nota";
    nota.textContent = "Cada fecha vencida genera un ingreso una sola vez. Los ingresos anteriores al saldo inicial no se repetirán.";
    form.append(nota, errorForm, accionesForm);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const dias = frecuencia.value === "mensual" ? diasMes.value.split(",").map((d) => Number(d.trim())) : [];
      if (frecuencia.value === "mensual" && (!dias.length || dias.some((d) => !Number.isInteger(d) || d < 1 || d > 31))) {
        errorForm.textContent = "Escribe días entre 1 y 31, separados por coma."; errorForm.hidden = false; return;
      }
      const datos = {
        regla_id: crypto.randomUUID(), nombre: nombre.value.trim(), cuenta_id: cuenta.value,
        monto: monto.value, moneda: "CRC", frecuencia: frecuencia.value,
        dias_mes: dias, dia_semana: frecuencia.value === "semanal" ? Number(diaSemana.value) : null,
        desde: inicio.value, hasta: fin.value || null,
      };
      dialog.close(); aviso("Guardando ingreso recurrente…");
      try {
        await peticion(`${base}/ingresos-recurrentes`, "POST", datos);
        await cargar(); aviso("Ingreso recurrente guardado.");
      } catch (reason) { mostrarError(reason.message); }
    });
    dialog.showModal();
  };

  const render = () => {
    if (!estado) return;
    contenido.replaceChildren(); recurrencias.replaceChildren(); acciones.replaceChildren();
    if (!estado.configurado) {
      contenido.textContent = "Todavía no hay cuentas configuradas para este cliente.";
      acciones.hidden = true; return;
    }
    fecha.textContent = `Al ${estado.fecha}`;
    acciones.hidden = false;
    acciones.append(
      boton("＋ Ingreso", () => abrirOperacion("ingreso")),
      boton("↻ Ingreso recurrente", abrirRecurrencia),
      boton("⇄ Transferir", () => abrirOperacion("transferencia")),
      boton("Pagar tarjeta", () => abrirOperacion("pago_tarjeta")),
    );
    const grid = document.createElement("div"); grid.className = "cuentas-grid";
    estado.cuentas.forEach((cuenta) => {
      const card = document.createElement("article"); card.className = "cuenta-card";
      const cabecera = document.createElement("div"); cabecera.className = "cuenta-card-cabecera";
      const nombre = document.createElement("h3"); nombre.textContent = cuenta.nombre;
      const tipo = document.createElement("span"); tipo.textContent = cuenta.tipo === "credito" ? "Crédito" :
        cuenta.tipo === "ahorro" ? "Ahorro" : "Banco · débito";
      cabecera.append(nombre, tipo); card.append(cabecera);
      const saldos = document.createElement("div"); saldos.className = "cuenta-saldos";
      const linea = (etiqueta, monto, moneda) => {
        const row = document.createElement("div");
        const label = document.createElement("span"); label.textContent = etiqueta;
        const value = document.createElement("strong"); value.textContent = dinero(monto, moneda);
        if (Number(monto) < 0) value.className = "cuenta-negativa";
        row.append(label, value); saldos.append(row);
      };
      linea(cuenta.tipo === "credito" ? "Deuda CRC" : "Saldo calculado", cuenta.saldo_crc, "CRC");
      if (cuenta.tipo === "credito") linea("Deuda USD", cuenta.saldo_usd, "USD");
      card.append(saldos);
      const corte = document.createElement("small"); corte.className = "cuenta-corte";
      const horaCorte = cuenta.corte_en ? ` · ${cuenta.corte_en.slice(11, 16)} hora CR` : "";
      corte.textContent = `Corte inicial: ${cuenta.fecha_corte}${horaCorte}`; card.append(corte);
      const detalle = document.createElement("details"); detalle.className = "cuenta-detalle";
      const summary = document.createElement("summary"); summary.textContent = "Ver movimientos"; detalle.append(summary);
      const lista = document.createElement("ul"); lista.className = "cuenta-movimientos";
      (cuenta.movimientos || []).forEach((movimiento) => {
        const item = document.createElement("li");
        const datos = document.createElement("span");
        const descripcion = document.createElement("strong"); descripcion.textContent = movimiento.descripcion;
        const fechaMov = document.createElement("small"); fechaMov.textContent = movimiento.fecha;
        datos.append(descripcion, fechaMov);
        const importe = document.createElement("span"); importe.textContent = dinero(movimiento.monto, movimiento.moneda);
        item.append(datos, importe);
        if (movimiento.origen === "operación") {
          const anular = boton("Anular", async () => {
            if (!window.confirm("¿Anular esta operación? El movimiento quedará en el historial de auditoría.")) return;
            try { await peticion(`${base}/operaciones/${encodeURIComponent(movimiento.id)}`, "DELETE");
              await cargar(); aviso("Operación anulada.");
            } catch (reason) { mostrarError(reason.message); }
          }, "cuenta-anular");
          item.append(anular);
        }
        lista.append(item);
      });
      if (!lista.childElementCount) {
        const item = document.createElement("li"); item.textContent = "Sin movimientos posteriores al corte."; lista.append(item);
      }
      detalle.append(lista); card.append(detalle); grid.append(card);
    });
    contenido.append(grid);
    (estado.advertencias || []).forEach((mensaje) => {
      const nota = document.createElement("p"); nota.className = "cuentas-sin-vincular";
      nota.textContent = mensaje; contenido.append(nota);
    });
    if (estado.sin_vincular?.length) {
      const nota = document.createElement("p"); nota.className = "cuentas-sin-vincular";
      nota.textContent = `${estado.sin_vincular.length} movimientos recientes tienen un método de pago sin cuenta vinculada y no afectan los saldos.`;
      contenido.append(nota);
    }
    const reglas = (estado.reglas || []).filter((r) => r.activo);
    if (reglas.length) {
      const titulo = document.createElement("h3"); titulo.textContent = "Ingresos recurrentes"; recurrencias.append(titulo);
      const lista = document.createElement("ul"); lista.className = "cuenta-reglas";
      reglas.forEach((regla) => {
        const item = document.createElement("li");
        const texto = document.createElement("span");
        const calendario = regla.frecuencia === "mensual" ?
          `días ${(regla.dias_mes || []).join(" y ")} de cada mes` :
          ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"][regla.dia_semana];
        const cuenta = estado.cuentas.find((c) => c.cuenta_id === regla.cuenta_id);
        texto.textContent = `${regla.nombre} · ${dinero(regla.monto)} · ${calendario} → ${cuenta?.nombre || regla.cuenta_id}`;
        const desactivar = boton("Desactivar", async () => {
          try { await peticion(`${base}/ingresos-recurrentes/${encodeURIComponent(regla.regla_id)}`, "DELETE");
            await cargar(); aviso("Recurrencia desactivada; se conservan los ingresos ya registrados.");
          } catch (reason) { mostrarError(reason.message); }
        });
        item.append(texto, desactivar); lista.append(item);
      });
      recurrencias.append(lista);
    }
  };
  const cargar = async () => {
    const solicitud = ++ultimaCarga;
    try {
      const resultado = await peticion(urlSaldos);
      if (solicitud !== ultimaCarga) return true;
      confirmado = resultado;
      pendientes.forEach((pendiente, id) => {
        if (pendiente.guardado) pendientes.delete(id);
      });
      mostrarError(""); recomponer();
      return true;
    } catch (reason) {
      if (!estado) contenido.textContent = "No pude cargar los saldos.";
      mostrarError(reason.message || "No pude cargar los saldos.");
      return false;
    }
  };
  window.fachaviCuentasDisponibles = () => (estado?.cuentas || []).filter((c) => c.tipo !== "credito");
  window.fachaviCuentaPagoPredeterminada = () =>
    (estado?.cuentas || []).find((c) => c.tipo === "credito" && c.cuenta_pago_default)?.cuenta_pago_default || "";
  window.fachaviCargarCuentas = cargar;
  window.addEventListener("fachavi:movimiento-editado", () => { void cargar(); });
  void cargar();
})();
