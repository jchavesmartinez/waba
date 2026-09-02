(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("dashboard-data").textContent);
  const byId = (id) => document.getElementById(id);
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
  const findKpi = (name) => data.kpis.find((k) => k.kpi === name);
  const rowObject = (kpi, row) => Object.fromEntries(kpi.columnas.map((c, i) => [c, row[i]]));

  byId("cliente").textContent = data.cliente.nombre;
  byId("periodo").textContent = data.periodo.etiqueta;
  byId("actualizado").textContent = "Actualizado: " + new Date(data.actualizado_en).toLocaleString("es-CR");

  const summary = findKpi("presupuesto_disponible") || data.kpis.find((k) => k.filas.length === 1 && k.columnas.some((c) => /presupuesto/i.test(c)));
  const summaryRow = summary?.filas?.[0] ? rowObject(summary, summary.filas[0]) : null;
  const summaryKeys = summaryRow ? Object.keys(summaryRow).filter((k) => !/pct/i.test(k)).slice(0, 4) : [];
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
      donut.style.background = `conic-gradient(var(--green) 0 ${pct}%, var(--lime) ${pct}% 100%)`;
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
  const visible = data.kpis.filter((k) => k !== summary);
  visible.forEach((kpi) => {
    const panel = document.createElement("article"); panel.className = "panel";
    const title = document.createElement("h2"); title.textContent = kpi.nombre;
    panel.append(title);
    if (kpi.descripcion) { const p = document.createElement("p"); p.className = "descripcion"; p.textContent = kpi.descripcion; panel.append(p); }
    if (!kpi.filas.length) { const p = document.createElement("p"); p.className = "vacio"; p.textContent = "Sin registros para este período."; panel.append(p); target.append(panel); return; }
    const wrap = document.createElement("div"); wrap.className = "tabla-wrap";
    const table = document.createElement("table");
    const head = document.createElement("thead"); const hr = document.createElement("tr");
    kpi.columnas.forEach((c) => { const th = document.createElement("th"); th.textContent = clean(c); hr.append(th); });
    head.append(hr); table.append(head);
    const body = document.createElement("tbody");
    kpi.filas.forEach((row) => { const tr = document.createElement("tr"); row.forEach((v, i) => { const td = document.createElement("td"); td.dataset.label = clean(kpi.columnas[i]); td.textContent = format(v, kpi.columnas[i], kpi.unidad); tr.append(td); }); body.append(tr); });
    table.append(body); wrap.append(table); panel.append(wrap); target.append(panel);
  });
  if (!data.kpis.length) target.innerHTML = '<article class="panel vacio">No hay KPIs habilitados para mostrar.</article>';
})();
