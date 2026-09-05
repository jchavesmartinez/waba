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
    budget: keyMatch(row, /^presupuesto$|monto_presupuestado|presupuesto_mensual/i),
    spent: keyMatch(row, /^gastado$|gasto_neto|gasto|monto|total/i),
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
        const value = document.createElement("strong"); value.textContent = format(row[key], key);
        line.append(labelEl, track, value); container.append(line);
      });
  };

  const renderHierarchy = (target) => {
    const byIdInsensitive = (ids) => data.kpis.find((k) => ids.includes(String(k.kpi || "").toLowerCase()));
    const conceptKpi = byIdInsensitive(["ejecucion_presupuesto_concepto", "gasto_por_concepto"]);
    const categoryKpi = byIdInsensitive(["gasto_por_categoria", "ejecucion_presupuesto_mes"]);
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
      categoryMeta.textContent = [
        categoryKeys.budget && `Presupuesto: ${format(bucket.totals[categoryKeys.budget], categoryKeys.budget)}`,
        categoryKeys.spent && `Gastado: ${format(bucket.totals[categoryKeys.spent], categoryKeys.spent)}`,
      ].filter(Boolean).join(" · ");
      categorySummary.append(categoryHeading, categoryMeta); categoryDetails.append(categorySummary);
      const conceptsWrap = document.createElement("div"); conceptsWrap.className = "nivel-hijos";
      const rows = bucket.rows.filter((row) => keyMatch(row, /^concepto$|rubro/i));
      rows.forEach((row) => {
        const conceptKey = keyMatch(row, /^concepto$|rubro/i);
        const conceptLineKey = keyMatch(row, /^linea_id$|linea_presupuesto_id/i);
        const conceptDetails = document.createElement("details"); conceptDetails.className = "nivel nivel-concepto";
        const conceptSummary = document.createElement("summary");
        const conceptHeading = document.createElement("span"); conceptHeading.className = "nivel-titulo"; conceptHeading.textContent = row[conceptKey];
        const conceptKeys = metricKeys(row); const conceptMeta = document.createElement("span"); conceptMeta.className = "nivel-meta";
        conceptMeta.textContent = [
          conceptKeys.budget && `Presupuesto: ${format(row[conceptKeys.budget], conceptKeys.budget)}`,
          conceptKeys.spent && `Gastado: ${format(row[conceptKeys.spent], conceptKeys.spent)}`,
        ].filter(Boolean).join(" · ");
        conceptSummary.append(conceptHeading, conceptMeta); conceptDetails.append(conceptSummary);
        const conceptBody = document.createElement("div"); conceptBody.className = "nivel-detalle"; appendMetricBars(conceptBody, row);
        const matches = movements.filter((movement, index) => {
          const sameLine = conceptLineKey && movementLine(movement) && movementLine(movement) === normalized(row[conceptLineKey]);
          const sameConcept = movementConcept(movement) === normalized(row[conceptKey]);
          const sameCategory = movementCategory(movement) && movementCategory(movement) === normalized(bucket.name);
          const sameName = normalized(movementName(movement)) === normalized(row[conceptKey]);
          if ((sameLine || sameConcept || sameName) && (!movementCategory(movement) || sameCategory)) { usedMovements.add(index); return true; }
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
            item.append(detail, value); movementList.append(item);
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

  byId("cliente").textContent = data.cliente.nombre;
  byId("periodo").textContent = data.periodo.etiqueta;
  byId("actualizado").textContent = "Actualizado: " + new Date(data.actualizado_en).toLocaleString("es-CR");

  const summary = findKpi("presupuesto_disponible") || data.kpis.find((k) => k.filas.length === 1 && k.columnas.some((c) => /presupuesto/i.test(c)));
  const summaryRow = summary?.filas?.[0] ? rowObject(summary, summary.filas[0]) : null;
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
  const hierarchyKpis = renderHierarchy(target);
  const visible = data.kpis
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
})();
