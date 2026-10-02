"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const NB1 = new Intl.NumberFormat("nb-NO", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const NB0 = new Intl.NumberFormat("nb-NO", { maximumFractionDigits: 0 });
const MONTHS = ["januar", "februar", "mars", "april", "mai", "juni", "juli", "august",
  "september", "oktober", "november", "desember"];
const HISTORY_QUARTERS = { wide: 56, narrow: 28 };

const state = { data: null, region: "norge", mode: "nominal", cursor: null };

// ---------- Formatering ----------
function signed(v, unit = " %") {
  if (Math.abs(v) < 0.05) return "0,0" + unit;
  return (v > 0 ? "+" : "\u2212") + NB1.format(Math.abs(v)) + unit;
}
function quarterText(q) {
  const [y, k] = q.split("K");
  return `${k}. kvartal ${y}`;
}
function monthText(m) {
  const [y, mm] = m.split("-");
  return `${MONTHS[Number(mm) - 1]} ${y}`;
}
function el(tag, attrs = {}, parent) {
  const isSvg = ["svg", "g", "path", "line", "rect", "text", "circle", "polyline"].includes(tag);
  const node = isSvg ? document.createElementNS(SVG_NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}
function niceStep(range, target) {
  const raw = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const steps = [1, 2, 2.5, 5, 10];
  return mag * steps.find(s => s * mag >= raw);
}

// ---------- Data ----------
function regionData() { return state.data.regions[state.region]; }

function series(width) {
  const r = regionData();
  const base = r.last_index;
  const n = r.history.quarters.length;
  const keep = width < 600 ? HISTORY_QUARTERS.narrow : HISTORY_QUARTERS.wide;
  const start = Math.max(0, n - keep);
  const hq = r.history.quarters.slice(start);
  const hv = r.history[state.mode].slice(start).map(v => (v / base) * 100);
  const fc = r.forecast[state.mode];
  const norm = arr => [100, ...arr.map(v => (v / base) * 100)];
  return {
    hq, hv,
    fq: [hq[hq.length - 1], ...r.forecast.quarters],
    p10: norm(fc.p10), p25: norm(fc.p25), p50: norm(fc.p50),
    p75: norm(fc.p75), p90: norm(fc.p90),
  };
}

// ---------- Viftediagram ----------
function drawChart() {
  const box = document.getElementById("chart");
  box.textContent = "";
  const W = Math.max(320, box.clientWidth || 960);
  const s = series(W);
  const r = regionData();
  const H = W < 600 ? 360 : 460;
  const m = { l: 46, r: 18, t: W < 600 ? 52 : 46, b: 30 };
  const nH = s.hq.length;
  const nTotal = nH + s.fq.length - 1;
  const x = i => m.l + (i / (nTotal - 1)) * (W - m.l - m.r);

  const all = [...s.hv, ...s.p10, ...s.p90];
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.06;
  lo -= pad; hi += pad;
  const step = niceStep(hi - lo, W < 600 ? 4 : 6);
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil(hi / step) * step;
  const y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: "0",
    "aria-label": `Prisnivå for ${r.name} med prognose. Bruk piltastene for å lese av verdier.`,
  }, box);

  // Rutenett og akser
  const grid = el("g", { class: "grid" }, svg);
  const axis = el("g", { class: "axis" }, svg);
  for (let v = lo; v <= hi + 1e-9; v += step) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
    el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", text: NB0.format(v) }, axis);
  }
  const labels = [...s.hq, ...s.fq.slice(1)];
  const yearEvery = W < 600 ? 4 : 2;
  labels.forEach((q, i) => {
    const [yr, k] = q.split("K");
    if (k === "1" && Number(yr) % yearEvery === 0) {
      el("text", { x: x(i), y: H - 8, "text-anchor": "middle", text: yr }, axis);
    }
  });

  // Bånd og linjer
  const fx = j => x(nH - 1 + j);
  const band = (a, b) => {
    const up = a.map((v, j) => `${fx(j)},${y(v)}`);
    const down = b.map((v, j) => `${fx(j)},${y(v)}`).reverse();
    return `M${up.join("L")}L${down.join("L")}Z`;
  };
  el("path", { class: "band-80", d: band(s.p10, s.p90) }, svg);
  el("path", { class: "band-50", d: band(s.p25, s.p75) }, svg);
  el("polyline", { class: "history-line", points: s.hv.map((v, i) => `${x(i)},${y(v)}`).join(" ") }, svg);
  el("polyline", { class: "median-line", points: s.p50.map((v, j) => `${fx(j)},${y(v)}`).join(" ") }, svg);

  // Skillelinje for siste kjente kvartal
  const xn = x(nH - 1);
  el("line", { class: "now-line", x1: xn, x2: xn, y1: m.t - 6, y2: H - m.b }, svg);
  el("text", { class: "now-label", x: xn - 8, y: H - m.b - 8, "text-anchor": "end", text: "Kjent" }, svg);
  el("text", { class: "now-label", x: xn + 8, y: H - m.b - 8, text: "Prognose" }, svg);

  // Horisontmarkører, forskjøvet i to rader for å unngå overlapp
  r.horizons.forEach((hz, idx) => {
    const xh = fx(hz.h);
    const row = idx % 2;
    const ly = m.t - 26 + row * 16;
    el("line", { class: "hz-tick", x1: xh, x2: xh, y1: ly + 4, y2: y(s.p50[hz.h]) }, svg);
    el("text", { class: "hz-label", x: xh, y: ly, "text-anchor": "middle", text: hz.label }, svg);
    el("circle", { class: "hz-dot", cx: xh, cy: y(s.p50[hz.h]), r: 4.5 }, svg);
  });

  // Avlesning
  const cursor = el("line", { class: "cursor", y1: m.t, y2: H - m.b, visibility: "hidden" }, svg);
  const overlay = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  const pick = i => {
    i = Math.max(0, Math.min(nTotal - 1, i));
    state.cursor = i;
    cursor.setAttribute("x1", x(i));
    cursor.setAttribute("x2", x(i));
    cursor.setAttribute("visibility", "visible");
    readout(i, s, labels);
  };
  overlay.addEventListener("pointermove", ev => {
    const pt = svg.getBoundingClientRect();
    const px = ((ev.clientX - pt.left) / pt.width) * W;
    pick(Math.round(((px - m.l) / (W - m.l - m.r)) * (nTotal - 1)));
  });
  overlay.addEventListener("pointerleave", () => {
    cursor.setAttribute("visibility", "hidden");
    state.cursor = null;
    defaultReadout();
  });
  svg.addEventListener("keydown", ev => {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    ev.preventDefault();
    const cur = state.cursor ?? nH - 1;
    pick(cur + (ev.key === "ArrowRight" ? 1 : -1));
  });
  svg.addEventListener("blur", () => {
    cursor.setAttribute("visibility", "hidden");
    state.cursor = null;
    defaultReadout();
  });

  const note = state.mode === "real"
    ? `Justert for inflasjon til kroneverdien i ${quarterText(r.last_quarter)}. `
    : "";
  document.getElementById("chart-note").textContent =
    `Prisnivå der ${quarterText(r.last_quarter)} = 100. ${note}Mørkt felt: 50 % sannsynlighet. Lyst felt: 80 % sannsynlighet.`;
}

function readout(i, s, labels) {
  const out = document.getElementById("readout");
  const nH = s.hq.length;
  const q = labels[i];
  if (i < nH) {
    const diff = s.hv[i] - 100;
    const rel = Math.abs(diff) < 0.05 ? "siste kjente prisnivå"
      : `${NB1.format(Math.abs(diff))} % ${diff < 0 ? "lavere" : "høyere"} enn siste kjente nivå`;
    out.innerHTML = `<strong>${quarterText(q)}:</strong> ${rel}`;
  } else {
    const j = i - nH + 1;
    out.innerHTML = `<strong>${quarterText(q)}:</strong> mest sannsynlig ${signed(s.p50[j] - 100)}, ` +
      `80 % sannsynlig mellom ${signed(s.p10[j] - 100)} og ${signed(s.p90[j] - 100)}`;
  }
}

function defaultReadout() {
  const r = regionData();
  const hz = r.horizons.find(h => h.months === 12) || r.horizons[0];
  const v = hz[state.mode];
  document.getElementById("readout").innerHTML =
    `<strong>${r.name} om ${hz.label}:</strong> mest sannsynlig ${signed(v.p50)}, ` +
    `80 % sannsynlig mellom ${signed(v.p10)} og ${signed(v.p90)}`;
}

// ---------- Spennstriper ----------
function drawStrips() {
  const r = regionData();
  const wrap = document.getElementById("strips");
  wrap.textContent = "";
  const vals = r.horizons.flatMap(h => [h[state.mode].p10, h[state.mode].p90, 0]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const step = niceStep(hi - lo, 5);
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil(hi / step) * step;
  const pos = v => ((v - lo) / (hi - lo)) * 1000;

  document.getElementById("hz-note").textContent = state.mode === "real"
    ? `Prosentvis endring i inflasjonsjustert pris fra ${quarterText(r.last_quarter)}. Den tykke streken er mest sannsynlig utfall, feltene viser 50 % og 80 % sannsynlighet.`
    : `Prosentvis endring i pris fra ${quarterText(r.last_quarter)}. Den tykke streken er mest sannsynlig utfall, feltene viser 50 % og 80 % sannsynlighet.`;

  for (const hz of r.horizons) {
    const v = hz[state.mode];
    const row = el("div", { class: "strip" }, wrap);
    const lab = el("div", { class: "strip-label" }, row);
    lab.append(hz.label);
    el("small", { text: `til ${quarterText(hz.quarter).replace("kvartal", "kv.")}` }, lab);
    el("div", { class: "strip-median", text: signed(v.p50) }, row);
    const svg = el("svg", { viewBox: "0 0 1000 34", preserveAspectRatio: "none", "aria-hidden": "true" }, row);
    el("rect", { class: "outer", x: pos(v.p10), y: 6, width: pos(v.p90) - pos(v.p10), height: 22 }, svg);
    el("rect", { class: "inner", x: pos(v.p25), y: 6, width: pos(v.p75) - pos(v.p25), height: 22 }, svg);
    el("line", { class: "zero", x1: pos(0), x2: pos(0), y1: 0, y2: 34, "vector-effect": "non-scaling-stroke" }, svg);
    el("line", { class: "mid", x1: pos(v.p50), x2: pos(v.p50), y1: 2, y2: 32, "vector-effect": "non-scaling-stroke" }, svg);
    el("div", { class: "strip-range", text: `80 %: ${signed(v.p10, "")} til ${signed(v.p90)}` }, row);
  }
  // Akse
  const axisRow = el("div", { class: "strip-axis", "aria-hidden": "true" }, wrap);
  el("div", {}, axisRow);
  el("div", {}, axisRow);
  const cell = el("div", { class: "axis-cell", style: "position:relative;height:20px" }, axisRow);
  for (let t = lo; t <= hi + 1e-9; t += step) {
    el("span", {
      text: signed(t).replace(",0", "").replace(" %", "\u00a0%"),
      style: `position:absolute;left:${pos(t) / 10}%;transform:translateX(-50%);font-size:11px;color:var(--slate)`,
    }, cell);
  }
  el("div", {}, axisRow);
}

// ---------- Drivere og treffsikkerhet ----------
function drawDrivers() {
  const list = document.getElementById("drivers");
  list.textContent = "";
  const d = regionData().drivers || [];
  if (!d.length) {
    el("li", { text: "Driverforklaring er ikke tilgjengelig for denne kjøringen." }, list);
    return;
  }
  const max = Math.max(...d.map(x => Math.abs(x.pp)), 0.1);
  for (const item of d) {
    const li = el("li", {}, list);
    el("span", { text: item.label }, li);
    const bar = el("span", { class: "driver-bar", "aria-hidden": "true" }, li);
    el("span", { class: item.pp >= 0 ? "pos" : "neg", style: `width:${(Math.abs(item.pp) / max) * 50}%` }, bar);
    el("span", { class: "driver-value", text: signed(item.pp, " pp") }, li);
  }
}

function drawAccuracy() {
  const body = document.querySelector("#accuracy tbody");
  body.textContent = "";
  for (const a of state.data.accuracy || []) {
    const tr = el("tr", {}, body);
    el("th", { scope: "row", text: a.label }, tr);
    el("td", { text: `${NB1.format(a.mae_model)} pp` }, tr);
    el("td", { text: a.mae_uendret == null ? "–" : `${NB1.format(a.mae_uendret)} pp` }, tr);
  }
}

function drawAbout() {
  const d = state.data;
  const a = d.assumptions;
  const upd = d.monthly_update || {};
  const p = [
    `Prognosen bygger på SSBs prisindeks for brukte boliger, som kommer hvert kvartal. Siste kjente kvartal er ${quarterText(d.last_quarter)}. ` +
    (upd.last_month
      ? `Siden oppdateres hver måned, og mellom kvartalstallene brukes ferskere tall for renter, prisvekst og kreditt (til og med ${monthText(upd.last_month)}).`
      : "Siden oppdateres hver måned med de nyeste tilgjengelige tallene."),
    "For hver horisont er det laget en egen modell for endringen i inflasjonsjustert pris. Den bygger på boliglånsrenten etter skatt og inflasjon, renteendringer, kredittvekst, oljeprisen, prisnivået sammenlignet med byggekostnader og med resten av landet, og prisutviklingen den siste tiden. En utvidet variant tar også med befolkningsvekst, boligbygging og husholdningenes inntektsvekst. Disse kombineres med to enkle referanser, historisk gjennomsnittlig vekst og uendret pris, og vektes etter hvor godt de har truffet tidligere.",
    "Spennene bygger på hvor mye modellen faktisk har bommet når den er testet bakover i tid fra 2005, med bare de dataene som var kjent på hvert tidspunkt. At 80 %-spennet er bredt, er en ærlig beskrivelse av hvor vanskelig det er å spå boligpriser.",
    `Prisene i kroner er regnet ut fra prognosen for inflasjonsjustert pris og en antakelse om at prisveksten går fra dagens ${NB1.format(a.inflation_now)} % til inflasjonsmålet på ${NB0.format(a.inflation_target)} % i løpet av to år. Usikkerheten i inflasjonen er ikke med i spennet.`,
    "Modellen vet ingenting om fremtidige rentebeslutninger, politiske endringer eller sjokk som ikke ligner noe som har skjedd før. Bruk prognosene som et utgangspunkt for egne vurderinger.",
  ];
  const box = document.getElementById("about");
  box.textContent = "";
  p.forEach(t => el("p", { text: t }, box));
}

// ---------- Oppsett ----------
function renderRegionButtons() {
  const wrap = document.getElementById("regions");
  wrap.textContent = "";
  for (const [key, r] of Object.entries(state.data.regions)) {
    const b = el("button", { type: "button", "aria-pressed": String(key === state.region), text: r.name }, wrap);
    b.addEventListener("click", () => {
      state.region = key;
      history.replaceState(null, "", `#${key}`);
      render();
      document.querySelector('#regions button[aria-pressed="true"]').focus();
    });
  }
}

function render() {
  renderRegionButtons();
  document.querySelectorAll(".mode button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode)));
  drawChart();
  defaultReadout();
  drawStrips();
  drawDrivers();
  drawAccuracy();
}

function setSourceLink() {
  const host = location.hostname;
  if (!host.endsWith("github.io")) return;
  const user = host.split(".")[0];
  const repo = location.pathname.split("/").filter(Boolean)[0];
  if (!repo) return;
  const p = document.getElementById("source-link");
  p.append("Kildekode, data og tidligere prognoser: ");
  el("a", { href: `https://github.com/${user}/${repo}`, text: `github.com/${user}/${repo}` }, p);
}

async function init() {
  try {
    const res = await fetch("forecast.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
  } catch (err) {
    const main = document.querySelector("main");
    main.innerHTML = "";
    el("p", {
      class: "error",
      text: `Fant ikke prognosedataene (forecast.json: ${err.message}). Kjør workflowen «Oppdater data og prognose» i GitHub Actions, så publiseres de på nytt.`,
    }, main);
    return;
  }
  const fromHash = location.hash.slice(1);
  if (state.data.regions[fromHash]) state.region = fromHash;
  document.getElementById("stamp").textContent =
    `Oppdatert ${monthText(state.data.asof_month)}. Boligprisene er kjent til og med ${quarterText(state.data.last_quarter)}.`;
  document.querySelectorAll(".mode button").forEach(b => b.addEventListener("click", () => {
    state.mode = b.dataset.mode;
    render();
  }));
  setSourceLink();
  drawAbout();
  render();
  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => { drawChart(); }, 150);
  });
}

init();
