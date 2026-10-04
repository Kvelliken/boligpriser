"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const NB1 = new Intl.NumberFormat("nb-NO", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const NB0 = new Intl.NumberFormat("nb-NO", { maximumFractionDigits: 0 });
const MONTHS = ["januar", "februar", "mars", "april", "mai", "juni", "juli", "august",
  "september", "oktober", "november", "desember"];
const HORIZONS = [["6", "6 mnd"], ["12", "1 år"], ["36", "3 år"], ["60", "5 år"]];
const TABS = [
  ["prognose", "Prognose"], ["drivere", "Hva driver prognosen"],
  ["renter", "Renter og kreditt"], ["befolkning", "Befolkning og bygging"],
  ["okonomi", "Inntekt og olje"], ["prisniva", "Prisnivå og trend"],
  ["treff", "Treffsikkerhet"], ["om", "Om modellen"],
];
const GROUP_TABS = ["renter", "befolkning", "okonomi", "prisniva"];

const state = { data: null, region: "norge", mode: "nominal", tab: "prognose", h: "12" };

// ---------- Formatering ----------
function signed(v, unit = " %") {
  if (v == null || !isFinite(v)) return "–";
  if (Math.abs(v) < 0.05) return "0,0" + unit;
  return (v > 0 ? "+" : "\u2212") + NB1.format(Math.abs(v)) + unit;
}
function plain(v, unit) {
  if (v == null || !isFinite(v)) return "–";
  return `${v < 0 ? "\u2212" : ""}${NB1.format(Math.abs(v))} ${unit}`;
}
function monthName(t) { return `${MONTHS[((t % 12) + 12) % 12]} ${Math.floor(t / 12)}`; }
function quarterText(q) { const [y, k] = q.split("K"); return `${k}. kvartal ${y}`; }
function mIndex(label) { const [y, m] = label.split("-").map(Number); return y * 12 + m - 1; }
function qIndex(label) { const [y, k] = label.split("K").map(Number); return y * 12 + (k - 1) * 3 + 1; }
function el(tag, attrs = {}, parent) {
  const svgTags = ["svg", "g", "path", "line", "rect", "text", "circle", "polyline"];
  const node = svgTags.includes(tag) ? document.createElementNS(SVG_NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}
function niceStep(range, target) {
  if (!(range > 0)) return 1;
  const raw = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  return mag * [1, 2, 2.5, 5, 10].find(s => s * mag >= raw);
}
const R = () => state.data.regions[state.region];

// ---------- Generisk linjediagram ----------
// cfg.layers: {type: "line"|"band"|"dot", pts: [[t, v]] eller [[t, lo, hi]], cls}
function lineChart(box, cfg) {
  box.textContent = "";
  const W = Math.max(300, box.clientWidth || 900);
  const narrow = W < 600;
  const H = cfg.height ? cfg.height(narrow) : (narrow ? 340 : 440);
  const m = { l: 48, r: 14, t: cfg.markers && cfg.markers.length ? (narrow ? 50 : 44) : 14, b: 28 };
  const ts = [], vs = [];
  for (const L of cfg.layers) for (const p of L.pts) {
    ts.push(p[0]);
    for (const v of p.slice(1)) if (v != null && isFinite(v)) vs.push(v);
  }
  if (cfg.refY != null) vs.push(cfg.refY);
  const t0 = Math.min(...ts), t1 = Math.max(...ts);
  let lo = Math.min(...vs), hi = Math.max(...vs);
  const pad = (hi - lo || 1) * 0.07;
  lo -= pad; hi += pad;
  const step = niceStep(hi - lo, narrow ? 4 : 6);
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const x = t => m.l + ((t - t0) / (t1 - t0 || 1)) * (W - m.l - m.r);
  const y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: "0", "aria-label": cfg.aria || "" }, box);
  const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
  const dec = step < 1 ? 1 : 0;
  for (let v = lo; v <= hi + 1e-9; v += step) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
    const txt = dec ? NB1.format(v) : NB0.format(v);
    el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", text: (cfg.yFmt ? cfg.yFmt(v, txt) : txt) }, axis);
  }
  const years = (t1 - t0) / 12;
  const every = years > 24 ? 5 : years > 12 ? (narrow ? 4 : 2) : narrow ? 2 : 1;
  for (let yr = Math.ceil(t0 / 12); yr * 12 <= t1; yr++) {
    if (yr % every) continue;
    el("text", { x: x(yr * 12), y: H - 8, "text-anchor": "middle", text: String(yr) }, axis);
  }
  if (cfg.refY != null) el("line", { class: "ref-line", x1: m.l, x2: W - m.r, y1: y(cfg.refY), y2: y(cfg.refY) }, svg);

  for (const L of cfg.layers) {
    if (!L.pts.length) continue;
    if (L.type === "band") {
      const up = L.pts.map(p => `${x(p[0])},${y(p[1])}`);
      const dn = L.pts.map(p => `${x(p[0])},${y(p[2])}`).reverse();
      el("path", { class: L.cls, d: `M${up.join("L")}L${dn.join("L")}Z` }, svg);
    } else if (L.type === "line") {
      el("polyline", { class: L.cls, points: L.pts.map(p => `${x(p[0])},${y(p[1])}`).join(" ") }, svg);
    } else if (L.type === "dot") {
      for (const p of L.pts) el("circle", { class: L.cls, cx: x(p[0]), cy: y(p[1]), r: L.r || 4.5 }, svg);
    }
  }
  for (const v of cfg.vlines || []) {
    el("line", { class: v.cls || "now-line", x1: x(v.t), x2: x(v.t), y1: m.t - 4, y2: H - m.b }, svg);
    if (v.label) el("text", { class: "now-label", x: x(v.t) + (v.anchor === "end" ? -6 : 6), y: H - m.b - 8,
      "text-anchor": v.anchor || "start", text: v.label }, svg);
  }
  (cfg.markers || []).forEach((mk, i) => {
    const ly = m.t - 26 + (i % 2) * 16;
    el("line", { class: "hz-tick", x1: x(mk.t), x2: x(mk.t), y1: ly + 4, y2: y(mk.v) }, svg);
    el("text", { class: "hz-label", x: x(mk.t), y: ly, "text-anchor": "middle", text: mk.label }, svg);
    el("circle", { class: "hz-dot", cx: x(mk.t), cy: y(mk.v), r: 4.5 }, svg);
  });

  if (!cfg.hover) return;
  const hts = cfg.hoverTs;
  const cursor = el("line", { class: "cursor", y1: m.t, y2: H - m.b, visibility: "hidden" }, svg);
  const overlay = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  let cur = null;
  const pick = i => {
    cur = Math.max(0, Math.min(hts.length - 1, i));
    cursor.setAttribute("x1", x(hts[cur])); cursor.setAttribute("x2", x(hts[cur]));
    cursor.setAttribute("visibility", "visible");
    cfg.hover(hts[cur]);
  };
  const nearest = t => {
    let best = 0;
    for (let i = 1; i < hts.length; i++) if (Math.abs(hts[i] - t) < Math.abs(hts[best] - t)) best = i;
    return best;
  };
  const reset = () => { cursor.setAttribute("visibility", "hidden"); cur = null; if (cfg.leave) cfg.leave(); };
  overlay.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    pick(nearest(t0 + ((px - m.l) / (W - m.l - m.r)) * (t1 - t0)));
  });
  overlay.addEventListener("pointerleave", reset);
  svg.addEventListener("blur", reset);
  svg.addEventListener("keydown", ev => {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    ev.preventDefault();
    pick((cur ?? hts.length - 1) + (ev.key === "ArrowRight" ? 1 : -1));
  });
}

// ---------- Fane: Prognose ----------
function drawFan() {
  const r = R(), c = r.chart, mode = state.mode;
  const box = document.getElementById("fan");
  const narrow = (box.clientWidth || 900) < 600;
  const ts = c.months.map(mIndex);
  const aT = mIndex(state.data.anchor_month);
  const iA = c.months.indexOf(state.data.anchor_month);
  const base = c.fan[mode].p50[iA];
  const n = v => (v == null ? null : (v / base) * 100);
  const from = aT - (narrow ? 7 : 14) * 12;
  const sel = (arr) => ts.map((t, i) => [t, n(arr[i])]).filter(p => p[0] >= from && p[1] != null);
  const band = (a, b) => ts.map((t, i) => [t, n(c.fan[mode][a][i]), n(c.fan[mode][b][i])]).filter(p => p[1] != null);
  const hist = sel(c.hist[mode]), now = sel(c.now[mode]), med = sel(c.fan[mode].p50);
  const lastKnown = hist.length ? hist[hist.length - 1][0] : aT;
  const markers = r.horizons.map(h => {
    const t = mIndex(h.target);
    return { t, v: n(c.fan[mode].p50[c.months.indexOf(h.target)]), label: h.label };
  });
  const lookup = {};
  ts.forEach((t, i) => { lookup[t] = i; });
  const hoverTs = [...hist, ...now.slice(1), ...med.slice(1)].map(p => p[0]);
  const out = document.getElementById("readout");
  const hover = t => {
    const i = lookup[t];
    if (t <= lastKnown) {
      const d = n(c.hist[mode][i]) - 100;
      out.innerHTML = `<strong>${monthName(t)}:</strong> ${Math.abs(d) < 0.05 ? "samme nivå som i dag" :
        `${NB1.format(Math.abs(d))} % ${d < 0 ? "lavere" : "høyere"} enn i dag`}`;
    } else if (t <= aT) {
      const d = n(c.now[mode][i]) - 100;
      out.innerHTML = `<strong>${monthName(t)} (anslått):</strong> ${NB1.format(Math.abs(d))} % ${d < 0 ? "lavere" : "høyere"} enn i dag`;
    } else {
      const f = c.fan[mode];
      out.innerHTML = `<strong>${monthName(t)}:</strong> mest sannsynlig ${signed(n(f.p50[i]) - 100)}, ` +
        `80 % sannsynlig mellom ${signed(n(f.p10[i]) - 100)} og ${signed(n(f.p90[i]) - 100)}`;
    }
  };
  lineChart(box, {
    aria: `Prisnivå for ${r.name} med prognose. Bruk piltastene for å lese av verdier.`,
    layers: [
      { type: "band", pts: band("p10", "p90"), cls: "band-80" },
      { type: "band", pts: band("p25", "p75"), cls: "band-50" },
      { type: "line", pts: hist, cls: "history-line" },
      { type: "line", pts: now, cls: "nowcast-line" },
      { type: "line", pts: med, cls: "median-line" },
    ],
    vlines: [{ t: lastKnown, label: narrow ? "" : "Siste kjente", anchor: "end", cls: "known-line" },
      { t: aT, label: "Nå", cls: "now-line" }],
    markers, hover, hoverTs, leave: defaultReadout,
  });
  defaultReadout();
  document.getElementById("fan-note").textContent =
    `Prisnivå der i dag (${monthName(aT)}) = 100. Heltrukket blå linje er kjente priser til og med ${quarterText(r.last_quarter)}; ` +
    `den stiplede biten frem til i dag er anslått. ${mode === "real" ? "Justert for inflasjon til dagens kroneverdi. " : ""}` +
    `Mørkt felt: 50 % sannsynlighet. Lyst felt: 80 % sannsynlighet.`;
}

function defaultReadout() {
  const r = R();
  const hz = r.horizons.find(h => h.months === 12);
  const v = hz[state.mode];
  document.getElementById("readout").innerHTML =
    `<strong>${r.name} om 12 måneder:</strong> mest sannsynlig ${signed(v.p50)}, ` +
    `80 % sannsynlig mellom ${signed(v.p10)} og ${signed(v.p90)}`;
}

function drawStrips() {
  const r = R(), wrap = document.getElementById("strips");
  wrap.textContent = "";
  const vals = r.horizons.flatMap(h => [h[state.mode].p10, h[state.mode].p90, 0]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const step = niceStep(hi - lo, 5);
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const pos = v => ((v - lo) / (hi - lo)) * 1000;
  document.getElementById("hz-note").textContent =
    `Prosentvis endring ${state.mode === "real" ? "i inflasjonsjustert pris " : "i pris "}fra ${monthName(mIndex(state.data.anchor_month))}. ` +
    "Den tykke streken er mest sannsynlig utfall, feltene viser 50 % og 80 % sannsynlighet.";
  for (const hz of r.horizons) {
    const v = hz[state.mode];
    const row = el("div", { class: "strip" }, wrap);
    const lab = el("div", { class: "strip-label" }, row);
    lab.append(hz.label);
    el("small", { text: `til ${monthName(mIndex(hz.target))}` }, lab);
    el("div", { class: "strip-median", text: signed(v.p50) }, row);
    const svg = el("svg", { viewBox: "0 0 1000 34", preserveAspectRatio: "none", "aria-hidden": "true" }, row);
    el("rect", { class: "outer", x: pos(v.p10), y: 6, width: pos(v.p90) - pos(v.p10), height: 22 }, svg);
    el("rect", { class: "inner", x: pos(v.p25), y: 6, width: pos(v.p75) - pos(v.p25), height: 22 }, svg);
    el("line", { class: "zero", x1: pos(0), x2: pos(0), y1: 0, y2: 34, "vector-effect": "non-scaling-stroke" }, svg);
    el("line", { class: "mid", x1: pos(v.p50), x2: pos(v.p50), y1: 2, y2: 32, "vector-effect": "non-scaling-stroke" }, svg);
    el("div", { class: "strip-range", text: `80 %: ${signed(v.p10, "")} til ${signed(v.p90)}` }, row);
  }
  const axisRow = el("div", { class: "strip-axis", "aria-hidden": "true" }, wrap);
  el("div", {}, axisRow); el("div", {}, axisRow);
  const cell = el("div", { class: "axis-cell" }, axisRow);
  for (let t = lo; t <= hi + 1e-9; t += step) {
    el("span", { text: signed(t).replace(",0", ""), style: `left:${pos(t) / 10}%` }, cell);
  }
  el("div", {}, axisRow);
}

// ---------- Fane: Hva driver prognosen ----------
function horizonPickers() {
  document.querySelectorAll(".horizon-pick").forEach(box => {
    box.textContent = "";
    for (const [k, label] of HORIZONS) {
      const b = el("button", { type: "button", "aria-pressed": String(k === state.h), text: label }, box);
      b.addEventListener("click", () => { state.h = k; renderTab(); });
    }
  });
}

function drawWaterfall() {
  const r = R(), d = r.decomposition[state.h];
  const hzLabel = HORIZONS.find(h => h[0] === state.h)[1];
  const small = d.items.filter(i => Math.abs(i.pp) < 0.05);
  const big = d.items.filter(i => Math.abs(i.pp) >= 0.05);
  const steps = [{ label: `Normal utvikling for ${r.name}`, pp: d.normal_pp, kind: "base" }];
  big.forEach(i => steps.push({ label: i.label, pp: i.pp, kind: "step", key: i.key }));
  if (small.length) steps.push({ label: `Andre variabler (${small.length})`, pp: small.reduce((s, i) => s + i.pp, 0), kind: "step" });
  steps.push({ label: "Endring justert for inflasjon", pp: d.real_pp, kind: "total" });
  if (state.mode === "nominal") {
    steps.push({ label: "Generell prisstigning", pp: d.inflation_pp, kind: "infl" });
    steps.push({ label: "Endring i kroner", pp: d.nominal_pp, kind: "total" });
  }
  let run = 0;
  const bars = steps.map(s => {
    let a, b;
    if (s.kind === "total") { a = 0; b = s.pp; run = s.pp; }
    else if (s.kind === "base") { a = 0; b = s.pp; run = s.pp; }
    else { a = run; b = run + s.pp; run = b; }
    return { ...s, a, b };
  });
  const ext = bars.flatMap(b => [b.a, b.b, 0]);
  let lo = Math.min(...ext), hi = Math.max(...ext);
  const pad = (hi - lo || 1) * 0.05; lo -= pad; hi += pad;
  const pos = v => ((v - lo) / (hi - lo)) * 100;
  const box = document.getElementById("waterfall");
  box.textContent = "";
  for (const b of bars) {
    const row = el("div", { class: `wf-row wf-${b.kind}` }, box);
    el("div", { class: "wf-label", text: b.label }, row);
    const track = el("div", { class: "wf-track" }, row);
    el("span", { class: "wf-zero", style: `left:${pos(0)}%` }, track);
    const left = Math.min(pos(b.a), pos(b.b)), width = Math.max(Math.abs(pos(b.b) - pos(b.a)), 0.4);
    const cls = b.kind === "total" ? "wf-bar total" : b.kind === "infl" ? "wf-bar infl" : b.pp >= 0 ? "wf-bar pos" : "wf-bar neg";
    el("span", { class: cls, style: `left:${left}%;width:${width}%` }, track);
    el("div", { class: "wf-value", text: b.kind === "total" ? signed(b.pp) : signed(b.pp, " pp") }, row);
  }
  document.getElementById("waterfall-note").textContent =
    `Prisendring de neste ${hzLabel === "6 mnd" ? "6 månedene" : hzLabel === "1 år" ? "12 månedene" : hzLabel.replace(" år", " årene")} fra ${monthName(mIndex(state.data.anchor_month))}. ` +
    "Normal utvikling er det modellen venter hvis alle forklaringsvariabler lå på sitt historiske snitt for området, inkludert den historiske trenden. " +
    "Bidragene er vektet slik modellene faktisk teller i prognosen.";

  const wbox = document.getElementById("weights");
  wbox.textContent = "";
  const w = d.weights;
  for (const [name, v] of Object.entries(w)) {
    const row = el("div", { class: "w-row" }, wbox);
    el("span", { text: name }, row);
    const tr = el("span", { class: "w-track" }, row);
    el("span", { class: "w-bar", style: `width:${v * 100}%` }, tr);
    el("span", { class: "w-val", text: `${NB0.format(v * 100)} %` }, row);
  }

  document.querySelector("#inputs-table thead th:last-child").textContent = `Bidrag (${hzLabel})`;
  const tb = document.querySelector("#inputs-table tbody");
  tb.textContent = "";
  for (const inp of r.inputs) {
    const tr = el("tr", {}, tb);
    const th = el("th", { scope: "row" }, tr);
    const a = el("a", { href: `#${state.region}/${inp.group}`, text: inp.label }, th);
    a.addEventListener("click", ev => { ev.preventDefault(); setTab(inp.group); });
    el("small", { text: inp.used_in.length === 2 ? "Begge modellene" : inp.used_in[0] }, th);
    el("td", { text: plain(inp.current, inp.unit) }, tr);
    el("td", { text: plain(inp.normal, inp.unit) }, tr);
    el("td", { class: inp.contrib[state.h] >= 0 ? "pos" : "neg", text: signed(inp.contrib[state.h], " pp") }, tr);
  }
}

// ---------- Faner: grupper av forklaringsvariabler ----------
function drawGroup(group) {
  const r = R(), box = document.getElementById("group-inputs");
  box.textContent = "";
  const items = r.inputs.filter(i => i.group === group);
  if (!items.length) {
    el("p", { class: "error", text: "Ingen variabler i denne gruppen har data for dette området." }, box);
    return;
  }
  const aT = mIndex(state.data.anchor_month);
  for (const inp of items) {
    const sec = el("section", { class: "input" }, box);
    const head = el("div", { class: "input-head" }, sec);
    el("h2", { text: inp.label }, head);
    el("p", { class: "section-note", text: inp.explain }, head);
    const body = el("div", { class: "input-body" }, sec);
    const chartBox = el("div", { class: "chart small" }, body);
    const facts = el("div", { class: "input-facts" }, body);

    const histPts = inp.history.quarters.map((q, i) => [qIndex(q), inp.history.values[i]]).filter(p => p[1] != null);
    const lastT = histPts.length ? histPts[histPts.length - 1][0] : aT;
    const curT = inp.current_note ? aT : lastT;
    const projPts = inp.projection ? [[curT, inp.current], ...inp.projection.quarters.map((q, i) => [qIndex(q) + (curT - lastT), inp.projection.values[i]])] : [];
    const readout = el("p", { class: "readout small" }, chartBox);
    const chartEl = el("div", {}, chartBox);
    const lookup = new Map([...histPts, ...projPts].map(p => [p[0], p[1]]));
    const show = () => { readout.innerHTML = `<strong>I dag:</strong> ${plain(inp.current, inp.unit)} (normalt ${plain(inp.normal, inp.unit)})`; };
    lineChart(chartEl, {
      height: n => (n ? 220 : 260),
      refY: inp.normal,
      layers: [
        { type: "line", pts: histPts, cls: "history-line" },
        { type: "line", pts: projPts, cls: "projection-line" },
        { type: "dot", pts: inp.current != null ? [[curT, inp.current]] : [], cls: "current-dot", r: 5 },
      ],
      vlines: [{ t: aT, cls: "now-line", label: "Nå" }],
      hoverTs: [...lookup.keys()].sort((a, b) => a - b),
      hover: t => {
        const v = lookup.get(t);
        readout.innerHTML = `<strong>${t > aT ? "Typisk forløp, " : ""}${t === curT ? "i dag" : monthName(t)}:</strong> ${plain(v, inp.unit)}`;
      },
      leave: show,
    });
    show();

    el("p", { class: "fact-line" }, facts).innerHTML =
      `<span>Brukes i</span><strong>${inp.used_in.join(" og ")}</strong>`;
    el("p", { class: "fact-line" }, facts).innerHTML =
      `<span>I dag mot normalt</span><strong>${plain(inp.current, inp.unit)} mot ${plain(inp.normal, inp.unit)}</strong>`;
    el("h3", { text: "Bidrag til prognosen" }, facts);
    const bars = el("div", { class: "mini-bars" }, facts);
    const mx = Math.max(0.3, ...HORIZONS.map(([k]) => Math.abs(inp.contrib[k] || 0)));
    for (const [k, label] of HORIZONS) {
      const v = inp.contrib[k] || 0;
      const row = el("div", { class: "mb-row" }, bars);
      el("span", { text: label }, row);
      const tr = el("span", { class: "mb-track" }, row);
      el("span", { class: `mb-bar ${v >= 0 ? "pos" : "neg"}`, style: `width:${(Math.abs(v) / mx) * 50}%;${v >= 0 ? "left:50%" : "right:50%"}` }, tr);
      el("span", { class: "mb-val", text: signed(v, " pp") }, row);
    }
    const s = inp.sensitivity.pp[state.h];
    const hzLabel = HORIZONS.find(h => h[0] === state.h)[1];
    if (s != null) {
      const effect = Math.abs(s) < 0.05 ? "nesten ikke endret seg"
        : `vært ${NB1.format(Math.abs(s))} prosentpoeng ${s >= 0 ? "høyere" : "lavere"}`;
      el("p", { class: "sensitivity", text:
        `Hvis ${inp.label.toLowerCase()} var ${inp.sensitivity.step_text}, ville prognosen for ${hzLabel} ${effect}.` }, facts);
    }
  }
}

// ---------- Fane: Treffsikkerhet ----------
function drawAccuracy() {
  const r = R(), a = r.accuracy[state.h];
  const box = document.getElementById("acc-chart");
  const facts = document.getElementById("acc-facts");
  facts.textContent = "";
  if (!a) { box.textContent = "Ingen historiske prognoser for denne horisonten."; return; }
  const v = a[state.mode];
  const ts = a.quarters.map(qIndex);
  const pts = arr => ts.map((t, i) => [t, arr[i]]).filter(p => p[1] != null);
  const hzLabel = HORIZONS.find(h => h[0] === state.h)[1];
  const out = document.getElementById("acc-readout");
  const show = () => { out.innerHTML = `<strong>Prisendring over ${hzLabel}:</strong> faktisk utvikling mot modellens anslag laget ${hzLabel} før`; };
  const idx = new Map(ts.map((t, i) => [t, i]));
  lineChart(box, {
    refY: 0,
    yFmt: (val, txt) => `${txt} %`,
    layers: [
      { type: "band", pts: ts.map((t, i) => [t, v.p10[i], v.p90[i]]), cls: "band-80" },
      { type: "line", pts: pts(v.pred), cls: "median-line" },
      { type: "line", pts: pts(v.actual), cls: "history-line" },
    ],
    hoverTs: ts,
    hover: t => {
      const i = idx.get(t);
      out.innerHTML = `<strong>Frem til ${quarterText(a.quarters[i])}:</strong> faktisk ${signed(v.actual[i])}, ` +
        `anslått ${signed(v.pred[i])} (80 %: ${signed(v.p10[i], "")} til ${signed(v.p90[i])})`;
    },
    leave: show,
  });
  show();
  const lg = document.getElementById("acc-legend");
  lg.innerHTML = `<span class="lg lg-actual">Faktisk</span><span class="lg lg-pred">Modellens anslag</span><span class="lg lg-band">80 %-spenn</span>`;
  const add = (dt, dd) => { el("dt", { text: dt }, facts); el("dd", { text: dd }, facts); };
  add("Gjennomsnittlig bom, justert for inflasjon", `${NB1.format(a.mae)} prosentpoeng`);
  if (a.mae_uendret != null) add("Til sammenligning: å anta uendret realpris", `${NB1.format(a.mae_uendret)} prosentpoeng`);
  add("Utfall innenfor 80 %-spennet", `${NB0.format(a.coverage)} %`);
  add("Antall prognoser", NB0.format(a.n));
}

// ---------- Fane: Om modellen ----------
function drawAbout() {
  const d = state.data, a = d.assumptions;
  const p = [
    `Prognosene regnes fra ${monthName(mIndex(d.anchor_month))}, som er siste måned med prisstatistikk fra SSB. SSBs boligprisindeks kommer bare hvert kvartal og er kjent til og med ${quarterText(d.last_quarter)}. Prisene mellom dette og i dag er anslått av modellen, og vises stiplet i diagrammet.`,
    "For hver horisont er det laget egne modeller for endringen i inflasjonsjustert pris. Hovedmodellen bygger på variabler med historikk fra 1990-tallet: boliglånsrenten etter skatt og inflasjon, renteendringer, kredittvekst, oljeprisen, prisnivået sammenlignet med byggekostnader og resten av landet, og prisutviklingen den siste tiden. Den utvidede modellen bruker i tillegg befolkningsvekst, boligbygging og husholdningenes inntektsvekst, som har kortere historikk. Disse kombineres med to enkle referanser, historisk gjennomsnittlig vekst og uendret pris, og vektes etter hvor godt de har truffet tidligere.",
    "Modellen bruker dagens verdi av hver forklaringsvariabel. Den har lært av historien hvordan priser har utviklet seg videre fra lignende situasjoner, og trenger derfor ikke egne prognoser for renter eller befolkning. Fremskrivningene som vises under hver variabel, er en illustrasjon av hvordan variabelen typisk har beveget seg videre.",
    "Spennene bygger på hvor mye modellen faktisk har bommet når den er testet bakover i tid fra 2005, med bare de dataene som var kjent på hvert tidspunkt. Fanen Treffsikkerhet viser disse prognosene mot det som faktisk skjedde.",
    `Prisene i kroner er regnet ut fra prognosen for inflasjonsjustert pris og en antakelse om at prisveksten går fra dagens ${NB1.format(a.inflation_now)} % til inflasjonsmålet på ${NB0.format(a.inflation_target)} % i løpet av to år. Usikkerheten i inflasjonen er ikke med i spennet.`,
    "Modellen vet ingenting om fremtidige rentebeslutninger, endringer i skatt og regulering, eller krefter som ikke finnes i dataene, som at mange utleiere selger samtidig. Bruk prognosene som et utgangspunkt for egne vurderinger.",
  ];
  const box = document.getElementById("about");
  box.textContent = "";
  p.forEach(t => el("p", { text: t }, box));
}

// ---------- Oppsett og navigasjon ----------
function renderRegionButtons() {
  const wrap = document.getElementById("regions");
  wrap.textContent = "";
  for (const [key, r] of Object.entries(state.data.regions)) {
    const b = el("button", { type: "button", "aria-pressed": String(key === state.region), text: r.name }, wrap);
    b.addEventListener("click", () => {
      state.region = key; updateHash(); renderAll();
      document.querySelector('#regions button[aria-pressed="true"]').focus();
    });
  }
}

function renderTabs() {
  const nav = document.getElementById("tabs");
  nav.textContent = "";
  TABS.forEach(([key, label]) => {
    const b = el("button", { type: "button", role: "tab", id: `tab-${key}`, "aria-selected": String(key === state.tab),
      tabindex: key === state.tab ? "0" : "-1", text: label }, nav);
    b.addEventListener("click", () => setTab(key));
    b.addEventListener("keydown", ev => {
      const i = TABS.findIndex(t => t[0] === key);
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
        ev.preventDefault();
        const next = TABS[(i + (ev.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length][0];
        setTab(next);
        document.getElementById(`tab-${next}`).focus();
      }
    });
  });
}

function setTab(key) {
  state.tab = key; updateHash(); renderTabs(); renderTab();
}

function renderTab() {
  const isGroup = GROUP_TABS.includes(state.tab);
  const panels = { prognose: "panel-prognose", drivere: "panel-drivere", treff: "panel-treff", om: "panel-om" };
  for (const id of Object.values(panels)) document.getElementById(id).hidden = true;
  document.getElementById("panel-gruppe").hidden = !isGroup;
  if (!isGroup) document.getElementById(panels[state.tab]).hidden = false;
  else document.getElementById("panel-gruppe").setAttribute("aria-labelledby", `tab-${state.tab}`);
  horizonPickers();
  if (state.tab === "prognose") { drawFan(); drawStrips(); }
  else if (state.tab === "drivere") drawWaterfall();
  else if (state.tab === "treff") drawAccuracy();
  else if (state.tab === "om") drawAbout();
  else drawGroup(state.tab);
}

function renderAll() {
  renderRegionButtons();
  document.querySelectorAll(".mode button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode)));
  renderTabs();
  renderTab();
}

function updateHash() { history.replaceState(null, "", `#${state.region}/${state.tab}`); }

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
    if (!state.data.anchor_month) throw new Error("gammelt dataformat – kjør workflowen på nytt");
  } catch (err) {
    const main = document.querySelector("main");
    main.innerHTML = "";
    el("p", { class: "error", text: `Fant ikke gyldige prognosedata (forecast.json: ${err.message}). Kjør workflowen «Oppdater data og prognose» i GitHub Actions, så publiseres de på nytt.` }, main);
    return;
  }
  const [hr, ht] = location.hash.slice(1).split("/");
  if (state.data.regions[hr]) state.region = hr;
  if (TABS.some(t => t[0] === ht)) state.tab = ht;
  document.getElementById("stamp").textContent =
    `Prognosene regnes fra ${monthName(mIndex(state.data.anchor_month))}, siste måned med tall fra SSB. ` +
    `Boligprisindeksen er kjent til og med ${quarterText(state.data.last_quarter)}; prisene etter det er anslått.`;
  document.querySelectorAll(".mode button").forEach(b => b.addEventListener("click", () => {
    state.mode = b.dataset.mode; renderAll();
  }));
  setSourceLink();
  renderAll();
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderTab, 150); });
}

init();
