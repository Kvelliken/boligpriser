"""Kjører prognosemodellen og skriver output/forecast.json.

Prognosene regnes fra «nå», definert som siste måned med KPI fra SSB.
Boligprisindeksen er kvartalsvis og kommer senere, så prisnivået frem til
«nå» anslås med modellens korttidsprognoser.

Bruk: python -m boligprognose.forecast [--asof 2026-10]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (DISPLAY_ALIAS, EXTENDED_FEATURES, FEATURE_LABELS, FEATURE_META, GROUPS,
                       LONG_FEATURES, REGIONS, ar1_projection, build_panel, display_value,
                       origin_overrides)
from .model import (COMPONENTS, H_MAX, accuracy_table, available_features, backtest,
                    ensemble_errors, fit_component, interval_offsets, origin_rows, weights_at)
from .panel import load_series

log = logging.getLogger("forecast")
ROOT = Path(__file__).resolve().parents[2]

USER_HORIZONS = [("6 mnd", 6), ("1 år", 12), ("3 år", 36), ("5 år", 60)]
ACCURACY_H = {6: 2, 12: 4, 36: 12, 60: 20}
COMP_LABELS = {"ridge_lang": "Hovedmodellen", "ridge_utvidet": "Utvidet modell",
               "drift": "Historisk trend", "uendret": "Uendret pris"}
RIDGE = ("ridge_lang", "ridge_utvidet")
INFLATION_TARGET = 0.02
INFLATION_CONVERGENCE_MONTHS = 24
CHART_FROM = pd.Period("2010-01", "M")
INPUT_HISTORY_FROM = pd.Period("2000Q1", "Q")
QKEYS = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]


def qlabel(p):
    return f"{p.year}K{p.quarter}"


def mlabel(m):
    return f"{m.year}-{m.month:02d}"


def rnd(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# -- Inflasjon -----------------------------------------------------------------
def inflation_path_monthly(kpi_m, end_month):
    """KPI per måned: faktiske tall, deretter tolvmånedersvekst som glir mot 2 % over 24 mnd."""
    last_m = kpi_m.index.max()
    infl_now = float(kpi_m.iloc[-1] / kpi_m[kpi_m.index <= last_m - 12].iloc[-1] - 1)
    level, vals = float(kpi_m.iloc[-1]), {}
    for i, m in enumerate(pd.period_range(last_m + 1, end_month, freq="M"), start=1):
        w = min(i / INFLATION_CONVERGENCE_MONTHS, 1.0)
        level *= (1 + (1 - w) * infl_now + w * INFLATION_TARGET) ** (1 / 12)
        vals[m] = level
    path = pd.concat([kpi_m, pd.Series(vals, dtype=float)])
    path.index = pd.PeriodIndex(path.index, freq="M")
    return path, infl_now


def inflation_path(kpi_monthly, last_q, h_max, asof=None):
    """Kvartalsvis KPI-bane (brukes i tester og for bakoverkompatibilitet)."""
    end = (last_q + h_max).asfreq("M", how="end")
    path, infl_now = inflation_path_monthly(kpi_monthly, end)
    return path.groupby(path.index.asfreq("Q")).mean(), infl_now


# -- Interpolering mellom kvartalshorisonter --------------------------------------
def interp(vals, f):
    """Lineær interpolasjon i en liste/array indeksert med h = 0..H (vals[0] = 0)."""
    f = float(np.clip(f, 0, len(vals) - 1))
    lo = int(np.floor(f))
    hi = min(lo + 1, len(vals) - 1)
    w = f - lo
    return np.asarray(vals[lo]) * (1 - w) + np.asarray(vals[hi]) * w


def change_offsets(off_t, off_a):
    """Usikkerhet for endringen fra «nå» til målet, gitt usikkerhet målt fra siste kjente kvartal.

    Antar at bommene akkumuleres omtrent som uavhengige tillegg (som en tilfeldig gange):
    variansen i endringen = variansen til målet minus variansen ved «nå».
    """
    off_t, off_a = np.asarray(off_t), np.asarray(off_a)
    q = np.sign(off_t) * np.sqrt(np.maximum(off_t ** 2 - off_a ** 2, 0.0))
    return np.sort(q)


# -- Nedbrytning ------------------------------------------------------------------
def decompose(fns, w, Xo, region, idx, region_means):
    """Del ensembleprognosen i normalnivå + bidrag fra hver variabel (eksakt sum)."""
    preds = {}
    for c, fn in fns.items():
        if fn is None:
            continue
        p = fn(Xo.loc[[idx]]).iloc[0]
        if np.isfinite(p):
            preds[c] = float(p)
    if not preds:
        return np.nan, np.nan, {}, {}, {}
    ws = np.array([w.get(c, 0.0) for c in preds])
    if ws.sum() <= 0:
        ws = np.ones(len(preds))
    wmap = dict(zip(preds, ws / ws.sum()))
    normal, contrib, sens = 0.0, defaultdict(float), defaultdict(float)
    for c, wc in wmap.items():
        if c in RIDGE:
            m, feats = fns[c].model, fns[c].feats
            x = Xo.loc[idx, feats]
            normal += wc * m.const.get(region, np.mean(list(m.const.values())))
            for j in feats:
                mr = region_means.get((region, j), m.mu[j])
                contrib[j] += wc * m.coef[j] * (x[j] - mr) / m.sd[j]
                normal += wc * m.coef[j] * (mr - m.mu[j]) / m.sd[j]
                sens[j] += wc * m.coef[j] / m.sd[j]
        elif c == "drift":
            normal += wc * preds[c]
    point = sum(wc * preds[c] for c, wc in wmap.items())
    return point, normal, dict(contrib), dict(sens), wmap


def region_feature_means(panel, feats):
    means = {}
    for r, g in panel.groupby(level="region"):
        for f in feats:
            if f in g:
                v = g[f].dropna()
                if len(v):
                    means[(r, f)] = float(v.mean())
    return means


def _alias(key):
    return DISPLAY_ALIAS.get(key, key)


# -- Hovedløp ---------------------------------------------------------------------
def run(data_dir, asof):
    series = load_series(Path(data_dir) / "raw")
    panel, nat = build_panel(series)
    feats_long = available_features(panel, LONG_FEATURES)
    feats_ext = available_features(panel, EXTENDED_FEATURES)
    dropped = sorted(set(EXTENDED_FEATURES) - set(feats_ext))
    if dropped:
        log.warning("Variabler uten data (droppet): %s", dropped)
    all_feats = sorted(set(feats_long) | set(feats_ext))

    regions = [r for r in REGIONS if r in set(panel.index.get_level_values("region"))]
    T = series["bpi_norge"].index.max()
    kpi_m = series["kpi"]
    anchor = min(kpi_m.index.max(), asof)
    mid_T = T.asfreq("M", how="start") + 1
    f_of = lambda m: (m - mid_T).n / 3  # noqa: E731
    f_a = f_of(anchor)
    if f_a < 0:
        raise ValueError(f"Siste KPI-måned {anchor} er eldre enn siste boligpriskvartal {T}")
    log.info("Siste boligpriskvartal %s, «nå» = %s (%.2f kvartaler etter), kjøremåned %s",
             T, anchor, f_a, asof)

    bt = backtest(panel, feats_long, feats_ext, T)
    ens = ensemble_errors(bt)
    offsets = interval_offsets(ens, regions)

    Xo = origin_rows(panel, T).copy()
    overrides = origin_overrides(series, nat, T, anchor)
    for k in ("real_rate_at", "d_rate4", "credit_real"):
        if k in overrides and k in Xo:
            Xo[k] = overrides[k]
    rmeans = region_feature_means(panel, all_feats)

    # Prognose og nedbrytning per kvartalshorisont h = 1..H
    point = {r: [0.0] for r in regions}
    normal = {r: [0.0] for r in regions}
    contrib = {r: [dict()] for r in regions}
    sens = {r: [dict()] for r in regions}
    wmaps = {r: [dict()] for r in regions}
    global_w = {}
    for h in range(1, H_MAX + 1):
        w = weights_at(bt, h, T + H_MAX + 1)
        global_w[h] = w
        fns = {c: fit_component(c, panel, feats_long, feats_ext, h, T) for c in COMPONENTS}
        for r in regions:
            p, n, c, s, wm = decompose(fns, w, Xo, r, (r, T), rmeans)
            point[r].append(p)
            normal[r].append(n)
            contrib[r].append(c)
            sens[r].append(s)
            wmaps[r].append(wm)

    end_month = anchor + 60
    if f_of(end_month) > H_MAX:
        log.warning("Lengste horisont går %0.1f kvartaler forbi H_MAX", f_of(end_month) - H_MAX)
    kpi_path, infl_now = inflation_path_monthly(kpi_m, end_month)
    kpi_mid = float(kpi_path[mid_T])
    kpi_A = float(kpi_path[anchor])

    out_regions = {}
    for r in regions:
        pts = np.array(point[r])
        if not np.all(np.isfinite(pts)):
            raise ValueError(f"Prognosen for {r} mangler verdier")
        offs = [np.zeros(7)] + [np.asarray(offsets[r][h]) for h in range(1, H_MAX + 1)]
        bpi = series[REGIONS[r]["bpi"]]
        bpi_T = float(bpi[T])
        L = lambda f: float(interp(pts, f))  # noqa: E731
        O = lambda f: interp(offs, f)  # noqa: E731

        # Månedlig historikk (log-lineær interpolasjon mellom kvartalsmidtpunktene)
        qmid = pd.PeriodIndex([q.asfreq("M", how="start") + 1 for q in bpi.index], freq="M")
        logb = pd.Series(np.log(bpi.values), index=qmid)
        months = pd.period_range(CHART_FROM, end_month, freq="M")
        hist_months = months[months <= mid_T]
        full = logb.reindex(logb.index.union(hist_months)).sort_index()
        num = pd.Series(full.values, index=[(m - CHART_FROM).n for m in full.index])
        hist_log = num.interpolate(method="index").values
        hist_nom = pd.Series(np.exp(hist_log), index=full.index).reindex(hist_months)

        base_nom_A = bpi_T * np.exp(L(f_a)) * kpi_A / kpi_mid
        chart = {"months": [mlabel(m) for m in months],
                 "hist": {"nominal": [], "real": []}, "now": {"nominal": [], "real": []},
                 "fan": {"nominal": {k: [] for k in QKEYS}, "real": {k: [] for k in QKEYS}}}
        for m in months:
            k_m = float(kpi_path[m])
            if m <= mid_T:
                v = float(hist_nom[m])
                chart["hist"]["nominal"].append(rnd(v))
                chart["hist"]["real"].append(rnd(v / k_m * kpi_A))
            else:
                chart["hist"]["nominal"].append(None)
                chart["hist"]["real"].append(None)
            if mid_T <= m <= anchor:
                lvl = bpi_T * np.exp(L(f_of(m)))
                chart["now"]["nominal"].append(rnd(lvl * k_m / kpi_mid))
                chart["now"]["real"].append(rnd(lvl * kpi_A / kpi_mid))
            else:
                chart["now"]["nominal"].append(None)
                chart["now"]["real"].append(None)
            if m >= anchor:
                f = f_of(m)
                dl = L(f) - L(f_a)
                co = change_offsets(O(f), O(f_a))
                for k, o in zip(QKEYS, co):
                    real = base_nom_A * np.exp(dl + o)
                    chart["fan"]["real"][k].append(rnd(real))
                    chart["fan"]["nominal"][k].append(rnd(real * k_m / kpi_A))
            else:
                for k in QKEYS:
                    chart["fan"]["real"][k].append(None)
                    chart["fan"]["nominal"][k].append(None)

        horizons, decomp = [], {}
        for label, months_ahead in USER_HORIZONS:
            m_t = anchor + months_ahead
            f_t = f_of(m_t)
            dl = L(f_t) - L(f_a)
            co = change_offsets(O(f_t), O(f_a))
            infl = float(kpi_path[m_t]) / kpi_A
            row = {"label": label, "months": months_ahead, "target": mlabel(m_t), "real": {}, "nominal": {}}
            for k, o in zip(QKEYS, co):
                row["real"][k] = rnd((np.exp(dl + o) - 1) * 100, 1)
                row["nominal"][k] = rnd((np.exp(dl + o) * infl - 1) * 100, 1)
            horizons.append(row)

            real_total = np.exp(dl) - 1
            scale = real_total / dl if abs(dl) > 1e-9 else 1.0
            keys = set().union(*[set(c) for c in contrib[r]])
            items = defaultdict(float)
            sens_d = defaultdict(float)
            for j in keys:
                cj = [c.get(j, 0.0) for c in contrib[r]]
                sj = [s.get(j, 0.0) for s in sens[r]]
                items[_alias(j)] += (float(interp(cj, f_t)) - float(interp(cj, f_a))) * scale * 100
                sens_d[_alias(j)] += (float(interp(sj, f_t)) - float(interp(sj, f_a))) * scale * 100
            normal_pp = (float(interp(normal[r], f_t)) - float(interp(normal[r], f_a))) * scale * 100
            h_near = int(np.clip(round(f_t), 1, H_MAX))
            decomp[str(months_ahead)] = {
                "normal_pp": rnd(normal_pp, 2),
                "items": sorted([{"key": k, "label": FEATURE_LABELS.get(k, k), "pp": rnd(v, 2)}
                                 for k, v in items.items()], key=lambda d: -abs(d["pp"] or 0)),
                "real_pp": rnd(real_total * 100, 2),
                "nominal_pp": rnd((np.exp(dl) * infl - 1) * 100, 2),
                "inflation_pp": rnd((np.exp(dl) * infl - np.exp(dl)) * 100, 2),
                "weights": {COMP_LABELS[c]: rnd(v, 3) for c, v in wmaps[r][h_near].items()},
                "sensitivity": {k: rnd(v, 3) for k, v in sens_d.items()},
            }

        # Forklaringsvariablene
        inputs = []
        g = panel.xs(r, level="region")
        for key, meta in FEATURE_META.items():
            if key not in all_feats or (key == "relval_gap" and r == "norge"):
                continue
            hist = g[key][(g.index >= INPUT_HISTORY_FROM) & (g.index <= T)]
            cur_raw = float(Xo.loc[(r, T), key]) if np.isfinite(Xo.loc[(r, T), key]) else None
            proj = ar1_projection(g[key][g.index >= pd.Period("1995Q1", "Q")], cur_raw)
            step = meta["step"]
            inputs.append({
                "key": key, "label": FEATURE_LABELS[key], "group": meta["group"], "unit": meta["unit"],
                "explain": meta["explain"],
                "used_in": [lbl for lbl, fs in (("Hovedmodellen", feats_long), ("Utvidet modell", feats_ext)) if key in fs],
                "history": {"quarters": [qlabel(q) for q in hist.index],
                            "values": [rnd(display_value(key, v), 3) for v in hist.values]},
                "current": rnd(display_value(key, cur_raw), 3),
                "current_note": "oppdatert med månedstall" if key in ("real_rate_at", "d_rate4", "credit_real") and overrides else "",
                "normal": rnd(display_value(key, rmeans.get((r, key))), 3),
                "projection": None if proj is None else {
                    "quarters": [qlabel(T + i) for i in range(1, len(proj) + 1)],
                    "values": [rnd(display_value(key, v), 3) for v in proj]},
                "contrib": {str(mo): next((it["pp"] for it in decomp[str(mo)]["items"] if it["key"] == key), 0.0)
                            for _, mo in USER_HORIZONS},
                "sensitivity": {"step_text": meta["step_text"],
                                "pp": {str(mo): rnd((decomp[str(mo)]["sensitivity"].get(key) or 0) * step, 2)
                                       for _, mo in USER_HORIZONS}},
            })

        # Treffsikkerhet: anslag laget h kvartaler før mot faktisk utvikling
        accuracy = {}
        for _, mo in USER_HORIZONS:
            h = ACCURACY_H[mo]
            e = ens[(ens.region == r) & (ens.h == h)].sort_values("origin")
            if e.empty:
                continue
            tq = [o + h for o in e.origin]
            infl = np.array([nat.loc[t, "kpi_q"] / nat.loc[o, "kpi_q"] for o, t in zip(e.origin, tq)])
            lo, hi = offsets[r][h][1], offsets[r][h][5]
            a_r, p_r = np.exp(e.actual.values) - 1, np.exp(e.pred.values) - 1
            l_r, h_r = np.exp(e.pred.values + lo) - 1, np.exp(e.pred.values + hi) - 1
            inside = (e.actual.values >= e.pred.values + lo) & (e.actual.values <= e.pred.values + hi)
            rw = bt[(bt.region == r) & (bt.h == h) & (bt.component == "uendret")]
            to_list = lambda a: [rnd(v * 100, 2) for v in a]  # noqa: E731
            accuracy[str(mo)] = {
                "h": h, "quarters": [qlabel(t) for t in tq],
                "real": {"actual": to_list(a_r), "pred": to_list(p_r), "p10": to_list(l_r), "p90": to_list(h_r)},
                "nominal": {"actual": to_list((1 + a_r) * infl - 1), "pred": to_list((1 + p_r) * infl - 1),
                            "p10": to_list((1 + l_r) * infl - 1), "p90": to_list((1 + h_r) * infl - 1)},
                "mae": rnd(np.mean(np.abs(e.error)) * 100, 2),
                "mae_uendret": rnd(np.mean(np.abs(rw.error)) * 100, 2) if len(rw) else None,
                "coverage": rnd(inside.mean() * 100, 1),
                "n": int(len(e)),
            }

        out_regions[r] = {
            "name": REGIONS[r]["name"], "last_quarter": qlabel(T), "chart": chart,
            "horizons": horizons, "decomposition": decomp, "inputs": inputs, "accuracy": accuracy,
        }

    weights_out = {}
    for _, mo in USER_HORIZONS:
        h_near = int(np.clip(round(f_of(anchor + mo)), 1, H_MAX))
        weights_out[str(mo)] = {COMP_LABELS[c]: rnd(v, 3) for c, v in global_w[h_near].items()}

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asof_month": str(asof),
        "anchor_month": mlabel(anchor),
        "last_quarter": qlabel(T),
        "monthly_update": overrides,
        "assumptions": {
            "inflation_now": rnd(infl_now * 100, 2),
            "inflation_target": INFLATION_TARGET * 100,
            "inflation_convergence_months": INFLATION_CONVERGENCE_MONTHS,
        },
        "features": {"long": feats_long, "extended": feats_ext, "dropped": dropped},
        "groups": GROUPS,
        "weights": weights_out,
        "accuracy_overall": accuracy_table(ens, bt),
        "regions": out_regions,
    }


def sanitize(obj):
    """Erstatt NaN med None, og stopp hvis selve prognosetallene mangler.

    Et feilet prognosesteg lar forrige gyldige prognose bli liggende på nettsiden.
    """
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (pd.Period,)):
            return str(x)
        return x
    obj = clean(obj)
    for r, d in obj["regions"].items():
        vals = [v for hz in d["horizons"] for v in hz["nominal"].values()]
        if any(v is None for v in vals):
            raise ValueError(f"Prognosen for {r} inneholder manglende verdier – avbryter")
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "output"))
    ap.add_argument("--asof", help="Måned, f.eks. 2026-10 (standard: inneværende måned)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asof = pd.Period(args.asof, "M") if args.asof else pd.Period(dt.date.today(), "M")

    result = sanitize(run(args.data_dir, asof))
    out = Path(args.out_dir)
    (out / "archive").mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out / "forecast.json").write_text(text, encoding="utf-8")
    (out / "archive" / f"forecast-{asof}.json").write_text(text, encoding="utf-8")
    for r, d in result["regions"].items():
        h1 = next(x for x in d["horizons"] if x["months"] == 12)
        log.info("%-10s 12 mnd fra %s: nominelt %+.1f %% (80 %%: %+.1f til %+.1f)", r,
                 result["anchor_month"], h1["nominal"]["p50"], h1["nominal"]["p10"], h1["nominal"]["p90"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
