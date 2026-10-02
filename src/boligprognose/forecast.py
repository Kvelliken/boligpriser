"""Kjører prognosemodellen og skriver output/forecast.json.

Bruk: python -m boligprognose.forecast [--asof 2026-10]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (EXTENDED_FEATURES, FEATURE_LABELS, LONG_FEATURES, REGIONS,
                       build_panel, origin_overrides)
from .model import (COMPONENTS, H_MAX, QUANTILES, accuracy_table, backtest, combine,
                    ensemble_errors, fit_component, interval_offsets, origin_rows,
                    available_features, weights_at)
from .panel import load_series

log = logging.getLogger("forecast")
ROOT = Path(__file__).resolve().parents[2]

USER_HORIZONS = [("6 mnd", 6), ("1 år", 12), ("3 år", 36), ("5 år", 60)]
INFLATION_TARGET = 0.02
INFLATION_CONVERGENCE_MONTHS = 24
HISTORY_FROM = pd.Period("2005Q1", "Q")
QKEYS = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]


def qlabel(p):
    return f"{p.year}K{p.quarter}"


def inflation_path(kpi_monthly, last_q, h_max, asof):
    """KPI per kvartal fremover: dagens tolvmånedersvekst glir lineært mot 2 % over 24 mnd."""
    last_m = kpi_monthly.index.max()
    infl_now = float(kpi_monthly.iloc[-1] / kpi_monthly[kpi_monthly.index <= last_m - 12].iloc[-1] - 1)
    end_m = (last_q + h_max).asfreq("M", how="end")
    months = pd.period_range(last_m + 1, end_m, freq="M")
    level, vals = float(kpi_monthly.iloc[-1]), {}
    for i, m in enumerate(months, start=1):
        w = min(i / INFLATION_CONVERGENCE_MONTHS, 1.0)
        annual = (1 - w) * infl_now + w * INFLATION_TARGET
        level *= (1 + annual) ** (1 / 12)
        vals[m] = level
    path = pd.concat([kpi_monthly, pd.Series(vals)])
    path.index = pd.PeriodIndex(path.index, freq="M")
    q = path.groupby(path.index.asfreq("Q")).mean()
    return q, infl_now


def run(data_dir, asof):
    series = load_series(Path(data_dir) / "raw")
    panel, nat = build_panel(series)
    feats_long = available_features(panel, LONG_FEATURES)
    feats_ext = available_features(panel, EXTENDED_FEATURES)
    dropped = sorted(set(EXTENDED_FEATURES) - set(feats_ext))
    if dropped:
        log.warning("Variabler uten data (droppet): %s", dropped)

    regions = [r for r in REGIONS if r in set(panel.index.get_level_values("region"))]
    last_obs = series["bpi_norge"].index.max()
    log.info("Siste boligprisdata: %s, prognosemåned: %s", last_obs, asof)

    bt = backtest(panel, feats_long, feats_ext, last_obs)
    ens = ensemble_errors(bt)
    offsets = interval_offsets(ens, regions)

    # Startpunkt med ferskere månedsdata for nasjonale variabler
    Xo = origin_rows(panel, last_obs).copy()
    overrides = origin_overrides(series, nat, last_obs, asof)
    for k in ("real_rate_at", "d_rate4", "credit_real"):
        if k in overrides and k in Xo:
            Xo[k] = overrides[k]

    # Prognoser per horisont
    point = {r: {} for r in regions}
    weights, drivers_model = {}, None
    h_driver = 4
    for h in range(1, H_MAX + 1):
        w = weights_at(bt, h, last_obs + H_MAX + 1)
        weights[h] = w.round(3).to_dict()
        preds = {}
        for comp in COMPONENTS:
            fn = fit_component(comp, panel, feats_long, feats_ext, h, last_obs)
            preds[comp] = None if fn is None else fn(Xo)
            if comp == "ridge_lang" and h == h_driver and fn is not None:
                drivers_model = fn
        for r in regions:
            idx = (r, last_obs)
            comp_preds = {c: (p[idx] if p is not None and idx in p.index else None) for c, p in preds.items()}
            point[r][h] = combine(comp_preds, w)

    kpi_q_path, infl_now = inflation_path(series["kpi"], last_obs, H_MAX, asof)
    kpi_T = kpi_q_path[last_obs]

    out_regions = {}
    for r in regions:
        bpi = series[REGIONS[r]["bpi"]]
        hist = bpi[bpi.index >= HISTORY_FROM]
        kpi_hist = nat["kpi_q"].reindex(hist.index)
        real_hist = hist / kpi_hist * kpi_T
        bpi_T = float(bpi[last_obs])
        fq, nominal, real = [], {k: [] for k in QKEYS}, {k: [] for k in QKEYS}
        for h in range(1, H_MAX + 1):
            q = last_obs + h
            fq.append(qlabel(q))
            infl_factor = float(kpi_q_path[q] / kpi_T)
            for k, off in zip(QKEYS, offsets[r][h]):
                lr = point[r][h] + off
                real[k].append(round(bpi_T * np.exp(lr), 2))
                nominal[k].append(round(bpi_T * np.exp(lr) * infl_factor, 2))
        horizons = []
        for label, months in USER_HORIZONS:
            target_q = (asof + months).asfreq("Q")
            h = (target_q - last_obs).n
            h = int(min(max(h, 1), H_MAX))
            infl_factor = float(kpi_q_path[last_obs + h] / kpi_T)
            row = {"label": label, "months": months, "quarter": qlabel(last_obs + h), "h": h,
                   "real": {}, "nominal": {}}
            for k, off in zip(QKEYS, offsets[r][h]):
                lr = point[r][h] + off
                row["real"][k] = round((np.exp(lr) - 1) * 100, 1)
                row["nominal"][k] = round((np.exp(lr) * infl_factor - 1) * 100, 1)
            horizons.append(row)
        drivers = []
        if drivers_model is not None:
            xo = Xo.loc[[(r, last_obs)]]
            if xo[drivers_model.feats].notna().all(axis=1).iloc[0]:
                contrib = drivers_model.model.contributions(xo[drivers_model.feats]).iloc[0] * 100
                structural_zero = {"oil_g4_stav"} if r != "stavanger" else set()
                if r == "norge":
                    structural_zero.add("relval_gap")
                for f, v in contrib.sort_values(key=np.abs, ascending=False).items():
                    if f in structural_zero:
                        continue
                    drivers.append({"key": f, "label": FEATURE_LABELS.get(f, f), "pp": round(float(v), 2)})
        out_regions[r] = {
            "name": REGIONS[r]["name"],
            "last_quarter": qlabel(last_obs),
            "last_index": round(bpi_T, 2),
            "history": {"quarters": [qlabel(p) for p in hist.index],
                        "nominal": [round(float(v), 2) for v in hist.values],
                        "real": [round(float(v), 2) for v in real_hist.values]},
            "forecast": {"quarters": fq, "nominal": nominal, "real": real},
            "horizons": horizons,
            "drivers": drivers,
        }

    acc = accuracy_table(ens, bt)
    acc_user = []
    first = next(iter(out_regions.values()))
    for hz in first["horizons"]:
        row = next((a for a in acc if a["h"] == hz["h"]), None)
        if row:
            acc_user.append({"label": hz["label"], **row})

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "asof_month": str(asof),
        "last_quarter": qlabel(last_obs),
        "monthly_update": overrides,
        "assumptions": {
            "inflation_now": round(infl_now * 100, 2),
            "inflation_target": INFLATION_TARGET * 100,
            "inflation_convergence_months": INFLATION_CONVERGENCE_MONTHS,
            "real_base": f"Realpriser er i kroneverdi for {qlabel(last_obs)}",
        },
        "features": {"long": feats_long, "extended": feats_ext, "dropped": dropped},
        "weights": {str(h): w for h, w in weights.items()},
        "accuracy": acc_user,
        "regions": out_regions,
    }


def sanitize(obj):
    """Erstatt NaN med None, og stopp hvis selve prognosetallene mangler.

    Et feilet prognosesteg lar forrige gyldige prognose bli liggende på nettsiden.
    """
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x
    obj = clean(obj)
    for r, d in obj["regions"].items():
        vals = [v for arr in d["forecast"]["nominal"].values() for v in arr]
        vals += [v for hz in d["horizons"] for v in hz["nominal"].values()]
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
    text = json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False)
    (out / "forecast.json").write_text(text, encoding="utf-8")
    (out / "archive" / f"forecast-{asof}.json").write_text(text, encoding="utf-8")
    for r, d in result["regions"].items():
        h1 = next(x for x in d["horizons"] if x["months"] == 12)
        log.info("%-10s 1 år: nominelt %+.1f %% (80 %%: %+.1f til %+.1f)", r,
                 h1["nominal"]["p50"], h1["nominal"]["p10"], h1["nominal"]["p90"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
