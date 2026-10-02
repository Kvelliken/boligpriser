"""Bygger analysepaneler fra rådata.

Utdata i data/processed:
  panel_quarterly.csv  – alle serier på kvartalsfrekvens, uten utfylling.
                         Månedsserier er kvartalssnitt; ufullstendige kvartal
                         er markert i availability.csv. Inneholder også
                         reelle boligpriser (deflatert med KPI, 2015 = 100).
  panel_monthly.csv    – alle serier på månedsfrekvens der manglende verdier
                         fylles med siste tilgjengelige observasjon
                         (kvartalstall brukes til nytt kvartal foreligger).
  availability.csv     – siste observasjon og alder per serie ved kjøretidspunkt.
I tillegg lagres en datert kopi av kvartalspanelet i data/vintages/, slik at
vi senere kan backteste modellen på data slik de faktisk forelå.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from .ssb import parse_period

log = logging.getLogger("panel")
ROOT = Path(__file__).resolve().parents[2]
BASE_YEAR = 2015


def load_series(raw_dir):
    raw_dir = Path(raw_dir)
    out = {}
    ssb_file = raw_dir / "ssb_series.csv"
    if ssb_file.exists():
        df = pd.read_csv(ssb_file, dtype={"period": str})
        for name, g in df.groupby("series"):
            idx = [parse_period(p) for p in g["period"]]
            s = pd.Series(g["value"].values, index=idx).dropna()
            s = s[[i is not None for i in s.index]]
            freqs = {p.freqstr[0] for p in s.index}
            if len(freqs) != 1:
                log.warning("Serie %s har blandet frekvens %s – hoppes over", name, freqs)
                continue
            s.index = pd.PeriodIndex(s.index)
            out[name] = s.sort_index()
    for f in sorted(raw_dir.glob("market_*.csv")):
        name = f.stem.removeprefix("market_")
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")["value"]
        out[name] = d.groupby(d.index.to_period("M")).mean()
    return out


def _freq(s):
    return s.index.freqstr[0]  # 'M', 'Q' eller 'Y'


def to_quarterly(s):
    f = _freq(s)
    if f == "Q":
        return s
    if f == "M":
        return s.groupby(s.index.asfreq("Q")).mean()
    if f == "Y":
        return _year_to_q(s)
    raise ValueError(f)


def _year_to_q(s):
    rows = {}
    for y, v in s.items():
        for q in range(1, 5):
            rows[pd.Period(f"{y.year}Q{q}", "Q")] = v
    return pd.Series(rows).sort_index()


def to_monthly_step(s):
    """Legg hver observasjon på alle måneder den dekker."""
    f = _freq(s)
    if f == "M":
        return s
    rows = {}
    for p, v in s.items():
        for m in pd.period_range(p.start_time, p.end_time, freq="M"):
            rows[m] = v
    return pd.Series(rows).sort_index()


def months_in_quarter_observed(s):
    """For månedsserier: antall måneder med data per kvartal."""
    return s.groupby(s.index.asfreq("Q")).count()


def build_quarterly(series):
    cols = {name: to_quarterly(s) for name, s in series.items()}
    panel = pd.DataFrame(cols).sort_index()
    if "kpi" in series:
        counts = months_in_quarter_observed(series["kpi"])
        kpi_q = panel["kpi"].where(counts.reindex(panel.index).fillna(0) == 3)
        base = kpi_q[kpi_q.index.year == BASE_YEAR].mean()
        deflator = kpi_q / base
        for name in [c for c in panel.columns if c.startswith("bpi_")]:
            panel[f"{name}_real"] = panel[name] / deflator
    panel.index.name = "period"
    return panel


def build_monthly(series, asof):
    cols = {}
    for name, s in series.items():
        m = to_monthly_step(s)
        cols[name] = m
    panel = pd.DataFrame(cols).sort_index()
    idx = pd.period_range(panel.index.min(), asof, freq="M")
    panel = panel.reindex(idx).ffill()
    panel.index.name = "month"
    return panel


def availability(series, asof):
    rows = []
    for name, s in series.items():
        last = s.index.max()
        last_month = last.asfreq("M", how="end")
        age = (asof - last_month).n
        rows.append({"series": name, "freq": _freq(s), "first": str(s.index.min()),
                     "last": str(last), "age_months": age})
    return pd.DataFrame(rows).sort_values("series")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--asof", help="Måned, f.eks. 2026-09 (standard: inneværende måned)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_dir = Path(args.data_dir)
    asof = pd.Period(args.asof, "M") if args.asof else pd.Period(dt.date.today(), "M")
    series = load_series(data_dir / "raw")
    if not series:
        raise SystemExit("Ingen rådata funnet – kjør fetch først")

    proc = data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    q = build_quarterly(series)
    m = build_monthly(series, asof)
    av = availability(series, asof)
    q.to_csv(proc / "panel_quarterly.csv", float_format="%.6g")
    m.to_csv(proc / "panel_monthly.csv", float_format="%.6g")
    av.to_csv(proc / "availability.csv", index=False)

    vint = data_dir / "vintages" / dt.date.today().isoformat()
    vint.mkdir(parents=True, exist_ok=True)
    q.to_csv(vint / "panel_quarterly.csv.gz", float_format="%.6g", compression="gzip")
    print(av.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
