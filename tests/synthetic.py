"""Syntetiske rådata med samme seriesnavn og format som fetch.py produserer.

Brukes til å teste hele kjeden (panel -> modell -> nettside) uten nettverk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _qcode(p):
    return f"{p.year}K{p.quarter}"


def _mcode(p):
    return f"{p.year}M{p.month:02d}"


def make_raw(raw_dir, last_q="2026Q2", last_m="2026-08", seed=1):
    rng = np.random.default_rng(seed)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    months = pd.period_range("1979-01", last_m, freq="M")
    quarters = pd.period_range("1979Q4", last_q, freq="Q")
    rows = []

    # KPI: ca. 2,5 % årlig inflasjon med støy
    infl = 0.025 / 12 + rng.normal(0, 0.002, len(months))
    kpi = pd.Series(100 * np.exp(np.cumsum(infl)), index=months)
    kpi = kpi / kpi[kpi.index.year == 2025].mean() * 100
    rows += [("kpi", _mcode(m), v) for m, v in kpi.items()]

    # Rente: fallende trend + sykler
    t = np.arange(len(quarters))
    rate = np.clip(12 - 0.045 * t + 1.5 * np.sin(t / 9) + rng.normal(0, 0.2, len(t)), 1.5, None)
    rate_q = pd.Series(rate, index=quarters)
    rows += [("rente_utest_total", _qcode(q), v) for q, v in rate_q.items()]
    rows += [("rente_utest_bolig", _qcode(q), v - 0.3) for q, v in rate_q.items() if q.year >= 2002]
    for m in months[months >= pd.Period("2013-12", "M")]:
        rows.append(("rente_ny_total", _mcode(m), rate_q.get(m.asfreq("Q"), rate_q.iloc[-1]) + 0.1))

    # K2 og byggekostnader
    k2 = 6 + 2 * np.sin(np.arange(len(months)) / 40) + rng.normal(0, 0.3, len(months))
    rows += [("k2_hush_vekst", _mcode(m), v) for m, v in zip(months, k2) if m >= pd.Period("1986-12", "M")]
    bki = pd.Series(50 * np.exp(np.cumsum(0.03 / 12 + rng.normal(0, 0.002, len(months)))), index=months)
    rows += [("byggekostnad", _mcode(m), v) for m, v in bki.items()]

    # Boligpriser: realvekst avhenger av rentenivå og momentum
    bq = pd.period_range("1992Q1", last_q, freq="Q")
    kpi_q = kpi.groupby(kpi.index.asfreq("Q")).mean()
    regions = {"norge": 0.0, "oslo": 0.004, "bergen": 0.0, "trondheim": -0.002, "stavanger": -0.003}
    common = []
    g_prev = 0.0
    for q in bq:
        g = 0.002 + 0.4 * g_prev - 0.0015 * (rate_q[q] - 5) + rng.normal(0, 0.012)
        common.append(g)
        g_prev = g
    for reg, extra in regions.items():
        idio = rng.normal(0, 0.008, len(bq))
        lp_real = np.cumsum(np.array(common) + extra + idio)
        nominal = np.exp(lp_real) * kpi_q.reindex(bq).values
        nominal = nominal / nominal[(bq.year == 2015)].mean() * 100
        rows += [(f"bpi_{reg}", _qcode(q), v) for q, v in zip(bq, nominal)]

    # Befolkning, igangsatte, inntekt (kortere historikk)
    pq = pd.period_range("1997Q4", last_q, freq="Q")
    pops = {"norge": 4.4e6, "oslo": 5.0e5, "baerum": 1.0e5, "bergen": 2.3e5, "trondheim": 1.5e5,
            "stavanger": 1.1e5}
    for reg, p0 in pops.items():
        pop = p0 * np.exp(np.cumsum(0.0025 + rng.normal(0, 0.001, len(pq))))
        rows += [(f"bef_{reg}", _qcode(q), v) for q, v in zip(pq, pop)]
        if reg != "norge":
            rows += [(f"igang_{reg}", _qcode(q), max(0, p0 * 0.002 + rng.normal(0, p0 * 0.0004)))
                     for q in pq if q.year >= 2000]
    rows += [("igang_norge", _qcode(q), 7000 + rng.normal(0, 800)) for q in pq if q.year >= 2000]
    iq = pd.period_range("1999Q1", last_q, freq="Q")
    disp = 2e5 * np.exp(np.cumsum(0.012 + rng.normal(0, 0.006, len(iq))))
    rows += [("disp_inntekt_sa", _qcode(q), v) for q, v in zip(iq, disp)]
    pd.DataFrame(rows, columns=["series", "period", "value"]).to_csv(raw / "ssb_series.csv", index=False)

    days = pd.bdate_range("1991-01-01", pd.Period(last_m, "M").end_time.normalize())
    pol = pd.Series(np.interp(np.arange(len(days)), np.linspace(0, len(days), len(rate_q)), rate_q.values - 2),
                    index=days)
    pol.rename("value").rename_axis("date").to_csv(raw / "market_styringsrente.csv")
    brent = pd.Series(60 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))), index=days)
    brent.rename("value").rename_axis("date").to_csv(raw / "market_brent_usd.csv")
    return raw
