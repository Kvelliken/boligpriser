"""Forklaringsvariabler for prognosemodellen (kvartalsvis, per region).

Alle variabler er definert slik at de er kjent ved utgangen av kvartalet t,
og brukes til å forutsi realprisendringen fra t til t+h.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .panel import to_quarterly

log = logging.getLogger("features")

BASE_YEAR = 2015

REGIONS = {
    "norge":     {"name": "Hele landet",    "bpi": "bpi_norge",
                  "pop": ["bef_norge"], "starts": ["igang_norge"]},
    "oslo":      {"name": "Oslo med Bærum", "bpi": "bpi_oslo",
                  "pop": ["bef_oslo", "bef_baerum"], "starts": ["igang_oslo", "igang_baerum"]},
    "bergen":    {"name": "Bergen",         "bpi": "bpi_bergen",
                  "pop": ["bef_bergen"], "starts": ["igang_bergen"]},
    "trondheim": {"name": "Trondheim",      "bpi": "bpi_trondheim",
                  "pop": ["bef_trondheim"], "starts": ["igang_trondheim"]},
    "stavanger": {"name": "Stavanger",      "bpi": "bpi_stavanger",
                  "pop": ["bef_stavanger"], "starts": ["igang_stavanger"]},
}

# Skattesats for rentefradrag (alminnelig inntekt).
TAX_RATE = {2013: 0.28, 2014: 0.27, 2015: 0.27, 2016: 0.25, 2017: 0.24, 2018: 0.23}
TAX_RATE_FROM_2019 = 0.22

FEATURE_LABELS = {
    "mom1": "Prisutvikling siste kvartal",
    "mom4": "Prisutvikling siste år",
    "qval_gap": "Boligpris mot byggekostnader",
    "relval_gap": "Prisnivå mot resten av landet",
    "real_rate_at": "Realrente etter skatt",
    "d_rate4": "Renteendring siste år",
    "credit_real": "Kredittvekst hos husholdningene",
    "oil_g4": "Oljepris",
    "oil_g4_stav": "Oljepris (Stavanger-effekt)",
    "pop_g4": "Befolkningsvekst",
    "starts_pc": "Igangsatte boliger per innbygger",
    "income_g4": "Vekst i husholdningenes realinntekt",
}

# Visningsinformasjon for nettsiden. `kind`: "log" vises som (e^x - 1) * 100 %,
# "pct" som x * 100, "level" uendret. `step` er endringen i følsomhetsberegningen (råenheter).
FEATURE_META = {
    "real_rate_at": {"group": "renter", "kind": "pct", "unit": "%", "step": 0.01,
                     "step_text": "1 prosentpoeng høyere",
                     "explain": "Boliglånsrenten etter rentefradrag og fratrukket prisveksten siste år. Viser hva lånet egentlig koster."},
    "d_rate4": {"group": "renter", "kind": "pct", "unit": "pp", "step": 0.01,
                "step_text": "1 prosentpoeng høyere",
                "explain": "Hvor mye boliglånsrenten har endret seg det siste året. Renteoppganger demper markedet en stund etterpå."},
    "credit_real": {"group": "renter", "kind": "pct", "unit": "%", "step": 0.01,
                    "step_text": "1 prosentpoeng høyere",
                    "explain": "Veksten i husholdningenes gjeld siste tolv måneder, fratrukket prisveksten."},
    "pop_g4": {"group": "befolkning", "kind": "log", "unit": "%", "step": 0.005,
               "step_text": "0,5 prosentpoeng høyere",
               "explain": "Befolkningsveksten i området siste år, inkludert flytting og innvandring."},
    "starts_pc": {"group": "befolkning", "kind": "level", "unit": "per 1000", "step": 1.0,
                  "step_text": "1 bolig mer per 1000 innbyggere",
                  "explain": "Igangsettingstillatelser for boliger siste tolv måneder per 1000 innbyggere. Mange nye boliger øker tilbudet."},
    "income_g4": {"group": "okonomi", "kind": "log", "unit": "%", "step": 0.01,
                  "step_text": "1 prosentpoeng høyere",
                  "explain": "Vekst i husholdningenes disponible realinntekt per innbygger siste år (hele landet)."},
    "oil_g4": {"group": "okonomi", "kind": "log", "unit": "%", "step": 0.10,
               "step_text": "10 prosent høyere",
               "explain": "Endringen i oljeprisen (Brent, dollar) siste år. Påvirker norsk økonomi, og særlig Stavanger."},
    "qval_gap": {"group": "prisniva", "kind": "log", "unit": "%", "step": 0.10,
                 "step_text": "10 prosent høyere",
                 "explain": "Boligprisene sammenlignet med byggekostnadene, målt mot snittet de siste ti årene. Lavt nivå betyr at det er dyrt å bygge nytt i forhold til å kjøpe brukt."},
    "relval_gap": {"group": "prisniva", "kind": "log", "unit": "%", "step": 0.10,
                   "step_text": "10 prosent høyere",
                   "explain": "Prisene i byen sammenlignet med hele landet, målt mot snittet de siste ti årene. Negativt betyr at byen har falt bak."},
    "mom1": {"group": "prisniva", "kind": "log", "unit": "%", "step": 0.01,
             "step_text": "1 prosentpoeng høyere",
             "explain": "Den inflasjonsjusterte prisendringen siste kvartal. Boligmarkedet har fart, så trender varer gjerne en stund."},
    "mom4": {"group": "prisniva", "kind": "log", "unit": "%", "step": 0.05,
             "step_text": "5 prosentpoeng høyere",
             "explain": "Den inflasjonsjusterte prisendringen siste år."},
}
GROUPS = {
    "renter": "Renter og kreditt",
    "befolkning": "Befolkning og bygging",
    "okonomi": "Inntekt og olje",
    "prisniva": "Prisnivå og trend",
}
# Variabler som slås sammen med en annen i visningen
DISPLAY_ALIAS = {"oil_g4_stav": "oil_g4"}


def display_value(key, x):
    kind = FEATURE_META[key]["kind"]
    if x is None or not np.isfinite(x):
        return None
    if kind == "log":
        return float((np.exp(x) - 1) * 100)
    if kind == "pct":
        return float(x * 100)
    return float(x)


def ar1_projection(s, start_value, steps=20, min_obs=16):
    """Enkel AR(1)-fremskrivning mot historisk snitt (kun til illustrasjon)."""
    s = s[np.isfinite(s.astype(float))]
    if len(s) < min_obs or start_value is None or not np.isfinite(start_value):
        return None
    x, y = s.values[:-1].astype(float), s.values[1:].astype(float)
    X = np.column_stack([np.ones_like(x), x])
    try:
        c, phi = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if not (np.isfinite(c) and np.isfinite(phi)):
        return None
    phi = float(np.clip(phi, 0.0, 0.97))
    mean = float(s.mean()) if phi == 0 else float(np.mean(y) - phi * np.mean(x)) / (1 - phi)
    out, v = [], start_value
    for _ in range(steps):
        v = mean + phi * (v - mean)
        out.append(float(v))
    return out


LONG_FEATURES = ["mom1", "mom4", "qval_gap", "relval_gap", "real_rate_at",
                 "d_rate4", "credit_real", "oil_g4", "oil_g4_stav"]
EXTENDED_FEATURES = LONG_FEATURES + ["pop_g4", "starts_pc", "income_g4"]

GAP_WINDOW, GAP_MIN = 40, 20   # 10 års glidende snitt, minst 5 år


# Serier der verdiene er nivåer (indekser, beløp, antall) og logaritmen tas.
LEVEL_PREFIXES = ("bpi_", "kpi", "byggekostnad", "brent_usd", "bef_", "disp_inntekt")


def clean_levels(series):
    """Behandle verdier <= 0 i nivåserier som manglende, og logg hvor de finnes.

    SSB og andre kilder kan levere 0 der tallet egentlig mangler. Logaritmen av 0
    gir minus uendelig, som ellers ville ødelagt hele beregningen.
    """
    out = {}
    for name, s in series.items():
        if name.startswith(LEVEL_PREFIXES):
            bad = s[~(s > 0)]
            if len(bad):
                periods = [str(p) for p in bad.index]
                shown = ", ".join(periods[:6]) + (f" … (+{len(periods) - 6})" if len(periods) > 6 else "")
                log.warning("Serie %s har %d verdier <= 0 eller tomme (%s) – behandles som manglende",
                            name, len(bad), shown)
                s = s[s > 0]
        out[name] = s
    return out


def tax_rate(year):
    if year >= 2019:
        return TAX_RATE_FROM_2019
    return TAX_RATE.get(year, 0.28)


def splice(primary, fallback):
    """Forleng `primary` bakover med `fallback`, nivåjustert på overlappen."""
    if primary is None or primary.dropna().empty:
        return fallback
    if fallback is None or fallback.dropna().empty:
        return primary
    overlap = primary.dropna().index.intersection(fallback.dropna().index)
    shift = (primary[overlap] - fallback[overlap]).mean() if len(overlap) else 0.0
    out = (fallback + shift).combine_first(primary)
    out.loc[primary.dropna().index] = primary.dropna()
    return out.sort_index()


def _q(series, name):
    s = series.get(name)
    return None if s is None or s.empty else to_quarterly(s)


def complete_quarter_mean(s):
    """Kvartalssnitt av en månedsserie, bare for kvartaler med alle tre måneder."""
    counts = s.groupby(s.index.asfreq("Q")).count()
    means = s.groupby(s.index.asfreq("Q")).mean()
    return means.where(counts == 3)


def kpi_quarterly(series):
    kpi = series["kpi"]
    return complete_quarter_mean(kpi) if kpi.index.freqstr.startswith("M") else kpi


def mortgage_rate(series):
    """Boliglånsrente (utestående, husholdninger), forlenget bakover med totale utlån."""
    bolig, total = _q(series, "rente_utest_bolig"), _q(series, "rente_utest_total")
    rate = splice(bolig, total)
    if rate is None:
        pol = series.get("styringsrente")
        if pol is None:
            raise ValueError("Mangler både boliglånsrente og styringsrente")
        log.warning("Bruker styringsrente + 2 pp som rentemål")
        rate = to_quarterly(pol) + 2.0
    return rate


def national_frame(series):
    kpi_q = kpi_quarterly(series)
    infl4 = np.log(kpi_q).diff(4)
    rate = mortgage_rate(series)
    taxes = pd.Series([tax_rate(p.year) for p in rate.index], index=rate.index)
    df = pd.DataFrame({"kpi_q": kpi_q, "infl4": infl4, "rate": rate})
    df["real_rate_at"] = df["rate"] / 100 * (1 - taxes.reindex(df.index)) - df["infl4"]
    df["d_rate4"] = df["rate"].diff(4) / 100
    k2 = _q(series, "k2_hush_vekst")
    df["credit_real"] = (k2.reindex(df.index) / 100 - df["infl4"]) if k2 is not None else np.nan
    brent = _q(series, "brent_usd")
    df["oil_g4"] = np.log(brent).diff(4).reindex(df.index) if brent is not None else np.nan
    bki = _q(series, "byggekostnad")
    df["bki"] = bki.reindex(df.index) if bki is not None else np.nan
    disp = _q(series, "disp_inntekt_sa")
    pop = _q(series, "bef_norge")
    if disp is None or pop is None:
        log.warning("Inntektsvekst kan ikke beregnes: mangler %s",
                    " og ".join(n for n, s in (("disp_inntekt_sa", disp), ("bef_norge", pop)) if s is None))
    if disp is not None and pop is not None:
        real_pc = np.log(disp / (df["kpi_q"].reindex(disp.index) * pop.reindex(disp.index)))
        df["income_g4"] = real_pc.diff(4).reindex(df.index)
    else:
        df["income_g4"] = np.nan
    base = df["kpi_q"][df.index.year == BASE_YEAR].mean()
    df["deflator"] = df["kpi_q"] / base
    return df


def _gap(s):
    return s - s.rolling(GAP_WINDOW, min_periods=GAP_MIN).mean()


def _sum_series(series, names):
    parts = [_q(series, n) for n in names]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))


def region_frame(region, series, nat):
    meta = REGIONS[region]
    bpi = series[meta["bpi"]]
    norge = series["bpi_norge"]
    df = pd.DataFrame(index=nat.index.union(bpi.index))
    df["bpi"] = bpi
    df = df.join(nat)
    df["lp_real"] = np.log(df["bpi"] / df["deflator"])
    df["mom1"] = df["lp_real"].diff(1)
    df["mom4"] = df["lp_real"].diff(4)
    df["qval_gap"] = _gap(np.log(df["bpi"] / df["bki"]))
    rel = np.log(df["bpi"] / norge.reindex(df.index))
    df["relval_gap"] = 0.0 if region == "norge" else _gap(rel)
    df["oil_g4_stav"] = df["oil_g4"] if region == "stavanger" else 0.0
    pop = _sum_series(series, meta["pop"])
    starts = _sum_series(series, meta["starts"])
    df["pop_g4"] = np.log(pop).diff(4).reindex(df.index) if pop is not None else np.nan
    if pop is not None and starts is not None:
        df["starts_pc"] = (starts.rolling(4).sum() / pop * 1000).reindex(df.index)
    else:
        df["starts_pc"] = np.nan
    df["region"] = region
    return df


def build_panel(series):
    """Langt panel (region, kvartal) med alle variabler."""
    series = clean_levels(series)
    nat = national_frame(series).replace([np.inf, -np.inf], np.nan)
    frames = [region_frame(r, series, nat) for r in REGIONS if REGIONS[r]["bpi"] in series]
    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    panel.index.name = "period"
    return panel.reset_index().set_index(["region", "period"]).sort_index(), nat


def origin_overrides(series, nat, origin, asof):
    """Oppdater nasjonale variabler ved prognosestart med ferskere månedsdata.

    Boligprisene er kvartalsvise, men renter, KPI og kreditt kommer månedlig.
    For prognosestarten bruker vi de tre siste tilgjengelige månedene i stedet
    for snittet i kvartal `origin`, slik at prognosen endres når ny informasjon kommer.
    """
    out = {}
    kpi = series.get("kpi")
    if kpi is None or not kpi.index.freqstr.startswith("M"):
        return out
    last_m = min(kpi.index.max(), asof)
    if last_m <= origin.asfreq("M", how="end"):
        return out
    k3 = kpi[kpi.index <= last_m].tail(3).mean()
    k3_prev = kpi[(kpi.index <= last_m - 12)].tail(3).mean()
    infl4 = float(np.log(k3 / k3_prev))
    rate = nat.loc[origin, "rate"]
    ny = series.get("rente_ny_total")
    if ny is not None and not ny.empty:
        q_months = pd.period_range(origin.asfreq("M", how="start"), origin.asfreq("M", how="end"), freq="M")
        base = ny.reindex(q_months).mean()
        latest = ny[ny.index <= asof].tail(3).mean()
        if np.isfinite(base) and np.isfinite(latest):
            rate = rate + (latest - base)
    out["rate"] = float(rate)
    out["real_rate_at"] = float(rate / 100 * (1 - tax_rate(asof.year)) - infl4)
    year_ago = origin - 3
    if year_ago in nat.index and np.isfinite(nat.loc[year_ago, "rate"]):
        out["d_rate4"] = float((rate - nat.loc[year_ago, "rate"]) / 100)
    k2 = series.get("k2_hush_vekst")
    if k2 is not None and not k2.empty:
        out["credit_real"] = float(k2[k2.index <= asof].tail(3).mean() / 100 - infl4)
    out["infl4"] = infl4
    out["last_month"] = str(last_m)
    return out
