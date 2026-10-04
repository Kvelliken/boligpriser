"""Prognosemodell for boligpriser.

Metode
------
For hver horisont h = 1..H kvartaler estimeres en egen («direkte») modell for
realprisendringen log(P[t+h]) - log(P[t]). Da trenger vi ikke prognoser for
forklaringsvariablene – de inngår med verdiene som er kjent i dag.

Komponenter i ensemblet:
  ridge_lang     Ridge-regresjon på et panel av alle regionene, med variabler
                 som har lang historikk (fra 1990-tallet).
  ridge_utvidet  Som over, pluss befolkningsvekst, boligbygging og inntekt
                 (kortere historikk, fra ca. 2000).
  drift          Regionens historiske gjennomsnittlige realprisvekst.
  uendret        Ingen endring i realpris (tilfeldig gange).

Vektene settes ut fra hvor godt hver komponent har truffet historisk, krympet
mot like vekter. Usikkerhetsspennene er empiriske: de bygger på ensemblets
faktiske bom i en historisk test der modellen bare så data som var kjent på
hvert tidspunkt.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .features import (EXTENDED_FEATURES, LONG_FEATURES)

log = logging.getLogger("model")

H_MAX = 24
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
COMPONENTS = ["ridge_lang", "ridge_utvidet", "drift", "uendret"]
RIDGE_LAMBDA = 8.0
BACKTEST_START = pd.Period("2005Q1", "Q")
MIN_TRAIN_ROWS = 60
MIN_REALIZED_FOR_WEIGHTS = 8
WEIGHT_SHRINK = 0.5
DRIFT_WINDOW = 80


# -- Ridge ---------------------------------------------------------------------
class Ridge:
    """Ridge med ustraffede regionkonstanter. Variabler standardiseres."""

    def __init__(self, lam=RIDGE_LAMBDA):
        self.lam = lam

    def fit(self, X, y, regions):
        self.cols = list(X.columns)
        self.mu = X.mean()
        self.sd = X.std(ddof=0).replace(0, 1.0)
        Z = ((X - self.mu) / self.sd).to_numpy()
        self.regions = sorted(set(regions))
        D = np.column_stack([(np.asarray(regions) == r).astype(float) for r in self.regions])
        A = np.hstack([Z, D])
        P = np.diag([self.lam] * Z.shape[1] + [0.0] * D.shape[1])
        beta = np.linalg.solve(A.T @ A + P, A.T @ np.asarray(y, float))
        self.coef = pd.Series(beta[:Z.shape[1]], index=self.cols)
        self.const = dict(zip(self.regions, beta[Z.shape[1]:]))
        return self

    def standardized(self, X):
        return (X[self.cols] - self.mu) / self.sd

    def predict(self, X, regions):
        Z = self.standardized(X).to_numpy()
        const = np.array([self.const.get(r, np.mean(list(self.const.values()))) for r in regions])
        return Z @ self.coef.to_numpy() + const

    def contributions(self, X):
        return self.standardized(X).mul(self.coef, axis=1)


# -- Hjelpefunksjoner ----------------------------------------------------------
def available_features(panel, wanted):
    """Dropp variabler som mangler helt (f.eks. hvis en kilde feilet)."""
    return [f for f in wanted if f in panel and panel[f].notna().any()]


def targets(panel, h):
    lp = panel["lp_real"]
    return lp.groupby(level="region").shift(-h) - lp


def _training_rows(panel, feats, y, h, origin):
    periods = panel.index.get_level_values("period")
    ok = (periods + h <= origin) & np.isfinite(y) & np.isfinite(panel[feats]).all(axis=1)
    return panel[ok], y[ok]


def fit_component(name, panel, feats_long, feats_ext, h, origin):
    """Tilpass én komponent med data kjent ved `origin`. Returnerer predict-funksjon eller None."""
    y = targets(panel, h)
    if name in ("ridge_lang", "ridge_utvidet"):
        feats = feats_long if name == "ridge_lang" else feats_ext
        if not feats:
            return None
        Xtr, ytr = _training_rows(panel, feats, y, h, origin)
        if len(Xtr) < MIN_TRAIN_ROWS:
            return None
        regs = Xtr.index.get_level_values("region")
        model = Ridge().fit(Xtr[feats], ytr, regs)

        def predict(Xo, model=model, feats=feats):
            out = pd.Series(np.nan, index=Xo.index)
            ok = np.isfinite(Xo[feats].astype(float)).all(axis=1)
            if ok.any():
                out[ok] = model.predict(Xo.loc[ok, feats], Xo.index[ok].get_level_values("region"))
            return out
        predict.model = model
        predict.feats = feats
        return predict
    if name == "drift":
        mom = panel["mom1"]
        periods = panel.index.get_level_values("period")
        hist = mom[(periods <= origin) & (periods > origin - DRIFT_WINDOW)]
        means = hist.groupby(level="region").mean()

        def predict(Xo, means=means, h=h):
            regs = Xo.index.get_level_values("region")
            return pd.Series([means.get(r, np.nan) * h for r in regs], index=Xo.index)
        return predict
    if name == "uendret":
        return lambda Xo: pd.Series(0.0, index=Xo.index)
    raise ValueError(name)


def origin_rows(panel, origin):
    try:
        return panel.xs(origin, level="period", drop_level=False)
    except KeyError:
        return panel.iloc[0:0]


# -- Historisk test ------------------------------------------------------------
def backtest(panel, feats_long, feats_ext, last_obs, h_max=H_MAX):
    """Rullerende test. Returnerer DataFrame med prognoser og utfall per komponent."""
    rows = []
    origins = pd.period_range(max(BACKTEST_START, panel.index.get_level_values("period").min() + 8),
                              last_obs - 1, freq="Q")
    lp = panel["lp_real"]
    for h in range(1, h_max + 1):
        y = targets(panel, h)
        for origin in origins:
            if origin + h > last_obs:
                break
            Xo = origin_rows(panel, origin)
            if Xo.empty:
                continue
            actual = y.reindex(Xo.index)
            for comp in COMPONENTS:
                fn = fit_component(comp, panel, feats_long, feats_ext, h, origin)
                if fn is None:
                    continue
                pred = fn(Xo)
                for idx, p in pred.items():
                    if np.isfinite(p) and np.isfinite(actual[idx]):
                        rows.append((idx[0], origin, h, comp, float(p), float(actual[idx])))
    bt = pd.DataFrame(rows, columns=["region", "origin", "h", "component", "pred", "actual"])
    bt["error"] = bt["actual"] - bt["pred"]
    log.info("Historisk test: %d prognoser", len(bt))
    return bt


def weights_at(bt, h, origin):
    """Vekter for horisont h ved `origin`, basert på bom som var kjent da."""
    realized = bt[(bt.h == h) & (bt.origin + h <= origin)]
    comps = COMPONENTS
    equal = pd.Series(1.0 / len(comps), index=comps)
    if realized.origin.nunique() < MIN_REALIZED_FOR_WEIGHTS:
        return equal
    # Felles utvalg: bare (region, origin) der alle komponenter finnes
    piv = realized.pivot_table(index=["region", "origin"], columns="component", values="error")
    piv = piv.dropna(axis=1, how="all")
    common = piv.dropna()
    if len(common) < MIN_REALIZED_FOR_WEIGHTS:
        common = piv
    mse = (common ** 2).mean()
    inv = (1 / mse.replace(0, np.nan)).fillna(0)
    w = (inv / inv.sum()).reindex(comps).fillna(0)
    return WEIGHT_SHRINK * equal + (1 - WEIGHT_SHRINK) * w


def combine(preds, w):
    """Vektet snitt over tilgjengelige komponenter (renormaliserte vekter)."""
    avail = {c: p for c, p in preds.items() if p is not None and np.isfinite(p)}
    if not avail:
        return np.nan
    ws = np.array([w.get(c, 0.0) for c in avail])
    if ws.sum() <= 0:
        ws = np.ones(len(avail))
    return float(np.dot(ws / ws.sum(), list(avail.values())))


def ensemble_errors(bt):
    """Ensemblets bom i den historiske testen, med vekter kjent på hvert tidspunkt."""
    out = []
    for (h, origin), g in bt.groupby(["h", "origin"]):
        w = weights_at(bt, h, origin)
        for region, gr in g.groupby("region"):
            preds = dict(zip(gr.component, gr.pred))
            p = combine(preds, w)
            out.append((region, origin, h, p, gr.actual.iloc[0]))
    ens = pd.DataFrame(out, columns=["region", "origin", "h", "pred", "actual"])
    ens["error"] = ens["actual"] - ens["pred"]
    return ens


def interval_offsets(ens, regions, h_max=H_MAX):
    """Empiriske kvantiler for bom per region og horisont (sentrert på median).

    Bom skaleres med regionens typiske bom, kvantilene beregnes på det samlede
    utvalget, og skaleres tilbake. Spennene tvinges til ikke å krympe med h.
    """
    res = {r: {} for r in regions}
    for h in range(1, h_max + 1):
        e = ens[ens.h == h]
        if e.empty:
            continue
        sig_all = float(np.sqrt((e.error ** 2).mean()))
        sig = e.groupby("region").error.apply(lambda x: float(np.sqrt((x ** 2).mean())) if len(x) >= 10 else np.nan)
        z = e.apply(lambda r: r.error / (sig.get(r.region) if np.isfinite(sig.get(r.region, np.nan)) else sig_all),
                    axis=1)
        qz = np.quantile(z, QUANTILES)
        qz = qz - qz[QUANTILES.index(0.5)]
        for r in regions:
            s = sig.get(r, np.nan)
            s = s if np.isfinite(s) else sig_all
            res[r][h] = qz * s
    # Fyll manglende horisonter (lengst ut) ved å skalere siste kjente med sqrt(h)
    for r in regions:
        known = sorted(res[r])
        if not known:
            continue
        for h in range(1, h_max + 1):
            if h not in res[r]:
                base = max(k for k in known if k <= h) if any(k <= h for k in known) else known[0]
                res[r][h] = res[r][base] * np.sqrt(h / base)
        lo = np.minimum.accumulate([res[r][h][:3] for h in range(1, h_max + 1)], axis=0)
        hi = np.maximum.accumulate([res[r][h][4:] for h in range(1, h_max + 1)], axis=0)
        for i, h in enumerate(range(1, h_max + 1)):
            res[r][h] = np.concatenate([lo[i], [0.0], hi[i]])
    return res


def accuracy_table(ens, bt):
    """Gjennomsnittlig absolutt bom for ensemble og «uendret» per horisont (prosentpoeng)."""
    rows = []
    rw = bt[bt.component == "uendret"]
    for h, g in ens.groupby("h"):
        r = rw[rw.h == h]
        rows.append({"h": int(h), "mae_model": float(np.mean(np.abs(g.error)) * 100),
                     "mae_uendret": float(np.mean(np.abs(r.error)) * 100) if len(r) else None,
                     "n": int(g.origin.nunique())})
    return rows
