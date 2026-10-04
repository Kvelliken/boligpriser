import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import boligprognose.model as model
from boligprognose.features import splice
from boligprognose.forecast import inflation_path, run
from boligprognose.model import Ridge, interval_offsets


def test_ridge_recovers_signal_with_region_constants():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    regions = np.where(np.arange(n) % 2, "x", "y")
    y = 2 * X["a"] - 1 * X["b"] + np.where(regions == "x", 5, -5) + rng.normal(0, 0.1, n)
    m = Ridge(lam=0.01).fit(X, y, regions)
    pred = m.predict(X, regions)
    assert np.corrcoef(pred, y)[0, 1] > 0.99
    assert m.coef["a"] > 0 > m.coef["b"]


def test_splice_extends_backwards_with_level_shift():
    idx = pd.period_range("2000Q1", "2000Q4", freq="Q")
    long = pd.Series([5.0, 5.0, 5.0, 5.0], index=idx)
    short = pd.Series([4.5, 4.5], index=idx[2:])
    s = splice(short, long)
    assert s.tolist() == [4.5, 4.5, 4.5, 4.5]


def test_inflation_path_converges_to_target():
    m = pd.period_range("2020-01", "2026-08", freq="M")
    kpi = pd.Series(100 * 1.04 ** (np.arange(len(m)) / 12), index=m)
    q, now = inflation_path(kpi, pd.Period("2026Q2", "Q"), 24, pd.Period("2026-10", "M"))
    assert now == pytest.approx(0.04, abs=1e-3)
    late = q[pd.Period("2031Q1", "Q")] / q[pd.Period("2030Q1", "Q")] - 1
    assert late == pytest.approx(0.02, abs=1e-3)


def test_intervals_are_centered_ordered_and_widen():
    rng = np.random.default_rng(1)
    rows = []
    for h in range(1, 9):
        for o in range(40):
            for r in ("a", "b"):
                rows.append((r, o, h, 0.0, rng.normal(0, 0.02 * np.sqrt(h))))
    ens = pd.DataFrame(rows, columns=["region", "origin", "h", "pred", "actual"])
    ens["error"] = ens.actual - ens.pred
    off = interval_offsets(ens, ["a", "b"], h_max=8)
    for h in range(1, 9):
        q = off["a"][h]
        assert q[3] == 0 and all(np.diff(q) >= 0)
    widths = [off["a"][h][5] - off["a"][h][1] for h in range(1, 9)]
    assert all(np.diff(widths) >= 0)


def test_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    from synthetic import make_raw
    make_raw(tmp_path / "raw")
    monkeypatch.setattr(model, "BACKTEST_START", pd.Period("2016Q1", "Q"))
    out = run(tmp_path, pd.Period("2026-10", "M"))
    # «Nå» er siste KPI-måned i de syntetiske dataene
    assert out["anchor_month"] == "2026-08" and out["last_quarter"] == "2026K2"
    assert set(out["regions"]) == {"norge", "oslo", "bergen", "trondheim", "stavanger"}
    oslo = out["regions"]["oslo"]
    assert [h["label"] for h in oslo["horizons"]] == ["6 mnd", "1 år", "3 år", "5 år"]
    assert [h["target"] for h in oslo["horizons"]] == ["2027-02", "2027-08", "2029-08", "2031-08"]
    for hz in oslo["horizons"]:
        vals = [hz["nominal"][k] for k in ("p05", "p10", "p25", "p50", "p75", "p90", "p95")]
        assert vals == sorted(vals)
    # Nedbrytningen skal summere til reell endring, og inflasjonen til nominell
    for mo, d in oslo["decomposition"].items():
        total = d["normal_pp"] + sum(i["pp"] for i in d["items"])
        assert total == pytest.approx(d["real_pp"], abs=0.05), mo
        assert d["real_pp"] + d["inflation_pp"] == pytest.approx(d["nominal_pp"], abs=0.05)
    # Fanen starter i «nå», historikken slutter i midten av siste kjente kvartal
    ch = oslo["chart"]
    i_a = ch["months"].index("2026-08")
    assert ch["fan"]["nominal"]["p50"][i_a] == pytest.approx(ch["now"]["nominal"][i_a])
    assert ch["hist"]["nominal"][ch["months"].index("2026-05")] is not None
    assert ch["hist"]["nominal"][ch["months"].index("2026-06")] is None
    # Variabler fra begge modellene vises, med historikk som slutter i siste kjente kvartal
    keys = {i["key"] for i in oslo["inputs"]}
    assert {"real_rate_at", "pop_g4", "starts_pc", "income_g4"} <= keys
    pop = next(i for i in oslo["inputs"] if i["key"] == "pop_g4")
    assert pop["used_in"] == ["Utvidet modell"] and pop["history"]["quarters"][-1] == "2026K2"
    assert "relval_gap" not in {i["key"] for i in out["regions"]["norge"]["inputs"]}
    # Treffsikkerhet som tidsserie
    acc = oslo["accuracy"]["12"]
    assert len(acc["quarters"]) == len(acc["real"]["actual"]) == len(acc["real"]["pred"]) > 0
    assert 0 <= acc["coverage"] <= 100


def test_change_offsets_shrink_with_known_part():
    from boligprognose.forecast import change_offsets
    t = np.array([-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2])
    a = np.array([-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1])
    c = change_offsets(t, a)
    assert c[3] == 0 and all(np.diff(c) >= 0) and abs(c[0]) < abs(t[0])


def test_sanitize_rejects_missing_forecast_values():
    from boligprognose.forecast import sanitize
    good = {"regions": {"a": {"horizons": [{"nominal": {"p50": 2.0}}]}}, "x": float("nan")}
    assert sanitize(good)["x"] is None
    bad = {"regions": {"a": {"horizons": [{"nominal": {"p50": float("nan")}}]}}}
    with pytest.raises(ValueError):
        sanitize(bad)


def test_zeros_in_level_series_do_not_break_forecast(tmp_path, monkeypatch, caplog):
    """SSB kan levere 0 der tall mangler. Det skal behandles som manglende, ikke krasje."""
    from synthetic import make_raw
    raw = make_raw(tmp_path / "raw")
    df = pd.read_csv(raw / "ssb_series.csv", dtype={"period": str})
    early = df.series.eq("bpi_trondheim") & df.period.str[:4].astype(int).lt(1995)
    df.loc[early, "value"] = 0.0
    df.loc[df.series.eq("bef_baerum") & df.period.str.startswith("2003"), "value"] = 0.0
    df = df[df.series != "disp_inntekt_sa"]
    df.to_csv(raw / "ssb_series.csv", index=False)
    monkeypatch.setattr(model, "BACKTEST_START", pd.Period("2016Q1", "Q"))
    import logging
    with caplog.at_level(logging.WARNING):
        out = run(tmp_path, pd.Period("2026-10", "M"))
    assert "bpi_trondheim" in caplog.text and "disp_inntekt_sa" in caplog.text
    assert "income_g4" in out["features"]["dropped"]
    for r in out["regions"].values():
        for hz in r["horizons"]:
            assert all(np.isfinite(v) for v in hz["nominal"].values())
