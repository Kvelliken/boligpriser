import pandas as pd
import pytest

from boligprognose.filters import AmbiguousSeries, extract_series, matches
from boligprognose.market import parse_fred_csv, parse_sdmx_csv
from boligprognose.panel import availability, build_monthly, build_quarterly
from boligprognose.ssb import (build_selection, dimension_catalog, find_time_dim,
                               jsonstat2_to_frame, parse_period)


def jsonstat_bpi():
    regions = {"0": "Hele landet", "1": "Oslo med Bærum",
               "2": "Møre og Romsdal og Vestland uten Bergen", "3": "Bergen"}
    tid = ["2024K1", "2024K2", "2024K3"]
    return {
        "version": "2.0", "class": "dataset", "label": "Prisindeks for brukte boliger",
        "id": ["Region", "Boligtype", "ContentsCode", "Tid"],
        "size": [4, 1, 1, 3],
        "role": {"time": ["Tid"], "metric": ["ContentsCode"]},
        "dimension": {
            "Region": {"category": {"index": {k: i for i, k in enumerate(regions)}, "label": regions}},
            "Boligtype": {"category": {"index": {"00": 0}, "label": {"00": "Boliger i alt"}}},
            "ContentsCode": {"category": {"index": {"Indeks": 0},
                                          "label": {"Indeks": "Prisindeks (2015=100)"}}},
            "Tid": {"category": {"index": {t: i for i, t in enumerate(tid)},
                                 "label": {t: t for t in tid}}},
        },
        "value": [100, 101, 102, 110, 112, 111, 90, 91, None, 120, 121, 125],
    }


def test_jsonstat_parse_order_and_labels():
    df = jsonstat2_to_frame(jsonstat_bpi())
    assert len(df) == 12
    row = df[(df.Region == "3") & (df.period == "2024K3")].iloc[0]
    assert row.value == 125 and row.Region_label == "Bergen"
    assert df.value.isna().sum() == 1


def test_selection_and_cells():
    js = jsonstat_bpi()
    dims = dimension_catalog(js)
    codes, cells = build_selection(dims, find_time_dim(js), {"Region": "^(Hele landet|Bergen)$"})
    assert codes["Region"] == "0,3" and codes["Tid"] == "*" and cells == 6


def test_filter_does_not_confuse_region_names():
    df = jsonstat2_to_frame(jsonstat_bpi())
    s = extract_series(df, {"Region": "^Bergen"})
    assert list(s.values) == [120, 121, 125]
    with pytest.raises(AmbiguousSeries):
        extract_series(df, {"Region": "Bergen"})  # treffer også "... uten Bergen"


def test_matches_exclude():
    spec = {"match": "indeks", "exclude": "endring"}
    assert matches(spec, "x", "Prisindeks")
    assert not matches(spec, "x", "Indeks, endring fra året før")


def test_coalesce_changed_municipality_codes():
    df = pd.DataFrame({
        "Region": ["4601", "4601", "1201", "1201"],
        "Region_label": ["Bergen", "Bergen", "Bergen (-2019)", "Bergen (-2019)"],
        "period": ["2019K4", "2020K1", "2019K4", "2020K1"],
        "value": [None, 285.0, 283.0, None],
    })
    s = extract_series(df, {"Region": "^Bergen", "_combine": "coalesce"})
    assert s.sort_index().tolist() == [283.0, 285.0]


def test_parse_period():
    assert parse_period("2024K2") == pd.Period("2024Q2", "Q")
    assert parse_period("2024M11") == pd.Period("2024-11", "M")
    assert parse_period("2024") == pd.Period("2024", "Y")
    assert parse_period("2024U01") is None


def test_norgesbank_csv_semicolon():
    text = ("FREQ;Frequency;INSTRUMENT_TYPE;TIME_PERIOD;OBS_VALUE\n"
            "B;Business;KPRA;2025-06-19;4.25\nB;Business;KPRA;2025-06-20;4.25\n")
    s = parse_sdmx_csv(text)
    assert s.iloc[-1] == 4.25 and len(s) == 2


def test_fred_csv_missing_dot():
    s = parse_fred_csv("observation_date,DCOILBRENTEU\n2025-01-02,75.9\n2025-01-03,.\n")
    assert len(s) == 1


def _series_for_panel():
    kpi = pd.Series(100.0, index=pd.period_range("2015-01", "2016-08", freq="M"))
    kpi[kpi.index.year == 2016] = 110.0
    bpi = pd.Series([100.0, 101, 102, 103, 110, 111],
                    index=pd.period_range("2015Q1", "2016Q2", freq="Q"))
    return {"kpi": kpi, "bpi_oslo": bpi}


def test_quarterly_real_prices_and_partial_quarter():
    q = build_quarterly(_series_for_panel())
    assert q.loc[pd.Period("2016Q1", "Q"), "bpi_oslo_real"] == pytest.approx(100.0)
    # 2016K3 har bare to KPI-måneder -> kvartalet er ufullstendig, ingen realpris
    assert pd.isna(q["bpi_oslo_real"].get(pd.Period("2016Q3", "Q")))


def test_monthly_uses_last_quarterly_value():
    asof = pd.Period("2016-09", "M")
    m = build_monthly(_series_for_panel(), asof)
    assert m.index.max() == asof
    assert m.loc[pd.Period("2016-09", "M"), "bpi_oslo"] == 111.0  # siste kvartal videreført
    av = availability(_series_for_panel(), asof).set_index("series")
    assert av.loc["bpi_oslo", "age_months"] == 3 and av.loc["kpi", "age_months"] == 1


class _Resp:
    def __init__(self, status, payload, url):
        self.status_code, self._p, self.url = status, payload, url
        self.ok = status < 400
        self.text = str(payload)

    def json(self):
        return self._p


class _FakeSession:
    """v2 gir 404, v2-beta svarer – klienten skal falle tilbake og huske basen."""
    def __init__(self):
        self.headers, self.calls = {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if "/v2/" in url:
            return _Resp(404, "not found", url)
        return _Resp(200, jsonstat_bpi(), url)


def test_client_falls_back_to_beta_and_selects(monkeypatch):
    import boligprognose.ssb as ssb
    monkeypatch.setattr(ssb, "MIN_SECONDS_BETWEEN", 0)
    sess = _FakeSession()
    client = ssb.SSBClient(["https://x/api/pxwebapi/v2", "https://x/api/pxwebapi/v2-beta"],
                           session=sess)
    df, info = client.fetch_table("07221", {"Region": "^Bergen"})
    assert client._base.endswith("v2-beta")
    data_params = sess.calls[-1][1]
    assert data_params["valueCodes[Region]"] == "3" and data_params["valueCodes[Tid]"] == "*"
    assert info["cells"] == 3 and len(df) == 12  # fake-serveren ignorerer utvalget


def test_config_bpi_filters_match_real_codes():
    """Filtrene i config skal treffe nøyaktig én serie med kodene SSB faktisk bruker (07221)."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config" / "sources.yaml"))
    spec = cfg["ssb"]["tables"]["boligpriser"]
    regions = {"TOTAL": "Hele landet", "001": "Oslo med Bærum", "002": "Stavanger",
               "003": "Bergen", "004": "Trondheim", "009": "Møre og Romsdal og Vestland uten Bergen"}
    contents = {"Boligindeks": "Prisindeks for brukte boliger",
                "SesJustBoligindeks": "Prisindeks for brukte boliger, sesongjustert"}
    rows = [{"Region": r, "Region_label": rl, "Boligtype": "00", "Boligtype_label": "Alle boligtyper",
             "ContentsCode": c, "ContentsCode_label": cl, "period": "2026K2", "value": 1.0}
            for r, rl in regions.items() for c, cl in contents.items()]
    df = pd.DataFrame(rows)
    for name, filt in spec["series"].items():
        assert len(extract_series(df, filt)) == 1, name


def test_sum_over_building_types_then_coalesce_municipalities():
    """05889: summer bygningstyper per kommunekode, deretter slå sammen gamle/nye koder."""
    rows = []
    for code, label, per, vals in [
        ("1201", "Bergen (1972-2019)", "2019K4", (10.0, 5.0)),
        ("1201", "Bergen (1972-2019)", "2020K1", (None, None)),
        ("4601", "Bergen", "2019K4", (None, None)),
        ("4601", "Bergen", "2020K1", (7.0, 3.0)),
    ]:
        for btype, v in zip(("111", "142"), vals):
            rows.append({"Region": code, "Region_label": label, "Byggeareal": btype,
                         "Byggeareal_label": btype, "ContentsCode": "Igangsatte",
                         "ContentsCode_label": "Igangsatte", "period": per, "value": v})
    df = pd.DataFrame(rows)
    s = extract_series(df, {"Region": "^Bergen", "ContentsCode": "^Igangsatte$",
                            "_sum_over": ["Byggeareal"], "_combine": "coalesce"})
    assert s.sort_index().tolist() == [15.0, 10.0]


def test_coalesce_prefers_nonzero_over_zero_from_invalid_code():
    df = pd.DataFrame({
        "Region": ["3201", "3201", "0219", "0219"],
        "Region_label": ["Bærum", "Bærum", "Bærum (-2019)", "Bærum (-2019)"],
        "period": ["2019K4", "2024K1", "2019K4", "2024K1"],
        "value": [0.0, 300.0, 250.0, 0.0],
    })
    s = extract_series(df, {"Region": "^Bærum", "_combine": "coalesce"})
    assert s.sort_index().tolist() == [250.0, 300.0]
