"""Klienter for Norges Banks SDMX-API og FRED (oljepris)."""
from __future__ import annotations

import csv
import io
import time

import pandas as pd
import requests

UA = {"User-Agent": "boligprognose/0.1 (GitHub Actions)"}


def _get_text(url, params, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=90)
            if r.ok:
                return r.text
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url} feilet: {last}")


def parse_sdmx_csv(text):
    """Norges Banks CSV -> Series (dato -> verdi). Skilletegn detekteres."""
    first_line = text.splitlines()[0] if text else ""
    sep = csv.Sniffer().sniff(first_line, delimiters=";,\t").delimiter if first_line else ";"
    df = pd.read_csv(io.StringIO(text), sep=sep)
    cols = {c.upper(): c for c in df.columns}
    if "TIME_PERIOD" not in cols or "OBS_VALUE" not in cols:
        raise ValueError(f"Uventede kolonner fra Norges Bank: {list(df.columns)}")
    s = pd.Series(
        pd.to_numeric(df[cols["OBS_VALUE"]].astype(str).str.replace(",", "."), errors="coerce").values,
        index=pd.to_datetime(df[cols["TIME_PERIOD"]]),
    ).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def fetch_norgesbank(base_url, flow, key, start="1990-01-01"):
    text = _get_text(f"{base_url}/{flow}/{key}",
                     {"format": "csv", "startPeriod": start, "locale": "en"})
    return parse_sdmx_csv(text)


def parse_fred_csv(text):
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    val_col = df.columns[1]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]))
    return s.dropna().sort_index()


def fetch_fred(base_url, series_id):
    return parse_fred_csv(_get_text(base_url, {"id": series_id}))
