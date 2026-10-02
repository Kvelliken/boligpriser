"""Klient for SSBs PxWebApi v2 og parser for JSON-stat2."""
from __future__ import annotations

import logging
import math
import re
import time

import pandas as pd
import requests

from .filters import matches

log = logging.getLogger(__name__)

MAX_CELLS = 700_000          # SSB-grensen er 800 000 celler per uttak
MIN_SECONDS_BETWEEN = 2.1    # SSB tillater 30 spørringer per 60 sekunder
TIME_DIMS = ("Tid", "Time", "tid")


class SSBError(RuntimeError):
    pass


class SSBClient:
    def __init__(self, base_urls, lang="no", session=None):
        self.base_urls = list(base_urls)
        self.lang = lang
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "boligprognose/0.1 (GitHub Actions)"
        self._base = None
        self._last_call = 0.0

    # -- HTTP ---------------------------------------------------------------
    def _get(self, path, params):
        errors = []
        bases = [self._base] if self._base else self.base_urls
        for base in bases:
            url = f"{base}/{path}"
            resp = None
            for attempt in range(3):
                wait = MIN_SECONDS_BETWEEN - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.monotonic()
                try:
                    resp = self.session.get(url, params=params, timeout=90)
                except requests.RequestException as exc:
                    errors.append(f"{url}: {exc}")
                    resp = None
                    time.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(10 * (attempt + 1))
                    continue
                break
            if resp is not None and resp.ok:
                self._base = base
                return resp.json()
            if resp is not None:
                errors.append(f"{resp.url} -> HTTP {resp.status_code}: {resp.text[:300]}")
        raise SSBError("SSB-kall feilet:\n  " + "\n  ".join(errors))

    # -- API ----------------------------------------------------------------
    def search(self, query, page_size=8):
        js = self._get("tables", {"query": query, "lang": self.lang, "pageSize": page_size})
        out = []
        for t in js.get("tables", [])[:page_size]:
            out.append({
                "id": t.get("id"),
                "label": t.get("label"),
                "first": t.get("firstPeriod"),
                "last": t.get("lastPeriod"),
                "time_unit": t.get("timeUnit"),
                "updated": t.get("updated"),
            })
        return out

    def metadata(self, table):
        return self._get(f"tables/{table}/metadata",
                         {"lang": self.lang, "outputformat": "json-stat2"})

    def data(self, table, value_codes):
        params = {"lang": self.lang, "outputformat": "json-stat2"}
        for dim, codes in value_codes.items():
            params[f"valueCodes[{dim}]"] = codes
        return self._get(f"tables/{table}/data", params)

    # -- Høynivå ------------------------------------------------------------
    def fetch_table(self, table, select=None, time_from=None):
        """Hent en tabell med utvalg basert på regex mot kode/tekst.

        Dimensjoner uten filter hentes i sin helhet ("*").
        Returnerer (long DataFrame, info-dict).
        """
        meta = self.metadata(table)
        dims = dimension_catalog(meta)
        time_dim = find_time_dim(meta)
        value_codes, n_cells = build_selection(dims, time_dim, select or {}, time_from)
        if n_cells > MAX_CELLS:
            raise SSBError(f"Tabell {table}: utvalget gir {n_cells:,} celler (> {MAX_CELLS:,}). "
                           "Legg til flere filtre i config.")
        js = self.data(table, value_codes)
        df = jsonstat2_to_frame(js)
        info = {"table": table, "title": meta.get("label"), "cells": n_cells,
                "updated": meta.get("updated")}
        return df, info


# -- Hjelpefunksjoner (rene, testbare) ---------------------------------------
def _category_codes(dim):
    cat = dim["category"]
    idx = cat.get("index")
    if idx is None:
        return list(cat.get("label", {}).keys())
    if isinstance(idx, list):
        return idx
    return sorted(idx, key=idx.get)


def dimension_catalog(js):
    """{dim: [(kode, tekst), ...]} i tabellens rekkefølge."""
    out = {}
    for d in js["id"]:
        dim = js["dimension"][d]
        labels = dim["category"].get("label", {})
        out[d] = [(c, labels.get(c, c)) for c in _category_codes(dim)]
    return out


def find_time_dim(js):
    role = js.get("role", {}) or {}
    if role.get("time"):
        return role["time"][0]
    for d in js["id"]:
        if d in TIME_DIMS:
            return d
    raise SSBError("Fant ingen tidsdimensjon i tabellen")


def build_selection(dims, time_dim, select, time_from=None):
    value_codes, n_cells = {}, 1
    for d, cats in dims.items():
        if d == time_dim:
            value_codes[d] = f"from({time_from})" if time_from else "*"
            n_cells *= len(cats)
            continue
        if d in select:
            chosen = [c for c, lab in cats if matches(select[d], c, lab)]
            if not chosen:
                sample = ", ".join(f"{c}={lab}" for c, lab in cats[:10])
                raise SSBError(f"Filteret for {d} traff ingen verdier. Eksempler: {sample}")
            value_codes[d] = ",".join(chosen)
            n_cells *= len(chosen)
        else:
            value_codes[d] = "*"
            n_cells *= len(cats)
    return value_codes, n_cells


def jsonstat2_to_frame(js):
    """JSON-stat2 -> long DataFrame med kolonnene <dim>, <dim>_label, period, value."""
    ids, sizes = js["id"], js["size"]
    time_dim = find_time_dim(js)
    codes = [_category_codes(js["dimension"][d]) for d in ids]
    values = js.get("value", [])
    total = math.prod(sizes)
    if isinstance(values, dict):  # sparsom representasjon
        dense = [None] * total
        for k, v in values.items():
            dense[int(k)] = v
        values = dense
    if len(values) != total:
        raise SSBError(f"JSON-stat2: forventet {total} verdier, fikk {len(values)}")

    index = pd.MultiIndex.from_product(codes, names=ids)
    df = index.to_frame(index=False)
    df["value"] = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
    for d in ids:
        if d == time_dim:
            continue
        labels = js["dimension"][d]["category"].get("label", {})
        df[f"{d}_label"] = df[d].map(lambda c, lab=labels: lab.get(c, c))
    return df.rename(columns={time_dim: "period"})


_PERIOD_PATTERNS = [
    (re.compile(r"^(\d{4})K([1-4])$"), lambda m: pd.Period(f"{m[1]}Q{m[2]}", "Q")),
    (re.compile(r"^(\d{4})M(\d{2})$"), lambda m: pd.Period(f"{m[1]}-{m[2]}", "M")),
    (re.compile(r"^(\d{4})$"), lambda m: pd.Period(m[1], "Y")),
]


def parse_period(code):
    for pat, fn in _PERIOD_PATTERNS:
        m = pat.match(str(code))
        if m:
            return fn(m)
    return None
