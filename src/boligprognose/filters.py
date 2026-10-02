"""Regex-filtre mot kode/tekst og uttrekk av enkeltserier fra lange tabeller."""
from __future__ import annotations

import re

import pandas as pd


def matches(spec, code, label):
    """True hvis kode eller tekst matcher filteret.

    spec kan være en regex-streng eller {match: regex, exclude: regex}.
    """
    if isinstance(spec, dict):
        inc, exc = spec.get("match", ".*"), spec.get("exclude")
    else:
        inc, exc = spec, None
    hay = (str(code), str(label))
    ok = any(re.search(inc, h, re.IGNORECASE) for h in hay)
    if ok and exc:
        ok = not any(re.search(exc, h, re.IGNORECASE) for h in hay)
    return ok


class AmbiguousSeries(ValueError):
    pass


def extract_series(df, filters):
    """Plukk ut én tidsserie fra en lang SSB-tabell.

    `filters` er {dim: filter}. Spesialnøkler:
      _sum_over: [dim, ...]  summerer over disse dimensjonene først
                             (f.eks. alle bygningstyper når tabellen mangler «i alt»).
      _combine:  hva som skjer når flere serier fortsatt matcher per periode:
                 "coalesce" (første ikke-tomme, for kommuner med endrede koder)
                 eller "sum". Uten `_combine` er flere treff en feil.
    """
    filters = dict(filters or {})
    combine = filters.pop("_combine", None)
    sum_over = filters.pop("_sum_over", None) or []
    mask = pd.Series(True, index=df.index)
    for dim, spec in filters.items():
        if dim not in df.columns:
            raise KeyError(f"Dimensjonen {dim!r} finnes ikke. Har: {list(df.columns)}")
        lab_col = f"{dim}_label" if f"{dim}_label" in df.columns else dim
        mask &= pd.Series([matches(spec, c, lab) for c, lab in zip(df[dim], df[lab_col])],
                          index=df.index)
    sub = df[mask]
    if sub.empty:
        raise ValueError(f"Ingen rader matcher {filters}")

    key_cols = [c for c in df.columns
                if c not in ("period", "value") and not c.endswith("_label")]
    if sum_over:
        # Summer over angitte dimensjoner (f.eks. alle bygningstyper) før videre kombinering.
        key_cols = [c for c in key_cols if c not in sum_over]
        sub = (sub.groupby(key_cols + ["period"], sort=False)["value"]
                  .sum(min_count=1).reset_index())
    groups = sub.groupby(key_cols, sort=False) if key_cols else [((), sub)]
    parts = [g.set_index("period")["value"] for _, g in groups]
    if len(parts) == 1:
        return parts[0]
    if combine == "coalesce":
        # Første ikke-tomme verdi. En 0 erstattes hvis en annen kode har en verdi
        # ulik 0 for samme periode (ugyldig kommunekode i den perioden).
        out = parts[0]
        for p in parts[1:]:
            p = p.reindex(out.index.union(p.index))
            out = out.reindex(p.index)
            replace = out.isna() | ((out == 0) & p.notna() & (p != 0))
            out = out.where(~replace, p)
        return out
    if combine == "sum":
        return pd.concat(parts, axis=1).sum(axis=1, min_count=1)

    label_cols = [f"{c}_label" for c in key_cols if f"{c}_label" in sub.columns] or key_cols
    hits = sub[label_cols].drop_duplicates().to_dict("records")
    raise AmbiguousSeries(f"{len(parts)} serier matcher {filters}: {hits[:6]}")
