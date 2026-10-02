"""Verifiserer datakildene og skriver en markdown-rapport.

For hver SSB-tabell: tittel, dimensjoner med eksempelverdier, tidsspenn,
hvilke verdier filtrene treffer, og – hvis tabellnummeret mangler eller
tittelen ikke stemmer – kandidater fra SSBs tabellsøk.
"""
from __future__ import annotations

import argparse
import re
import sys

from .fetch import load_config
from .filters import extract_series, matches
from .market import fetch_fred, fetch_norgesbank
from .ssb import SSBClient, build_selection, dimension_catalog, find_time_dim


def _fmt_cats(cats, n=40):
    shown = "; ".join(f"`{c}` {lab}" for c, lab in cats[:n])
    more = f" … (+{len(cats) - n})" if len(cats) > n else ""
    return shown + more


def verify_ssb(cfg, out):
    client = SSBClient(cfg["base_urls"], lang=cfg.get("lang", "no"))
    out.append("## SSB")
    for name, spec in cfg["tables"].items():
        out.append(f"\n### {name}")
        table = spec.get("table")
        needs_search = not table
        if table:
            try:
                meta = client.metadata(table)
                title = meta.get("label", "")
                ok_title = bool(re.search(spec.get("expect_title", ".*"), title, re.I))
                out.append(f"- Tabell **{table}**: {title} {'✅' if ok_title else '⚠️ tittel matcher ikke'}")
                out.append(f"- API-base: `{client._base}`")
                dims = dimension_catalog(meta)
                tdim = find_time_dim(meta)
                t = dims[tdim]
                out.append(f"- Tid (`{tdim}`): {t[0][0]} – {t[-1][0]} ({len(t)} perioder)")
                for d, cats in dims.items():
                    if d == tdim:
                        continue
                    out.append(f"- `{d}` ({len(cats)}): {_fmt_cats(cats)}")
                    if d in (spec.get("select") or {}):
                        hit = [(c, lab) for c, lab in cats if matches(spec["select"][d], c, lab)]
                        out.append(f"  - filter treffer {len(hit)}: {_fmt_cats(hit, 20)}")
                _, cells = build_selection(dims, tdim, spec.get("select") or {}, spec.get("time_from"))
                out.append(f"- Estimert uttak: {cells:,} celler")
                if spec.get("series") and ok_title:
                    df, _ = client.fetch_table(table, spec.get("select"), spec.get("time_from"))
                    for sname, filt in spec["series"].items():
                        try:
                            s = extract_series(df, filt).dropna()
                            zeros = int((s == 0).sum())
                            out.append(f"  - serie `{sname}`: {s.index.min()} – {s.index.max()}, "
                                       f"n={s.size}, første {s.iloc[0]:.2f}, siste {s.iloc[-1]:.2f}"
                                       f"{f', {zeros} nuller ⚠️' if zeros else ''} ✅")
                        except Exception as exc:  # noqa: BLE001
                            out.append(f"  - serie `{sname}`: ❌ {exc}")
                needs_search = not ok_title
            except Exception as exc:  # noqa: BLE001
                out.append(f"- ❌ Tabell {table} feilet: {exc}")
                needs_search = True
        if needs_search and spec.get("search"):
            queries = spec["search"] if isinstance(spec["search"], list) else [spec["search"]]
            for query in queries:
                try:
                    hits = client.search(query)
                    out.append(f"- Søk «{query}» ga {len(hits)} treff:")
                    for h in hits:
                        out.append(f"  - **{h['id']}** {h['label']} ({h['first']} – {h['last']}, "
                                   f"{h['time_unit']})")
                except Exception as exc:  # noqa: BLE001
                    out.append(f"- ❌ Søk «{query}» feilet: {exc}")


def verify_market(nb_cfg, fred_cfg, out):
    out.append("\n## Norges Bank og FRED")
    for name, spec in (nb_cfg or {}).get("series", {}).items():
        try:
            s = fetch_norgesbank(nb_cfg["base_url"], spec["flow"], spec["key"], nb_cfg.get("start"))
            out.append(f"- `{name}` ({spec['flow']}/{spec['key']}): {s.index.min().date()} – "
                       f"{s.index.max().date()}, siste {s.iloc[-1]:.3f} ✅")
        except Exception as exc:  # noqa: BLE001
            out.append(f"- `{name}` ({spec['flow']}/{spec['key']}): ❌ {exc}")
    for name, sid in (fred_cfg or {}).get("series", {}).items():
        try:
            s = fetch_fred(fred_cfg["base_url"], sid)
            out.append(f"- `{name}` (FRED {sid}): {s.index.min().date()} – {s.index.max().date()}, "
                       f"siste {s.iloc[-1]:.2f} ✅")
        except Exception as exc:  # noqa: BLE001
            out.append(f"- `{name}` (FRED {sid}): ❌ {exc}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", default="verify-report.md")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = ["# Verifisering av datakilder", ""]
    verify_ssb(cfg["ssb"], out)
    verify_market(cfg.get("norgesbank"), cfg.get("fred"), out)
    text = "\n".join(out) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
