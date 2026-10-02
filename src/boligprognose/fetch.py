"""Henter alle konfigurerte kilder og lagrer rådata i data/raw.

Feil i én kilde stopper ikke de andre. Eksisterende filer overskrives bare
ved vellykket henting, slik at en midlertidig feil ikke sletter data.
Prosessen avslutter med feilkode hvis en kritisk kilde feiler.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from .filters import extract_series
from .market import fetch_fred, fetch_norgesbank
from .ssb import SSBClient

log = logging.getLogger("fetch")
ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    with open(path or ROOT / "config" / "sources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_ssb(cfg, raw_dir, status):
    client = SSBClient(cfg["base_urls"], lang=cfg.get("lang", "no"))
    series_frames = []
    for name, spec in cfg["tables"].items():
        entry = {"source": "SSB", "table": spec.get("table"), "critical": bool(spec.get("critical"))}
        status[name] = entry
        if not spec.get("table"):
            entry.update(ok=False, note="Ikke verifisert (table: null) – hoppes over")
            continue
        try:
            df, info = client.fetch_table(spec["table"], spec.get("select"), spec.get("time_from"))
            entry.update(title=info["title"], cells=info["cells"], rows=len(df))
            df.to_csv(raw_dir / f"ssb_{name}.csv", index=False)
            for sname, filt in (spec.get("series") or {}).items():
                s = extract_series(df, filt).dropna()
                series_frames.append(pd.DataFrame(
                    {"series": sname, "period": s.index, "value": s.values}))
                status[sname] = {"source": "SSB", "table": spec["table"], "ok": True,
                                 "first": str(s.index.min()), "last": str(s.index.max()),
                                 "n": int(s.size), "critical": entry["critical"]}
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001 – vi vil fange alt og rapportere
            log.exception("SSB %s feilet", name)
            entry.update(ok=False, error=str(exc)[:500])
    if series_frames:
        pd.concat(series_frames).to_csv(raw_dir / "ssb_series.csv", index=False)


def fetch_market(nb_cfg, fred_cfg, raw_dir, status):
    jobs = []
    for name, spec in (nb_cfg or {}).get("series", {}).items():
        jobs.append((name, "Norges Bank", spec.get("critical", False),
                     lambda s=spec: fetch_norgesbank(nb_cfg["base_url"], s["flow"], s["key"],
                                                     nb_cfg.get("start", "1990-01-01"))))
    for name, sid in (fred_cfg or {}).get("series", {}).items():
        jobs.append((name, "FRED", False, lambda i=sid: fetch_fred(fred_cfg["base_url"], i)))
    for name, source, critical, fn in jobs:
        try:
            s = fn()
            if s.empty:
                raise RuntimeError("Tom serie")
            s.rename("value").rename_axis("date").to_csv(raw_dir / f"market_{name}.csv")
            status[name] = {"source": source, "ok": True, "critical": critical,
                            "first": str(s.index.min().date()), "last": str(s.index.max().date()),
                            "n": int(s.size)}
        except Exception as exc:  # noqa: BLE001
            log.exception("%s %s feilet", source, name)
            status[name] = {"source": source, "ok": False, "critical": critical,
                            "error": str(exc)[:500]}


def status_markdown(status):
    lines = ["## Datastatus", "", "| Serie | Kilde | Status | Første | Siste | Merknad |",
             "|---|---|---|---|---|---|"]
    for name, e in status.items():
        flag = "✅" if e.get("ok") else ("❌" if e.get("critical") or e.get("error") else "⏸️")
        note = e.get("error") or e.get("note") or e.get("title") or ""
        lines.append(f"| {name} | {e.get('source','')} | {flag} | {e.get('first','')} | "
                     f"{e.get('last','')} | {str(note)[:120].replace('|','/')} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)

    status = {}
    fetch_ssb(cfg["ssb"], raw_dir, status)
    fetch_market(cfg.get("norgesbank"), cfg.get("fred"), raw_dir, status)

    report = {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "series": status}
    (data_dir / "processed" / "status.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = status_markdown(status)
    print(md)
    (data_dir / "processed" / "status.md").write_text(md + "\n", encoding="utf-8")

    failed = [n for n, e in status.items() if e.get("critical") and not e.get("ok")]
    if failed:
        log.error("Kritiske kilder feilet: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
