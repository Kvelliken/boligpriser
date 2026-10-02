"""Bygger den statiske nettsiden til _site/ (web/ + output/forecast.json)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build(web_dir, forecast_file, out_dir):
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(web_dir, out)
    forecast = Path(forecast_file)
    if forecast.exists():
        shutil.copy(forecast, out / "forecast.json")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", default=str(ROOT / "web"))
    ap.add_argument("--forecast", default=str(ROOT / "output" / "forecast.json"))
    ap.add_argument("--out", default=str(ROOT / "_site"))
    args = ap.parse_args(argv)
    out = build(args.web, args.forecast, args.out)
    print(f"Nettside bygget i {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
