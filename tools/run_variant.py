"""Run the production walk-forward benchmark with ONE knob changed.

    python tools/run_variant.py --name hl12 --halflife 12
    python tools/compare_benchmarks.py data/benchmark_vs_market.csv data/variants/hl12.csv

Keeps experiments out of the production CSV the site reads. Every knob
defaults to the shipped value, so `--name base` reproduces production.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_vs_market as bench  # noqa: E402
from cfbrank.epa_ridge import get_ep_model  # noqa: E402

OUT = ROOT / "data" / "variants"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--halflife", type=float)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(bench.SEASONS))
    args = parser.parse_args()

    if args.halflife is not None:
        bench.MARGIN_PARAMS = {**bench.MARGIN_PARAMS, "halflife": args.halflife}

    market, totals = bench.load_home_lines(), bench.load_total_lines()
    get_ep_model()
    frame = pd.concat([bench.walk_season(s, market, totals)
                       for s in args.seasons], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name}.csv"
    frame.to_csv(path, index=False)
    print(f"{args.name}: {len(frame)} games -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
