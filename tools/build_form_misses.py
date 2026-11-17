"""Compute and cache walk-forward offense/defense misses for many seasons.

    python tools/build_form_misses.py [--seasons 2018 2019 ...]

Writes data/form_misses.csv (one row per offense-side-of-a-game, both
statistics). ~13s per season for the play-by-play features, then it is
cached in-process; the 2020 COVID season is skipped by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.form import KINDS, add_defense_misses, walk_forward_misses  # noqa: E402

OUT = ROOT / "data" / "form_misses.csv"
DEFAULT_SEASONS = (2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    frames = []
    for season in args.seasons:
        for kind in KINDS:
            frame = add_defense_misses(walk_forward_misses(season, kind))
            print(f"{season} {kind:8s} {len(frame):5d} sides", flush=True)
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(result)} rows)")


if __name__ == "__main__":
    main()
