"""Does our prediction carry information the closing line lacks?

Fits actual ~ a*ours + b*line leave-one-season-out on the benchmark CSV
and scores the blend against the line alone on held-out seasons. If the
blend beats the line out of sample, the model knows something Vegas has
not priced in; if not, the in-sample stack weight was noise.

Usage: python tools/check_market_blend.py [data/benchmark_vs_market.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "benchmark_vs_market.csv"
    frame = pd.read_csv(path)
    blend = np.zeros(len(frame))
    print(f"{'held out':>8} {'w_ours':>7} {'w_line':>7} {'delta vs line':>14}")
    for season in sorted(frame.season.unique()):
        train, test = frame.season != season, frame.season == season
        x = frame.loc[train, ["ours", "line"]].values
        coef, *_ = np.linalg.lstsq(x, frame.loc[train, "actual"].values,
                                   rcond=None)
        blend[test.values] = frame.loc[test, ["ours", "line"]].values @ coef
        d = (np.abs(frame.actual - blend) - np.abs(frame.actual - frame.line))[test]
        print(f"{season:>8} {coef[0]:>+7.3f} {coef[1]:>+7.3f} {d.mean():>+14.3f}")
    delta = np.abs(frame.actual - blend) - np.abs(frame.actual - frame.line)
    ci = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
    print(f"\npooled {len(frame)} games: blend minus line = "
          f"{delta.mean():+.3f} +/- {ci:.3f} pts/game (negative = we add info)")


if __name__ == "__main__":
    main()
