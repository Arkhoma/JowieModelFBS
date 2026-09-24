"""Is the postseason gap to the line systematic (fixable) or just noise?

Reads data/benchmark_vs_market.csv. For postseason vs late regular season:
  bias    mean(actual - ours): do we lean toward one side?
  slope   actual ~ ours: < 1 means our margins are too extreme
  shrink  leave-one-season-out: does scaling our margin by a fitted
          factor actually beat the unscaled prediction on held-out bowls?

    python tools/check_postseason_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.scorecard import POSTSEASON  # noqa: E402


def slope(x: pd.Series, y: pd.Series) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "benchmark_vs_market.csv")
    post = frame[frame.season_type == POSTSEASON]
    late = frame[(frame.season_type != POSTSEASON) & (frame.week > 8)]

    print(f"{'slice':>6} {'n':>5} {'bias':>6} {'slope':>6} {'lineSlope':>9} "
          f"{'|ours|':>6} {'|line|':>6}")
    for label, f in (("late", late), ("post", post)):
        print(f"{label:>6} {len(f):>5} {(f.actual - f.ours).mean():>+6.2f} "
              f"{slope(f.ours, f.actual):>6.2f} {slope(f.line, f.actual):>9.2f} "
              f"{f.ours.abs().mean():>6.1f} {f.line.abs().mean():>6.1f}")

    gains = []
    for season in sorted(post.season.unique()):
        train, test = post[post.season != season], post[post.season == season]
        k = slope(train.ours, train.actual)
        base = np.abs(test.actual - test.ours).mean()
        shrunk = np.abs(test.actual - k * test.ours).mean()
        gains.append((shrunk - base) * len(test))
        print(f"  hold out {season}: k={k:.2f}  MAE {base:.2f} -> {shrunk:.2f}")
    print(f"Shrink, held out: {sum(gains) / len(post):+.3f} pts/game")


if __name__ == "__main__":
    main()
