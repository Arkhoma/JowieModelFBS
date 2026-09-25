"""Paired comparison of two benchmark_vs_market.csv runs on the same games.

Usage: python tools/compare_benchmarks.py OLD.csv NEW.csv
Negative deltas mean NEW is better.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

EARLY_WEEK = 8


def main() -> None:
    old, new = (pd.read_csv(p, dtype={"game_id": str}) for p in sys.argv[1:3])
    both = old.merge(new[["game_id", "ours"]], on="game_id",
                     suffixes=("_old", "_new"))
    err_old = (both.actual - both.ours_old).abs()
    err_new = (both.actual - both.ours_new).abs()
    line = (both.actual - both.line).abs()
    delta = err_new - err_old
    print(f"{len(both)} shared games (old {len(old)}, new {len(new)})\n")
    print(f"{'slice':<10} {'n':>5} {'old gap':>8} {'new gap':>8} "
          f"{'delta':>7} {'+/-':>6}")
    early = both.week <= EARLY_WEEK
    post = both.season_type == "postseason"
    slices = {"all": both.index == both.index, "wk<=8": early,
              "wk>8 reg": ~early & ~post, "post": post}
    slices.update({str(s): both.season == s for s in sorted(both.season.unique())})
    for name, mask in slices.items():
        d = delta[mask]
        print(f"{name:<10} {mask.sum():>5} "
              f"{(err_old - line)[mask].mean():>+8.3f} "
              f"{(err_new - line)[mask].mean():>+8.3f} {d.mean():>+7.3f} "
              f"{1.96 * d.std(ddof=1) / np.sqrt(len(d)):>6.3f}")


if __name__ == "__main__":
    main()
