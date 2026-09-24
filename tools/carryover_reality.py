"""How much does a team actually carry over, year to year?

No regression, no tuning -- just the raw question: if a team was good
last season, how good are they this season? Correlate final ratings for
consecutive seasons and look at the actual churn.

This is a reality check on the model's rho. If real teams only carry
over ~0.6 of their rating but the model leans on last year at 1.0, the
model is overweighting history no matter what MAE says.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}


def pearson(xs, ys) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    return cov / (var_x * var_y) ** 0.5


def slope(xs, ys) -> float:
    """Least-squares slope of this year on last year: the true carryover."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    return cov / var_x


def main() -> None:
    seasons = [s for s in available_seasons() if s >= 2015]
    ratings = {}
    for season in seasons:
        try:
            model = fit(load_season(season), **FITTED)
        except (FileNotFoundError, ValueError):
            continue
        ratings[season] = model.fbs_ratings()

    print("Year-over-year persistence of FINAL season ratings")
    print("(slope is the honest carryover: 1.0 = no change, "
          "0.5 = half regresses)\n")
    print(f"{'transition':>14} {'teams':>7} {'corr':>7} {'slope':>7}")
    print("-" * 40)

    slopes = []
    complete = [s for s in sorted(ratings) if s < max(ratings)]
    for season in complete:
        nxt = season + 1
        if nxt not in ratings:
            continue
        # 2026 is mid-season; its ratings are not a final verdict.
        if nxt == max(ratings):
            continue
        shared = sorted(set(ratings[season]) & set(ratings[nxt]))
        if len(shared) < 50:
            continue
        previous = [ratings[season][t] for t in shared]
        current = [ratings[nxt][t] for t in shared]
        r = pearson(previous, current)
        b = slope(previous, current)
        slopes.append(b)
        print(f"{season}->{nxt:>9} {len(shared):>7} {r:>7.3f} {b:>7.3f}")

    if slopes:
        mean_slope = sum(slopes) / len(slopes)
        print("-" * 40)
        print(f"{'MEAN':>14} {'':>7} {'':>7} {mean_slope:>7.3f}")
        print(f"\n  A team keeps about {mean_slope:.0%} of its rating "
              "edge into the next season.")
        print(f"  So roughly {1 - mean_slope:.0%} of what you were "
              "last year is gone by kickoff.")


if __name__ == "__main__":
    main()
