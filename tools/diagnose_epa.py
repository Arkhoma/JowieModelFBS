"""Why did EPA lose? Isolate the cause instead of guessing.

EPA losing to final margin contradicts SP+ and FEI, so the likely fault
is the implementation, not the concept. Three candidates, each testable:

1. NO PRIOR. The margin model shrinks toward last season; the EPA model
   shrank toward zero. That advantage alone was worth ~1.2 MAE earlier,
   which is most of the gap.

2. WRONG PENALTY. lambda=5 was cross-validated for margins, which are
   ~O(10-30). EPA edges are ~O(0.1-0.5) -- a hundred times smaller. The
   same lambda is therefore a far harsher penalty in relative terms, so
   EPA ratings are being crushed toward the target.

3. CONVERSION NOISE. points_per_epa is fit on the same games, so early
   in a season it is estimated from very little.

Suspect 2 is the strongest: penalty strength must scale with the
variance of what is being explained, and it was not rescaled.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.epa_ridge import build_epa_games, fit_epa_ratings, get_ep_model  # noqa: E402
from cfbrank.games import load_season  # noqa: E402

SEASON = 2024


def main() -> None:
    games = load_season(SEASON)
    epa_games = build_epa_games(SEASON, games, get_ep_model())
    print(f"{len(epa_games)} EPA games for {SEASON}")

    edges = np.array([g.epa_edge for g in epa_games])
    margins = np.array([g.margin for g in epa_games], dtype=float)

    print("\nSCALE COMPARISON -- the same lambda is not the same penalty")
    print(f"  EPA edge      sd {edges.std():8.4f}   "
          f"range [{edges.min():.3f}, {edges.max():.3f}]")
    print(f"  Final margin  sd {margins.std():8.4f}   "
          f"range [{margins.min():.0f}, {margins.max():.0f}]")
    ratio = margins.std() / edges.std()
    print(f"\n  Margin varies {ratio:.0f}x more than EPA edge.")
    print(f"  A penalty tuned for margins is therefore ~{ratio:.0f}x too")
    print(f"  harsh on EPA. Equivalent lambda would be ~{5 / ratio**2:.4f}.")

    correlation = float(np.corrcoef(edges, margins)[0, 1])
    print(f"\n  corr(EPA edge, margin) = {correlation:.3f}")

    print("\nLAMBDA SWEEP -- in-sample fit of margin on efficiency edge")
    print(f"{'lambda':>10} {'rating sd':>11} {'pts/EPA':>9} {'R^2':>8}")
    print("-" * 42)
    for lambda_ in (0.001, 0.01, 0.05, 0.2, 1.0, 5.0):
        model = fit_epa_ratings(epa_games, lambda_=lambda_)
        values = np.array(list(model.fbs_ratings().values()))
        predicted = np.array([
            model.predict_margin(g.home_team, g.away_team, g.neutral_site)
            for g in epa_games
        ])
        residual = margins - predicted
        r_squared = 1 - float(residual @ residual) / float(
            ((margins - margins.mean()) ** 2).sum())
        print(f"{lambda_:>10.3f} {values.std():>11.4f} "
              f"{model.points_per_epa:>9.1f} {r_squared:>8.3f}")


if __name__ == "__main__":
    main()
