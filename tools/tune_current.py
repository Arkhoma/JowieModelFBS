"""Re-fit the free parameters against the CURRENT season's data.

The shipped parameters (lambda=5, scale=28) were cross-validated on a
COMPLETE 2024 season -- ~800 games, 12+ per team. Week 3 of a live
season is a different problem: 260 games, 3 per team, and a conference
graph that barely connects. Using the full-season penalty on week-3 data
lets noise through, which is why the table looks silly.

This script measures the right penalty for the data we actually have.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.ridge import cross_validate  # noqa: E402

LAMBDAS = (5, 10, 20, 40, 80, 160, 320, 640)
SCALES = (14.0, 28.0)


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    games = load_season(season)
    print(f"{season}: {len(games)} games\n")

    best = cross_validate(
        games, lambdas=LAMBDAS, margin_scales=SCALES, halflives=(1e6,))

    print(f"{'lambda':>7} {'scale':>7} {'MAE':>8}")
    print("-" * 26)
    for row in best["all_results"]:
        print(f"{row['lambda']:>7} {row['margin_scale']:>7.0f} "
              f"{row['mae']:>8.3f}")

    print(f"\nBEST: lambda={best['lambda']} scale={best['margin_scale']:.0f} "
          f"MAE={best['mae']:.3f}")


if __name__ == "__main__":
    main()
