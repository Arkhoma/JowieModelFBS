"""Is the in-sample calibration slope trustworthy early in a season?

Flagged during the 2026 week-3 run: calibration came out at 2.393, far
above the ~1.5 seen on full seasons. The slope is fit IN SAMPLE on the
same games the ratings were fit to, and with three games per team ridge
shrinks ratings hard -- so the slope has to stretch further to reach the
observed margins. That is a small-sample artifact, not a discovery that
2026 teams are unusually spread out.

Danger: an inflated slope multiplies every predicted margin, so early
predictions come out wildly over-confident precisely when the model
knows least.

This measures the slope week by week and compares in-sample against an
honest out-of-sample estimate.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}


def out_of_sample_slope(season: int, through_week: int) -> float | None:
    """Fit ratings on weeks < W, measure the slope on week W itself."""
    games = load_season(season)
    history = [g for g in games if g.week < through_week]
    holdout = [g for g in games if g.week == through_week]
    if len(history) < 30 or not holdout:
        return None

    prior = build_prior_or_empty(season) or None
    model = fit(history, prior=prior, **FITTED)

    edges = []
    margins = []
    for game in holdout:
        if (game.home_team not in model.ratings
                or game.away_team not in model.ratings):
            continue
        edge = model.rating(game.home_team) - model.rating(game.away_team)
        location = 0.0 if game.neutral_site else model.home_field
        edges.append(edge)
        margins.append(game.margin - location)

    if len(edges) < 10:
        return None
    edges = np.array(edges)
    margins = np.array(margins)
    denominator = float(edges @ edges)
    return float(edges @ margins) / denominator if denominator else None


def main() -> None:
    print("Calibration slope: in-sample vs honest out-of-sample")
    print("(in-sample inflates when data is thin)\n")

    for season in (2024, 2025, 2026):
        if season not in available_seasons():
            continue
        games = load_season(season)
        max_week = max(g.week for g in games)
        print(f"{season}")
        print(f"{'through wk':>11} {'games':>7} {'in-sample':>11} "
              f"{'out-of-sample':>15}")
        print("-" * 48)

        for week in (3, 5, 8, 12, max_week):
            if week > max_week:
                continue
            subset = [g for g in games if g.week <= week]
            if len(subset) < 30:
                continue
            prior = build_prior_or_empty(season) or None
            model = fit(subset, prior=prior, **FITTED)
            honest = out_of_sample_slope(season, week)
            honest_text = f"{honest:>15.3f}" if honest else f"{'--':>15}"
            print(f"{week:>11} {len(subset):>7} {model.calibration:>11.3f} "
                  f"{honest_text}")
        print()


if __name__ == "__main__":
    main()
