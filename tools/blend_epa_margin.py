"""Combine margin and EPA instead of choosing between them.

EPA kept losing head to head, and the instinct to keep patching it was
the wrong one. The framing was wrong: SP+ and FEI do not discard the
scoreboard, and neither should we. Points are the actual objective;
efficiency is evidence about the process that produced them.

So stop asking "which observation is better" and ask "what is the best
combined estimate". Two ways to find out:

  BLEND    average the two models' predictions, weight swept
  STACK    regress actual margin on both predictions at once, letting
           least squares assign the weights

Stacking is the more honest version -- it never assumes the weights,
and if EPA carries no independent information its coefficient simply
goes to zero and we learn that too.

All weights are fit on 2024 and applied to 2025, so the reported number
is genuinely out of sample.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.epa_ridge import (  # noqa: E402
    build_epa_games, fit_epa_ratings_primed, get_ep_model,
)
from cfbrank.games import load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit as fit_margin  # noqa: E402

FIT_SEASON = 2024
TEST_SEASON = 2025
START_WEEK = 4
MIN_TRAINING = 50
RELATIVE_LAMBDA = 4.0
MARGIN_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}

_epa_cache: dict[int, list] = {}


def epa_games_for(season: int) -> list:
    if season not in _epa_cache:
        _epa_cache[season] = build_epa_games(
            season, load_season(season), get_ep_model())
    return _epa_cache[season]


def paired_predictions(season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk forward, returning (margin_pred, epa_pred, actual) triples."""
    games = sorted(load_season(season), key=lambda g: g.week)
    by_id = {g.game_id: g for g in epa_games_for(season)}

    margin_prior = build_prior_or_empty(season) or None

    from_margin: list[float] = []
    from_epa: list[float] = []
    actuals: list[float] = []

    for week in sorted({g.week for g in games}):
        if week < START_WEEK:
            continue
        history = [g for g in games if g.week < week]
        upcoming = [g for g in games if g.week == week]
        if len(history) < MIN_TRAINING or not upcoming:
            continue

        margin_model = fit_margin(history, prior=margin_prior, **MARGIN_PARAMS)
        epa_history = [by_id[str(g.game_id)] for g in history
                       if str(g.game_id) in by_id]
        if len(epa_history) < MIN_TRAINING:
            continue
        # Same shared-prior mechanism the production app uses -- see
        # epa_ridge.fit_epa_ratings_primed. No second, separately-fit
        # EPA-scale prior; one measured carryover, translated by unit.
        epa_model = fit_epa_ratings_primed(
            epa_history, margin_prior, relative_lambda=RELATIVE_LAMBDA)

        for game in upcoming:
            if (game.home_team not in margin_model.ratings
                    or game.away_team not in margin_model.ratings
                    or game.home_team not in epa_model.ratings
                    or game.away_team not in epa_model.ratings):
                continue
            from_margin.append(margin_model.predict_margin(
                game.home_team, game.away_team, game.neutral_site))
            from_epa.append(epa_model.predict_margin(
                game.home_team, game.away_team, game.neutral_site))
            actuals.append(float(game.margin))

    return (np.array(from_margin), np.array(from_epa), np.array(actuals))


def report(name: str, predicted: np.ndarray, actual: np.ndarray) -> float:
    mae = float(np.mean(np.abs(actual - predicted)))
    accuracy = float(np.mean((predicted > 0) == (actual > 0)))
    print(f"{name:<34} {mae:>8.3f} {accuracy:>9.1%}")
    return mae


def main() -> None:
    print("Fitting expected-points model ...")
    get_ep_model()

    print(f"\nWalking {FIT_SEASON} (weights fitted here) ...")
    margin_fit, epa_fit, actual_fit = paired_predictions(FIT_SEASON)
    print(f"  {len(actual_fit)} paired predictions")

    print(f"\nBLEND WEIGHT SWEEP on {FIT_SEASON}")
    print(f"{'weight on EPA':<34} {'MAE':>8} {'accuracy':>10}")
    print("-" * 54)
    best = (float("inf"), 0.0)
    for weight in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        blended = (1 - weight) * margin_fit + weight * epa_fit
        mae = report(f"  {weight:.1f}", blended, actual_fit)
        if mae < best[0]:
            best = (mae, weight)
    print(f"\n  best blend weight on EPA: {best[1]}")

    # Stacking: let least squares choose, no intercept (an even matchup
    # must predict zero).
    stacked = np.column_stack([margin_fit, epa_fit])
    coefficients, *_ = np.linalg.lstsq(stacked, actual_fit, rcond=None)
    print(f"\n  STACKED coefficients: margin {coefficients[0]:+.3f}, "
          f"EPA {coefficients[1]:+.3f}")
    if coefficients[1] > 0.05:
        print("  -> EPA carries information margin does not.")
    else:
        print("  -> EPA adds nothing once margin is known.")

    print(f"\n\nHELD-OUT TEST on {TEST_SEASON}")
    margin_test, epa_test, actual_test = paired_predictions(TEST_SEASON)
    print(f"{len(actual_test)} paired predictions\n")
    print(f"{'model':<34} {'MAE':>8} {'accuracy':>10}")
    print("-" * 54)
    report("margin only", margin_test, actual_test)
    report("EPA only", epa_test, actual_test)
    report(f"blend (w={best[1]:.1f}, fit on {FIT_SEASON})",
           (1 - best[1]) * margin_test + best[1] * epa_test, actual_test)
    report(f"stacked (fit on {FIT_SEASON})",
           coefficients[0] * margin_test + coefficients[1] * epa_test,
           actual_test)


if __name__ == "__main__":
    main()
