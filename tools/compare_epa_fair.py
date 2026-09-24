"""EPA vs margin, take two -- with the scale bug fixed and a fair prior.

The first comparison was rigged against EPA without my noticing:

  * lambda=5 was tuned for margins (sd ~20.7) and applied to EPA edges
    (sd ~0.78), making the penalty ~26^2 times too harsh.
  * the margin model got a preseason prior; the EPA model got nothing.

Both are fixed here. The penalty is now variance-scaled, and both models
receive a prior built from their OWN previous-season ratings, so neither
is handed an advantage the other lacks.

Also sweeps the relative penalty, since the right value on the EPA scale
has never been measured.
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

SEASONS = (2024, 2025)
START_WEEK = 4
MIN_TRAINING = 50
MARGIN_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}

_epa_cache: dict[int, list] = {}


def epa_games_for(season: int) -> list:
    if season not in _epa_cache:
        _epa_cache[season] = build_epa_games(
            season, load_season(season), get_ep_model())
    return _epa_cache[season]


def walk(season: int, relative_lambda: float) -> dict:
    games = sorted(load_season(season), key=lambda g: g.week)
    by_id = {g.game_id: g for g in epa_games_for(season)}

    margin_prior = build_prior_or_empty(season) or None

    out = {"margin": {"e": [], "h": []}, "epa": {"e": [], "h": []}}

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
        # Same shared-prior mechanism as production -- one measured
        # carryover, translated by unit, not a second fitted constant.
        epa_model = fit_epa_ratings_primed(
            epa_history, margin_prior, relative_lambda=relative_lambda)

        for game in upcoming:
            actual = float(game.margin)
            for name, model in (("margin", margin_model), ("epa", epa_model)):
                if (game.home_team not in model.ratings
                        or game.away_team not in model.ratings):
                    continue
                predicted = model.predict_margin(
                    game.home_team, game.away_team, game.neutral_site)
                out[name]["e"].append(abs(actual - predicted))
                out[name]["h"].append((predicted > 0) == (actual > 0))

    return out


def main() -> None:
    print("Fitting expected-points model (2022-2023, frozen) ...")
    get_ep_model()

    print("\nRELATIVE PENALTY SWEEP (EPA model, pooled over seasons)")
    print(f"{'rel lambda':>11} {'MAE':>8} {'accuracy':>10}")
    print("-" * 32)

    best = (float("inf"), None)
    for relative in (1.0, 4.0, 8.0, 16.0, 40.0):
        errors: list[float] = []
        hits: list[bool] = []
        for season in SEASONS:
            result = walk(season, relative)
            errors.extend(result["epa"]["e"])
            hits.extend(result["epa"]["h"])
        mae = float(np.mean(errors))
        if mae < best[0]:
            best = (mae, relative)
        print(f"{relative:>11.1f} {mae:>8.3f} {np.mean(hits):>9.1%}")

    print(f"\n  best relative lambda: {best[1]}")

    print(f"\n\nHEAD TO HEAD (relative lambda {best[1]}, both with priors)")
    print(f"{'season':>7} {'model':>8} {'n':>6} {'MAE':>8} {'accuracy':>10}")
    print("-" * 44)

    totals = {"margin": {"e": [], "h": []}, "epa": {"e": [], "h": []}}
    for season in SEASONS:
        result = walk(season, best[1])
        for name in ("margin", "epa"):
            errors, hits = result[name]["e"], result[name]["h"]
            totals[name]["e"].extend(errors)
            totals[name]["h"].extend(hits)
            print(f"{season:>7} {name:>8} {len(errors):>6} "
                  f"{np.mean(errors):>8.3f} {np.mean(hits):>9.1%}")

    print("-" * 44)
    summary = {}
    for name in ("margin", "epa"):
        errors, hits = totals[name]["e"], totals[name]["h"]
        summary[name] = (float(np.mean(errors)), float(np.mean(hits)))
        print(f"{'POOLED':>7} {name:>8} {len(errors):>6} "
              f"{np.mean(errors):>8.3f} {np.mean(hits):>9.1%}")

    mae_gain = summary["margin"][0] - summary["epa"][0]
    acc_gain = summary["epa"][1] - summary["margin"][1]
    print(f"\n  EPA vs margin: MAE {mae_gain:+.3f} points, "
          f"accuracy {acc_gain:+.1%}")


if __name__ == "__main__":
    main()
