"""Does EPA actually beat final margin? Walk-forward, out of sample.

The claim "per-play efficiency beats scoreboard" is the entire argument
for SP+ and FEI over simpler systems. It is also just a claim until
measured on this data with this code.

Protocol: for each week, fit BOTH models on prior weeks only, predict
that week's games, and compare. Identical training data, identical
opponent adjustment, identical solver. The only difference is whether a
game contributes its final margin or its net EPA per play.

The expected-points table is fit on 2022-2023 and frozen, so it never
sees the seasons being scored.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.epa_ridge import build_epa_games, fit_epa_ratings, get_ep_model  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit as fit_margin  # noqa: E402

SEASONS = (2024, 2025)
START_WEEK = 4
MIN_TRAINING = 50
MARGIN_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}


def evaluate(season: int) -> dict:
    games = sorted(load_season(season), key=lambda g: g.week)
    epa_games = build_epa_games(season, games, get_ep_model())
    epa_by_id = {g.game_id: g for g in epa_games}

    prior = build_prior_or_empty(season) or None

    results = {
        "margin": {"errors": [], "hits": []},
        "epa": {"errors": [], "hits": []},
    }

    for week in sorted({g.week for g in games}):
        if week < START_WEEK:
            continue
        history = [g for g in games if g.week < week]
        upcoming = [g for g in games if g.week == week]
        if len(history) < MIN_TRAINING or not upcoming:
            continue

        margin_model = fit_margin(history, prior=prior, **MARGIN_PARAMS)

        epa_history = [epa_by_id[str(g.game_id)] for g in history
                       if str(g.game_id) in epa_by_id]
        if len(epa_history) < MIN_TRAINING:
            continue
        epa_model = fit_epa_ratings(epa_history, lambda_=5.0)

        for game in upcoming:
            actual = float(game.margin)

            if (game.home_team in margin_model.ratings
                    and game.away_team in margin_model.ratings):
                predicted = margin_model.predict_margin(
                    game.home_team, game.away_team, game.neutral_site)
                results["margin"]["errors"].append(abs(actual - predicted))
                results["margin"]["hits"].append(
                    (predicted > 0) == (actual > 0))

            if (game.home_team in epa_model.ratings
                    and game.away_team in epa_model.ratings):
                predicted = epa_model.predict_margin(
                    game.home_team, game.away_team, game.neutral_site)
                results["epa"]["errors"].append(abs(actual - predicted))
                results["epa"]["hits"].append((predicted > 0) == (actual > 0))

    return results


def main() -> None:
    print("Fitting expected-points model (2022-2023, frozen) ...")
    get_ep_model()

    totals = {"margin": {"errors": [], "hits": []},
              "epa": {"errors": [], "hits": []}}

    print(f"\n{'season':>7} {'model':>8} {'n':>6} {'MAE':>8} {'accuracy':>10}")
    print("-" * 44)

    for season in SEASONS:
        results = evaluate(season)
        for name in ("margin", "epa"):
            errors = results[name]["errors"]
            hits = results[name]["hits"]
            if not errors:
                continue
            totals[name]["errors"].extend(errors)
            totals[name]["hits"].extend(hits)
            print(f"{season:>7} {name:>8} {len(errors):>6} "
                  f"{np.mean(errors):>8.3f} {np.mean(hits):>9.1%}")

    print("-" * 44)
    summary = {}
    for name in ("margin", "epa"):
        errors = totals[name]["errors"]
        hits = totals[name]["hits"]
        if not errors:
            continue
        summary[name] = (float(np.mean(errors)), float(np.mean(hits)))
        print(f"{'POOLED':>7} {name:>8} {len(errors):>6} "
              f"{np.mean(errors):>8.3f} {np.mean(hits):>9.1%}")

    if len(summary) == 2:
        mae_gain = summary["margin"][0] - summary["epa"][0]
        acc_gain = summary["epa"][1] - summary["margin"][1]
        print(f"\n  EPA vs margin: MAE {mae_gain:+.3f} points, "
              f"accuracy {acc_gain:+.1%}")
        if mae_gain > 0:
            print("  -> EPA wins. Per-play efficiency beats the scoreboard.")
        else:
            print("  -> Margin wins. EPA is not paying for itself here.")


if __name__ == "__main__":
    main()
