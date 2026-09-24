"""How good could the predictive model EVER get?

Answers Howie's third question. There are three numbers that matter:

1. What we do now, out of sample (walk-forward backtest).
2. The ORACLE floor: refit on the complete season, then "predict" games
   the model has already seen. This is cheating -- it uses knowledge of
   the future -- so it is not an achievable score. It is the error left
   over when team strength is known as well as a season of data can ever
   reveal it. Whatever remains is game-to-day noise: turnovers, weather,
   a backup quarterback, a bad spot on fourth down.
3. The gap between them, which is the only part better modelling can win.

If the oracle is close to our real score, the model is near the ceiling
and further effort should go elsewhere. If there is a wide gap, there is
signal still on the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.backtest import walk_forward  # noqa: E402
from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

SEASONS = (2021, 2022, 2023, 2024, 2025)
FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}


def oracle_scores(season: int) -> tuple[float, float, int]:
    """In-sample MAE and accuracy with full knowledge of the season."""
    games = load_season(season)
    model = fit(games, **FITTED)
    predictor = Predictor.from_model(model, games)

    errors, correct = [], []
    for game in games:
        if not (game.home_team in model.ratings
                and game.away_team in model.ratings):
            continue
        predicted = predictor.predict(
            game.home_team, game.away_team, game.neutral_site
        ).predicted_margin
        errors.append(abs(game.margin - predicted))
        correct.append((predicted > 0) == (game.margin > 0))
    return float(np.mean(errors)), float(np.mean(correct)), len(errors)


def main() -> None:
    have = set(available_seasons())
    print(f"{'season':>7} {'live MAE':>9} {'oracle':>8} {'gap':>6} "
          f"{'live acc':>9} {'oracle':>8}")
    print("-" * 52)

    live_mae, live_acc, orac_mae, orac_acc, weights = [], [], [], [], []

    for season in SEASONS:
        if season not in have:
            continue
        try:
            result = walk_forward(
                load_season(season), season,
                prior=build_prior_or_empty(season) or None)
        except (ValueError, FileNotFoundError):
            continue
        o_mae, o_acc, _ = oracle_scores(season)
        print(f"{season:>7} {result.mae:>9.2f} {o_mae:>8.2f} "
              f"{result.mae - o_mae:>6.2f} "
              f"{result.accuracy:>8.1%} {o_acc:>7.1%}")
        live_mae.append(result.mae)
        live_acc.append(result.accuracy)
        orac_mae.append(o_mae)
        orac_acc.append(o_acc)
        weights.append(result.n_predicted)

    w = np.array(weights, dtype=float)
    lm = float(np.average(live_mae, weights=w))
    om = float(np.average(orac_mae, weights=w))
    la = float(np.average(live_acc, weights=w))
    oa = float(np.average(orac_acc, weights=w))

    print("-" * 52)
    print(f"{'pooled':>7} {lm:>9.2f} {om:>8.2f} {lm - om:>6.2f} "
          f"{la:>8.1%} {oa:>7.1%}")

    print(f"\nHeadroom in MAE:      {lm - om:.2f} points "
          f"({(lm - om) / lm:.1%} of current error)")
    print(f"Headroom in accuracy: {oa - la:+.1%}")
    print(f"\nIrreducible noise is at least {om:.2f} points MAE -- that is "
          "what\nremains when team strength is known with hindsight.")


if __name__ == "__main__":
    main()
