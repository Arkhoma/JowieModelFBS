"""Is the prior scale secretly a calibration fix?

The sweep kept wanting MORE weight on last season, past 1.2, well beyond
the measured 0.63 carryover. Extra weight cannot be buying extra
carryover information that does not exist. So it must be buying
something else.

Hypothesis: ridge predictions are UNDER-DISPERSED. The penalty shrinks
ratings toward the target, so predicted margins are systematically too
close to zero -- the model says "by 7" when the honest answer is "by
10". Inflating the prior inflates every rating, which partly cancels the
compression and lowers MAE for a reason that has nothing to do with last
season.

Testable. Regress actual margin on predicted margin out of sample:

    actual = slope * predicted

slope = 1.0 means calibrated. slope > 1.0 means compressed, and the
right fix is to scale predictions by that slope -- one honest parameter
-- rather than to distort the prior into doing it sideways.

If the hypothesis holds, then AFTER calibration the best prior scale
should collapse back toward the measured 0.70.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

SEASONS = (2022, 2023, 2024, 2025)
SCALES = (0.0, 0.63, 0.70, 1.073, 1.2)
START_WEEK = 2
MIN_TRAINING = 20

_cache: dict[int, dict[str, float]] = {}


def base(season: int) -> dict[str, float]:
    if season not in _cache:
        model = fit(load_season(season - 1), lambda_=5.0,
                    margin_scale=28.0, halflife=1e6)
        _cache[season] = dict(model.ratings)
    return _cache[season]


def collect(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Walk forward, returning paired (predicted, actual) margins."""
    predicted: list[float] = []
    actual: list[float] = []

    for season in SEASONS:
        games = sorted(
            (g for g in load_season(season) if g.season == season),
            key=lambda g: g.week)
        prior = ({t: scale * r for t, r in base(season).items()}
                 if scale else None)

        for week in sorted({g.week for g in games}):
            if week < START_WEEK:
                continue
            history = [g for g in games if g.week < week]
            upcoming = [g for g in games if g.week == week]
            if len(history) < MIN_TRAINING or not upcoming:
                continue

            model = fit(history, lambda_=5.0, margin_scale=28.0,
                        halflife=1e6, prior=prior)
            predictor = Predictor.from_model(model, history)

            for game in upcoming:
                if (game.home_team not in model.ratings
                        or game.away_team not in model.ratings):
                    continue
                result = predictor.predict(
                    game.home_team, game.away_team, game.neutral_site)
                predicted.append(result.predicted_margin)
                actual.append(float(game.margin))

    return np.asarray(predicted), np.asarray(actual)


def main() -> None:
    print("Calibration of out-of-sample predictions (weeks 2+)")
    print("slope > 1 means predictions are too timid\n")
    print(f"{'prior':>7} {'slope':>8} {'MAE raw':>9} {'MAE calib':>10} "
          f"{'gain':>7}")
    print("-" * 46)

    rows = []
    for scale in SCALES:
        predicted, actual = collect(scale)
        # Slope through the origin: a margin of 0 should predict 0.
        slope = float((predicted @ actual) / (predicted @ predicted))
        mae_raw = float(np.mean(np.abs(actual - predicted)))
        mae_calibrated = float(np.mean(np.abs(actual - slope * predicted)))
        rows.append((scale, slope, mae_calibrated))
        print(f"{scale:>7.3f} {slope:>8.3f} {mae_raw:>9.3f} "
              f"{mae_calibrated:>10.3f} {mae_raw - mae_calibrated:>7.3f}")

    best = min(rows, key=lambda r: r[2])
    print("-" * 46)
    print(f"\nAfter calibration, best prior scale = {best[0]} "
          f"(MAE {best[2]:.3f})")
    print("\nIf that collapsed toward 0.70, the extra prior weight was")
    print("never about last season -- it was fixing compression, and a")
    print("calibration slope fixes it honestly and in one place.")


if __name__ == "__main__":
    main()
