"""Honest prior (0.70) versus the over-fit one (1.07). Out of sample.

The 1.073 figure was fit by least squares against early-season margins,
which quietly let it absorb the previous season's ridge shrinkage. The
0.70 figure comes from two independently measured quantities:

    0.63 carryover  (tools/carryover_reality.py)
  x 1.117 unshrink  (tools/measure_shrinkage.py)
  = 0.704

If 1.07 still predicts better, the extra weight is doing real work and
my reasoning about WHY is what needs correcting -- an over-fit knob that
genuinely helps is still worth knowing about. If they tie, prefer 0.70:
same accuracy, interpretable parameters, no cancelling errors.

Also sweeps neighbouring values so we can see the shape rather than
two isolated points.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.backtest import walk_forward  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

SEASONS = (2022, 2023, 2024, 2025)
SCALES = (0.0, 0.5, 0.63, 0.70, 0.85, 1.0, 1.073, 1.2)

EARLY = {"start_week": 2, "min_training_games": 20}
LATE = {"start_week": 9, "min_training_games": 300}

_cache: dict[int, dict[str, float]] = {}


def base(season: int) -> dict[str, float]:
    if season not in _cache:
        model = fit(load_season(season - 1), lambda_=5.0,
                    margin_scale=28.0, halflife=1e6)
        _cache[season] = dict(model.ratings)
    return _cache[season]


def score(scale: float, regime: dict) -> tuple[float, float, int]:
    n = mae = acc = 0
    for season in SEASONS:
        prior = ({t: scale * r for t, r in base(season).items()}
                 if scale else None)
        try:
            result = walk_forward(
                load_season(season), season, lambda_=5.0, margin_scale=28.0,
                halflife=1e6, prior=prior, **regime)
        except ValueError:
            continue
        n += result.n_predicted
        mae += result.mae * result.n_predicted
        acc += result.accuracy * result.n_predicted
    return (mae / n, acc / n, n) if n else (float("nan"), float("nan"), 0)


def sweep(label: str, regime: dict) -> None:
    print(f"\n{label}")
    print(f"{'prior scale':>12} {'MAE':>9} {'accuracy':>10}   note")
    print("-" * 56)
    notes = {
        0.0: "no prior at all",
        0.63: "raw carryover",
        0.70: "HONEST: 0.63 x 1.117",
        1.073: "old over-fit rho",
    }
    best = (float("inf"), None)
    for scale in SCALES:
        mae, accuracy, _ = score(scale, regime)
        if mae < best[0]:
            best = (mae, scale)
        print(f"{scale:>12.3f} {mae:>9.3f} {accuracy:>9.1%}   "
              f"{notes.get(scale, '')}")
    print(f"  -> lowest MAE at scale {best[1]}")


def main() -> None:
    sweep("EARLY SEASON (weeks 2-8)", EARLY)
    sweep("LATE SEASON (weeks 9+)", LATE)


if __name__ == "__main__":
    main()
