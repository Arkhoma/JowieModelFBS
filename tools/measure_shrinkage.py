"""Measure ridge shrinkage directly, instead of assuming it.

Defining unshrink = 1/carryover makes the two cancel to exactly 1.0 --
circular by construction, and no better than the rho=1.073 it replaced.

Shrinkage is measurable on its own terms. Split each season's games into
two halves at random. Fit ratings on each half independently. Both
halves estimate the SAME underlying team strength, so:

  * the covariance between halves reflects true signal variance
  * the variance within a half reflects signal + shrinkage + noise

Comparing them recovers how much the penalty compressed the true spread,
using nothing but one season's data. No reference to carryover at all,
so the two parameters stay genuinely independent.
"""

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
TRIALS = 6


def split_half_shrinkage(season: int, seed: int) -> float | None:
    """Ratio of true signal spread to fitted spread, from one split."""
    games = load_season(season)
    rng = random.Random(seed)
    shuffled = games[:]
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2

    try:
        left = fit(shuffled[:midpoint], **FITTED).fbs_ratings()
        right = fit(shuffled[midpoint:], **FITTED).fbs_ratings()
    except (ValueError, np.linalg.LinAlgError):
        return None

    shared = sorted(set(left) & set(right))
    if len(shared) < 50:
        return None

    a = np.array([left[t] for t in shared])
    b = np.array([right[t] for t in shared])
    a_centred, b_centred = a - a.mean(), b - b.mean()

    # Covariance across independent halves isolates true signal variance;
    # each half's own variance also carries noise. Their ratio is the
    # reliability, and 1/sqrt(reliability) is the shrinkage factor.
    covariance = float((a_centred * b_centred).mean())
    variance = float((a_centred ** 2).mean() + (b_centred ** 2).mean()) / 2.0
    if covariance <= 0 or variance <= 0:
        return None

    reliability = covariance / variance
    # Each half has half the data, so it is shrunk harder than a
    # full-season fit. Correct via the Spearman-Brown relation.
    full = 2 * reliability / (1 + reliability)
    if not 0 < full < 1:
        return None
    return float(1.0 / np.sqrt(full))


def main() -> None:
    seasons = [s for s in available_seasons() if 2015 <= s <= 2025]
    print("Ridge shrinkage, measured by split-half reliability")
    print("(>1.0 means fitted ratings are compressed vs true strength)\n")
    print(f"{'season':>8} {'unshrink':>10}")
    print("-" * 20)

    factors = []
    for season in seasons:
        estimates = [
            value for value in
            (split_half_shrinkage(season, seed) for seed in range(TRIALS))
            if value is not None
        ]
        if not estimates:
            continue
        mean = float(np.mean(estimates))
        factors.append(mean)
        print(f"{season:>8} {mean:>10.3f}")

    if factors:
        overall = float(np.mean(factors))
        print("-" * 20)
        print(f"{'MEAN':>8} {overall:>10.3f}")
        print(f"\n  Fitted ratings are compressed by about "
              f"{(overall - 1) * 100:.0f}%.")
        print(f"  Independent of carryover -- measured within seasons.")
        print(f"\n  Net prior scale = 0.63 (carryover) x {overall:.3f} "
              f"(unshrink) = {0.63 * overall:.3f}")


if __name__ == "__main__":
    main()
