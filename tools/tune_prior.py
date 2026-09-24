"""How much should last season count? Tune rho and lambda jointly.

Two knobs decide "how much last year matters" and they interact:

  rho     how much of last season's rating carries over (the TARGET)
  lambda  how hard ratings are pulled toward that target (the STRENGTH)

Shipping rho=1.073 while keeping lambda=5 was a mistake in method: that
lambda was cross-validated back when ridge shrank toward ZERO. Once the
target becomes a good prior rather than league average, the optimal pull
changes. Tuning one knob against a stale value of the other is not
tuning, it is guessing with extra steps.

Honesty protocol: rho was previously measured across ALL seasons and
then reported on those same seasons -- a mild leak. Here we tune on
early seasons and report on later ones the tuning never saw.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.backtest import walk_forward  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

TUNE_SEASONS = (2022, 2023)
TEST_SEASONS = (2024, 2025)

RHOS = (0.0, 0.25, 0.50, 0.75, 1.00)
LAMBDAS = (5, 20, 80)

# Early-season regime: this is where the prior earns its keep.
EARLY = {"start_week": 2, "min_training_games": 20}

_prior_cache: dict[int, dict[str, float]] = {}


def base_prior(season: int) -> dict[str, float]:
    """Previous season's ratings, unscaled. Cached -- it is the slow part."""
    if season not in _prior_cache:
        model = fit(load_season(season - 1), lambda_=5.0,
                    margin_scale=28.0, halflife=1e6)
        _prior_cache[season] = dict(model.ratings)
    return _prior_cache[season]


def score(seasons, rho: float, lambda_: float) -> tuple[float, float]:
    """Pooled out-of-sample MAE and accuracy for one (rho, lambda)."""
    total_n = 0
    total_mae = 0.0
    total_acc = 0.0

    for season in seasons:
        prior = {team: rho * rating
                 for team, rating in base_prior(season).items()} if rho else None
        result = walk_forward(
            load_season(season), season, lambda_=lambda_,
            margin_scale=28.0, halflife=1e6, prior=prior, **EARLY)
        total_n += result.n_predicted
        total_mae += result.mae * result.n_predicted
        total_acc += result.accuracy * result.n_predicted

    return total_mae / total_n, total_acc / total_n


def sweep() -> tuple[float, float]:
    print(f"TUNING on {TUNE_SEASONS} -- early season, out of sample")
    print("\nMAE (lower is better)")
    header = "  rho  " + "".join(f"{f'lam={lam}':>10}" for lam in LAMBDAS)
    print(header)
    print("-" * len(header))

    best = (float("inf"), None, None)
    for rho in RHOS:
        cells = []
        for lambda_ in LAMBDAS:
            mae, _ = score(TUNE_SEASONS, rho, lambda_)
            cells.append(f"{mae:>10.3f}")
            if mae < best[0]:
                best = (mae, rho, lambda_)
        print(f"{rho:>5.2f}  " + "".join(cells))

    print(f"\nBEST on tuning seasons: rho={best[1]}, lambda={best[2]} "
          f"(MAE {best[0]:.3f})")
    return best[1], best[2]


def report(rho: float, lambda_: float) -> None:
    print(f"\n\nHELD-OUT TEST on {TEST_SEASONS} (never used for tuning)")
    print(f"{'configuration':<34} {'MAE':>8} {'accuracy':>10}")
    print("-" * 54)

    configurations = [
        ("no prior (original)", 0.0, 5),
        ("shipped (rho=1.07, lam=5)", 1.073, 5),
        (f"tuned (rho={rho}, lam={lambda_})", rho, lambda_),
    ]
    for label, test_rho, test_lambda in configurations:
        mae, accuracy = score(TEST_SEASONS, test_rho, test_lambda)
        print(f"{label:<34} {mae:>8.3f} {accuracy:>9.1%}")


def decay(season: int, rho: float, lambda_: float) -> None:
    """Show the prior losing its grip as real games pile up."""
    print(f"\n\nPRIOR INFLUENCE OVER {season} "
          "-- does last year fade out on its own?")
    print(f"{'through week':>13} {'games':>7} {'corr w/ prior':>15} "
          f"{'mean |shift|':>14}")
    print("-" * 52)

    games = load_season(season)
    prior = {t: rho * r for t, r in base_prior(season).items()}

    for week in (3, 6, 9, 13):
        subset = [g for g in games if g.week <= week]
        if not subset:
            continue
        model = fit(subset, lambda_=lambda_, margin_scale=28.0,
                    halflife=1e6, prior=prior)
        shared = [t for t in model.fbs_ratings() if t in prior]
        if len(shared) < 10:
            continue
        current = [model.ratings[t] for t in shared]
        previous = [prior[t] for t in shared]
        mean_current = sum(current) / len(current)
        mean_previous = sum(previous) / len(previous)
        covariance = sum((a - mean_current) * (b - mean_previous)
                         for a, b in zip(current, previous))
        variance_current = sum((a - mean_current) ** 2 for a in current)
        variance_previous = sum((b - mean_previous) ** 2 for b in previous)
        correlation = covariance / (variance_current * variance_previous) ** 0.5
        shift = sum(abs(a - b) for a, b in zip(current, previous)) / len(shared)
        print(f"{week:>13} {len(subset):>7} {correlation:>15.3f} "
              f"{shift:>14.2f}")


def main() -> None:
    rho, lambda_ = sweep()
    report(rho, lambda_)
    decay(2025, rho, lambda_)


if __name__ == "__main__":
    main()
