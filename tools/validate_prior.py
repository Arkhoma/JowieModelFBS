"""Does the preseason prior actually predict better? Out of sample.

The week-3 table looking more sensible proves nothing -- it might just
match my expectations, which is the exact bias this project exists to
remove. The honest test is whether it predicts UNSEEN games better.

Reported separately for early weeks (where the prior should matter a
lot) and the full season (where it should wash out).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.backtest import walk_forward  # noqa: E402
from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty, measure_carryover  # noqa: E402

TEST_SEASONS = (2022, 2023, 2024, 2025)


def run(season: int, prior, start_week: int, min_games: int):
    try:
        return walk_forward(
            load_season(season), season, prior=prior,
            start_week=start_week, min_training_games=min_games,
        )
    except (ValueError, FileNotFoundError):
        return None


def compare(label: str, start_week: int, min_games: int, rho: float) -> None:
    print(f"\n{label}")
    print(f"{'season':>7} {'n':>5} {'MAE base':>9} {'MAE prior':>10} "
          f"{'acc base':>9} {'acc prior':>10}")
    print("-" * 56)

    totals = {"n": 0, "mae_b": 0.0, "mae_p": 0.0, "acc_b": 0.0, "acc_p": 0.0}

    for season in TEST_SEASONS:
        prior = build_prior_or_empty(season, rho)
        if not prior:
            continue
        base = run(season, None, start_week, min_games)
        with_prior = run(season, prior, start_week, min_games)
        if base is None or with_prior is None:
            continue

        print(f"{season:>7} {base.n_predicted:>5} {base.mae:>9.3f} "
              f"{with_prior.mae:>10.3f} {base.accuracy:>8.1%} "
              f"{with_prior.accuracy:>9.1%}")

        totals["n"] += base.n_predicted
        totals["mae_b"] += base.mae * base.n_predicted
        totals["mae_p"] += with_prior.mae * base.n_predicted
        totals["acc_b"] += base.accuracy * base.n_predicted
        totals["acc_p"] += with_prior.accuracy * base.n_predicted

    if not totals["n"]:
        print("  (no comparable seasons)")
        return

    n = totals["n"]
    mae_b, mae_p = totals["mae_b"] / n, totals["mae_p"] / n
    acc_b, acc_p = totals["acc_b"] / n, totals["acc_p"] / n
    print("-" * 56)
    print(f"{'POOLED':>7} {n:>5} {mae_b:>9.3f} {mae_p:>10.3f} "
          f"{acc_b:>8.1%} {acc_p:>9.1%}")
    print(f"\n  MAE improvement      {mae_b - mae_p:+.3f} points "
          f"({(mae_b - mae_p) / mae_b:+.1%})")
    print(f"  Accuracy improvement {acc_p - acc_b:+.1%}")


def main() -> None:
    seasons = [s for s in available_seasons() if s - 1 in available_seasons()]
    measured = measure_carryover(seasons)
    rho = measured["rho"]
    print(f"rho = {rho:.3f} (measured on {measured['n_games']} games, "
          f"R^2 {measured['r_squared']:.3f})")

    compare("EARLY SEASON (weeks 2-5, min 20 training games)",
            start_week=2, min_games=20, rho=rho)
    compare("FULL SEASON (weeks 4+, min 50 training games)",
            start_week=4, min_games=50, rho=rho)


if __name__ == "__main__":
    main()
