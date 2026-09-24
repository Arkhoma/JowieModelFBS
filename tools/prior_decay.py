"""Does last season overstay its welcome?

Two open questions the previous sweep could not answer:

1. It hit the GRID EDGE (rho=1.0, lambda=5 both at boundaries), so the
   true optimum was never bracketed. Extend both.

2. Boomer's claim was that the prior "fades on its own as evidence
   accumulates." That is the standard ridge intuition, but the decay
   table showed ratings still correlating 0.758 with last season at
   week 13. Intuition is not evidence. Test it directly: split results
   into EARLY and LATE and see whether the best rho differs.

If the prior helps early and hurts late, it needs to decay by week. If
it helps throughout, teams are simply persistent year to year and a
fixed prior is correct.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.backtest import walk_forward  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

SEASONS = (2022, 2023, 2024, 2025)

# Grid now extends BEYOND where the last sweep bottomed out, so an
# interior optimum can actually be found rather than assumed.
RHOS = (0.0, 0.5, 0.8, 1.0, 1.2)
LAMBDAS = (1, 2, 5, 15)

EARLY = {"start_week": 2, "min_training_games": 20}
LATE = {"start_week": 9, "min_training_games": 300}

_cache: dict[int, dict[str, float]] = {}


def base_prior(season: int) -> dict[str, float]:
    if season not in _cache:
        model = fit(load_season(season - 1), lambda_=5.0,
                    margin_scale=28.0, halflife=1e6)
        _cache[season] = dict(model.ratings)
    return _cache[season]


def score(rho: float, lambda_: float, regime: dict) -> tuple[float, float]:
    n = mae = acc = 0
    for season in SEASONS:
        prior = ({t: rho * r for t, r in base_prior(season).items()}
                 if rho else None)
        try:
            result = walk_forward(
                load_season(season), season, lambda_=lambda_,
                margin_scale=28.0, halflife=1e6, prior=prior, **regime)
        except ValueError:
            continue
        n += result.n_predicted
        mae += result.mae * result.n_predicted
        acc += result.accuracy * result.n_predicted
    return (mae / n, acc / n) if n else (float("nan"), float("nan"))


def grid(label: str, regime: dict) -> tuple[float, float]:
    print(f"\n{label} -- pooled MAE across {SEASONS}")
    header = "  rho  " + "".join(f"{f'lam={lam}':>9}" for lam in LAMBDAS)
    print(header)
    print("-" * len(header))

    best = (float("inf"), None, None)
    for rho in RHOS:
        cells = []
        for lambda_ in LAMBDAS:
            mae, _ = score(rho, lambda_, regime)
            cells.append(f"{mae:>9.3f}")
            if mae < best[0]:
                best = (mae, rho, lambda_)
        print(f"{rho:>5.2f}  " + "".join(cells))

    print(f"  -> best rho={best[1]}, lambda={best[2]}, MAE={best[0]:.3f}")
    return best[1], best[2]


def main() -> None:
    early_rho, early_lambda = grid("EARLY (weeks 2-8)", EARLY)
    late_rho, late_lambda = grid("LATE (weeks 9+)", LATE)

    print("\n\nVERDICT")
    print("-" * 58)
    print(f"  early season: rho={early_rho}, lambda={early_lambda}")
    print(f"  late season:  rho={late_rho}, lambda={late_lambda}")
    if late_rho is not None and early_rho is not None:
        if late_rho < early_rho:
            print("\n  Last season matters LESS once real games accumulate.")
            print("  A fixed prior overstays its welcome -> decay rho by week.")
        elif late_rho == early_rho:
            print("\n  Same weight works all season: teams are genuinely")
            print("  persistent, and the growing evidence already dilutes")
            print("  the prior without help.")
        else:
            print("\n  Unexpected: the prior matters MORE late. Investigate.")


if __name__ == "__main__":
    main()
