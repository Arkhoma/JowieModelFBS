"""How much does the predictive model improve as a season accumulates?

Answers Howie's first question with data instead of intuition: pool the
walk-forward backtest across seasons and report accuracy and MAE by week.

If "more games means better rankings" is true, error should fall and
accuracy should rise as the week number grows -- and it should flatten
once the ratings have enough evidence to stop moving.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.backtest import walk_forward  # noqa: E402
from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402

SEASONS = (2021, 2022, 2023, 2024, 2025)


def main() -> None:
    have = set(available_seasons())
    pooled: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    for season in SEASONS:
        if season not in have:
            continue
        try:
            result = walk_forward(
                load_season(season), season,
                prior=build_prior_or_empty(season) or None)
        except (ValueError, FileNotFoundError):
            continue
        print(f"{season}: {result.n_predicted} games, "
              f"MAE {result.mae:.2f}, acc {result.accuracy:.1%}")
        for week, stats in result.by_week.items():
            pooled[week].append((stats["n"], stats["mae"], stats["accuracy"]))

    print("\nPooled by week (all seasons, weighted by games):")
    print(f"{'week':>5} {'games':>6} {'MAE':>7} {'accuracy':>9}")
    print("-" * 32)

    rows = []
    for week in sorted(pooled):
        entries = pooled[week]
        counts = np.array([e[0] for e in entries], dtype=float)
        mae = float(np.average([e[1] for e in entries], weights=counts))
        acc = float(np.average([e[2] for e in entries], weights=counts))
        total = int(counts.sum())
        rows.append((week, total, mae, acc))
        print(f"{week:>5} {total:>6} {mae:>7.2f} {acc:>8.1%}")

    # Early vs late, split at the midpoint of the regular season.
    early = [r for r in rows if r[0] <= 8]
    late = [r for r in rows if r[0] > 8]
    for label, group in (("weeks <=8", early), ("weeks >8", late)):
        if not group:
            continue
        counts = np.array([r[1] for r in group], dtype=float)
        mae = float(np.average([r[2] for r in group], weights=counts))
        acc = float(np.average([r[3] for r in group], weights=counts))
        print(f"\n{label}: {int(counts.sum())} games, "
              f"MAE {mae:.2f}, accuracy {acc:.1%}")


if __name__ == "__main__":
    main()
