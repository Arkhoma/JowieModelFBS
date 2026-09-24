"""Does the RANKING converge even when game prediction does not improve?

This resolves an apparent paradox in the other two tools. Prediction
accuracy is nearly flat across the season (72% in week 5, 74% in week 13),
which seems to say the model learns nothing after September. But ranking
quality and prediction accuracy are different things.

Game outcomes are mostly irreducible noise -- tools/accuracy_ceiling.py
shows even a hindsight-perfect model only reaches ~80%. So prediction
accuracy is dominated by a term that no amount of learning can reduce, and
it cannot move much. That flat line is a ceiling effect, not evidence that
the ratings are static.

The ranking itself is a cleaner measure of what the model knows. Here we
ask: how close is the week-N ranking to the final end-of-season ranking?
Measured by Kendall tau (rank correlation, 1.0 = identical ordering) and
by how many of the final top 25 are already in the week-N top 25.

If those climb while prediction accuracy stays flat, the answer to "does
the top 25 get more accurate with time" is yes -- and the reason it does
not show up as better predictions is that football is noisy, not that the
rankings stopped improving.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

SEASONS = (2021, 2022, 2023, 2024, 2025)
FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}


def _ranking(ratings: dict[str, float]) -> dict[str, int]:
    ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
    return {team: rank for rank, (team, _) in enumerate(ordered, 1)}


def evaluate(season: int) -> dict[int, tuple[float, int]]:
    games = load_season(season)
    prior = build_prior_or_empty(season) or None

    final = fit(games, prior=prior, **FITTED)
    final_rank = _ranking(final.fbs_ratings())
    final_top25 = {t for t, r in final_rank.items() if r <= 25}

    out: dict[int, tuple[float, int]] = {}
    for week in sorted({g.week for g in games}):
        history = [g for g in games if g.week <= week]
        if len(history) < 50:
            continue
        model = fit(history, prior=prior, **FITTED)
        week_rank = _ranking(model.fbs_ratings())

        shared = sorted(set(week_rank) & set(final_rank))
        if len(shared) < 20:
            continue
        tau = kendalltau(
            [week_rank[t] for t in shared],
            [final_rank[t] for t in shared],
        ).statistic
        overlap = len({t for t in shared if week_rank[t] <= 25} & final_top25)
        out[week] = (float(tau), overlap)
    return out


def main() -> None:
    have = set(available_seasons())
    pooled: dict[int, list[tuple[float, int]]] = defaultdict(list)

    for season in SEASONS:
        if season not in have:
            continue
        for week, stats in evaluate(season).items():
            pooled[week].append(stats)

    print("How close is the week-N ranking to the FINAL ranking?")
    print("(pooled across seasons; tau 1.0 = identical order)\n")
    print(f"{'week':>5} {'tau':>7} {'top25 matched':>15}")
    print("-" * 30)
    for week in sorted(pooled):
        entries = pooled[week]
        tau = float(np.mean([e[0] for e in entries]))
        overlap = float(np.mean([e[1] for e in entries]))
        bar = "#" * int(round(tau * 40))
        print(f"{week:>5} {tau:>7.3f} {overlap:>10.1f} / 25  {bar}")


if __name__ == "__main__":
    main()
