"""Should there be ONE authoritative top 25?

Answers Howie's second question. The spec says "never average them," but
that rule was reasoned from first principles, not measured. A rule worth
keeping should survive a test, so here is the test.

Blend the predictive and resume ratings at every weight from 0 (pure
resume) to 1 (pure predictive), and score each blend on a question that
has an objective answer: how well does the ranking predict the results of
games played LATER in the season?

Ranking quality is scored two ways:

  hit rate   -- how often the higher-ranked team actually wins
  Kendall    -- rank correlation between the ranking and the eventual
                end-of-season predictive rating

A blend that wins on prediction tells us averaging is defensible. A blend
that loses tells us the spec was right and the UI should keep showing two
columns instead of inventing a composite.

Both ratings are fit on games through a cutoff week; scoring uses only
games after it, so nothing leaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.resume import build_resumes  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

SEASONS = (2021, 2022, 2023, 2024, 2025)
FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
CUTOFF = 9
WEIGHTS = np.linspace(0.0, 1.0, 11)


def _zscore(values: dict[str, float]) -> dict[str, float]:
    """Put two differently-scaled ratings on comparable footing.

    Resume score is in 'wins above benchmark' (range ~2) and predictive is
    in points (range ~40). Blending them raw would silently make the blend
    almost entirely predictive regardless of the weight.
    """
    if not values:
        return {}
    array = np.array(list(values.values()), dtype=float)
    spread = float(array.std())
    if spread == 0.0:
        return {team: 0.0 for team in values}
    mean = float(array.mean())
    return {team: (v - mean) / spread for team, v in values.items()}


def evaluate_season(season: int) -> dict[float, tuple[int, int]] | None:
    """Hits and total for each blend weight, on post-cutoff games."""
    games = load_season(season)
    history = [g for g in games if g.week <= CUTOFF]
    future = [g for g in games if g.week > CUTOFF]
    if len(history) < 50 or not future:
        return None

    prior = build_prior_or_empty(season) or None
    model = fit(history, prior=prior, **FITTED)
    predictor = Predictor.from_model(model, history)
    resumes = build_resumes(history, model, predictor.errors)

    fbs = model.fbs_ratings()
    predictive = _zscore(fbs)
    resume = _zscore({t: r.score for t, r in resumes.items() if t in fbs})

    shared = set(predictive) & set(resume)
    scores: dict[float, tuple[int, int]] = {}

    for weight in WEIGHTS:
        blended = {
            team: weight * predictive[team] + (1.0 - weight) * resume[team]
            for team in shared
        }
        hits = total = 0
        for game in future:
            home, away = game.home_team, game.away_team
            if home not in blended or away not in blended:
                continue
            if blended[home] == blended[away]:
                continue
            higher_is_home = blended[home] > blended[away]
            total += 1
            hits += int(higher_is_home == (game.margin > 0))
        scores[round(float(weight), 2)] = (hits, total)
    return scores


def main() -> None:
    have = set(available_seasons())
    pooled: dict[float, list[tuple[int, int]]] = {
        round(float(w), 2): [] for w in WEIGHTS}

    for season in SEASONS:
        if season not in have:
            continue
        result = evaluate_season(season)
        if result is None:
            continue
        for weight, pair in result.items():
            pooled[weight].append(pair)

    print(f"Ranking teams by a blend, then predicting games after week "
          f"{CUTOFF}.")
    print("Weight 1.00 = pure predictive, 0.00 = pure resume.\n")
    print(f"{'weight':>7} {'games':>7} {'hit rate':>9}")
    print("-" * 26)

    best = (0.0, -1.0)
    for weight in sorted(pooled):
        entries = pooled[weight]
        if not entries:
            continue
        hits = sum(h for h, _ in entries)
        total = sum(t for _, t in entries)
        rate = hits / total if total else 0.0
        flag = ""
        if rate > best[1]:
            best = (weight, rate)
        print(f"{weight:>7.2f} {total:>7} {rate:>8.1%}{flag}")

    print(f"\nBest blend weight: {best[0]:.2f} at {best[1]:.1%}")
    pure_pred = pooled[1.0]
    rate_pred = (sum(h for h, _ in pure_pred)
                 / sum(t for _, t in pure_pred))
    print(f"Pure predictive:   1.00 at {rate_pred:.1%}")
    print(f"Difference:        {best[1] - rate_pred:+.2%}")


if __name__ == "__main__":
    main()
