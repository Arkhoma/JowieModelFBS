"""Walk-forward backtesting -- the honesty layer.

Trains on weeks 1..N, predicts week N+1, never letting future results
leak backward. This is the only fair way to ask "would this model have
been right?"

Reported metrics:
  MAE        mean absolute error on margin, in points
  accuracy   straight win/loss hit rate
  Brier      calibration of the win probabilities (lower is better)
  baseline   what you get by always picking the home team

A model that cannot beat the home-team baseline is not a model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .games import Game
from .predict import Predictor
from .ridge import fit as fit_ridge

MIN_TRAINING_GAMES = 50


@dataclass(slots=True)
class BacktestResult:
    """Out-of-sample accuracy for one season."""

    season: int
    n_predicted: int
    mae: float
    accuracy: float
    brier: float
    home_baseline_accuracy: float
    by_week: dict[int, dict]

    def summary(self) -> str:
        lines = [
            f"Season {self.season}: {self.n_predicted} games predicted "
            "out of sample",
            f"  MAE            {self.mae:6.2f} points",
            f"  Win accuracy   {self.accuracy:6.1%}",
            f"  Brier score    {self.brier:6.4f}",
            f"  Home baseline  {self.home_baseline_accuracy:6.1%}",
        ]
        edge = self.accuracy - self.home_baseline_accuracy
        lines.append(f"  Edge over baseline {edge:+.1%}")
        return "\n".join(lines)


def walk_forward(
    games: list[Game],
    season: int,
    lambda_: float = 5.0,
    margin_scale: float = 28.0,
    halflife: float = 1e6,
    start_week: int = 4,
    prior: dict[str, float] | None = None,
    min_training_games: int = MIN_TRAINING_GAMES,
) -> BacktestResult:
    """Predict each week using only games played before it.

    A `prior` (last season's ratings) is legitimate here and does not
    leak: it is information that genuinely existed before kickoff. It
    also lets us score the earliest weeks, where there is otherwise too
    little current-season evidence to fit anything.
    """
    season_games = sorted(
        (g for g in games if g.season == season),
        key=lambda g: g.week,
    )
    if not season_games:
        raise ValueError(f"No games found for season {season}")

    weeks = sorted({g.week for g in season_games})
    errors: list[float] = []
    correct: list[bool] = []
    brier_terms: list[float] = []
    home_wins: list[bool] = []
    by_week: dict[int, dict] = {}

    for week in weeks:
        if week < start_week:
            continue
        history = [g for g in season_games if g.week < week]
        upcoming = [g for g in season_games if g.week == week]
        if len(history) < min_training_games or not upcoming:
            continue

        model = fit_ridge(history, lambda_, margin_scale, halflife,
                          prior=prior)
        predictor = Predictor.from_model(model, history)

        week_errors: list[float] = []
        week_correct: list[bool] = []

        for game in upcoming:
            # Only score teams the training data has actually seen.
            if (game.home_team not in model.ratings
                    or game.away_team not in model.ratings):
                continue
            prediction = predictor.predict(
                game.home_team, game.away_team, game.neutral_site)

            error = game.margin - prediction.predicted_margin
            hit = (prediction.predicted_margin > 0) == (game.margin > 0)
            actual_home_win = 1.0 if game.margin > 0 else 0.0

            week_errors.append(abs(error))
            week_correct.append(hit)
            errors.append(abs(error))
            correct.append(hit)
            brier_terms.append(
                (prediction.home_win_probability - actual_home_win) ** 2)
            home_wins.append(game.margin > 0)

        if week_errors:
            by_week[week] = {
                "n": len(week_errors),
                "mae": float(np.mean(week_errors)),
                "accuracy": float(np.mean(week_correct)),
            }

    if not errors:
        raise ValueError(
            f"No out-of-sample predictions produced for {season}. "
            "Season may be too short to backtest."
        )

    return BacktestResult(
        season=season,
        n_predicted=len(errors),
        mae=float(np.mean(errors)),
        accuracy=float(np.mean(correct)),
        brier=float(np.mean(brier_terms)),
        home_baseline_accuracy=float(np.mean(home_wins)),
        by_week=by_week,
    )


def backtest_seasons(
    games_by_season: dict[int, list[Game]], **kwargs
) -> dict[int, BacktestResult]:
    """Run the walk-forward test across several seasons."""
    results = {}
    for season, games in sorted(games_by_season.items()):
        try:
            results[season] = walk_forward(games, season, **kwargs)
        except ValueError:
            continue
    return results


def aggregate(results: dict[int, BacktestResult]) -> dict:
    """Pool several seasons into one honest headline number."""
    if not results:
        return {}
    weights = np.array([r.n_predicted for r in results.values()], dtype=float)
    return {
        "seasons": sorted(results),
        "n_predicted": int(weights.sum()),
        "mae": float(np.average(
            [r.mae for r in results.values()], weights=weights)),
        "accuracy": float(np.average(
            [r.accuracy for r in results.values()], weights=weights)),
        "brier": float(np.average(
            [r.brier for r in results.values()], weights=weights)),
        "home_baseline": float(np.average(
            [r.home_baseline_accuracy for r in results.values()],
            weights=weights)),
    }
