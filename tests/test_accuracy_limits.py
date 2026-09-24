"""Tests for the accuracy-ceiling and convergence findings.

These pin conclusions that drive product decisions, so that a future
change cannot quietly invalidate them without the suite noticing:

- The model is much closer to the noise floor than to perfection, so
  "just make it more accurate" has a hard limit.
- The ranking genuinely converges even though prediction accuracy is
  flat across the season.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfbrank.backtest import walk_forward
from cfbrank.games import available_seasons, load_season
from cfbrank.predict import Predictor
from cfbrank.prior import build_prior_or_empty
from cfbrank.ridge import fit

FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
SEASON = 2024

pytestmark = pytest.mark.skipif(
    SEASON not in available_seasons(),
    reason=f"{SEASON} schedule data not on disk",
)


@pytest.fixture(scope="module")
def games():
    return load_season(SEASON)


@pytest.fixture(scope="module")
def oracle(games):
    """Hindsight model: fit on the whole season, scored on that season."""
    model = fit(games, **FITTED)
    predictor = Predictor.from_model(model, games)
    errors, correct = [], []
    for game in games:
        if not (game.home_team in model.ratings
                and game.away_team in model.ratings):
            continue
        margin = predictor.predict(
            game.home_team, game.away_team, game.neutral_site
        ).predicted_margin
        errors.append(abs(game.margin - margin))
        correct.append((margin > 0) == (game.margin > 0))
    return float(np.mean(errors)), float(np.mean(correct))


@pytest.fixture(scope="module")
def live(games):
    return walk_forward(
        games, SEASON, prior=build_prior_or_empty(SEASON) or None)


class TestNoiseFloor:
    """College football is mostly irreducible variance."""

    def test_even_hindsight_cannot_predict_most_of_the_margin(self, oracle):
        oracle_mae, _ = oracle
        # A model that already knows how the season turned out still misses
        # by ~10.5 points a game. That is the noise floor, not a defect.
        assert 8.0 < oracle_mae < 13.0

    def test_even_hindsight_loses_one_game_in_five(self, oracle):
        _, oracle_accuracy = oracle
        assert 0.75 < oracle_accuracy < 0.87

    def test_live_model_is_nearer_the_floor_than_to_perfection(
            self, live, oracle):
        """The remaining headroom is smaller than the noise floor.

        This is the finding that says 'stop optimising accuracy'. If it
        ever flips, there is real signal being left on the table and the
        advice should change.
        """
        oracle_mae, _ = oracle
        headroom = live.mae - oracle_mae
        assert headroom < oracle_mae


class TestRankingConverges:
    """Prediction accuracy is flat, but the ranking is not."""

    def test_later_rankings_agree_more_with_the_final_ranking(self, games):
        final = fit(games, **FITTED).fbs_ratings()

        def agreement(week: int) -> float:
            history = [g for g in games if g.week <= week]
            partial = fit(history, **FITTED).fbs_ratings()
            shared = sorted(set(partial) & set(final))
            return float(np.corrcoef(
                [partial[t] for t in shared],
                [final[t] for t in shared],
            )[0, 1])

        assert agreement(11) > agreement(4)

    def test_early_ranking_is_already_informative(self, games):
        """Week 4 is noisy but not random -- the prior does real work."""
        final = fit(games, **FITTED).fbs_ratings()
        early = fit([g for g in games if g.week <= 4], **FITTED).fbs_ratings()
        shared = sorted(set(early) & set(final))
        correlation = float(np.corrcoef(
            [early[t] for t in shared], [final[t] for t in shared])[0, 1])
        assert correlation > 0.5
