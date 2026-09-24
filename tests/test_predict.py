"""Tests for prediction and backtesting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank.backtest import walk_forward  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5, "margin_scale": 28.0, "halflife": 1e6}


@pytest.fixture(scope="module")
def predictor_2024():
    games = load_season(2024)
    return Predictor.from_model(fit(games, **FITTED), games)


@pytest.fixture(scope="module")
def backtest_2024():
    return walk_forward(load_season(2024), 2024)


def test_prediction_has_all_parts(predictor_2024):
    prediction = predictor_2024.predict("Ohio State", "Michigan")
    assert prediction.home_team == "Ohio State"
    assert 0.0 <= prediction.home_win_probability <= 1.0
    assert prediction.error_std > 0
    assert prediction.factors


def test_scores_reconstruct_the_margin(predictor_2024):
    prediction = predictor_2024.predict("Georgia", "Alabama")
    assert (prediction.home_score - prediction.away_score) == pytest.approx(
        prediction.predicted_margin)


def test_scores_are_realistic(predictor_2024):
    prediction = predictor_2024.predict("Texas", "Oklahoma")
    assert 0 < prediction.home_score < 80
    assert 0 < prediction.away_score < 80


def test_stronger_team_gets_higher_win_probability(predictor_2024):
    strong = predictor_2024.predict(
        "Ohio State", "Kent State", neutral_site=True)
    even = predictor_2024.predict(
        "Ohio State", "Notre Dame", neutral_site=True)
    assert strong.home_win_probability > even.home_win_probability
    assert strong.home_win_probability > 0.85


def test_win_probability_tracks_margin_direction(predictor_2024):
    prediction = predictor_2024.predict(
        "Kent State", "Ohio State", neutral_site=True)
    assert prediction.predicted_margin < 0
    assert prediction.home_win_probability < 0.5


def test_explain_mentions_both_teams(predictor_2024):
    text = predictor_2024.predict("Oregon", "Washington").explain()
    assert "Oregon" in text and "Washington" in text
    assert "Predicted" in text and "Win prob" in text


def test_favourite_matches_margin_sign(predictor_2024):
    prediction = predictor_2024.predict(
        "Ohio State", "Kent State", neutral_site=True)
    assert prediction.favourite == "Ohio State"
    assert prediction.spread > 0


def test_backtest_beats_the_home_team_baseline(backtest_2024):
    """The bar every rating system must clear. Always picking the home
    team wins ~57% of the time; a real model must do better."""
    assert backtest_2024.accuracy > backtest_2024.home_baseline_accuracy
    assert backtest_2024.accuracy > 0.65


def test_backtest_error_is_reasonable(backtest_2024):
    """Published computer ratings land around 13-16 points MAE in CFB."""
    assert 10.0 < backtest_2024.mae < 18.0


def test_backtest_is_well_calibrated(backtest_2024):
    """Brier below 0.25 beats always guessing 50%."""
    assert backtest_2024.brier < 0.25


def test_backtest_predicted_a_full_season(backtest_2024):
    assert backtest_2024.n_predicted > 800
    assert len(backtest_2024.by_week) > 8


def test_unknown_team_rates_as_average(predictor_2024):
    """A team the model has never seen gets a neutral rating rather than
    an exception -- important for early-season and new FBS members."""
    prediction = predictor_2024.predict("Ohio State", "Nonexistent Tech")
    assert prediction.predicted_margin > 0
