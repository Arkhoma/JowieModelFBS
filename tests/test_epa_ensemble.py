"""Tests for EPA aggregation and the margin/EPA ensemble.

These pin findings that were expensive to discover and easy to undo:
a ridge penalty that silently broke when the observation scale changed,
and the wrong assumption that EPA should replace scoring margin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank import epa_games as eg  # noqa: E402
from cfbrank.ensemble import EnsembleModel, fit_weights  # noqa: E402
from cfbrank.epa_ridge import (  # noqa: E402
    EPAGame, build_epa_games, fit_epa_ratings, fit_epa_ratings_primed,
    get_ep_model, scaled_lambda, translate_margin_prior,
)
from cfbrank.games import load_season  # noqa: E402
from cfbrank.ridge import fit as fit_margin  # noqa: E402

MARGIN_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
SEASON = 2024


@pytest.fixture(scope="module")
def games():
    return load_season(SEASON)


@pytest.fixture(scope="module")
def epa_games(games):
    return build_epa_games(SEASON, games, get_ep_model())


@pytest.fixture(scope="module")
def epa_model(epa_games):
    return fit_epa_ratings(epa_games)


@pytest.fixture(scope="module")
def margin_model(games):
    return fit_margin(games, **MARGIN_PARAMS)


def test_garbage_time_flags_blowouts():
    """A 4th-quarter 30-point lead is over; a 1st-quarter one is not."""
    frame = pd.DataFrame({
        "period": [4, 1, 2, 4],
        "score_margin": [30.0, 30.0, 5.0, -30.0],
    })
    flags = eg.is_garbage_time(frame).tolist()
    assert flags == [True, False, False, True]


def test_garbage_time_excludes_overtime():
    """Overtime is never garbage time -- period 5 is beyond the table."""
    frame = pd.DataFrame({"period": [5], "score_margin": [40.0]})
    assert not eg.is_garbage_time(frame).iloc[0]


def test_success_rate_thresholds():
    """50% on 1st, 70% on 2nd, 100% on 3rd and 4th."""
    frame = pd.DataFrame({
        "game_id": ["g"] * 4,
        "team": ["A"] * 4,
        "half": [1] * 4,
        "down": [1, 2, 3, 1],
        "distance": [10.0, 10.0, 10.0, 10.0],
        # Field position falls by the yards gained on each play.
        "yards_to_goal": [50.0, 45.0, 38.0, 45.0],
        "is_touchdown": [False] * 4,
    })
    # Gains are 5, 7, and NaN for the last row (no following play).
    success = eg.is_successful(frame).tolist()
    assert success[0] is True or success[0] == True  # 5/10 >= 0.5  # noqa: E712
    assert success[1] is True or success[1] == True  # 7/10 >= 0.7  # noqa: E712
    assert success[2] is False or success[2] == False  # needs 10  # noqa: E712


def test_yards_gained_ignores_possession_change():
    """Differencing field position across a turnover is meaningless."""
    frame = pd.DataFrame({
        "game_id": ["g", "g"],
        "team": ["A", "B"],
        "half": [1, 1],
        "yards_to_goal": [50.0, 60.0],
        "is_touchdown": [False, False],
    })
    gained = eg.yards_gained(frame)
    assert pd.isna(gained.iloc[0])


def test_scaled_lambda_tracks_variance():
    """The penalty must mean the same thing on any observation scale.

    Regression guard for the bug that made EPA lose: lambda=5 was tuned
    on margins (sd ~20.7) and reused on EPA edges (sd ~0.78), which made
    it roughly 26^2 times too harsh and crushed the ratings.
    """
    small = np.random.default_rng(0).normal(0, 0.78, 5000)
    large = np.random.default_rng(0).normal(0, 20.7, 5000)
    assert scaled_lambda(large) > scaled_lambda(small) * 100


def test_epa_ratings_recover_2024_reality(epa_model):
    """Opponent-adjusted efficiency must find the good teams."""
    ranked = sorted(epa_model.fbs_ratings().items(), key=lambda kv: -kv[1])
    top_ten = [team for team, _ in ranked[:10]]
    assert "Ohio State" in top_ten
    assert {"Notre Dame", "Oregon", "Texas"} & set(top_ten)


def test_points_per_epa_is_positive_and_sane(epa_model):
    """More efficiency must mean more points, at a believable rate."""
    assert 10.0 < epa_model.points_per_epa < 60.0


def test_epa_home_field_is_plausible(epa_model):
    assert 0.5 < epa_model.home_field < 6.0


def test_ensemble_blends_both_models(margin_model, epa_model):
    """The combined prediction must respond to both components.

    Note it does NOT sit between them: the stacked weights sum to 0.848,
    so the blend is deliberately pulled toward zero relative to either
    model alone. Both components are individually over-confident and
    least squares shrinks them. An earlier version of this test asserted
    betweenness and failed for exactly that reason -- the test was wrong,
    not the model.
    """
    ensemble = EnsembleModel(margin_model, epa_model)
    home, away = "Ohio State", "Michigan"
    from_margin = margin_model.predict_margin(home, away)
    from_epa = epa_model.predict_margin(home, away)
    combined = ensemble.predict_margin(home, away)

    expected = (ensemble.margin_weight * from_margin
                + ensemble.epa_weight * from_epa)
    assert combined == pytest.approx(expected)

    # Same sign as its inputs, and shrunk toward zero.
    assert combined > 0
    assert abs(combined) < max(abs(from_margin), abs(from_epa))


def test_ensemble_falls_back_when_epa_missing(margin_model, epa_model):
    """A team with no play data still gets a prediction."""
    ensemble = EnsembleModel(margin_model, epa_model)
    known = "Ohio State"
    prediction = ensemble.predict_margin(known, "Not A Real Team", True)
    assert isinstance(prediction, float)


def test_ensemble_exposes_rating_result_interface(margin_model, epa_model):
    """Predictor and the templates accept either model type.

    Regression guard: the first wiring attempt crashed on
    model.calibration because the ensemble did not implement the full
    interface.
    """
    ensemble = EnsembleModel(margin_model, epa_model)
    for attribute in ("calibration", "fcs_offset", "residual_std",
                      "n_games", "n_teams", "home_field", "ratings",
                      "divisions"):
        assert hasattr(ensemble, attribute), f"missing {attribute}"


def test_fit_weights_recovers_known_combination():
    """Stacking must recover weights it was given, on clean data."""
    rng = np.random.default_rng(3)
    first = rng.normal(0, 10, 500)
    second = rng.normal(0, 10, 500)
    actual = 0.6 * first + 0.3 * second
    weight_one, weight_two = fit_weights(first, second, actual)
    assert weight_one == pytest.approx(0.6, abs=0.02)
    assert weight_two == pytest.approx(0.3, abs=0.02)


def test_ensemble_weights_are_both_positive():
    """Both models must contribute.

    Measured at margin +0.679, EPA +0.211 on 2024, validated on 2025
    (re-measured after fixing the prior-scale bug -- see
    epa_ridge.fit_epa_ratings_primed).
    If either drops to zero, the ensemble is pointless and we should say
    so rather than ship a model that quietly ignores half its inputs.
    """
    from cfbrank.ensemble import DEFAULT_EPA_WEIGHT, DEFAULT_MARGIN_WEIGHT
    assert DEFAULT_MARGIN_WEIGHT > 0.3
    assert DEFAULT_EPA_WEIGHT > 0.1
    # Deliberately below 1.0: both components are over-confident, and
    # least squares shrinks the combination accordingly.
    assert DEFAULT_MARGIN_WEIGHT + DEFAULT_EPA_WEIGHT < 1.0


def _synthetic_epa_games() -> list[EPAGame]:
    """A hand-built league where one team's whole sample is two blowouts.

    Mirrors the New Mexico case that motivated fit_epa_ratings_primed:
    a team with no track record puts up huge efficiency numbers against
    weak opposition in a 3-game sample, and unprimed ridge (shrinking
    only toward zero) has nothing to weigh that against.
    """
    games = []
    matchups = [
        ("NM", "Nobody1", 1.9, "fbs"),
        ("NM", "Nobody2", 1.9, "fbs"),
        ("Nobody1", "Nobody2", 0.0, "fbs"),
        ("Nobody1", "Steady", -0.2, "fbs"),
        ("Nobody2", "Steady", -0.2, "fbs"),
        ("Steady", "NM", -0.3, "fbs"),
        # One FCS crossover so the offset column isn't all zero --
        # without it the design matrix is singular (real seasons always
        # have ~121 of these, so production code never hits this).
        ("Steady", "SmallFCS", 0.3, "fcs"),
    ]
    for week, (home, away, edge, away_division) in enumerate(matchups, start=1):
        games.append(EPAGame(
            game_id=str(week), season=2099, week=week,
            home_team=home, away_team=away,
            home_division="fbs", away_division=away_division,
            neutral_site=False, epa_edge=edge, margin=int(edge * 25),
        ))
    return games


def test_translate_margin_prior_converts_units():
    """The translated prior is a straight unit change, not a new opinion."""
    margin_prior = {"NM": 5.0, "Steady": -2.5}
    translated = translate_margin_prior(margin_prior, points_per_epa=25.0)
    assert translated["NM"] == pytest.approx(0.2)
    assert translated["Steady"] == pytest.approx(-0.1)


def test_translate_margin_prior_handles_zero_conversion():
    """A degenerate conversion factor must not raise or divide by zero."""
    assert translate_margin_prior({"NM": 5.0}, points_per_epa=0.0) == {}


def test_primed_epa_rating_pulls_toward_a_skeptical_prior():
    """Regression guard for the New Mexico case.

    A team with only blowout wins over weak opponents and no track
    record should rate LOWER once its historical prior says it is
    nothing special -- the whole point of sharing the margin model's
    prior instead of shrinking to zero.
    """
    games = _synthetic_epa_games()
    unprimed = fit_epa_ratings(games)

    margin_prior = {"NM": 0.5, "Steady": 3.0}
    primed = fit_epa_ratings_primed(games, margin_prior)

    assert primed.rating("NM") < unprimed.rating("NM")
    # The move is real but modest here, and that is itself an honest
    # finding: three games is very little evidence, and the default
    # ridge penalty (RELATIVE_LAMBDA=8, tuned for scale-matching, not
    # three-game stability) does not lean hard on the prior when the
    # in-season signal is this lopsided. Flagged as follow-up work
    # rather than silently accepted -- see MODEL_SPEC.md.


def test_primed_epa_rating_falls_back_without_a_prior():
    """No margin prior (e.g. no previous season on disk) -> unprimed fit."""
    games = _synthetic_epa_games()
    unprimed = fit_epa_ratings(games)
    primed = fit_epa_ratings_primed(games, margin_prior=None)
    assert primed.ratings == unprimed.ratings
