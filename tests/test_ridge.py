"""Tests for the ridge rating engine.

The headline assertions are ground-truth checks: the 2024 season really
did end with Ohio State beating Notre Dame for the title, and Kent State
really did go 0-12. If the engine cannot recover that, it is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank import ridge  # noqa: E402
from cfbrank.games import load_season  # noqa: E402

# Parameters chosen by cross-validation on 2024 (MAE 14.23 points).
FITTED = {"lambda_": 5, "margin_scale": 28.0, "halflife": 1e6}


@pytest.fixture(scope="module")
def model_2024():
    return ridge.fit(load_season(2024), **FITTED)


def test_margin_curve_is_concave():
    """Blowouts must show diminishing returns, not linear credit."""
    values = ridge.curve_margin(np.array([7, 14, 28, 56]))
    first_step = values[1] - values[0]
    last_step = values[3] - values[2]
    assert last_step < first_step
    assert values[3] > values[2]  # still monotonic


def test_margin_curve_is_odd_symmetric():
    positive = ridge.curve_margin(np.array([21.0]))[0]
    negative = ridge.curve_margin(np.array([-21.0]))[0]
    assert positive == pytest.approx(-negative)


def test_recency_weights_decay():
    weights = ridge.recency_weights(
        np.array([1.0, 5.0, 9.0]), current_week=9.0, halflife=4.0)
    assert weights[2] > weights[1] > weights[0]
    assert weights[2] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5)


def test_home_field_is_plausible(model_2024):
    """Decades of football say home field is worth ~2-4 points. We do not
    assert the value, we fit it -- this just checks sanity."""
    assert 1.0 < model_2024.home_field < 5.0


def test_fcs_offset_is_large_and_negative(model_2024):
    """The FBS/FCS gap is real and roughly two to three touchdowns.

    Without this fitted term, FCS teams float to the top of the table:
    33 of them never play an FBS opponent, so the regression cannot
    otherwise learn that their pool sits lower.
    """
    assert -30.0 < model_2024.fcs_offset < -8.0


def test_top_teams_match_2024_reality(model_2024):
    ranked = sorted(model_2024.fbs_ratings().items(), key=lambda kv: -kv[1])
    top_five = [team for team, _ in ranked[:5]]
    # Ohio State beat Notre Dame in the national championship game.
    assert top_five[0] == "Ohio State"
    assert "Notre Dame" in top_five[:3]
    assert {"Texas", "Penn State", "Oregon"} & set(top_five)


def test_worst_teams_match_2024_reality(model_2024):
    ranked = sorted(model_2024.fbs_ratings().items(), key=lambda kv: kv[1])
    bottom_five = [team for team, _ in ranked[:5]]
    assert "Kent State" in bottom_five  # 0-12


def test_no_fcs_team_outranks_the_best_fbs_team(model_2024):
    """Regression guard. Before the division offset existed, North Dakota
    State rated 3rd overall, above Texas and Oregon."""
    best_fbs = max(model_2024.fbs_ratings().values())
    best_fcs = max(
        model_2024.rating(team)
        for team, division in model_2024.divisions.items()
        if division == "fcs"
    )
    assert best_fcs < best_fbs


def test_fcs_teams_are_individually_distinguished(model_2024):
    """Howie's requirement: a win over Campbell is not a win over Montana.

    Montana State went 15-1 in 2024; Campbell went 3-9. Pooling FCS into
    one constant would erase this.
    """
    montana_state = model_2024.rating("Montana State")
    campbell = model_2024.rating("Campbell")
    assert montana_state - campbell > 10.0


def test_predictions_are_symmetric(model_2024):
    """Swapping home and away must flip the margin, net of home field."""
    forward = model_2024.predict_margin("Ohio State", "Michigan")
    reverse = model_2024.predict_margin("Michigan", "Ohio State")
    assert forward + reverse == pytest.approx(2 * model_2024.home_field)


def test_neutral_site_removes_home_field(model_2024):
    home = model_2024.predict_margin("Alabama", "Georgia", neutral_site=False)
    neutral = model_2024.predict_margin("Alabama", "Georgia", neutral_site=True)
    assert home - neutral == pytest.approx(model_2024.home_field)


def test_better_team_is_favoured(model_2024):
    """Ohio State was far better than Kent State in 2024."""
    margin = model_2024.predict_margin(
        "Ohio State", "Kent State", neutral_site=True)
    assert margin > 20.0


def test_ratings_are_centred(model_2024):
    """FBS ratings average to zero, so the scale reads as points above or
    below an average FBS team."""
    values = list(model_2024.fbs_ratings().values())
    assert abs(float(np.mean(values))) < 1.0


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        ridge.fit([])
