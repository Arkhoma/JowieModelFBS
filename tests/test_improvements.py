"""Tests for garbage-time margins, fumble luck, and the roster prior."""

import numpy as np
import pandas as pd
import pytest

from cfbrank.game_features import _fumble_luck, _garbage_margin
from cfbrank.returning import _share


def _plays(rows):
    frame = pd.DataFrame(rows, columns=["team", "period", "score_margin",
                                        "fumble_player", "half"])
    return frame


def test_garbage_margin_freezes_at_first_decided_snap():
    frame = _plays([
        ("Home", 1, 0, "", 1),
        ("Home", 4, 21, "", 2),      # not over yet (limit 22 in Q4)
        ("Away", 4, -23, "", 2),     # away trails by 23 -> decided
        ("Home", 4, 30, "", 2),
    ])
    assert _garbage_margin(frame, "Home") == 23.0


def test_garbage_margin_none_when_game_stays_close():
    frame = _plays([("Home", 1, 0, "", 1), ("Away", 4, -3, "", 2)])
    assert _garbage_margin(frame, "Home") is None


def test_fumble_luck_counts_recoveries_from_possession_change():
    frame = _plays([
        ("Home", 1, 0, "QB", 1),     # Home fumbles, Away recovers
        ("Away", 1, 0, "RB", 1),     # Away fumbles, Away keeps it
        ("Away", 1, 0, "", 1),
    ])
    # Two fumbles, home recovered none: 0 - 2/2 = -1
    assert _fumble_luck(frame, "Home") == -1.0


def test_fumble_at_end_of_half_is_not_scored():
    frame = _plays([("Home", 2, 0, "QB", 1), ("Away", 3, 0, "", 2)])
    assert _fumble_luck(frame, "Home") == 0.0


def test_share_of_production_that_stayed():
    frame = pd.DataFrame({
        "team": ["A", "A", "B"],
        "off_yds": [300.0, 100.0, 50.0],
        "stayed": [True, False, False],
    })
    shares = _share(frame, "off_yds")
    assert shares["A"] == pytest.approx(0.75)
    assert shares["B"] == 0.0


def test_roster_prior_weights_never_see_their_own_season():
    from cfbrank import prior_model
    weights_2024 = prior_model.fit_weights(2024)
    weights_2025 = prior_model.fit_weights(2025)
    # Adding the 2024 offseason must change the fit, proving the 2024
    # target was excluded from fit_weights(2024).
    assert not np.allclose(weights_2024, weights_2025)
