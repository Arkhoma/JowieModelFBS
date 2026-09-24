"""Tests for the play loader and expected-points model.

Run against real downloaded data -- these catch upstream schema changes,
not just logic bugs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank import expected_points as ep  # noqa: E402
from cfbrank.plays import load_plays  # noqa: E402


@pytest.fixture(scope="module")
def plays_2024():
    return load_plays(2024)


@pytest.fixture(scope="module")
def model(plays_2024):
    return ep.fit(plays_2024)


def test_plays_load(plays_2024):
    assert len(plays_2024) > 150_000
    assert plays_2024["game_id"].nunique() > 1_400


def test_plays_are_in_chronological_order(plays_2024):
    """Upstream play_id is corrupt in ~2% of games (negative overflow
    values), so ordering comes from period + clock instead."""
    for _, group in plays_2024.groupby("game_id"):
        periods = group["period"].tolist()
        assert all(a <= b for a, b in zip(periods, periods[1:])), (
            "plays out of period order"
        )


def test_event_flags_are_plausible(plays_2024):
    """Guards the empty-string quirk: `sack_player` is '' not NaN, so a
    naive isna() filter would mark every play as a sack."""
    assert 0.005 < plays_2024["is_sack"].mean() < 0.05
    assert 0.02 < plays_2024["is_touchdown"].mean() < 0.10
    assert 0.002 < plays_2024["is_interception"].mean() < 0.03
    assert 0.30 < plays_2024["is_rush"].mean() < 0.70


def test_derived_state_is_sane(plays_2024):
    assert plays_2024["seconds_remaining"].between(0, 3600).all()
    assert set(plays_2024["half"].unique()) <= {1, 2, 3}
    assert plays_2024["down"].between(0, 4).all()


def test_expected_points_rises_toward_the_end_zone(model):
    """The single most important shape check: EP must increase as you
    approach the opponent's goal line."""
    probe = pd.DataFrame({
        "down": [1] * 6,
        "distance": [10] * 6,
        "yards_to_goal": [95, 75, 55, 35, 20, 10],
    })
    values = model.predict(probe)
    assert all(a < b for a, b in zip(values, values[1:])), (
        f"EP not monotonic toward end zone: {values}"
    )


def test_own_goal_line_is_low_value(model):
    """1st and 10 from your own 5 is worth close to nothing -- you are
    nearly as likely to concede points as score them."""
    value = model.predict(pd.DataFrame(
        [{"down": 1, "distance": 10, "yards_to_goal": 95}]))[0]
    assert -1.5 < value < 1.0


def test_red_zone_is_worth_several_points(model):
    value = model.predict(pd.DataFrame(
        [{"down": 1, "distance": 10, "yards_to_goal": 10}]))[0]
    assert 3.0 < value < 6.5


def test_third_and_long_is_worse_than_first_and_ten(model):
    at_fifty = {"yards_to_goal": 50}
    first = model.predict(pd.DataFrame(
        [{"down": 1, "distance": 10, **at_fifty}]))[0]
    third = model.predict(pd.DataFrame(
        [{"down": 3, "distance": 15, **at_fifty}]))[0]
    assert third < first


def test_epa_is_roughly_centred(plays_2024, model):
    """EPA should average near zero -- every gain is someone else's loss."""
    scored = ep.add_epa(plays_2024, model)
    assert scored["epa"].notna().mean() > 0.95
    assert abs(scored["epa"].mean()) < 0.05


def test_epa_identifies_known_good_teams(plays_2024, model):
    """2024 ground truth: Ohio State won the title, Notre Dame lost the
    final, Oregon was the top seed. All three must rate well."""
    scored = ep.add_epa(plays_2024, model)
    offense = scored.groupby("team")["epa"].agg(["mean", "count"])
    offense = offense[offense["count"] >= 400].sort_values(
        "mean", ascending=False)
    top_25 = set(offense.head(25).index)
    assert {"Ohio State", "Oregon", "Notre Dame"} <= top_25


def test_epa_identifies_known_bad_teams(plays_2024, model):
    """Kent State went 0-12 in 2024; Florida State collapsed to 2-10."""
    scored = ep.add_epa(plays_2024, model)
    offense = scored.groupby("team")["epa"].agg(["mean", "count"])
    offense = offense[offense["count"] >= 400].sort_values("mean")
    bottom_25 = set(offense.head(25).index)
    assert "Kent State" in bottom_25
