"""Time Machine math (cfbrank.time_machine).

The anchor: a "cross-season" game between two teams from the SAME
season must reproduce the regular neutral-site Predictor. Everything the
Time Machine adds is just reading the two sides from different seasons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import get_season  # noqa: E402
from cfbrank.time_machine import matchup, season_snapshot  # noqa: E402

SEASON = 2026
PAIRS = [("Ohio State", "Notre Dame"), ("Boise State", "Kansas"),
         ("Ole Miss", "Sacramento State")]


@pytest.fixture(scope="module")
def data():
    return get_season(SEASON)


@pytest.fixture(scope="module")
def snap(data):
    return season_snapshot(data.rankings, data.predictor, str(SEASON))


def test_every_season_is_centred_on_an_average_fbs_team(snap):
    """The whole cross-season idea rests on this: 0 = average FBS team."""
    teams = snap["teams"].values()
    for key in ("rating", "off", "def"):
        assert abs(np.mean([t[key] for t in teams])) < 0.01


@pytest.mark.parametrize("a,b", PAIRS)
def test_same_season_matches_the_regular_neutral_prediction(data, snap, a, b):
    live = data.predictor.predict(a, b, neutral_site=True)
    flipped = data.predictor.predict(b, a, neutral_site=True)
    ours = matchup(snap, a, snap, b)
    assert ours["margin"] == pytest.approx(live.predicted_margin, abs=0.02)
    assert ours["total"] == pytest.approx(live.total, abs=0.05)
    # The regular predictor's residuals are home-side and slightly skewed,
    # so its A-vs-B and B-vs-A don't quite sum to 1. Averaged over both
    # orderings it's the neutral answer the mirrored quantiles give.
    neutral = (live.home_win_probability + 1 - flipped.home_win_probability) / 2
    assert ours["win_a"] == pytest.approx(neutral, abs=0.01)


def test_matchup_is_symmetric(snap):
    ab = matchup(snap, "Ohio State", snap, "Kansas")
    ba = matchup(snap, "Kansas", snap, "Ohio State")
    assert ab["margin"] == pytest.approx(-ba["margin"])
    assert ab["score_a"] == pytest.approx(ba["score_b"])
    assert ab["win_a"] + ba["win_a"] == pytest.approx(1.0, abs=0.006)


def test_cross_season_uses_each_sides_own_season(snap):
    """Fake 'other season' where everyone is 5 points better relative to
    their own field -- the matchup must read each side from its own."""
    other = {**snap, "teams": {t: {**v, "rating": v["rating"] + 5}
                               for t, v in snap["teams"].items()}}
    same = matchup(snap, "Kansas", snap, "Kansas")
    cross = matchup(other, "Kansas", snap, "Kansas")
    assert same["margin"] == 0
    assert cross["margin"] == pytest.approx(5)
