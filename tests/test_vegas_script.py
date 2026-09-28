"""The Vegas Script: market-implied ratings blended into a side number."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from cfbrank.games import Game
from cfbrank.predict import Predictor
from cfbrank.ridge import fit
from cfbrank.static_export import predictor_payload
from cfbrank.vegas_script import (
    MARKET_WEIGHT, OURS_WEIGHT, VegasScript, fit_market_ratings, priced_games,
    script_alert,
)

TEAMS = ["A", "B", "C", "D", "E", "F", "G"]
STRENGTH = {"A": 14, "B": 7, "C": 3, "D": 0, "E": -7, "F": -14, "G": -30}
FCS = {"G"}   # the ridge engine's division column needs real crossover games


def _game(i, home, away, margin, week=1):
    return Game(game_id=str(i), season=2024, week=week, season_type="regular",
                kickoff=datetime(2024, 9, 1, tzinfo=timezone.utc),
                home_team=home, away_team=away,
                home_division="fcs" if home in FCS else "fbs",
                away_division="fcs" if away in FCS else "fbs", home_points=max(margin, 0) + 20,
                away_points=max(-margin, 0) + 20, neutral_site=False,
                conference_game=True, home_conference="X", away_conference="X")


@pytest.fixture(scope="module")
def schedule():
    games, i = [], 0
    for week in range(1, 11):
        for h in TEAMS:
            for a in TEAMS:
                if h != a and (week + TEAMS.index(h) + TEAMS.index(a)) % 3 == 0:
                    games.append(_game(i, h, a, STRENGTH[h] - STRENGTH[a] + 3, week))
                    i += 1
    return games


@pytest.fixture(scope="module")
def lines(schedule):
    # A "market" that knows the true strengths exactly, home field 2.5.
    return {g.game_id: STRENGTH[g.home_team] - STRENGTH[g.away_team] + 2.5
            for g in schedule}


def test_priced_games_carry_the_line_as_margin(schedule, lines):
    priced = priced_games(schedule[:3], lines)
    assert [g.margin for g in priced] == pytest.approx(
        [lines[g.game_id] for g in schedule[:3]])


def test_too_few_lines_means_no_script(schedule, lines):
    assert fit_market_ratings(schedule[:10], lines) is None


def test_market_ratings_recover_the_market(schedule, lines):
    market = fit_market_ratings(schedule, lines)
    assert market is not None
    # Light ridge penalty, so a hair of shrinkage is expected.
    assert market.predict_margin("A", "F") == pytest.approx(28 + 2.5, abs=1.0)


def test_blend_uses_the_shipped_weights(schedule, lines):
    vegas = VegasScript(fit_market_ratings(schedule, lines))
    mkt = vegas.market_margin("B", "E", False)
    assert vegas.blend(10.0, "B", "E", False) == pytest.approx(
        OURS_WEIGHT * 10.0 + MARKET_WEIGHT * mkt)


def test_predictor_and_static_payload_agree(schedule, lines):
    """predict.js rebuilds the Vegas Script from the payload alone."""
    vegas = VegasScript(fit_market_ratings(schedule, lines))
    model = fit(schedule, lambda_=5.0, margin_scale=28.0, halflife=1e6)
    predictor = Predictor.from_model(model, schedule, vegas=vegas)
    live = predictor.predict("C", "A", neutral_site=False)
    assert live.vegas_margin is not None

    p = predictor_payload(predictor, TEAMS)["vegas"]
    market = p["calibrated"]["C"] - p["calibrated"]["A"] + p["home_field"]
    rebuilt = p["ours_weight"] * live.predicted_margin + p["market_weight"] * market
    assert rebuilt == pytest.approx(live.vegas_margin)


@pytest.mark.parametrize("ours,script,expected", [
    (5.0, 7.9, None),        # 2.9-point split: no alert
    (5.0, 8.0, "in"),        # 3.0: the script is in
    (-3.0, 4.0, "flip"),     # different winner too
    (2.0, -1.5, "flip"),
    (5.0, None, None),       # no script yet
])
def test_script_alert_levels(ours, script, expected):
    assert script_alert(ours, script) == expected


def test_no_script_leaves_prediction_alone(schedule):
    model = fit(schedule, lambda_=5.0, margin_scale=28.0, halflife=1e6)
    p = Predictor.from_model(model, schedule).predict("A", "B")
    assert p.vegas_margin is None and p.vegas_favourite is None
    assert "vegas" not in predictor_payload(Predictor.from_model(model, schedule), TEAMS)
