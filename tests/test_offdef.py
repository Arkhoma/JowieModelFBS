"""Offense/defense engine: recovers known truth, and keeps its contracts."""

from __future__ import annotations

import numpy as np
import pytest

from cfbrank import offdef
from cfbrank.games import Game

TRUE_OFF = {"A": 10.0, "B": 0.0, "C": -4.0, "D": -6.0}
TRUE_DEF = {"A": -3.0, "B": 8.0, "C": 0.0, "D": -5.0}
MEAN, HFA = 28.0, 3.0


def _game(i: int, home: str, away: str, neutral: bool = False,
          noise: tuple[float, float] = (0.0, 0.0)) -> Game:
    venue = 0.0 if neutral else 1.0
    home_pts = MEAN + TRUE_OFF[home] - TRUE_DEF[away] + venue * HFA / 2
    away_pts = MEAN + TRUE_OFF[away] - TRUE_DEF[home] - venue * HFA / 2
    return Game(str(i), 2024, 1 + i // 6, "regular", None, home, away,
                "fbs", "fbs", round(home_pts + noise[0]),
                round(away_pts + noise[1]), neutral, True, "X", "X")


def _round_robin(repeats: int = 6) -> list[Game]:
    teams = list(TRUE_OFF)
    games, i = [], 0
    for _ in range(repeats):
        for h in teams:
            for a in teams:
                if h != a:
                    games.append(_game(i, h, a))
                    i += 1
    return games


@pytest.fixture(scope="module")
def model() -> offdef.OffDefResult:
    return offdef.fit(offdef.sides_from_games(_round_robin()), lambda_=1e-6)


def test_recovers_offense_and_defense_separately(model):
    for team in TRUE_OFF:
        assert model.offense_rating(team) == pytest.approx(
            TRUE_OFF[team] - np.mean(list(TRUE_OFF.values())), abs=0.3)
        assert model.defense_rating(team) == pytest.approx(
            TRUE_DEF[team] - np.mean(list(TRUE_DEF.values())), abs=0.3)


def test_home_field_is_on_the_margin_scale(model):
    # Home offense gets +HFA/2 and away -HFA/2: the fitted coefficient
    # must equal the full-margin value, comparable with ridge.fit.
    assert model.home_field == pytest.approx(HFA, abs=0.3)


def test_total_and_margin_are_consistent(model):
    home, away = model.predict_scores("A", "B")
    assert model.predict_total("A", "B") == pytest.approx(home + away)
    truth_total = (2 * MEAN + TRUE_OFF["A"] - TRUE_DEF["B"]
                   + TRUE_OFF["B"] - TRUE_DEF["A"])
    assert model.predict_total("A", "B") == pytest.approx(truth_total, abs=0.6)


def test_margin_prediction_matches_truth(model):
    truth = ((TRUE_OFF["A"] + TRUE_DEF["A"])
             - (TRUE_OFF["D"] + TRUE_DEF["D"]) + HFA)
    assert model.predict_margin("A", "D") == pytest.approx(truth, abs=0.8)
    assert model.predict_margin("A", "D", neutral_site=True) == pytest.approx(
        truth - HFA, abs=0.8)


def test_net_rating_is_offense_plus_defense(model):
    for team in TRUE_OFF:
        assert model.ratings[team] == pytest.approx(
            model.offense_rating(team) + model.defense_rating(team))


def test_each_side_shrinks_toward_its_own_prior():
    # One game of evidence and a heavy penalty: ratings should sit near
    # the prior, and offense/defense priors must not bleed into each other.
    sides = offdef.sides_from_games([_game(0, "A", "B"), _game(1, "C", "D")])
    prior = ({"A": 7.0}, {"A": -2.0})
    fitted = offdef.fit(sides, lambda_=1e4, prior=prior)
    assert fitted.offense["A"] - fitted.offense["B"] == pytest.approx(7.0, abs=0.1)
    assert fitted.defense["A"] - fitted.defense["B"] == pytest.approx(-2.0, abs=0.1)


def test_success_rate_scale_converts_to_points():
    # A statistic that is exactly margin / 100 must convert back with a
    # slope of ~100, home field included.
    games = _round_robin()
    sides = [offdef.SideGame(s.game_id, s.season, s.week, s.offense,
                             s.defense, "fbs", "fbs", s.venue,
                             s.value / 100.0, s.margin)
             for s in offdef.sides_from_games(games)]
    fitted = offdef.fit(sides, lambda_=1e-8, is_points=False)
    assert fitted.edge_scale == pytest.approx(100.0, rel=0.05)
    assert fitted.home_field_points == pytest.approx(HFA, abs=0.4)


def test_predictor_uses_matchup_total_but_keeps_margin(model):
    # The off/def model only sets the TOTAL; the margin must still come
    # from the primary model, since the split did not improve spreads.
    from cfbrank.predict import Predictor
    games = _round_robin()
    plain = Predictor.from_model(model, games)
    with_totals = Predictor.from_model(model, games, total_model=model)
    a = plain.predict("A", "D")
    b = with_totals.predict("A", "D")
    assert b.predicted_margin == pytest.approx(a.predicted_margin)
    assert a.total == pytest.approx(plain.average_total)
    assert b.total == pytest.approx(model.predict_total("A", "D"))
    assert b.total != pytest.approx(a.total, abs=0.5)
    # Unknown team falls back to league average instead of raising.
    assert with_totals.predict_total("A", "Nobody") == plain.average_total


def test_efficiency_sides_skip_games_missing_one_team():
    from cfbrank.epa_games import TeamGameEPA
    games = [_game(0, "A", "B"), _game(1, "C", "D")]
    rec = lambda gid, t, o: TeamGameEPA(gid, 2024, 1, t, o, 0.1, 0.0, 0.45,
                                        0.40, 60, 60, 0.0)
    records = [rec("0", "A", "B"), rec("0", "B", "A"), rec("1", "C", "D")]
    sides = offdef.sides_from_efficiency(games, records)
    assert {s.game_id for s in sides} == {"0"}
    assert len(sides) == 2
