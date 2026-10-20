"""Tests for rest-of-season projections (projected record, bowl odds,
conference race odds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank.games import load_remaining_schedule, load_season  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.projection import (  # noqa: E402
    project_all,
    project_conference_races,
    project_team,
    simulate_remaining,
)
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5, "margin_scale": 28.0, "halflife": 1e6}


@pytest.fixture(scope="module")
def played_2026():
    return load_season(2026)


@pytest.fixture(scope="module")
def predictor_2026(played_2026):
    return Predictor.from_model(fit(played_2026, **FITTED), played_2026)


@pytest.fixture(scope="module")
def remaining_2026():
    return load_remaining_schedule(2026)


@pytest.fixture(scope="module")
def sim_2026(remaining_2026, predictor_2026):
    return simulate_remaining(remaining_2026, predictor_2026, n_sims=500, seed=1)


def test_simulation_has_remaining_games(sim_2026):
    assert sim_2026.games
    assert sim_2026.home_wins.shape == (500, len(sim_2026.games))


def test_win_probabilities_are_valid(sim_2026):
    assert all(0.0 <= sg.home_win_probability <= 1.0 for sg in sim_2026.games)


def test_projected_wins_never_exceed_games_remaining(sim_2026):
    """expected_wins - current wins can't be more than games left to play."""
    projection = project_team("Ohio State", wins=5, losses=0, sim=sim_2026)
    assert projection.expected_wins - 5 <= projection.games_remaining + 1e-9
    assert projection.expected_wins >= 5 - 1e-9


def test_undefeated_team_cant_have_negative_losses(sim_2026):
    projection = project_team("Georgia", wins=6, losses=0, sim=sim_2026)
    assert projection.most_likely_record[1] >= 0


def test_team_with_no_games_left_is_locked_in(sim_2026):
    projection = project_team("Nonexistent Tech", wins=8, losses=2, sim=sim_2026)
    assert projection.games_remaining == 0
    assert projection.most_likely_record == (8, 2)
    assert projection.bowl_eligible_pct == 100.0


def test_strong_team_has_high_bowl_odds(predictor_2026, sim_2026):
    """A team already at 6+ wins is bowl-eligible with certainty."""
    projection = project_team("Ohio State", wins=6, losses=0, sim=sim_2026)
    assert projection.bowl_eligible_pct == 100.0


def test_project_all_covers_every_requested_team(sim_2026):
    teams = ["Ohio State", "Michigan", "Georgia"]
    records = {"Ohio State": (6, 0), "Michigan": (5, 1), "Georgia": (5, 1)}
    projections = project_all(teams, records, sim_2026)
    assert set(projections) == set(teams)


def test_conference_race_odds_sum_to_100(played_2026, sim_2026):
    conference_of = {
        g.home_team: g.home_conference for g in played_2026 if g.home_conference
    }
    conference_of.update({
        g.away_team: g.away_conference for g in played_2026 if g.away_conference
    })
    races = project_conference_races(conference_of, played_2026, sim_2026)
    assert races
    for conf, standings in races.items():
        total = sum(pct for _, pct in standings)
        assert total == pytest.approx(100.0, abs=0.5), conf


def test_conference_race_is_deterministic_for_same_seed(remaining_2026, predictor_2026):
    sim_a = simulate_remaining(remaining_2026, predictor_2026, n_sims=200, seed=7)
    sim_b = simulate_remaining(remaining_2026, predictor_2026, n_sims=200, seed=7)
    assert np.array_equal(sim_a.home_wins, sim_b.home_wins)
