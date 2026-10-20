"""Tests for playoff field selection, seeding, and bracket simulation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.playoff import (  # noqa: E402
    AUTO_BIDS,
    BYE_SEEDS,
    FIELD_SIZE,
    select_field,
    simulate_bracket,
)
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.resume import build_resumes, rank_resumes  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5, "margin_scale": 28.0, "halflife": 1e6}


@pytest.fixture(scope="module")
def games_2024():
    return load_season(2024)


@pytest.fixture(scope="module")
def model_2024(games_2024):
    return fit(games_2024, **FITTED)


@pytest.fixture(scope="module")
def predictor_2024(games_2024, model_2024):
    return Predictor.from_model(model_2024, games_2024)


@pytest.fixture(scope="module")
def resume_rank_2024(games_2024, model_2024, predictor_2024):
    resumes = build_resumes(games_2024, model_2024, predictor_2024.errors)
    return {r.team: rank for rank, r in rank_resumes(resumes, model_2024)}


@pytest.fixture(scope="module")
def conference_of_2024(games_2024, model_2024):
    conference_of = {}
    for g in games_2024:
        if model_2024.divisions.get(g.home_team) == "fbs" and g.home_conference:
            conference_of.setdefault(g.home_team, g.home_conference)
        if model_2024.divisions.get(g.away_team) == "fbs" and g.away_conference:
            conference_of.setdefault(g.away_team, g.away_conference)
    return {t: c for t, c in conference_of.items() if "independent" not in c.lower()}


@pytest.fixture(scope="module")
def champions_2024(games_2024, conference_of_2024, resume_rank_2024):
    """Real 2024 regular-season conference winners, picked by actual wins
    (not simulated -- the season is long over), so field selection can be
    tested against ground truth instead of noisy projected odds."""
    by_conference: dict[str, list[str]] = {}
    for team, conf in conference_of_2024.items():
        by_conference.setdefault(conf, []).append(team)

    wins: dict[str, int] = {}
    for g in games_2024:
        if g.conference_game:
            wins[g.winner] = wins.get(g.winner, 0) + 1

    return {
        conf: max(teams, key=lambda t: wins.get(t, 0))
        for conf, teams in by_conference.items() if len(teams) > 1
    }


@pytest.fixture(scope="module")
def field_2024(resume_rank_2024, conference_of_2024, champions_2024):
    return select_field(resume_rank_2024, conference_of_2024, champions_2024)


def test_field_has_twelve_teams(field_2024):
    assert len(field_2024) == FIELD_SIZE
    assert [t.seed for t in field_2024] == list(range(1, FIELD_SIZE + 1))


def test_field_has_no_duplicate_teams(field_2024):
    teams = [t.team for t in field_2024]
    assert len(teams) == len(set(teams))


def test_top_four_seeds_are_all_conference_champions(field_2024):
    """The bye seeds are specifically the top-ranked champions -- an
    at-large team can never hold a bye."""
    top_four = [t for t in field_2024 if t.seed <= BYE_SEEDS]
    assert len(top_four) == BYE_SEEDS
    assert all(t.champion for t in top_four)


def test_at_most_five_champions_in_the_field(field_2024):
    assert sum(1 for t in field_2024 if t.champion) <= AUTO_BIDS


def test_seeds_are_ordered_by_resume_rank_within_each_group(field_2024):
    byes = [t.rank for t in field_2024 if t.seed <= BYE_SEEDS]
    rest = [t.rank for t in field_2024 if t.seed > BYE_SEEDS]
    assert byes == sorted(byes)
    assert rest == sorted(rest)


def test_too_few_teams_raises_instead_of_shipping_a_short_bracket():
    with pytest.raises(ValueError):
        select_field({"Team A": 1, "Team B": 2}, {}, {})


def test_bracket_odds_cover_every_field_team(field_2024, predictor_2024):
    odds = simulate_bracket(field_2024, predictor_2024, n_sims=300, seed=1)
    teams = {t.team for t in field_2024}
    assert set(odds.won_it_all) == teams
    assert set(odds.reached_championship) == teams


def test_bye_seeds_always_reach_the_quarterfinal(field_2024, predictor_2024):
    odds = simulate_bracket(field_2024, predictor_2024, n_sims=300, seed=1)
    for t in field_2024:
        if t.seed <= BYE_SEEDS:
            assert odds.reached_quarterfinal[t.team] == 100.0


def test_round_odds_are_monotonically_non_increasing(field_2024, predictor_2024):
    """Nobody can have better odds of winning the title than of simply
    reaching the championship game, and so on back through each round."""
    odds = simulate_bracket(field_2024, predictor_2024, n_sims=500, seed=2)
    for t in field_2024:
        assert odds.reached_quarterfinal[t.team] >= odds.reached_semifinal[t.team]
        assert odds.reached_semifinal[t.team] >= odds.reached_championship[t.team]
        assert odds.reached_championship[t.team] >= odds.won_it_all[t.team]


def test_title_odds_sum_to_100_percent(field_2024, predictor_2024):
    odds = simulate_bracket(field_2024, predictor_2024, n_sims=500, seed=3)
    assert sum(odds.won_it_all.values()) == pytest.approx(100.0, abs=0.5)


def test_top_seed_has_better_title_odds_than_bottom_seed(field_2024, predictor_2024):
    odds = simulate_bracket(field_2024, predictor_2024, n_sims=1500, seed=4)
    top_seed = next(t.team for t in field_2024 if t.seed == 1)
    bottom_seed = next(t.team for t in field_2024 if t.seed == 12)
    assert odds.won_it_all[top_seed] > odds.won_it_all[bottom_seed]


def test_bracket_is_deterministic_for_the_same_seed(field_2024, predictor_2024):
    a = simulate_bracket(field_2024, predictor_2024, n_sims=200, seed=99)
    b = simulate_bracket(field_2024, predictor_2024, n_sims=200, seed=99)
    assert a.won_it_all == b.won_it_all
