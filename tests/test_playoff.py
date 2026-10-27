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
    BracketOdds,
    PlayoffTeam,
    consensus_field,
    select_field,
    simulate_bracket,
    simulate_fields,
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
    odds = simulate_bracket([field_2024] * 300, predictor_2024, seed=1)
    teams = {t.team for t in field_2024}
    assert set(odds.won_it_all) == teams
    assert set(odds.reached_championship) == teams


def test_bye_seeds_always_reach_the_quarterfinal(field_2024, predictor_2024):
    odds = simulate_bracket([field_2024] * 300, predictor_2024, seed=1)
    for t in field_2024:
        if t.seed <= BYE_SEEDS:
            assert odds.reached_quarterfinal[t.team] == 100.0


def test_round_odds_are_monotonically_non_increasing(field_2024, predictor_2024):
    """Nobody can have better odds of winning the title than of simply
    reaching the championship game, and so on back through each round."""
    odds = simulate_bracket([field_2024] * 500, predictor_2024, seed=2)
    for t in field_2024:
        assert odds.reached_quarterfinal[t.team] >= odds.reached_semifinal[t.team]
        assert odds.reached_semifinal[t.team] >= odds.reached_championship[t.team]
        assert odds.reached_championship[t.team] >= odds.won_it_all[t.team]


def test_title_odds_sum_to_100_percent(field_2024, predictor_2024):
    odds = simulate_bracket([field_2024] * 500, predictor_2024, seed=3)
    assert sum(odds.won_it_all.values()) == pytest.approx(100.0, abs=0.5)


def test_top_seed_has_better_title_odds_than_bottom_seed(field_2024, predictor_2024):
    odds = simulate_bracket([field_2024] * 1500, predictor_2024, seed=4)
    top_seed = next(t.team for t in field_2024 if t.seed == 1)
    bottom_seed = next(t.team for t in field_2024 if t.seed == 12)
    assert odds.won_it_all[top_seed] > odds.won_it_all[bottom_seed]


def test_bracket_is_deterministic_for_the_same_seed(field_2024, predictor_2024):
    a = simulate_bracket([field_2024] * 200, predictor_2024, seed=99)
    b = simulate_bracket([field_2024] * 200, predictor_2024, seed=99)
    assert a.won_it_all == b.won_it_all


# --- Fields rebuilt per simulated season ------------------------------


def test_each_trial_builds_its_own_field_from_its_own_champions():
    """Conference A's champion is its #8 team in one trial and its #1 team
    in the next. The #1 team is a top-seed champion only in the second --
    one "likeliest champion" repeated every trial would miss exactly this."""
    resume_rank = {f"T{i}": i for i in range(1, 21)}
    conference_of = {"T1": "A", "T8": "A", **{f"T{i}": c for i, c in
                     zip(range(3, 8), "BCDEF")}}
    champions = {"A": ["T8", "T1"], **{c: [f"T{i}"] * 2 for i, c in
                 zip(range(3, 8), "BCDEF")}}
    first, second = simulate_fields(resume_rank, conference_of, champions, n_sims=2)

    assert [t.team for t in second][:BYE_SEEDS] == ["T1", "T3", "T4", "T5"]
    assert next(t for t in second if t.team == "T1").champion
    # In the first trial T1 still makes it, but as an at-large -- no bye, no auto bid.
    t1_first = next(t for t in first if t.team == "T1")
    assert not t1_first.champion and t1_first.seed > BYE_SEEDS


def test_odds_account_for_every_trials_field(field_2024, predictor_2024):
    """12 teams, 5 auto bids and 4 byes per field -- so those totals are
    fixed no matter how the fields vary."""
    odds = simulate_bracket([field_2024] * 100, predictor_2024, seed=5)
    assert sum(odds.made_field.values()) == pytest.approx(FIELD_SIZE * 100)
    assert sum(odds.auto_bid.values()) == pytest.approx(AUTO_BIDS * 100)
    assert sum(odds.bye.values()) == pytest.approx(BYE_SEEDS * 100)


def test_a_team_in_only_some_trials_gets_partial_odds(field_2024, predictor_2024):
    """Swap the 12th seed for the first team outside the field in half
    the trials: both end up at 50% to make it."""
    last = field_2024[-1]
    outsider = PlayoffTeam(seed=last.seed, team="Outsider U", conference="X",
                           rank=99, champion=False)
    # Unrated teams fall back to a coin flip, so no need to touch the model.
    fields = [field_2024, field_2024[:-1] + [outsider]] * 50
    odds = simulate_bracket(fields, predictor_2024, seed=6)
    assert odds.made_field[last.team] == pytest.approx(50.0)
    assert odds.made_field["Outsider U"] == pytest.approx(50.0)
    assert odds.made_field[field_2024[0].team] == pytest.approx(100.0)


def _odds(made, auto, bye):
    zeros = {t: 0.0 for t in made}
    return BracketOdds(made_field=made, auto_bid=auto, bye=bye,
                       reached_quarterfinal=zeros, reached_semifinal=zeros,
                       reached_championship=zeros, won_it_all=zeros)


def test_consensus_field_is_the_most_frequent_teams_with_likeliest_byes():
    teams = [f"T{i}" for i in range(1, 15)]
    resume_rank = {t: i for i, t in enumerate(teams, 1)}
    # T13/T14 are the least-often-in; T12 has the best bye odds despite
    # the worst resume of the byes, so it takes a bye seed.
    made = {t: 100.0 - i for i, t in enumerate(teams)}
    bye = {t: 0.0 for t in teams}
    bye.update({"T12": 90.0, "T2": 80.0, "T3": 70.0, "T4": 60.0})
    auto = {t: 0.0 for t in teams}
    auto.update({"T12": 90.0})
    field = consensus_field(_odds(made, auto, bye), resume_rank, {})

    assert [t.team for t in field][:BYE_SEEDS] == ["T2", "T3", "T4", "T12"]
    assert {t.team for t in field} == set(teams[:FIELD_SIZE])
    assert [t.seed for t in field] == list(range(1, FIELD_SIZE + 1))
    rest = [t.rank for t in field if t.seed > BYE_SEEDS]
    assert rest == sorted(rest)
    assert next(t for t in field if t.team == "T12").champion
    assert not next(t for t in field if t.team == "T2").champion
