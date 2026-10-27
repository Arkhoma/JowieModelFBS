"""Tests for playoff field selection, seeding, and bracket simulation.

Field rules under test are the 2026-27 CFP's (see cfbrank.playoff): four
Power 4 champions, the top Group-of-6 TEAM, Notre Dame if top 12, the
rest at-large, seeded straight by ranking.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.playoff import (  # noqa: E402
    BYE_SEEDS,
    FIELD_SIZE,
    NOTRE_DAME,
    POWER_CONFERENCES,
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


# --- The 2026 rules, on a small league we control ---------------------
#
# 40 teams ranked 1..40. T20-T23 are the Power 4 champions (far from the
# top), T30 and T31 play in the Group of 6, and Notre Dame can be dropped
# in at any rank.

POWER_CHAMPS = {"SEC": "T20", "ACC": "T21", "Big Ten": "T22", "Big 12": "T23"}


def league(nd_rank=None):
    """(resume_rank, conference_of, champions) for the toy league."""
    names = [NOTRE_DAME if i == nd_rank else f"T{i}" for i in range(1, 41)]
    resume_rank = {name: i for i, name in enumerate(names, 1)}
    conference_of = {team: conf for conf, team in POWER_CHAMPS.items()}
    conference_of.update({"T30": "Sun Belt", "T31": "Pac-12"})
    return resume_rank, conference_of, dict(POWER_CHAMPS)


def teams_of(field):
    return [t.team for t in field]


def test_power_four_champions_are_in_however_low_they_rank():
    field = select_field(*league())
    assert set(POWER_CHAMPS.values()) <= set(teams_of(field))
    assert all(t.auto_bid for t in field if t.team in POWER_CHAMPS.values())


def test_group_of_six_bid_is_the_highest_ranked_team_not_a_champion():
    resume_rank, conference_of, champions = league()
    champions["Sun Belt"] = "T31"          # a lower-ranked team "won" it
    conference_of["T31"] = "Sun Belt"
    field = select_field(resume_rank, conference_of, champions)
    by_team = {t.team: t for t in field}
    assert by_team["T30"].auto_bid          # T30 outranks the champion
    assert "T31" not in by_team


def test_only_power_four_champions_get_champion_bids():
    """A Group-of-6 champion has no bid of its own anymore."""
    resume_rank, conference_of, champions = league()
    champions.update({"Mountain West": "T35", "Sun Belt": "T31"})
    conference_of.update({"T35": "Mountain West", "T31": "Sun Belt"})
    assert not {"T35", "T31"} & set(teams_of(select_field(resume_rank, conference_of, champions)))


def test_seeds_are_straight_by_ranking_so_byes_go_to_the_four_best_teams():
    field = select_field(*league())
    assert [t.rank for t in field] == sorted(t.rank for t in field)
    assert teams_of(field)[:BYE_SEEDS] == ["T1", "T2", "T3", "T4"]
    # ...at-large teams, not champions: nobody gets a bye for a trophy.
    assert not any(t.auto_bid for t in field[:BYE_SEEDS])


def test_auto_bids_ranked_outside_the_top_12_sit_at_the_bottom_in_rank_order():
    # Auto bids T20-T23 and T30 rank far below the 7 at-large teams (T1-T7).
    field = select_field(*league())
    assert teams_of(field)[:7] == [f"T{i}" for i in range(1, 8)]
    assert teams_of(field)[-5:] == ["T20", "T21", "T22", "T23", "T30"]


@pytest.mark.parametrize("nd_rank", [1, 7, 12])
def test_notre_dame_in_the_top_12_gets_an_auto_bid_and_leaves_six_at_large(nd_rank):
    field = select_field(*league(nd_rank))
    by_team = {t.team: t for t in field}
    assert by_team[NOTRE_DAME].auto_bid
    assert len(field) == FIELD_SIZE
    assert sum(not t.auto_bid for t in field) == 6


@pytest.mark.parametrize("nd_rank", [13, 25])
def test_notre_dame_outside_the_top_12_gets_no_bid(nd_rank):
    field = select_field(*league(nd_rank))
    assert NOTRE_DAME not in teams_of(field)
    assert sum(not t.auto_bid for t in field) == 7


def test_notre_dame_is_not_a_conference_champion_anywhere():
    """It's recognised by name -- it has no conference to win."""
    resume_rank, conference_of, champions = league(nd_rank=5)
    assert NOTRE_DAME not in conference_of
    assert NOTRE_DAME not in champions.values()
    assert any(t.team == NOTRE_DAME for t in select_field(resume_rank, conference_of, champions))


def test_too_few_teams_raises_instead_of_shipping_a_short_bracket():
    with pytest.raises(ValueError):
        select_field({"Team A": 1, "Team B": 2}, {}, {})


# --- Real 2024 data: structural invariants -----------------------------


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
def champions_2024(games_2024, conference_of_2024):
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
    teams = teams_of(field_2024)
    assert len(teams) == len(set(teams))


def test_every_power_four_champion_is_in_the_field(field_2024, champions_2024):
    in_field = set(teams_of(field_2024))
    for conf in POWER_CONFERENCES & set(champions_2024):
        assert champions_2024[conf] in in_field, conf


def test_seeds_follow_resume_rank(field_2024):
    ranks = [t.rank for t in field_2024]
    assert ranks == sorted(ranks)


def test_at_most_six_auto_bids(field_2024):
    """4 Power 4 champions + 1 Group of 6 + Notre Dame, never more."""
    assert 4 <= sum(t.auto_bid for t in field_2024) <= 6


# --- Bracket simulation ------------------------------------------------


def test_bracket_odds_cover_every_field_team(field_2024, predictor_2024):
    odds = simulate_bracket([field_2024] * 300, predictor_2024, seed=1)
    teams = set(teams_of(field_2024))
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


def test_each_trial_builds_its_own_field_from_its_own_ranking_and_champions():
    """Two trials, same league. In the second, T20 (a SEC champion that
    was 20th) and T1 swap ranks, and the SEC champion changes."""
    resume_rank, conference_of, champions = league()
    teams = list(resume_rank)
    first = [resume_rank[t] for t in teams]
    swapped = dict(resume_rank, T1=20, T20=1)
    second = [swapped[t] for t in teams]
    per_trial = {conf: [team, team] for conf, team in champions.items()}
    per_trial["SEC"] = ["T20", "T40"]

    a, b = simulate_fields(teams, np.array([first, second]), conference_of, per_trial)

    assert teams_of(a)[0] == "T1"           # trial 0: today's order
    assert teams_of(b)[0] == "T20"          # trial 1: T20 climbed to #1
    assert "T40" in teams_of(b)             # trial 1: new SEC champion in
    assert "T40" not in teams_of(a)


def test_odds_account_for_every_trials_field(field_2024, predictor_2024):
    """12 teams and 4 byes per field, however the fields vary; auto bids
    are whatever the field says."""
    odds = simulate_bracket([field_2024] * 100, predictor_2024, seed=5)
    assert sum(odds.made_field.values()) == pytest.approx(FIELD_SIZE * 100)
    assert sum(odds.bye.values()) == pytest.approx(BYE_SEEDS * 100)
    assert sum(odds.auto_bid.values()) == pytest.approx(
        sum(t.auto_bid for t in field_2024) * 100)


def test_a_team_in_only_some_trials_gets_partial_odds(field_2024, predictor_2024):
    """Swap the 12th seed for an outsider in half the trials: both end up
    at 50% to make it."""
    last = field_2024[-1]
    outsider = PlayoffTeam(seed=last.seed, team="Outsider U", conference="X",
                           rank=99, auto_bid=False)
    # Unrated teams fall back to a coin flip, so no need to touch the model.
    fields = [field_2024, field_2024[:-1] + [outsider]] * 50
    odds = simulate_bracket(fields, predictor_2024, seed=6)
    assert odds.made_field[last.team] == pytest.approx(50.0)
    assert odds.made_field["Outsider U"] == pytest.approx(50.0)
    assert odds.made_field[field_2024[0].team] == pytest.approx(100.0)


def test_mean_seed_is_the_average_over_trials_the_team_made_it(field_2024, predictor_2024):
    last = field_2024[-1]
    # Second trial: the top two seeds trade places.
    swapped = [replace(field_2024[0], seed=2), replace(field_2024[1], seed=1),
               *field_2024[2:]]
    odds = simulate_bracket([field_2024, swapped], predictor_2024, seed=7)
    assert odds.mean_seed[field_2024[0].team] == pytest.approx(1.5)
    assert odds.mean_seed[last.team] == pytest.approx(float(last.seed))


def _odds(made, auto, bye, mean_seed):
    zeros = {t: 0.0 for t in made}
    return BracketOdds(made_field=made, auto_bid=auto, bye=bye,
                       mean_seed=mean_seed, reached_quarterfinal=zeros,
                       reached_semifinal=zeros, reached_championship=zeros,
                       won_it_all=zeros)


def test_consensus_field_is_the_most_frequent_teams_with_the_likeliest_byes():
    teams = [f"T{i}" for i in range(1, 15)]
    resume_rank = {t: i for i, t in enumerate(teams, 1)}
    made = {t: 100.0 - i for i, t in enumerate(teams)}   # T13/T14 least often in
    mean_seed = {t: float(i) for i, t in enumerate(teams, 1)}
    mean_seed["T12"] = 0.5     # flattering average seed, but...
    bye = {t: 0.0 for t in teams}
    bye.update({"T5": 80.0, "T6": 70.0, "T7": 60.0, "T8": 50.0})   # ...no bye odds
    auto = {t: 0.0 for t in teams}
    auto["T12"] = 90.0
    field = consensus_field(_odds(made, auto, bye, mean_seed), resume_rank, {})

    # Byes go to the likeliest four (ordered by average seed), then
    # everyone else by average seed.
    assert teams_of(field) == ["T5", "T6", "T7", "T8", "T12",
                               "T1", "T2", "T3", "T4", "T9", "T10", "T11"]
    assert [t.seed for t in field] == list(range(1, FIELD_SIZE + 1))
    assert next(t for t in field if t.team == "T12").auto_bid
    assert not next(t for t in field if t.team == "T5").auto_bid
