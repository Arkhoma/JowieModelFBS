"""Tests for cfbrank.scenarios: conditional playoff odds and game impact.

Built on a tiny league we control, not a fitted model: team A makes the
field exactly when it beats B in game 0, so every expected number can be
worked out by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank.games import ScheduledGame  # noqa: E402
from cfbrank.projection import SeasonSimulation, SimGame  # noqa: E402
from cfbrank.scenarios import MIN_TRIALS, PlayoffScenarios  # noqa: E402

N = 400


def _game(week: int, home: str, away: str) -> ScheduledGame:
    return ScheduledGame(2026, week, None, home, away, "fbs", "fbs", False, True)


def _league(g1_a_wins: int | None = None):
    """Three games, N simulated seasons.

        g0 (wk 12)  A hosts B   -- decides who is in the field
        g1 (wk 13)  C hosts A   -- doesn't affect the field
        g2 (wk 12)  D hosts E   -- irrelevant to everyone
        g3 (wk 12)  F hosts G   -- a 99% blowout: almost never goes the other way

    `g1_a_wins`: force A to win g1 in exactly that many seasons (to test
    the too-few-seasons guard).
    """
    rng = np.random.default_rng(3)
    wins = rng.random((N, 4)) < [0.5, 0.5, 0.5, 0.99]
    if g1_a_wins is not None:
        wins[:, 1] = True                       # C (home) wins ...
        wins[:g1_a_wins, 1] = False             # ... except these: A wins
    games = [SimGame(_game(12, "A", "B"), 0.5), SimGame(_game(13, "C", "A"), 0.5),
             SimGame(_game(12, "D", "E"), 0.5), SimGame(_game(12, "F", "G"), 0.99)]
    sim = SeasonSimulation(games=games, home_wins=wins, n_sims=N)
    fields = [[SimpleNamespace(team="A" if wins[i, 0] else "B"),
               SimpleNamespace(team="Z")]          # Z is in every field
              for i in range(N)]
    return sim, fields, PlayoffScenarios.from_fields(sim, fields)


def test_odds_are_the_share_of_simulated_seasons_in_the_field():
    sim, _, sc = _league()
    assert sc.made_odds("A") == pytest.approx(sim.home_wins[:, 0].mean() * 100)
    assert sc.made_odds("Z") == 100.0
    assert sc.made_odds("Nobody") == 0.0         # never in any field


def test_a_team_s_own_game_flips_its_odds_by_the_result():
    _, _, sc = _league()
    first = sc.team_games("A")[0]                # g0, week 12
    assert (first.game.game.home_team, first.if_win, first.if_lose) == ("A", 100.0, 0.0)
    assert first.team_win_probability == 0.5


def test_a_game_that_does_not_matter_leaves_the_odds_alone():
    sim, _, sc = _league()
    second = sc.team_games("A")[1]               # g1, week 13: C hosts A
    a_wins = ~sim.home_wins[:, 1]
    expected = sim.home_wins[a_wins, 0].mean() * 100
    assert second.if_win == pytest.approx(expected)
    assert abs(second.if_win - second.if_lose) < 15   # sampling noise only
    assert second.team_win_probability == 0.5        # 1 - C's chance


def test_team_games_come_back_in_schedule_order():
    _, _, sc = _league()
    assert [g.game.game.week for g in sc.team_games("A")] == [12, 13]


def test_final_records_split_the_seasons_and_condition_the_odds():
    _, _, sc = _league()
    records = sc.by_final_record("A", wins=5, losses=1)
    assert [(r.wins, r.losses) for r in records] == [(7, 1), (6, 2), (5, 3)]  # best first
    assert sum(r.share for r in records) == pytest.approx(100.0)
    assert records[0].playoff == 100.0           # won both: beat B, so in
    assert records[-1].playoff == 0.0            # lost both: lost to B, so out
    assert 20 < records[1].playoff < 80          # split: depends which one it won
    assert sc.by_final_record("D", 0, 0)[0].share > 0   # D plays one game


def test_a_team_with_nothing_left_has_no_records():
    _, _, sc = _league()
    assert sc.by_final_record("Nobody", 8, 4) == []


def test_too_few_simulated_seasons_gives_none_not_a_noisy_number():
    _, _, sc = _league(g1_a_wins=MIN_TRIALS - 1)
    second = sc.team_games("A")[1]
    assert second.if_win is None                 # A won g1 in only 29 seasons
    assert second.if_lose is not None
    rare = [r for r in sc.by_final_record("A", 0, 0) if r.n < MIN_TRIALS]
    assert rare, "expected a record too rare to estimate"
    assert all(r.playoff is None for r in rare)


def test_decisive_games_outrank_irrelevant_and_foregone_ones():
    _, _, sc = _league()
    impacts = {(i.game.game.home_team): i for i in sc.game_impacts()}
    assert sc.game_impacts()[0].game.game.home_team == "A"       # biggest first
    assert impacts["A"].impact > 10 * impacts["D"].impact        # D v E: noise only
    assert impacts["F"].impact == 0.0 and impacts["F"].movers == ()   # 99% blowout


def test_movers_are_the_teams_whose_odds_hinge_on_the_game():
    _, _, sc = _league()
    decisive = sc.game_impacts()[0]
    movers = {m.team: m for m in decisive.movers}
    assert set(movers) == {"A", "B"}             # Z is in either way: no swing
    assert (movers["A"].if_home_wins, movers["A"].if_away_wins) == (100.0, 0.0)
    assert (movers["B"].if_home_wins, movers["B"].if_away_wins) == (0.0, 100.0)


def test_noise_is_not_reported_as_a_game_that_matters():
    """D v E and C v A cannot affect who makes the field. Comparing halves of
    a finite sample still shows small differences; those must be ignored."""
    _, _, sc = _league()
    impacts = {i.game.game.home_team: i for i in sc.game_impacts()}
    assert impacts["D"].impact == 0.0 and impacts["D"].movers == ()
    assert impacts["C"].impact == 0.0
    # The only game worth watching for A is the one that decides its bid.
    assert [r.game.game.home_team for r in sc.rooting_guide("A")] == ["A"]


def test_rooting_guide_says_who_to_pull_for():
    _, _, sc = _league()
    for team, side in (("A", "A"), ("B", "B")):   # A hosts B in the decisive game
        top = sc.rooting_guide(team)[0]
        assert top.root_for == side
        assert (top.odds_if_it_wins, top.odds_if_it_loses) == (100.0, 0.0)
        assert top.swing > 40                     # ~2*.5*.5*100
    assert sc.rooting_guide("Nobody") == []
    assert sc.rooting_guide("Z") == []            # in every field: nothing hinges


def test_rooting_guide_can_leave_out_the_teams_own_games():
    _, _, sc = _league()
    assert [r.game.game.home_team for r in sc.rooting_guide("A")] == ["A"]
    assert sc.rooting_guide("A", include_own=False) == []   # nothing else matters to A
    # Z is in every field and A's game doesn't move it: nobody else matters either.
    assert sc.rooting_guide("Z", include_own=False) == []


def test_mismatched_fields_and_simulation_are_rejected():
    sim, fields, _ = _league()
    with pytest.raises(ValueError):
        PlayoffScenarios.from_fields(sim, fields[:-1])
