"""Tests for the game loader.

These run against the real downloaded CSVs, not fixtures. Slightly
unorthodox, but the whole point is catching the day the upstream mirror
changes its schema under us.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank.games import (  # noqa: E402
    available_seasons,
    load_season,
    team_divisions,
)


@pytest.fixture(scope="module")
def games_2024():
    return load_season(2024)


def test_seasons_are_on_disk():
    seasons = available_seasons()
    assert seasons, "No schedule files found. Run tools/fetch_mirror.py"
    assert 2024 in seasons


def test_loads_a_plausible_number_of_games(games_2024):
    # 2024 audit: 799 FBS-v-FBS + 121 FBS-v-FCS + 692 FCS-v-FCS
    assert 1500 < len(games_2024) < 1700


def test_every_game_is_completed(games_2024):
    assert all(isinstance(g.home_points, int) for g in games_2024)
    assert all(isinstance(g.away_points, int) for g in games_2024)


def test_only_rated_divisions_survive(games_2024):
    divisions = {g.home_division for g in games_2024}
    divisions |= {g.away_division for g in games_2024}
    assert divisions <= {"fbs", "fcs"}, f"Unexpected divisions: {divisions}"


def test_no_ties_in_modern_football(games_2024):
    """Overtime has existed since 1996, so a surviving tie is a data bug.

    Upstream encodes cancelled games as 0-0 rather than null -- five such
    rows exist across 2022-2024, all FCS-vs-FCS. The loader drops them.
    """
    assert [g.game_id for g in games_2024 if g.margin == 0] == []


def test_phantom_zero_zero_games_are_dropped():
    """Regression guard for the cancelled-game bug, pinned to known ids."""
    known_phantoms = {"401637161", "401638341", "401638672"}
    loaded_ids = {g.game_id for g in load_season(2024)}
    assert not (known_phantoms & loaded_ids)


def test_winner_and_loser_agree_with_margin(games_2024):
    for game in games_2024:
        if game.margin > 0:
            assert game.winner == game.home_team
            assert game.loser == game.away_team
        else:
            assert game.winner == game.away_team
            assert game.loser == game.home_team


def test_fcs_teams_are_present_and_distinct(games_2024):
    """The whole Montana-vs-Campbell argument depends on this."""
    divisions = team_divisions(games_2024)
    fcs_teams = {t for t, d in divisions.items() if d == "fcs"}
    assert len(fcs_teams) > 100, "Expected ~129 FCS teams"
    assert {"Montana State", "Montana", "Campbell"} <= fcs_teams


def test_fcs_records_differ_sharply(games_2024):
    """Montana State was 15-1, Campbell 3-9. If pooling were used, these
    would be indistinguishable -- which is exactly what we refuse to do."""

    def wins(team: str) -> tuple[int, int]:
        played = [g for g in games_2024 if team in (g.home_team, g.away_team)]
        won = sum(1 for g in played if g.winner == team)
        return won, len(played) - won

    montana_state = wins("Montana State")
    campbell = wins("Campbell")
    assert montana_state[0] >= 14
    assert campbell[1] >= 8
    assert montana_state[0] > campbell[0] * 3


def test_no_self_games(games_2024):
    assert all(g.home_team != g.away_team for g in games_2024)


def test_neutral_site_is_boolean(games_2024):
    assert all(isinstance(g.neutral_site, bool) for g in games_2024)
    # 2024 had 80 neutral-site games among FBS-involved matchups.
    assert sum(1 for g in games_2024 if g.neutral_site) > 20


def test_live_season_loads_partial(  ):
    """2026 is in progress: some games played, many not."""
    games = load_season(2026)
    assert games, "2026 should have completed games by week 3"
    assert all(g.season == 2026 for g in games)
    assert max(g.week for g in games) <= 5
