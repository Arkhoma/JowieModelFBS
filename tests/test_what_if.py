"""Tests for the what-if pages: Slate, Games That Matter Most, Path to the
playoff. Real season data, rendered the way the static builder does.

The schedule changes weekly (and in the offseason nothing is left to play),
so tests that need unplayed games skip rather than fail when there are none.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app.main import get_record, get_season  # noqa: E402
from app.pages import (  # noqa: E402
    LIVE_URLS, base_context, playoff_context, slate_context, team_context,
)
from app.scenario_view import slate, team_path  # noqa: E402
from build_static_site import STATIC_URLS, _env  # noqa: E402

SEASON = 2026


@pytest.fixture(scope="module")
def data():
    return get_season(SEASON)


@pytest.fixture(scope="module")
def live(data):
    if not data.remaining or data.scenarios is None:
        pytest.skip("no games left to play: nothing to analyse")
    return data


def _render(template, active, **extra):
    return _env().get_template(template).render(
        base_context(active, SEASON, urls=STATIC_URLS, **extra))


# --- Slate -------------------------------------------------------------

def test_slate_is_only_the_next_weeks_games(live):
    s = slate(live)
    rated = live.predictor.model.ratings
    upcoming = [g for g in live.remaining if g.home_team in rated and g.away_team in rated]
    assert s["week"] == min(g.week for g in upcoming)
    assert len(s["rows"]) == sum(g.week == s["week"] for g in upcoming)
    assert {r["week"] for r in s["rows"]} == {s["week"]}


def test_slate_leads_with_the_games_that_matter(live):
    impacts = [r["impact"] for r in slate(live)["rows"]]
    assert impacts == sorted(impacts, reverse=True)
    assert slate(live)["max_impact"] >= max(impacts)


def test_slate_picks_are_sane(live):
    for row in slate(live)["rows"]:
        # Win probability comes from the model's real error distribution, so on
        # a near coin-flip the favourite by margin can sit a hair under 50%.
        assert 49 <= row["favourite_pct"] <= 100
        assert row["spread"] >= 0
        assert row["favourite"] in (row["home"], row["away"])


def test_slate_page_has_a_row_per_game_and_logos_but_none_in_the_sentences(live):
    s = slate(live)
    html = _render("slate.html", "slate", **slate_context(live))
    assert html.count('<tr class="border-t border-ink/20 tr-top">') == len(s["rows"])
    assert 'class="logo-slot logo-sm"' in html                      # in the game column
    for sentence in re.findall(r'<ul class="text-xs space-y-1">.*?</ul>', html, re.S):
        assert "logo-slot" not in sentence                          # "who it moves" lines


def test_every_slate_row_has_a_phone_card_so_nothing_is_lost_on_small_screens(live):
    """Phones hide the pick / Script / impact columns; each game's cell must
    then carry the same content, with no extra rows (they'd break the zebra)."""
    s = slate(live)
    html = _render("slate.html", "slate", **slate_context(live))
    assert html.count('class="sm:hidden mt-2 font-normal"') == len(s["rows"])
    assert html.count("<tr") - 1 == len(s["rows"])          # header row + one per game
    assert html.count("hidden sm:table-cell") >= 3 * len(s["rows"])


def test_slate_page_without_games_says_so():
    html = _render("slate.html", "slate", slate=None)
    assert "No games left to predict" in html and "<table" not in html


def test_slate_has_its_own_tab_on_every_site():
    assert LIVE_URLS.nav["slate"] == "/slate"
    assert STATIC_URLS.nav["slate"] == "slate.html"
    html = _render("slate.html", "slate", slate=None)
    assert 'href="slate.html"' in html
    assert re.search(r'href="slate.html"\s+class="[^"]*"\s+aria-current="page"', html)


def test_live_server_serves_the_slate(data):
    from fastapi.testclient import TestClient
    from app.main import app
    response = TestClient(app).get("/slate")
    assert response.status_code == 200 and "This Week" in response.text


# --- How our picks have done -------------------------------------------------

@pytest.fixture(scope="module")
def record():
    result = get_record(SEASON)
    if result is None:
        pytest.skip("no lined games to grade yet")
    return result


def test_record_grades_real_walk_forward_picks(data, record):
    overall = record["overall"]
    assert overall["n"] > 0 and overall["n"] >= overall["lined"]
    assert 0.5 < overall["su"] < 1.0
    line = overall["vs_line"]
    if line is not None:                          # lines can be missing for a season
        assert 0.5 < line["su"] < 1.0 and 0.5 < line["line_su"] < 1.0
        assert line["gap_ci"] > 0
    assert sum(w["n"] for w in record["weeks"]) <= overall["n"]
    assert all(w["week"] is None or w["week"] >= 4 for w in record["weeks"])   # START_WEEK
    assert record["through"] <= data.max_week


def test_record_shows_on_the_slate_page(data, record):
    html = _render("slate.html", "slate", **slate_context(data, record))
    assert 'aria-labelledby="record"' in html and "How our picks have done" in html
    assert f"<strong>{record['overall']['n']}</strong> games" in html
    assert html.count('<th scope="row"') == len(record["weeks"])


def test_record_still_shows_when_there_is_no_slate(record):
    """Offseason: nothing to predict, but the season's record is still news."""
    from app.scenario_view import record_view
    html = _render("slate.html", "slate", slate=None, record=record_view(record))
    assert "No games left to predict" in html and "How our picks have done" in html


def test_slate_without_a_record_has_no_record_section(data):
    html = _render("slate.html", "slate", **slate_context(data))     # record defaults to None
    assert 'aria-labelledby="record"' not in html


# --- Games that matter most -----------------------------------------------

def test_playoff_page_ranks_the_games_that_matter(live):
    context = playoff_context(live)
    games = context["key_games"]
    assert 0 < len(games) <= 10
    assert [g["impact"] for g in games] == sorted((g["impact"] for g in games), reverse=True)
    assert all(g["impact"] > 0 for g in games)
    html = _render("playoff.html", "playoff", **context)
    assert "Games That Matter Most" in html
    assert html.count('<tr class="border-t border-ink/20 tr-top">') == len(games)


def test_every_mover_is_a_team_with_a_real_swing(live):
    for game in playoff_context(live)["key_games"]:
        for m in game["movers"]:
            assert abs(m["if_home_wins"] - m["if_away_wins"]) >= 1.0


# --- Path to the playoff ---------------------------------------------------

def _team_with_odds(data):
    return max(data.playoff_odds.made_field, key=data.playoff_odds.made_field.get)


def _team_without_odds(data):
    playing = {t for sg in data.scenarios.sim.games
               for t in (sg.game.home_team, sg.game.away_team)}
    for row in data.rankings:
        if row["team"] in playing and data.scenarios.made_odds(row["team"]) == 0:
            return row
    pytest.skip("every team still playing has some playoff odds")


def test_path_agrees_with_the_playoff_page(live):
    team = _team_with_odds(live)
    row = next(r for r in live.rankings if r["team"] == team)
    path = team_path(live, team, row["wins"], row["losses"])
    if path is None:
        pytest.skip(f"{team} has no games left")
    assert path["baseline"] == pytest.approx(live.playoff_odds.made_field[team])
    assert sum(r.share for r in path["records"]) == pytest.approx(100.0)
    assert path["n_sims"] == live.scenarios.sim.n_sims


def test_rooting_guide_never_tells_a_team_to_root_for_itself(live):
    for team in live.playoff_odds.made_field:
        row = next((r for r in live.rankings if r["team"] == team), None)
        path = team_path(live, team, row["wins"], row["losses"]) if row else None
        for r in (path or {"rooting": []})["rooting"]:
            assert team not in (r["home"], r["away"])


def test_slate_flags_the_vegas_script_for_phones_and_desktops(live):
    s = slate(live)
    flagged = [r for r in s["rows"] if r["script_alert"]]
    html = _render("slate.html", "slate", **slate_context(live))
    assert "Vegas Script" in html
    # Once in the Script column, once in the phone-only pick cell.
    assert html.count("Script is in") == 2 * len(flagged)
    for r in flagged:
        assert r["vegas"] is not None and r["script_gap"] >= 3.0


def test_a_team_that_cannot_get_in_is_told_so_not_shown_hopeful_numbers(live):
    row = _team_without_odds(live)
    path = team_path(live, row["team"], row["wins"], row["losses"])
    assert path["out_of_reach"] and path["rooting"] == []
    html = _render("team.html", "rankings", **team_context(live, row["team"]))
    assert "No simulated season puts" in html and 'aria-labelledby="path"' in html


def test_a_team_with_nothing_left_has_no_path_section(live):
    assert team_path(live, "Nobody U", 0, 0) is None
    html = _render("team.html", "rankings", **{**team_context(live, "Ohio State"), "path": None})
    assert 'aria-labelledby="path"' not in html


def test_team_page_shows_records_games_and_a_rooting_guide(live):
    # A team whose odds hinge on someone ELSE's game (its own are in the table).
    bubble = [t for t in live.playoff_odds.made_field
              if 0 < live.scenarios.made_odds(t) < 100
              and live.scenarios.rooting_guide(t, include_own=False)]
    if not bubble:
        pytest.skip("no team's odds currently hinge on someone else's game")
    team = bubble[0]
    html = _render("team.html", "rankings", **team_context(live, team))
    assert 'aria-labelledby="path"' in html
    assert "By final record" in html and "Remaining games" in html
    assert "Who to root for" in html
    section = re.search(r'<section aria-labelledby="path".*?</section>', html, re.S).group(0)
    assert re.search(r'team-lead">(vs|at)</span>', section)           # logo, then vs/at, then name
    for bullet in re.findall(r"<li>.*?</li>", section, re.S):
        assert "logo-slot" not in bullet                              # no logos in sentences
