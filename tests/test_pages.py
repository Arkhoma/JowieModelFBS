"""Tests for app.pages: shared page context, URL schemes, season menus.

Mostly plain function calls -- no server. The template checks render
real pages with Jinja directly, the same way the static builder does.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app.main import get_season  # noqa: E402
from app.pages import (  # noqa: E402
    LIVE_URLS, SiteUrls, base_context, conference_spread, conferences_context,
    link_teams, logo_ids, percent,
    playoff_context, predict_context, rankings_context, team_context,
    team_index, team_link,
)
from build_static_site import STATIC_URLS, _env, _latest  # noqa: E402
from check_static_links import broken_links  # noqa: E402
from cfbrank.games import available_seasons, fbs_seasons, fbs_teams  # noqa: E402
from cfbrank.logos import team_logo_id  # noqa: E402
from cfbrank.time_machine import season_snapshot  # noqa: E402

SEASON = 2026


@pytest.fixture(scope="module")
def data():
    return get_season(SEASON)


def test_base_context_defaults_to_live_urls():
    ctx = base_context("about", SEASON)
    assert ctx["team_url"] is LIVE_URLS.team
    assert ctx["predict_url"] is LIVE_URLS.predict
    assert ctx["nav"] is LIVE_URLS.nav


def test_base_context_uses_the_given_site():
    custom = SiteUrls(nav={}, rank=lambda s, k: "r", team=lambda t, s: "t",
                      predict=lambda s: "p", playoff=lambda s: "o",
                      conferences=lambda s: "c")
    ctx = base_context("rankings", SEASON, urls=custom)
    assert ctx["team_url"]("X", SEASON) == "t"
    assert ctx["playoff_url"](SEASON) == "o"


def test_static_urls_keep_latest_stable_and_suffix_older_seasons():
    latest = _latest()
    assert STATIC_URLS.team("Texas A&M", latest) == "team-texas-a-m.html"
    assert STATIC_URLS.team("Texas A&M", 2019) == "team-texas-a-m-2019.html"
    assert STATIC_URLS.predict(latest) == "predict.html"
    assert STATIC_URLS.playoff(2019) == "playoff-2019.html"


def test_fbs_teams_matches_the_rankings_table(data):
    """The team season menu trusts fbs_teams() (schedule only, no fit)
    to know which team pages exist. It must agree with the table."""
    assert fbs_teams(SEASON) == {r["team"] for r in data.rankings}


def test_fbs_seasons_tracks_a_team_moving_up():
    """Sam Houston joined FBS in 2023, so no earlier team page exists."""
    assert fbs_seasons("Sam Houston")[0] == 2023
    assert fbs_seasons("Ohio State") == available_seasons()


def _render(template, active, data, **extra):
    return _env().get_template(template).render(
        base_context(active, SEASON, urls=STATIC_URLS, **extra))


def _menu_links(html):
    menu = re.search(r'aria-label="Choose season">(.*?)</nav>', html, re.S)
    return re.findall(r'href="([^"]+)"', menu.group(1))


def test_team_page_menu_lists_only_that_teams_fbs_seasons(data):
    html = _render("team.html", "rankings", data,
                   **team_context(data, "Sam Houston"))
    links = _menu_links(html)
    assert len(links) == len(fbs_seasons("Sam Houston"))
    assert "team-sam-houston-2023.html" in links
    assert "team-sam-houston-2022.html" not in links


def test_team_page_renders_the_form_section_from_real_games(data):
    # A steady team (its points charts are the default), whoever that is this week.
    team = next(t for t, f in data.form.teams.items()
                if not f.shifts and len(f.games) >= 4)
    html = _render("team.html", "rankings", data, **team_context(data, team))
    games = data.form.teams[team].games
    bars = html.count('class="form-up"') + html.count('class="form-down"')
    assert 'id="form"' in html
    assert html.count('<svg class="form-chart"') == 2
    assert "form-break" not in html
    # one offense bar per game, one defense bar per game with a fitted opponent
    assert bars == len(games) + sum(g.def_miss is not None for g in games)
    assert "never changes a rating or a prediction" in html


def test_a_flagged_teams_page_marks_the_break_and_names_the_statistic(data):
    flagged = [t for t, f in data.form.teams.items() if f.shifts]
    if not flagged:
        pytest.skip("no team is flagged in this season's data right now")
    html = _render("team.html", "rankings", data, **team_context(data, flagged[0]))
    assert "Worth a look" in html
    assert 'class="form-break"' in html
    assert "Dashed line marks the break" in html


def test_nav_tabs_keep_the_season_being_viewed(data):
    """Browsing 2026 vs browsing an older season: the tabs follow."""
    html = _render("playoff.html", "playoff", data, **playoff_context(data))
    assert f'href="{STATIC_URLS.rank(SEASON, "predictive")}"' in html
    assert f'href="{STATIC_URLS.predict(SEASON)}"' in html
    assert len(_menu_links(html)) == len(available_seasons())


def test_predict_page_links_to_the_time_machine(data):
    html = _render("predict.html", "predict", data, **predict_context(data))
    assert 'href="time-machine.html"' in html


def test_time_machine_page_renders_pickers_and_data(data):
    snap = season_snapshot(data.rankings, data.predictor, str(SEASON))
    teams = [{"name": r["team"], "conference": r["conference"]}
             for r in data.rankings]
    html = _render("time_machine.html", "predict", data,
                   snapshots={str(SEASON): snap}, teams=teams,
                   logo_ids=logo_ids(t["name"] for t in teams))
    assert 'id="a_team"' in html and 'id="b_season"' in html
    assert html.count('role="option"') == 2 * len(teams)
    assert '<option value="2019" selected>' in html       # 2019 LSU default
    assert 'id="snapshots"' in html and "time_machine.js" in html
    assert 'id="team-logos"' in html                      # JS draws logos from this
    assert "Vegas Script" not in html


# --- Logos everywhere a team name appears --------------------------------

def test_every_rankings_row_has_a_logo_slot(data):
    html = _render("rankings.html", "rankings", data, **rankings_context(data, "predictive"))
    assert html.count('class="team-chip"><span class="logo-slot logo-sm"') >= len(data.rankings)
    assert "data-logo='" in html                    # search.js reads the CDN from here


def test_team_page_header_has_a_large_logo(data):
    html = _render("team.html", "rankings", data, **team_context(data, "Ohio State"))
    assert 'class="logo-slot logo-lg"' in html
    assert "/logos/128/" in html


def test_conference_page_shows_conference_logos(data):
    html = _render("conferences.html", "conferences", data, **conferences_context(data))
    assert html.count("espncdn.com/i/teamlogos/ncaa_conf/") >= len(data.conference_races)
    assert 'class="section-title conf-name' in html


def test_team_pickers_put_a_logo_on_every_option_and_in_the_field(data):
    html = _render("predict.html", "predict", data, **predict_context(data))
    options = re.findall(r'<li id="[^"]+" role="option".*?</li>', html, re.S)
    assert options and all('class="logo-slot logo-sm"' in o for o in options)
    assert html.count("data-combo-logo") == 2                    # away + home fields
    assert html.count("combo-field") == 2


def test_static_predict_page_embeds_logo_ids_for_its_js(data):
    html = _render("predict.html", "predict", data, static_build=True, model_json={},
                   **predict_context(data))
    ids = re.search(r'id="team-logos">(.*?)</script>', html, re.S).group(1)
    assert f'"Ohio State": {team_logo_id("Ohio State")}' in ids


def test_prediction_result_shows_both_teams_logos_on_the_scoreboard(data):
    prediction = data.predictor.predict("Ohio State", "Michigan", False)
    html = _render("_prediction_result.html", "predict", data,
                   prediction=prediction, error=None)
    assert html.count('class="logo-slot logo-md"') == 2
    assert "logos-dark/" in html                  # artwork meant for the dark scoreboard
    assert "logo-xs" not in html                   # no logos in the "At ..." / "by" lines


@pytest.mark.parametrize("value, shown", [
    (0, "0%"),          # truly zero (or no trial hit it) stays 0
    (0.05, "<1%"),      # one trial in 2,000: possible, so not "0%"
    (0.99, "<1%"),
    (1, "1%"),
    (1.4, "1%"),
    (49.6, "50%"),
    (98.9, "99%"),
    (99, "99%"),
    (99.6, ">99%"),     # not certain, so not "100%"
    (100, "100%"),      # finished season / locked in
])
def test_percent_never_overstates_certainty(value, shown):
    assert percent(value) == shown


def test_templates_get_percent_from_the_page_context():
    assert base_context("playoff", SEASON, STATIC_URLS)["percent"](0.3) == "<1%"


def test_filtered_out_picker_options_really_disappear():
    """Options are `.flex`; without this rule that beats [hidden] and the
    filter only reorders the list instead of narrowing it."""
    css = (ROOT / "app" / "static" / "retro.css").read_text(encoding="utf-8")
    assert ".combo-list [hidden] { display: none; }" in css


def test_team_link_links_fbs_and_leaves_fcs_as_text():
    fbs = str(team_link("Ohio State", SEASON, STATIC_URLS))
    assert 'href="team-ohio-state.html"' in fbs
    fcs = str(team_link("Montana State", SEASON, STATIC_URLS))
    assert "<a " not in fcs and "Montana State" in fcs      # no page, no dead link


def test_every_team_name_comes_with_a_logo_slot():
    """FBS or not, linked or not: the fixed-size slot is always there."""
    for team in ("Ohio State", "Montana State", "Nowhere Tech"):
        html = str(team_link(team, SEASON, STATIC_URLS))
        assert html.startswith('<span class="team-chip"><span class="logo-slot logo-sm"')
    assert "cdn.collegefootballdata.com/logos/64/" in str(
        team_link("Ohio State", SEASON, STATIC_URLS))
    assert "<img" not in str(team_link("Nowhere Tech", SEASON, STATIC_URLS))


def test_running_text_gets_linked_names_but_no_logos():
    html = str(link_teams("Best win: Texas", SEASON, STATIC_URLS))
    assert "logo-slot" not in html and "<img" not in html
    assert 'href="team-texas.html" class="team-link">Texas</a>' in html


def test_a_lead_sits_between_the_logo_and_the_name():
    """Rank / vs / at follow the logo, so the logo always starts the line."""
    html = str(team_link("Ohio State", SEASON, STATIC_URLS, lead="#4"))
    assert html.index("logo-slot") < html.index("team-lead") < html.index("team-link")
    assert '<span class="team-lead">#4</span>' in html


def test_size_none_is_the_bare_link():
    html = str(team_link("Ohio State", SEASON, STATIC_URLS, size=None))
    assert html.startswith("<a ") and "team-chip" not in html
    assert str(team_link("Montana State", SEASON, STATIC_URLS, size=None)) == "Montana State"


def test_texas_logo_is_upside_down_but_texas_am_is_not():
    texas = str(team_link("Texas", SEASON, STATIC_URLS))
    assert 'class="logo-flipped"' in texas
    assert 'class="logo-flipped"' not in str(team_link("Texas A&M", SEASON, STATIC_URLS))
    assert 'class="logo-flipped"' not in str(team_link("Oklahoma", SEASON, STATIC_URLS))


def test_playoff_and_schedule_rows_put_rank_and_vs_after_the_logo(data):
    playoff = _render("playoff.html", "playoff", data, **playoff_context(data))
    assert re.search(r'logo-slot logo-sm.*?team-lead">#\d+</span>', playoff, re.S)
    assert not re.search(r'#\d+ <span class="team-chip"', playoff)
    team = _render("team.html", "rankings", data, **team_context(data, "Ohio State"))
    assert not re.search(r'(?:vs|at) <span class="team-chip"', team)


def test_search_index_carries_each_teams_logo_id():
    index = team_index(STATIC_URLS)
    assert isinstance(index["Ohio State"]["l"], int)
    assert index["Ohio State"]["l"] == team_logo_id("Ohio State")


def test_link_teams_prefers_the_longest_name_and_escapes():
    html = str(link_teams("W 35-10 vs North Texas <b>, then Texas", SEASON, STATIC_URLS))
    assert '>North Texas</a>' in html
    assert 'href="team-texas.html" class="team-link">Texas</a>' in html
    assert html.count("<a ") == 2                 # "Texas" inside "North Texas" not re-linked
    assert "&lt;b&gt;" in html                    # free text is escaped


def test_search_index_covers_every_fbs_team_season():
    index = team_index(STATIC_URLS)
    assert list(index["Sam Houston"]["u"])[0] == "2023"
    assert index["Ohio State"]["u"][str(_latest())] == "team-ohio-state.html"
    assert index["Ohio State"]["c"] == "Big Ten"
    assert sum(len(e["u"]) for e in index.values()) == sum(
        len(fbs_teams(s)) for s in available_seasons())


def test_conference_races_have_their_own_page_with_team_links(data):
    html = _render("conferences.html", "conferences", data, **conferences_context(data))
    assert html.count('class="team-link"') >= len(data.conference_races) * 4
    assert 'aria-current="page"' in html and "Confs" in html
    rankings = _render("rankings.html", "rankings", data,
                       **rankings_context(data, "predictive"))
    assert "Conference Races</h3>" not in rankings


def test_live_server_404s_an_unratable_season():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    assert client.get("/?season=1999").status_code == 404
    assert client.get("/conferences?season=1999").status_code == 404
    index = client.get("/search-index.json").json()
    assert index["LSU"]["u"]["2019"] == "/team/LSU?season=2019"


def test_conference_spread_stats():
    rows = [{"conference": "A", "rank": r} for r in (1, 3, 5, 7, 9)]
    rows += [{"conference": "B", "rank": r} for r in (2, 40)]
    rows += [{"conference": "", "rank": 4}]          # independent: left out
    a, b = conference_spread(rows)
    assert (a["conference"], a["best"], a["q1"], a["median"], a["q3"], a["worst"],
            a["mean"]) == ("A", 1, 3, 5, 7, 9, 5)
    assert b["conference"] == "B" and b["median"] == 21    # sorted by median


def test_conference_spread_on_real_data(data):
    spread = conference_spread(data.rankings)
    for s in spread:
        assert s["best"] <= s["q1"] <= s["median"] <= s["q3"] <= s["worst"]
    medians = [s["median"] for s in spread]
    assert medians == sorted(medians)
    html = _render("conferences.html", "conferences", data, **conferences_context(data))
    assert html.count('class="spread-box"') == len(spread)


def test_link_checker_catches_a_dead_link(tmp_path):
    """CI trusts a clean report from this, so prove it can fail."""
    (tmp_path / "a.html").write_text(
        '<a href="b.html">ok</a> <a href="gone.html?x=1">dead</a>'
        ' <a href="#top">anchor</a> <a href="https://x.org">ext</a>',
        encoding="utf-8")
    (tmp_path / "b.html").write_text("", encoding="utf-8")
    assert broken_links(tmp_path) == {"gone.html": ["a.html"]}


def test_conference_races_never_include_fcs_teams(data):
    """FCS teams are rated for graph connectivity (cfbrank.games) but
    never shown -- rankings, Predict and resume all filter to FBS alone,
    and Conference Races must too. Regression test for a real bug: an
    unfiltered conference map let a handful of FBS-vs-FCS games drag
    whole FCS conferences (Big Sky, Ivy League, ...) into this section.
    """
    fbs = set(data.model.fbs_ratings())
    for conf, standings in data.conference_races.items():
        for team, _pct in standings:
            assert team in fbs, f"{team} ({conf}) is not an FBS team"
