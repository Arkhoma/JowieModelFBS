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
    LIVE_URLS, SiteUrls, base_context, playoff_context, predict_context,
    team_context,
)
from build_static_site import STATIC_URLS, _env, _latest  # noqa: E402
from check_static_links import broken_links  # noqa: E402
from cfbrank.games import available_seasons, fbs_seasons, fbs_teams  # noqa: E402
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
                      predict=lambda s: "p", playoff=lambda s: "o")
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
                   snapshots={str(SEASON): snap}, teams=teams)
    assert 'id="a_team"' in html and 'id="b_season"' in html
    assert html.count('role="option"') == 2 * len(teams)
    assert '<option value="2019" selected>' in html       # 2019 LSU default
    assert 'id="snapshots"' in html and "time_machine.js" in html
    assert "Vegas Script" not in html


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
