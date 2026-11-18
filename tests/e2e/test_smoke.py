"""Browser smoke tests: the pages load clean and the interactive bits work.

Everything here is something the Python tests can't see, because it only
happens in a browser: JavaScript errors, the combobox filter, logos drawn
by JS, layout overflow. (The team-picker filter once reordered options
without hiding any; no Python test could have caught that.)

See conftest.py for how to run them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Playwright is a test-only dependency: without it these skip, not break collection.
expect = pytest.importorskip("playwright.sync_api").expect

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from build_static_site import STATIC_URLS, _latest  # noqa: E402

pytestmark = pytest.mark.e2e

LATEST = _latest()
PAGES = {
    "home": "index.html",
    "rankings": STATIC_URLS.rank(LATEST, "predictive"),
    "conferences": STATIC_URLS.conferences(LATEST),
    "playoff": STATIC_URLS.playoff(LATEST),
    "predict": STATIC_URLS.predict(LATEST),
    "this week": STATIC_URLS.nav["slate"],
    "time machine": STATIC_URLS.nav["time_machine"],
    "about": STATIC_URLS.nav["about"],
}


@pytest.mark.parametrize("path", PAGES.values(), ids=PAGES.keys())
def test_page_loads_clean(browser_page, path):
    browser_page.open(path)
    page = browser_page.page
    expect(page.locator("main h1, main h2").first).to_be_visible()
    expect(page.locator("nav[aria-label=Main] a")).to_have_count(6)
    assert browser_page.problems == []


@pytest.mark.parametrize("width", [390, 1280], ids=["phone", "desktop"])
def test_nav_fits_without_sideways_scrolling(browser_page, width):
    page = browser_page.page
    page.set_viewport_size({"width": width, "height": 800})
    browser_page.open(PAGES["rankings"])
    expect(page.locator("nav[aria-label=Main] a").last).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_header_search_shows_logos_and_texas_is_upside_down(browser_page):
    browser_page.open(PAGES["home"])
    page = browser_page.page
    search = page.locator("#site-search")
    search.click()
    search.press_sequentially("texas", delay=40)
    options = page.locator("#site-search-list [role=option]")
    expect(options.first).to_be_visible()
    texas = options.filter(has_text="Texas").first
    assert "logo-flipped" in (texas.locator("img").get_attribute("class") or "")
    other = options.filter(has_text="Texas A&M")
    if other.count():
        assert "logo-flipped" not in (other.locator("img").get_attribute("class") or "")


def test_picker_filters_for_real_and_shows_the_chosen_teams_logo(browser_page):
    browser_page.open(PAGES["predict"])
    page = browser_page.page
    away = page.locator("#away_team")
    field_logo = away.locator("xpath=../span[@data-combo-logo]//img")

    away.click()
    away.press("Control+A")
    away.press_sequentially("ala", delay=40)       # real key presses, not .value =
    names = page.locator("#away_team-list [role=option]:visible").evaluate_all(
        "els => els.map(e => e.dataset.value)")
    assert names and all("ala" in n.lower() for n in names)
    assert "Air Force" not in names                # hidden, not just reordered
    expect(field_logo).to_have_count(0)            # partial text: empty slot

    away.press_sequentially("bama", delay=40)      # now exactly "Alabama"
    expect(field_logo).to_have_count(1)
    assert browser_page.problems == []


def test_predict_shows_a_scoreboard_with_logos_but_none_in_the_sentences(browser_page):
    browser_page.open(PAGES["predict"])
    page = browser_page.page
    page.get_by_role("button", name="Predict").click()
    board = page.locator("#result .scoreboard")
    expect(board).to_be_visible()
    expect(board.locator(".logo-md")).to_have_count(2)
    expect(board.locator(".text-center .logo-slot")).to_have_count(0)  # "At ..." / "by" lines
    # Default matchup is Texas at Oklahoma: Texas is upside down on the board
    # and in its win-probability row.
    expect(page.locator("#result img.logo-flipped")).to_have_count(2)
    assert browser_page.problems == []


def test_swap_moves_the_names_and_their_logos(browser_page):
    browser_page.open(PAGES["predict"])
    page = browser_page.page
    home_logo = page.locator("#home_team").locator("xpath=../span[@data-combo-logo]//img")
    page.locator("#swap-teams").click()
    expect(page.locator("#away_team")).to_have_value("Oklahoma")
    expect(page.locator("#home_team")).to_have_value("Texas")
    assert "logo-flipped" in (home_logo.get_attribute("class") or "")   # Texas is home now


def test_time_machine_runs_a_matchup(browser_page):
    browser_page.open(PAGES["time machine"])
    page = browser_page.page
    page.get_by_role("button", name="Run it").click()
    board = page.locator("#result .scoreboard")
    expect(board).to_be_visible()
    expect(board.locator(".text-center .logo-slot")).to_have_count(0)
    assert browser_page.problems == []


def test_this_week_lists_games_most_important_first_or_says_none_are_left(browser_page):
    browser_page.open(PAGES["this week"])
    page = browser_page.page
    rows = page.locator(".stat-table tbody tr")
    if rows.count() == 0:
        expect(page.get_by_text("No games left to predict")).to_be_visible()
        return
    widths = page.locator(".stat-table tbody tr .bg-brick").evaluate_all(
        "els => els.map(e => parseFloat(e.style.width))")
    assert widths == sorted(widths, reverse=True)
    assert browser_page.problems == []


def test_this_week_on_a_phone_shows_the_playoff_impact_without_sideways_scrolling(browser_page):
    """The impact column once sat off-screen on phones. Now each game's cell
    carries it, and exactly one copy of the bar is visible per game."""
    page = browser_page.page
    page.set_viewport_size({"width": 390, "height": 800})
    browser_page.open(PAGES["this week"])
    first = page.locator(".stat-table tbody tr").first
    if first.count() == 0:
        pytest.skip("no games this week")
    expect(first.locator(".bg-brick:visible")).to_have_count(1)
    assert page.evaluate("""() => {
        const wrap = document.querySelector('.stat-table').parentElement;
        return wrap.scrollWidth <= wrap.clientWidth + 1;
    }""")


def test_the_season_record_fits_a_phone_and_opens_week_by_week(browser_page):
    """The record table once spilled off a phone screen (seven columns)."""
    page = browser_page.page
    page.set_viewport_size({"width": 390, "height": 800})
    browser_page.open(PAGES["this week"])
    record = page.locator("section[aria-labelledby=record]")
    if record.count() == 0:
        pytest.skip("no graded games yet")
    record.locator("summary").click()
    expect(record.locator("tbody tr").first).to_be_visible()
    assert page.evaluate("""() => {
        const wrap = document.querySelector('section[aria-labelledby=record] .overflow-x-auto');
        return wrap.scrollWidth <= wrap.clientWidth + 1;
    }""")
    # "Week 4" must not wrap onto two lines.
    label = record.locator("tbody th").first
    assert label.evaluate("el => el.getBoundingClientRect().height") < 60
    assert browser_page.problems == []


def test_a_team_page_opens_from_the_rankings(browser_page):
    browser_page.open(PAGES["rankings"])
    page = browser_page.page
    page.locator(".stat-table .team-link").first.click()
    expect(page.locator("h2.page-title")).to_be_visible()
    expect(page.locator(".logo-lg")).to_have_count(1)
    assert browser_page.problems == []
