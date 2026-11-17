"""Tests for cfbrank.logos and app.logo_view: where logos live, and the slot.

Logo coverage is checked against the schedule data itself, so a team that
joins FBS without a logo shows up here rather than as a silent gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app.logo_view import (  # noqa: E402
    FILE_PX, conference_logo, logo_config, logo_slot, team_logo,
)
from build_logo_index import build_index  # noqa: E402
from cfbrank.games import available_seasons, fbs_teams, load_season  # noqa: E402
from cfbrank.logos import (  # noqa: E402
    CONFERENCE_IDS, TEAM_LOGO_TEMPLATE, UPSIDE_DOWN_TEAMS, conference_logo_url,
    team_logo_id, team_logo_ids, team_logo_url, upside_down_logo_ids,
)


def test_team_logo_url_fills_the_cdn_template():
    logo_id = team_logo_id("Oklahoma")
    assert logo_id is not None
    assert team_logo_url("Oklahoma", 64) == TEAM_LOGO_TEMPLATE.format(
        variant="logos", size=64, id=logo_id)
    assert team_logo_url("Oklahoma", 64).endswith(f"/logos/64/{logo_id}.png")


def test_dark_variant_is_the_same_logo_redrawn_for_dark_backgrounds():
    light, dark = team_logo_url("Navy", 96), team_logo_url("Navy", 96, dark=True)
    assert "/logos/96/" in light and "/logos-dark/96/" in dark
    assert light.rsplit("/", 1)[1] == dark.rsplit("/", 1)[1]


def test_unknown_team_has_no_logo_not_an_error():
    assert team_logo_id("Nowhere Tech") is None
    assert team_logo_url("Nowhere Tech", 64) is None
    assert conference_logo_url("Nowhere Conference") is None


def test_every_current_fbs_team_has_a_logo():
    """If this fails a team was promoted: run tools/build_logo_index.py."""
    latest = max(available_seasons())
    missing = sorted(t for t in fbs_teams(latest) if team_logo_id(t) is None)
    assert not missing, f"no logo for: {missing}"


def test_nearly_every_team_ever_in_fbs_has_a_logo():
    """Older seasons feed the Time Machine; a few defunct names may lack one."""
    ever = set().union(*(fbs_teams(s) for s in available_seasons()))
    covered = sum(team_logo_id(t) is not None for t in ever) / len(ever)
    assert covered >= 0.95, f"only {covered:.0%} of {len(ever)} teams have logos"


def test_every_conference_label_in_the_data_has_a_logo():
    labels = set()
    for season in available_seasons():
        if season >= 2018:
            labels |= {g.home_conference for g in load_season(season)
                       if g.home_division == "fbs" and g.home_conference}
    assert len(labels) >= 10, "found no conference labels; did the data columns change?"
    assert labels <= set(CONFERENCE_IDS), f"unmapped: {sorted(labels - set(CONFERENCE_IDS))}"
    assert all(conference_logo_url(c) for c in labels)


def test_build_index_keeps_only_teams_with_logos_first_record_wins():
    teams = [{"school": "A", "id": 1, "logos": ["x"]},
             {"school": "B", "id": 2, "logos": []},
             {"school": "A", "id": 99, "logos": ["y"]},
             {"school": "C", "id": 3}]
    assert build_index(teams) == {"A": 1}


def test_the_committed_index_is_plain_name_to_id():
    ids = team_logo_ids()
    assert len(ids) > 100
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in ids.items())


def test_texas_is_upside_down_and_the_flip_is_exactly_that_team():
    assert UPSIDE_DOWN_TEAMS == {"Texas"}
    assert upside_down_logo_ids() == [team_logo_id("Texas")]
    assert logo_config()["flipped"] == [team_logo_id("Texas")]
    assert 'class="logo-flipped"' in str(team_logo("Texas"))
    assert 'class="logo-flipped"' in str(team_logo("Texas", "md", dark=True))
    for other in ("Texas A&M", "Texas Tech", "Oklahoma"):
        assert "logo-flipped" not in str(team_logo(other))


def test_scoreboard_logos_use_the_dark_artwork():
    assert "/logos-dark/96/" in str(team_logo("Navy", "md", dark=True))
    assert "/logos/96/" in str(team_logo("Navy", "md"))


def test_js_gets_the_same_cdn_sizes_variants_and_flips():
    config = logo_config()
    assert config["template"] == TEAM_LOGO_TEMPLATE
    assert config["files"] == FILE_PX
    assert (config["light"], config["dark"]) == ("logos", "logos-dark")


# --- slot markup ---------------------------------------------------------

def test_slot_with_a_logo_is_decorative_and_lazy():
    html = str(logo_slot("https://example.test/a.png", "sm"))
    assert html == ('<span class="logo-slot logo-sm" aria-hidden="true">'
                    '<img src="https://example.test/a.png" alt="" '
                    'loading="lazy" decoding="async"></span>')


def test_slot_without_a_logo_is_still_there_but_empty():
    assert str(logo_slot(None, "md")) == '<span class="logo-slot logo-md" aria-hidden="true"></span>'


def test_slot_escapes_the_url():
    html = str(logo_slot('x"><script>', "sm"))
    assert "<script>" not in html
    assert 'src="x&#34;&gt;&lt;script&gt;"' in html


def test_unknown_slot_size_is_a_programming_error():
    with pytest.raises(ValueError):
        logo_slot(None, "huge")


@pytest.mark.parametrize("size", sorted(FILE_PX))
def test_each_size_requests_its_own_file_width(size):
    assert f"/logos/{FILE_PX[size]}/" in str(team_logo("Oklahoma", size))


def test_conference_logo_uses_the_espn_cdn_and_unknown_is_empty():
    assert "espncdn.com/i/teamlogos/ncaa_conf/500/8.png" in str(conference_logo("SEC"))
    assert "<img" not in str(conference_logo("Nowhere Conference"))
