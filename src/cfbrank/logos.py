"""Where a team's or conference's logo lives.

Team logos: CollegeFootballData serves every size (16-500 px) at a
predictable URL, so the committed index (data/team_logos.json, built by
tools/build_logo_index.py) only needs each team's numeric id.

Conference logos: CFBD has none, so these come from ESPN's CDN via a
short hand-kept table. There are only eleven FBS labels, the same every
season since 2018, and each id was checked by eye against its logo.
A name with no logo returns None -- callers show an empty slot, never a
broken image.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "team_logos.json"

# `{variant}` is "logos" (for light backgrounds) or "logos-dark" (artwork
# redrawn for dark ones). `{size}` is a pixel width CFBD actually serves.
# JS fills the same placeholders, so the CDN is named in exactly one place.
TEAM_LOGO_TEMPLATE = "https://cdn.collegefootballdata.com/{variant}/{size}/{id}.png"
LIGHT_VARIANT, DARK_VARIANT = "logos", "logos-dark"
CONFERENCE_LOGO_URL = "https://a.espncdn.com/i/teamlogos/ncaa_conf/500/{id}.png"

# Shown rotated 180 degrees, everywhere. (Texas, on purpose.)
UPSIDE_DOWN_TEAMS = frozenset({"Texas"})

# Keys are the conference labels used in the schedule data.
CONFERENCE_IDS = {
    "ACC": 1, "Big 12": 4, "Big Ten": 5, "SEC": 8, "Pac-12": 9,
    "Conference USA": 12, "Mid-American": 15, "Mountain West": 17,
    "FBS Independents": 18, "Sun Belt": 37, "American Athletic": 151,
}


@lru_cache(maxsize=1)
def team_logo_ids() -> dict[str, int]:
    """{team: CFBD id}; empty (not an error) if the index is missing."""
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))["ids"]
    except (FileNotFoundError, KeyError, ValueError):
        return {}


def team_logo_id(team: str) -> int | None:
    return team_logo_ids().get(team)


def team_logo_url(team: str, size: int, dark: bool = False) -> str | None:
    logo_id = team_logo_id(team)
    if logo_id is None:
        return None
    return TEAM_LOGO_TEMPLATE.format(
        variant=DARK_VARIANT if dark else LIGHT_VARIANT, size=size, id=logo_id)


def upside_down_logo_ids() -> list[int]:
    return sorted(i for t in UPSIDE_DOWN_TEAMS if (i := team_logo_id(t)) is not None)


def conference_logo_url(conference: str) -> str | None:
    logo_id = CONFERENCE_IDS.get(conference)
    return None if logo_id is None else CONFERENCE_LOGO_URL.format(id=logo_id)
