"""Logo slots: the one place server-rendered logo markup is built.

Every logo sits in a fixed-size, centered, transparent box (the
`.logo-slot` family in retro.css), so rows line up whether or not a team
has a logo and nothing shifts while images load. A team with no logo
still gets its empty box.

The logo is decorative -- the team's name is always printed beside it --
so the slot is hidden from screen readers and the image has an empty alt.

Logos belong next to names in tables, headings, cards and pickers, not
inside running text. Where a rank or "vs" accompanies a name, the logo
leads (see app.pages.team_link).

app/static/search.js (`logoSlot`) writes the same markup for the places
JavaScript builds (header search, Predict results, Time Machine). Keep
the two in step; tests/test_logos.py pins the Python side.
"""

from __future__ import annotations

from markupsafe import Markup

from cfbrank.logos import (
    DARK_VARIANT, LIGHT_VARIANT, TEAM_LOGO_TEMPLATE, UPSIDE_DOWN_TEAMS,
    conference_logo_url, team_logo_url, upside_down_logo_ids,
)

# Slot size -> pixel width of the CFBD file to request. The slot's display
# size is CSS (.logo-sm/-md/-lg); each file is ~2x its slot so logos stay
# crisp on high-density screens without shipping 500 px everywhere.
FILE_PX = {"sm": 64, "md": 96, "lg": 128}


def logo_config() -> dict:
    """What search.js needs to draw the same logos (-> .site-search[data-logo])."""
    return {"template": TEAM_LOGO_TEMPLATE, "files": FILE_PX,
            "light": LIGHT_VARIANT, "dark": DARK_VARIANT,
            "flipped": upside_down_logo_ids()}


def logo_slot(url: str | None, size: str, flipped: bool = False) -> Markup:
    """A fixed-size box holding `url`, or empty when there is none."""
    if size not in FILE_PX:
        raise ValueError(f"unknown logo size {size!r}; use one of {sorted(FILE_PX)}")
    image = (Markup('<img src="{}" alt="" loading="lazy" decoding="async"{}>').format(
        url, Markup(' class="logo-flipped"') if flipped else Markup(""))
        if url else Markup(""))
    return Markup('<span class="logo-slot logo-{}" aria-hidden="true">{}</span>'
                  ).format(size, image)


def team_logo(team: str, size: str = "sm", dark: bool = False) -> Markup:
    """`dark=True` picks the artwork meant for dark backgrounds."""
    return logo_slot(team_logo_url(team, FILE_PX[size], dark), size,
                     flipped=team in UPSIDE_DOWN_TEAMS)


def conference_logo(conference: str, size: str = "md") -> Markup:
    return logo_slot(conference_logo_url(conference), size)
