"""Tests for app.main's shared page-context builder.

No TestClient/httpx here on purpose: _base_context is a plain function,
and the bug this guards against (team_url=None silently falling back to
the live /team/ route instead of staying off) doesn't need a running
server to catch -- it needs exactly one dict comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.main import _base_context, _rank_url, _team_url  # noqa: E402


def test_team_url_defaults_to_the_live_route_when_omitted():
    """Not passing team_url at all -- the common case for every page that
    isn't a rankings table -- should still get a working link."""
    ctx = _base_context("about", 2026)
    assert ctx["team_url"] is _team_url


def test_team_url_stays_none_when_explicitly_disabled():
    """build_static_site.py passes team_url=None for every season but the
    latest, since only the latest season gets static team pages. That
    None must survive, not get silently replaced by the live route --
    which is exactly the bug this regression test catches.
    """
    ctx = _base_context("rankings", 2024, team_url=None)
    assert ctx["team_url"] is None


def test_team_url_keeps_a_real_override():
    def fake_url(team, season):
        return f"team-{team}.html"

    ctx = _base_context("rankings", 2026, team_url=fake_url)
    assert ctx["team_url"] is fake_url


def test_rank_url_still_defaults_normally():
    """rank_url never needs the None-means-disabled trick team_url does --
    confirm the sentinel fix left its simpler default alone."""
    ctx = _base_context("rankings", 2026)
    assert ctx["rank_url"] is _rank_url
