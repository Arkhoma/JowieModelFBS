"""Per-game home-site context: crowd, stadium, altitude, travel.

Research input only; the production model does not use it (yet). Built
to test whether home field varies by environment, see
tools/probe_home_field.py.

Sources, all from tools/fetch_cfbd.py --only games:
  <raw>/<season>/games.json.gz  attendance, venueId, pregame Elo
  <raw>/venues.json.gz          capacity, elevation, dome, lat/long
  <raw>/teams_all.json.gz       each team's home location (for travel)
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .games import PROJECT_ROOT

RAW = PROJECT_ROOT / "data" / "raw"
EARTH_RADIUS_MILES = 3958.8


@dataclass(frozen=True, slots=True)
class GameContext:
    game_id: str
    attendance: float | None
    capacity: float | None
    elevation_m: float | None   # CFBD reports metres (Falcon Stadium ~2025)
    dome: bool
    travel_miles: float | None   # away team's campus -> venue
    home_elo: float | None
    away_elo: float | None

    @property
    def fill(self) -> float | None:
        """Share of capacity filled, capped at 1.1 (standing room)."""
        if not self.attendance or not self.capacity:
            return None
        return min(self.attendance / self.capacity, 1.1)


def _load(path: Path) -> list:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(h))


def _latlong(record: dict) -> tuple[float, float] | None:
    location = record.get("location") or record
    lat, lon = location.get("latitude"), location.get("longitude")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


@lru_cache(maxsize=1)
def venues() -> dict[int, dict]:
    return {v["id"]: v for v in _load(RAW / "venues.json.gz") if v.get("id")}


@lru_cache(maxsize=1)
def team_locations() -> dict[str, tuple[float, float]]:
    out = {}
    for team in _load(RAW / "teams_all.json.gz"):
        where = _latlong(team.get("location") or {})
        if where and team.get("school"):
            out[team["school"]] = where
    return out


@lru_cache(maxsize=None)
def season_context(season: int) -> dict[str, GameContext]:
    """game_id -> context for every game CFBD has in that season."""
    stadiums, homes = venues(), team_locations()
    out = {}
    for game in _load(RAW / str(season) / "games.json.gz"):
        venue = stadiums.get(game.get("venueId"), {})
        site = _latlong(venue)
        away_home = homes.get(game.get("awayTeam", ""))
        gid = str(game.get("id", ""))
        out[gid] = GameContext(
            game_id=gid,
            attendance=_num(game.get("attendance")),
            capacity=_num(venue.get("capacity")),
            elevation_m=_num(venue.get("elevation")),
            dome=bool(venue.get("dome")),
            travel_miles=(_miles(away_home, site)
                          if site and away_home else None),
            home_elo=_num(game.get("homePregameElo")),
            away_elo=_num(game.get("awayPregameElo")),
        )
    return out
