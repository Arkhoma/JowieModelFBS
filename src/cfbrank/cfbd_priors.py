"""Preseason roster signals from the CFBD API (tools/fetch_cfbd.py --only priors).

Everything here is known before a season's first kickoff:

  talent     247 team talent composite -- the recruiting stars of the
             players actually on this season's roster
  recruiting average recruiting-class points over the last four classes
  portal_net summed 247 rating of incoming minus outgoing transfers
             (the portal endpoint starts in 2021; earlier seasons are 0)
  cfbd_ret   CFBD's returning share of last season's PPA, which unlike
             our returning.py counts every player, tackles included
  new_coach  1 if the head coach differs from last season's

Each function returns {team: value}; teams missing from a file are left
out and the caller treats them as average.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
RECRUITING_CLASSES = 4
# An unrated transfer is roughly a low three-star (247 scale 0-1).
UNRATED_TRANSFER = 0.80


@lru_cache(maxsize=None)
def load_records(season: int | None, name: str) -> tuple[dict, ...]:
    """Records from data/raw/<season>/<name>.json.gz; () if absent."""
    folder = RAW / str(season) if season is not None else RAW
    path = folder / f"{name}.json.gz"
    if not path.exists():
        return ()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return tuple(json.load(handle))


def talent(season: int) -> dict[str, float]:
    return {r["team"]: float(r["talent"])
            for r in load_records(season, "talent") if r.get("talent")}


def recruiting(season: int) -> dict[str, float]:
    """Mean class points over the classes that make up this roster."""
    points: dict[str, list[float]] = defaultdict(list)
    for year in range(season - RECRUITING_CLASSES + 1, season + 1):
        for r in load_records(year, "recruiting_teams"):
            points[r["team"]].append(float(r["points"]))
    return {team: sum(p) / RECRUITING_CLASSES for team, p in points.items()}


def portal_net(season: int) -> dict[str, float]:
    net: dict[str, float] = defaultdict(float)
    for r in load_records(season, "portal"):
        value = r.get("rating") or UNRATED_TRANSFER
        if r.get("destination"):
            net[r["destination"]] += value
        if r.get("origin"):
            net[r["origin"]] -= value
    return dict(net)


def cfbd_returning(season: int) -> dict[str, float]:
    return {r["team"]: float(r["percentPPA"])
            for r in load_records(season, "returning_production")
            if r.get("percentPPA") is not None}


@lru_cache(maxsize=None)
def _coach_seasons() -> dict[tuple[str, int], dict[str, int]]:
    """(school, year) -> {coach name: games coached}."""
    out: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for coach in load_records(None, "coaches"):
        name = f"{coach['firstName']} {coach['lastName']}"
        for s in coach.get("seasons", []):
            out[(s["school"], int(s["year"]))][name] = int(s.get("games") or 0)
    return dict(out)


def new_coach(season: int) -> dict[str, float]:
    """1.0 where last season's main head coach coached NO games this season.

    Deliberately not "this season's main coach differs": that would flag
    a coach fired mid-season -- a symptom of the very season we are
    predicting -- and leak the outcome into the preseason prior. A coach
    who is still there for week 1 counts as returning, however it ends.
    Teams with no coach record yet this season are omitted (unknown).
    """
    table = _coach_seasons()
    out = {}
    for (school, year), coaches in table.items():
        if year != season - 1 or (school, season) not in table:
            continue
        main = max(coaches, key=coaches.get)
        out[school] = float(main not in table[(school, season)])
    return out
