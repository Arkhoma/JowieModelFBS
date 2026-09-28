"""Closing betting lines, keyed by ESPN/CFBD game id.

Two sources, merged:
  <raw>/betting/cfb_line_odds.csv.gz  sportsdataverse mirror (no key needed)
  <raw>/<season>/lines.json.gz        CFBD /lines (fills the current season,
                                      which the mirror publishes late)

The two agree to within 1.5 points on 99.3% of 6,893 shared games with no
sign flips (tools/check_lines.py), so the mirror wins where both exist
and CFBD only fills gaps.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .games import PROJECT_ROOT

RAW = PROJECT_ROOT / "data" / "raw"
MIRROR_PATH = RAW / "betting" / "cfb_line_odds.csv.gz"
CFBD_BOOKS = ("consensus", "Bovada", "DraftKings", "ESPN Bet",
              "William Hill (New Jersey)")


def _consensus_or_median(lines: pd.DataFrame) -> pd.Series:
    """game_id -> the 'consensus' book's line, else the median of books."""
    lines = lines.assign(game_id=lines.game_id.astype("int64").astype(str))
    consensus = lines[lines.book == "consensus"].groupby("game_id").lines.first()
    return consensus.combine_first(lines.groupby("game_id").lines.median())


def _mirror(kind: str) -> pd.DataFrame:
    if not MIRROR_PATH.exists():
        return pd.DataFrame(columns=["market_type", "lines", "game_id", "book",
                                     "abbr", "game_desc"])
    lines = pd.read_csv(MIRROR_PATH)
    return lines[(lines.market_type == kind) & lines.lines.notna()
                 & lines.game_id.notna()]


def _cfbd(field: str) -> dict[str, float]:
    """game_id -> the preferred book's `field` from every CFBD lines file.
    `spread` is quoted home-side (negative = home favoured)."""
    out: dict[str, float] = {}
    for path in sorted(RAW.glob("*/lines.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            games = json.load(handle)
        for game in games:
            books = {b.get("provider"): b.get(field)
                     for b in game.get("lines") or [] if b.get(field) is not None}
            if books:
                out[str(game["id"])] = next(
                    (float(books[p]) for p in CFBD_BOOKS if p in books),
                    float(np.median(list(books.values()))))
    return out


@lru_cache(maxsize=1)
def home_lines() -> dict[str, float]:
    """game_id -> market's predicted HOME margin (minus the spread)."""
    lines = _mirror("spread")
    lines = lines[lines.abbr == lines.game_desc.str.split("@").str[1]]
    mirror = (-_consensus_or_median(lines)).to_dict() if len(lines) else {}
    extra = {gid: -spread for gid, spread in _cfbd("spread").items()}
    return {**extra, **mirror}


@lru_cache(maxsize=1)
def total_lines() -> dict[str, float]:
    """game_id -> market's over/under (combined points)."""
    lines = _mirror("total")
    lines = lines[lines.abbr == "over"]
    mirror = _consensus_or_median(lines).to_dict() if len(lines) else {}
    return {**_cfbd("overUnder"), **mirror}
