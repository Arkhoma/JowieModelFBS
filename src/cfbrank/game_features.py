"""Per-game signals extracted from play data, keyed by game id.

Two corrections to the final score, both from MODEL_SPEC.md:

GARBAGE-TIME MARGIN -- the home margin at the moment the game was
decided (first snap past the epa_games.GARBAGE_TIME_MARGINS threshold).
Points traded by backups after that tell us little about the starters.
Games that never reach garbage time keep their real final margin.

FUMBLE LUCK -- recovering a fumble is close to a coin flip, so the side
that falls on more loose balls got possessions it did not earn. Measured
from possession changes rather than the `fumble_recovered_player`
column, which is blank on most recoveries in this data.

    home_fumble_luck = home recoveries - total fumbles / 2

Positive means the home side got lucky. Interceptions are NOT luck and
are deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .epa_games import is_garbage_time
from .games import Game
from .plays import load_plays


@dataclass(frozen=True, slots=True)
class GameFeatures:
    game_id: str
    garbage_margin: float | None   # home margin when decided; None = never
    home_fumble_luck: float


def _home_sign(frame: pd.DataFrame, home_team: str) -> np.ndarray:
    return np.where(frame["team"].to_numpy() == home_team, 1.0, -1.0)


def _garbage_margin(frame: pd.DataFrame, home_team: str) -> float | None:
    decided = is_garbage_time(frame).to_numpy()
    if not decided.any():
        return None
    first = int(np.argmax(decided))
    sign = _home_sign(frame.iloc[[first]], home_team)[0]
    return float(sign * frame["score_margin"].iloc[first])


def _fumble_luck(frame: pd.DataFrame, home_team: str) -> float:
    if "fumble_player" not in frame.columns:
        return 0.0
    fumbled = (frame["fumble_player"].fillna("").astype(str).str.strip()
               != "").to_numpy()
    if not fumbled.any():
        return 0.0
    offense = frame["team"].to_numpy()
    next_offense = np.append(offense[1:], None)
    same_half = np.append(frame["half"].to_numpy()[1:] == frame["half"]
                          .to_numpy()[:-1], False)
    lost = (next_offense != offense) & same_half
    # A lost fumble means the DEFENCE recovered.
    home_recovered = 0.0
    total = 0.0
    for i in np.flatnonzero(fumbled):
        if not same_half[i]:
            continue          # end of half: nobody knows who recovered
        total += 1.0
        recoverer = next_offense[i] if lost[i] else offense[i]
        home_recovered += recoverer == home_team
    return home_recovered - total / 2.0


def season_features(season: int, games: list[Game]) -> dict[str, GameFeatures]:
    """Features for every scheduled game that has play data."""
    plays = load_plays(season)
    homes = {str(g.game_id): g.home_team for g in games}
    built: dict[str, GameFeatures] = {}
    for game_id, frame in plays.groupby("game_id", sort=False):
        home = homes.get(str(game_id))
        if home is None:
            continue
        built[str(game_id)] = GameFeatures(
            game_id=str(game_id),
            garbage_margin=_garbage_margin(frame, home),
            home_fumble_luck=_fumble_luck(frame, home),
        )
    return built
