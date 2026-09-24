"""Returning production -- what the market knows in August that we did not.

The old prior was "last season's rating, times a flat 0.63". Every team
got the same decay, whether it returned its whole two-deep or lost its
quarterback and eight starters to the draft and the portal. That is the
single largest reason we trailed the closing line by 0.6 points in weeks
4-8 and only 0.3 after.

Here, per team, per season, from play-level data plus next-season
rosters (player ids join at 100%):

  ret_off   share of last season's offensive yards (rush + receive) by
            players still on this team
  ret_qb    same, passing yards only -- the quarterback is the single
            most valuable returning piece
  ret_def   share of last season's defensive "splash" plays (sacks,
            interceptions, breakups, forced fumbles) still on the team.
            There is no tackle data, so this is a noisier proxy.
  portal    offensive yards brought in from OTHER teams, as a share of an
            average FBS team's yards

How much each matters is not assumed -- see prior_model.py, which fits
the weights on past offseasons only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROSTER_DIR = PROJECT_ROOT / "data" / "raw" / "rosters"
PLAYS_DIR = PROJECT_ROOT / "data" / "raw" / "plays"

DEFENSIVE_EVENTS = ("sack_player_id", "interception_player_id",
                    "pass_breakup_player_id", "fumble_forced_player_id")


@dataclass(frozen=True, slots=True)
class Returning:
    ret_off: float
    ret_qb: float
    ret_def: float
    portal: float


@lru_cache(maxsize=None)
def load_roster(season: int) -> pd.DataFrame:
    path = ROSTER_DIR / f"rosters_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No roster for {season} at {path}")
    roster = pd.read_parquet(path, columns=["athlete_id", "team"])
    roster = roster.dropna()
    return roster.assign(athlete_id=roster.athlete_id.astype("int64"))


def _credit(plays: pd.DataFrame, id_col: str, team_col: str,
            value: pd.Series | float, label: str) -> pd.DataFrame:
    mask = plays[id_col].notna()
    values = value[mask] if isinstance(value, pd.Series) else value
    return pd.DataFrame({
        "player_id": plays.loc[mask, id_col].astype("int64"),
        "team": plays.loc[mask, team_col],
        label: values,
    })


@lru_cache(maxsize=None)
def player_production(season: int) -> pd.DataFrame:
    """One row per (player, team): off_yds, qb_yds, def_plays."""
    columns = ["team", "opponent", "rush_player_id", "rush_yds",
               "reception_player_id", "reception_yds",
               "completion_player_id", "completion_yds", *DEFENSIVE_EVENTS]
    plays = pd.read_parquet(PLAYS_DIR / f"plays_{season}.parquet",
                            columns=columns)
    parts = [
        _credit(plays, "rush_player_id", "team",
                plays.rush_yds.fillna(0).clip(lower=0), "off_yds"),
        _credit(plays, "reception_player_id", "team",
                plays.reception_yds.fillna(0).clip(lower=0), "off_yds"),
        _credit(plays, "completion_player_id", "team",
                plays.completion_yds.fillna(0).clip(lower=0), "qb_yds"),
        *(_credit(plays, col, "opponent", 1.0, "def_plays")
          for col in DEFENSIVE_EVENTS),
    ]
    stacked = pd.concat(parts, ignore_index=True).fillna(0.0)
    return stacked.groupby(["player_id", "team"], as_index=False).sum()


def _share(frame: pd.DataFrame, column: str) -> pd.Series:
    total = frame.groupby("team")[column].sum()
    kept = frame[frame.stayed].groupby("team")[column].sum()
    return (kept.reindex(total.index, fill_value=0.0)
            / total.where(total > 0)).fillna(0.5)


def returning_production(season: int) -> dict[str, Returning]:
    """Returning-production features for every team with last-season data.

    Teams absent from the result (newcomers, missing files) should be
    treated as average by the caller.
    """
    previous = player_production(season - 1)
    roster = load_roster(season)
    now_team = roster.drop_duplicates("athlete_id").set_index(
        "athlete_id").team

    previous = previous.assign(new_team=previous.player_id.map(now_team))
    previous = previous.assign(stayed=previous.new_team == previous.team)

    ret_off = _share(previous, "off_yds")
    ret_qb = _share(previous, "qb_yds")
    ret_def = _share(previous, "def_plays")

    team_off = previous.groupby("team").off_yds.sum()
    average_team_off = float(team_off[team_off > 0].median())
    moved = previous[previous.new_team.notna() & ~previous.stayed]
    portal = moved.groupby("new_team").off_yds.sum() / average_team_off

    return {
        team: Returning(
            ret_off=float(ret_off[team]),
            ret_qb=float(ret_qb[team]),
            ret_def=float(ret_def[team]),
            portal=float(portal.get(team, 0.0)),
        )
        for team in ret_off.index
    }
