"""Per-game EPA aggregation -- the bridge from plays to ratings.

This is the module that was missing. The expected-points model and the
play loader both existed and were tested, but nothing imported them:
the ridge engine was fed FINAL SCORES only. That is the single biggest
methodological gap versus SP+ / FEI, which rate on per-play efficiency.

Why per-play beats final score: a 24-21 win where you outgained the
opponent by 2.5 yards per play is stronger evidence than a 35-3 win
built on two pick-sixes. Final margin is a lossy, high-variance summary
of ~150 plays. EPA per play uses all of them.

Two corrections applied here, both specified long ago and never built:

GARBAGE TIME -- once a game is decided, both teams stop playing real
football. Leaving those snaps in punishes teams for calling off the dogs
and rewards stat-padding against backups. The thresholds widen by period
because a 28-point lead means something very different in the first
quarter than the fourth.

TURNOVER LUCK -- fumble RECOVERY is close to a coin flip (recovering
teams hover near 50% regardless of skill), while forcing fumbles and
throwing interceptions are repeatable. EPA charges the full swing of a
recovery to whoever fell on the ball, so a team can look elite on pure
luck. We measure the luck component separately so it can be shown in its
own column and, optionally, regressed out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import expected_points as ep
from .plays import load_plays

# Garbage time: score margins beyond which the game is effectively over.
# Widest early (a 4th-quarter 22-point lead is safe; a 1st-quarter one is
# not), tightening as time runs out. Mirrors the convention used by most
# public efficiency models.
GARBAGE_TIME_MARGINS = {1: 43, 2: 37, 3: 27, 4: 22}

# Fumble recoveries are ~50/50 regardless of who forced them, so the
# recovery half of a fumble is luck. Interceptions are far more
# repeatable and are NOT treated as luck.
FUMBLE_RECOVERY_BASE_RATE = 0.5

MIN_PLAYS_FOR_RATING = 20


@dataclass(frozen=True, slots=True)
class TeamGameEPA:
    """One team's efficiency in one game."""

    game_id: str
    season: int
    week: int
    team: str
    opponent: str
    offense_epa_per_play: float
    defense_epa_per_play: float
    offense_success_rate: float
    defense_success_rate: float
    offense_plays: int
    defense_plays: int
    turnover_luck: float

    @property
    def net_epa_per_play(self) -> float:
        """Offense minus defense: the single-number summary."""
        return self.offense_epa_per_play - self.defense_epa_per_play


def is_garbage_time(frame: pd.DataFrame) -> pd.Series:
    """True where the game is already decided.

    Uses the absolute score margin against a period-dependent threshold.
    Overtime is never garbage time.
    """
    period = frame["period"].astype(float)
    margin = frame["score_margin"].abs()

    threshold = pd.Series(np.inf, index=frame.index)
    for period_number, limit in GARBAGE_TIME_MARGINS.items():
        threshold = threshold.mask(period == period_number, limit)

    return (margin > threshold) & (period <= 4)


def yards_gained(frame: pd.DataFrame) -> pd.Series:
    """Yards gained per play, derived from field position.

    There is no `yards_gained` column in this data -- only per-event
    columns (`rush_yds`, `reception_yds`, `completion_yds`) that are
    blank whenever that event did not occur. Stitching those together
    would miss sacks, scrambles, penalties and returns.

    Field position is the honest source: yards_to_goal before minus
    yards_to_goal after, within the same possession and game. A
    possession change or a score makes the difference meaningless, so
    those rows yield NaN and drop out of the success-rate average.
    """
    yards_to_goal = frame["yards_to_goal"].astype(float)
    next_ytg = yards_to_goal.shift(-1)

    same_game = frame["game_id"].shift(-1) == frame["game_id"]
    same_team = frame["team"].shift(-1) == frame["team"]
    same_half = frame["half"].shift(-1) == frame["half"]
    comparable = same_game & same_team & same_half

    gained = (yards_to_goal - next_ytg).where(comparable)

    # A touchdown gains exactly the remaining distance; there is no
    # following snap to difference against.
    if "is_touchdown" in frame.columns:
        scored = frame["is_touchdown"].astype(bool)
        gained = gained.mask(scored, yards_to_goal)

    return gained


def is_successful(frame: pd.DataFrame) -> pd.Series:
    """Standard success-rate definition.

    50% of needed yards on 1st down, 70% on 2nd, 100% on 3rd and 4th.
    Success rate stabilises faster than EPA -- it measures consistency
    where EPA measures explosiveness -- which matters a lot in September
    when every team has three games.
    """
    down = frame["down"].astype(float)
    distance = frame["distance"].astype(float).replace(0, np.nan)
    gained = yards_gained(frame)
    fraction = gained / distance

    required = pd.Series(1.0, index=frame.index)
    required = required.mask(down == 1, 0.5)
    required = required.mask(down == 2, 0.7)

    return (fraction >= required).fillna(False)


def _turnover_luck(frame: pd.DataFrame) -> float:
    """Expected-value swing from fumble recoveries that went our way.

    A team that recovers 4 of 4 loose balls got roughly two extra
    possessions the average team would not have. Positive means lucky.
    """
    if "is_fumble_recovered" not in frame.columns:
        return 0.0
    recovered = float(frame["is_fumble_recovered"].sum())
    forced = float(frame.get(
        "is_fumble_forced", pd.Series(dtype=float)).sum())
    total = recovered + forced
    if total == 0:
        return 0.0
    return recovered - total * FUMBLE_RECOVERY_BASE_RATE


def _side_metrics(side: pd.DataFrame) -> tuple[float, float, int]:
    """EPA per play, success rate, and play count for one side."""
    epa = side["epa"].dropna()
    if len(epa) == 0:
        return 0.0, 0.0, 0
    return (
        float(epa.mean()),
        float(is_successful(side).mean()),
        int(len(epa)),
    )


def aggregate_game(frame: pd.DataFrame) -> list[TeamGameEPA]:
    """Split one game's plays into per-team efficiency records."""
    records: list[TeamGameEPA] = []
    teams = [t for t in frame["team"].dropna().unique()]
    if len(teams) != 2:
        return records

    for team in teams:
        offense = frame[frame["team"] == team]
        defense = frame[frame["team"] != team]
        if offense.empty or defense.empty:
            continue

        offense_epa, offense_success, offense_plays = _side_metrics(offense)
        defense_epa, defense_success, defense_plays = _side_metrics(defense)
        if (offense_plays < MIN_PLAYS_FOR_RATING
                or defense_plays < MIN_PLAYS_FOR_RATING):
            continue

        records.append(TeamGameEPA(
            game_id=str(frame["game_id"].iloc[0]),
            season=int(frame["season"].iloc[0]),
            week=int(frame["week"].iloc[0]),
            team=str(team),
            opponent=str([t for t in teams if t != team][0]),
            offense_epa_per_play=offense_epa,
            defense_epa_per_play=defense_epa,
            offense_success_rate=offense_success,
            defense_success_rate=defense_success,
            offense_plays=offense_plays,
            defense_plays=defense_plays,
            turnover_luck=_turnover_luck(offense),
        ))

    return records


def season_epa(
    season: int,
    model: ep.ExpectedPointsModel,
    drop_garbage_time: bool = True,
) -> list[TeamGameEPA]:
    """Per-team, per-game EPA for a whole season."""
    plays = load_plays(season)
    plays = ep.add_epa(plays, model)

    if drop_garbage_time:
        plays = plays[~is_garbage_time(plays)]

    records: list[TeamGameEPA] = []
    for _, game in plays.groupby("game_id", sort=False):
        records.extend(aggregate_game(game))
    return records


def to_frame(records: list[TeamGameEPA]) -> pd.DataFrame:
    """Tabular view, convenient for inspection and joins."""
    return pd.DataFrame([
        {
            "game_id": r.game_id,
            "season": r.season,
            "week": r.week,
            "team": r.team,
            "opponent": r.opponent,
            "off_epa": r.offense_epa_per_play,
            "def_epa": r.defense_epa_per_play,
            "net_epa": r.net_epa_per_play,
            "off_sr": r.offense_success_rate,
            "def_sr": r.defense_success_rate,
            "off_plays": r.offense_plays,
            "def_plays": r.defense_plays,
            "turnover_luck": r.turnover_luck,
        }
        for r in records
    ])
