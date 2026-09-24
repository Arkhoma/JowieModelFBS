"""Load and clean play-level data into an EPA-ready frame.

The raw `player_stats` parquet files are one row per play with full
down / distance / field-position state, 2014-2026. This module turns them
into something the expected-points model can trust.

Three upstream quirks this handles, all verified on 2024 data:

1. `play_id` is UNRELIABLE for ordering. Some rows carry negative ids
   (e.g. -19467), evidently an integer overflow on the 20-digit values.
   33 of 1,596 games in 2024 are affected. Sorting on it scrambles play
   order and drops period-1 plays at the end of the game. We sort by
   actual game state instead: period ascending, clock descending.
2. Player columns use EMPTY STRINGS, not NaN, for "did not happen".
   `sack_player.isna()` reports 0% null, so a naive filter marks all
   1.69M plays as sacks.
3. Overtime scoring is not reflected in `team_score` / `opponent_score`.
   ~5% of games disagree with the schedule file's final score. Play data
   is reliable for in-game STATE, not for final outcomes -- so final
   scores always come from the schedule file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLAYS_DIR = PROJECT_ROOT / "data" / "raw" / "plays"

REGULATION_PERIODS = 4
PERIOD_SECONDS = 15 * 60

# Columns that name a player when an event occurred, blank otherwise.
EVENT_PLAYER_COLUMNS = (
    "touchdown_player",
    "field_goal_made_player",
    "field_goal_missed_player",
    "field_goal_blocked_player",
    "interception_player",
    "fumble_recovered_player",
    "fumble_forced_player",
    "sack_player",
    "sack_taken_player",
    "pass_breakup_player",
    "completion_player",
    "incompletion_player",
    "rush_player",
    "reception_player",
)


def _event_occurred(series: pd.Series) -> pd.Series:
    """True where an event column names a real player.

    Guards against quirk #2: blanks are empty strings, not NaN.
    """
    return series.notna() & (series.astype(str).str.strip() != "")


def load_plays(season: int, directory: Path = PLAYS_DIR) -> pd.DataFrame:
    """Load one season of plays, sorted into true chronological order."""
    path = directory / f"plays_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No play file for {season} at {path}. Run tools/fetch_mirror.py"
        )

    frame = pd.read_parquet(path)

    frame["game_id"] = frame["game_id"].astype(str)

    # Quirk #1: play_id is corrupt in ~2% of games (negative overflow
    # values), so order by real game state rather than by id. Clock counts
    # DOWN within a period, hence the descending sort on time remaining.
    frame["period"] = pd.to_numeric(frame["period"], errors="coerce")
    frame["clock_minutes"] = pd.to_numeric(frame["clock_minutes"], errors="coerce")
    frame["clock_seconds"] = pd.to_numeric(frame["clock_seconds"], errors="coerce")
    frame = frame.dropna(subset=["period", "clock_minutes", "clock_seconds"])

    frame["_period_clock"] = (frame["clock_minutes"] * 60
                              + frame["clock_seconds"])
    frame = frame.sort_values(
        ["game_id", "period", "_period_clock"],
        ascending=[True, True, False],
        kind="mergesort",  # stable: preserves file order for identical state
    ).drop(columns="_period_clock").reset_index(drop=True)

    return _add_derived_columns(frame)


def _add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the state columns the EPA model needs."""
    frame = frame.copy()

    # Boolean event flags, safe against the empty-string quirk.
    for column in EVENT_PLAYER_COLUMNS:
        if column in frame.columns:
            flag = column.replace("_player", "")
            frame[f"is_{flag}"] = _event_occurred(frame[column])

    # Seconds remaining in regulation. Overtime periods clamp to zero.
    period = frame["period"].astype(float)
    clock = (frame["clock_minutes"].astype(float) * 60
             + frame["clock_seconds"].astype(float))
    periods_left = (REGULATION_PERIODS - period).clip(lower=0)
    frame["seconds_remaining"] = (periods_left * PERIOD_SECONDS + clock).clip(lower=0)
    frame["is_overtime"] = period > REGULATION_PERIODS

    # Half matters for EPA: the next scoring event cannot cross halftime.
    frame["half"] = np.where(period <= 2, 1, 2)
    frame["half"] = np.where(frame["is_overtime"], 3, frame["half"])

    # Score margin from the perspective of the team with the ball.
    frame["score_margin"] = (frame["team_score"].astype(float)
                             - frame["opponent_score"].astype(float))

    # Possession changes drive the "next score" labeling.
    frame["possession_change"] = (
        frame["team"] != frame.groupby("game_id")["team"].shift()
    )

    return frame


def load_seasons(seasons, directory: Path = PLAYS_DIR) -> pd.DataFrame:
    """Concatenate several seasons of plays."""
    frames = [load_plays(season, directory) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def available_seasons(directory: Path = PLAYS_DIR) -> list[int]:
    seasons = []
    for path in directory.glob("plays_*.parquet"):
        try:
            seasons.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(seasons)
