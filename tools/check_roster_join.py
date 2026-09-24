"""Check that play-data player ids and team names join to rosters."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
plays = pd.read_parquet(ROOT / "data/raw/plays/plays_2023.parquet")
roster23 = pd.read_parquet(ROOT / "data/raw/rosters/rosters_2023.parquet")
roster24 = pd.read_parquet(ROOT / "data/raw/rosters/rosters_2024.parquet")

ids = set()
for col in ("rush_player_id", "reception_player_id", "completion_player_id"):
    ids |= set(plays[col].dropna().astype("int64"))
roster_ids = set(roster23.athlete_id.astype("int64"))
print(f"play ids {len(ids)}, in 2023 roster {len(ids & roster_ids)} "
      f"({len(ids & roster_ids) / len(ids):.1%})")

play_teams = set(plays.team.dropna())
roster_teams = set(roster23.team.dropna())
print(f"play teams {len(play_teams)}, missing from roster: "
      f"{len(play_teams - roster_teams)}")
print(sorted(play_teams - roster_teams)[:25])

sched = pd.read_csv(ROOT / "data/raw/schedules/schedules_2023.csv")
fbs = set(sched[sched.home_division == "fbs"].home_team)
print(f"FBS teams missing from roster: {sorted(fbs - roster_teams)}")
print(f"FBS teams missing from plays: {sorted(fbs - play_teams)}")
print(roster24.groupby("team").size().describe())
print(roster24.year.value_counts().sort_index())
