"""Audit play-level data across all seasons before trusting it.

Checks the three things that would silently wreck an EPA model:
  1. schema drift between seasons
  2. nulls in fields EPA depends on
  3. game coverage versus the schedule files
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PLAYS_DIR = ROOT / "data" / "raw" / "plays"
SCHEDULES_DIR = ROOT / "data" / "raw" / "schedules"

EPA_CRITICAL = ["down", "distance", "yards_to_goal", "period", "team", "opponent"]


def scheduled_game_ids(season: int) -> set[str]:
    path = SCHEDULES_DIR / f"schedules_{season}.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            row["game_id"]
            for row in csv.DictReader(handle)
            if row.get("home_points") not in ("", "NA", None)
            and "fbs" in (row.get("home_division"), row.get("away_division"))
        }


def main() -> None:
    paths = sorted(PLAYS_DIR.glob("plays_*.parquet"))
    if not paths:
        raise SystemExit(f"No play files in {PLAYS_DIR}. Run tools/fetch_mirror.py")

    header = (f"{'season':>7} {'plays':>9} {'games':>7} {'teams':>6} "
              f"{'pl/gm':>6} {'sched':>7} {'cover':>7} {'maxwk':>6}")
    print(header)
    print("-" * len(header))

    schemas: dict[int, tuple] = {}
    issues: list[str] = []

    for path in paths:
        season = int(path.stem.split("_")[-1])
        frame = pd.read_parquet(path)
        schemas[season] = tuple(frame.columns)

        games = frame["game_id"].nunique()
        plays = len(frame)
        teams = frame["team"].nunique()
        scheduled = scheduled_game_ids(season)
        play_ids = set(frame["game_id"].astype(str))
        covered = len(play_ids & scheduled) / len(scheduled) if scheduled else 0.0

        print(f"{season:>7} {plays:>9,} {games:>7} {teams:>6} "
              f"{plays / games:>6.1f} {len(scheduled):>7} "
              f"{covered:>6.1%} {int(frame['week'].max()):>6}")

        for column in EPA_CRITICAL:
            null_rate = frame[column].isna().mean()
            if null_rate > 0.01:
                issues.append(f"{season}: {column} is {null_rate:.2%} null")

        if plays / games < 100:
            issues.append(f"{season}: only {plays / games:.0f} plays/game "
                          "-- expected ~120-140, data may be partial")
        if scheduled and covered < 0.90:
            issues.append(f"{season}: only {covered:.1%} of scheduled FBS "
                          "games have play data")

    # Schema drift check -- a column that vanishes mid-history means any
    # feature built on it silently becomes NaN for those years.
    reference_season = max(schemas)
    reference = set(schemas[reference_season])
    for season, columns in sorted(schemas.items()):
        missing = reference - set(columns)
        extra = set(columns) - reference
        if missing or extra:
            issues.append(
                f"{season}: schema differs from {reference_season} "
                f"(missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]})"
            )

    print(f"\nColumns: {len(reference)} (reference season {reference_season})")

    if issues:
        print("\nISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues. Schema is stable across all seasons and EPA "
              "fields are populated.")


if __name__ == "__main__":
    main()
