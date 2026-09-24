"""Audit the raw schedule files before we trust them.

Cheap insurance: every modeling bug I have ever chased started with an
assumption about the data that nobody checked. Run this after fetching.

    python tools/audit_data.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "schedules"

NULLISH = {"", "NA", "NULL", "None"}


def is_blank(value: str | None) -> bool:
    return value is None or value.strip() in NULLISH


def audit_season(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fbs_rows = [
        r for r in rows
        if r.get("home_division") == "fbs" or r.get("away_division") == "fbs"
    ]
    both_fbs = [
        r for r in fbs_rows
        if r.get("home_division") == "fbs" and r.get("away_division") == "fbs"
    ]
    played = [r for r in fbs_rows if not is_blank(r.get("home_points"))]

    ties = [
        r for r in played
        if r.get("home_points") == r.get("away_points")
    ]
    missing_week = [r for r in fbs_rows if is_blank(r.get("week"))]
    neutral = [r for r in played if str(r.get("neutral_site")).upper() == "TRUE"]

    teams = {r["home_team"] for r in both_fbs} | {r["away_team"] for r in both_fbs}

    return {
        "season": path.stem.replace("schedules_", ""),
        "all_rows": len(rows),
        "fbs_involved": len(fbs_rows),
        "fbs_v_fbs": len(both_fbs),
        "played": len(played),
        "fbs_teams": len(teams),
        "ties": len(ties),
        "neutral": len(neutral),
        "missing_week": len(missing_week),
    }


def main() -> None:
    paths = sorted(RAW_DIR.glob("schedules_*.csv"))
    if not paths:
        raise SystemExit(f"No schedule files in {RAW_DIR}. Run fetch_mirror.py first.")

    header = (f"{'season':>7} {'rows':>7} {'fbs_inv':>8} {'fbs_v_fbs':>10} "
              f"{'played':>7} {'teams':>6} {'ties':>5} {'neutral':>8} {'no_wk':>6}")
    print(header)
    print("-" * len(header))

    problems = []
    for path in paths:
        stats = audit_season(path)
        print(f"{stats['season']:>7} {stats['all_rows']:>7} {stats['fbs_involved']:>8} "
              f"{stats['fbs_v_fbs']:>10} {stats['played']:>7} {stats['fbs_teams']:>6} "
              f"{stats['ties']:>5} {stats['neutral']:>8} {stats['missing_week']:>6}")
        if stats["ties"]:
            problems.append(f"{stats['season']}: {stats['ties']} tie(s) -- "
                            "overtime should make these impossible post-1995")
        if stats["missing_week"]:
            problems.append(f"{stats['season']}: {stats['missing_week']} rows "
                            "missing a week number")

    print("\nNotes:")
    print("  fbs_inv   = games with at least one FBS team (includes FCS buy games)")
    print("  fbs_v_fbs = both teams FBS (the ridge connectivity graph)")

    if problems:
        print("\nPOTENTIAL ISSUES:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nNo anomalies detected.")

    # Team-name consistency across seasons is a classic silent killer:
    # "Miami (FL)" vs "Miami" would split one team into two ratings.
    latest = paths[-1]
    with latest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    conferences = Counter(
        r["home_conference"] for r in rows
        if r.get("home_division") == "fbs" and not is_blank(r.get("home_conference"))
    )
    print(f"\nConferences in {latest.stem.replace('schedules_', '')}: "
          f"{len(conferences)}")
    for conference, count in conferences.most_common():
        print(f"  {conference:<24} {count:>4} home games")


if __name__ == "__main__":
    main()
