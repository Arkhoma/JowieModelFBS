"""How well do roster recruit_ids join to CFBD recruiting records?

Usage: python tools/check_recruit_join.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
FIRST, LAST = 2014, 2026


def load(year: int, name: str) -> pd.DataFrame:
    path = RAW / str(year) / f"{name}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))


def main() -> None:
    recruits = pd.concat([load(y, "recruits") for y in range(FIRST, LAST + 1)])
    print("recruit columns:", recruits.columns.tolist())
    print("sample:", recruits.iloc[0].to_dict(), "\n")
    known = set(recruits["id"].astype(str))
    athletes = set(recruits["athleteId"].dropna().astype(str))

    for year in (2016, 2019, 2022, 2025):
        roster = pd.read_parquet(RAW / "rosters" / f"rosters_{year}.parquet")
        fbs = set(load(year, "talent")["team"]) if year > FIRST else None
        if fbs:
            roster = roster[roster["team"].isin(fbs)]
        ids = roster["recruit_ids"].dropna().explode().dropna().astype(str)
        by_recruit = ids[ids.isin(known)].index.nunique()
        by_athlete = roster["athlete_id"].astype(str).isin(athletes).sum()
        print(f"{year}: {len(roster):>6} talent-team players | recruit_id "
              f"match {by_recruit / len(roster):.0%} | athlete_id match "
              f"{by_athlete / len(roster):.0%}")

    portal = load(2025, "portal")
    print("\nportal columns:", portal.columns.tolist())
    print(f"portal 2025: rating present {portal['rating'].notna().mean():.0%}, "
          f"destination present {portal['destination'].notna().mean():.0%}")


if __name__ == "__main__":
    main()
