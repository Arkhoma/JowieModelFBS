"""Which CFBD team names fail to match our schedule team names?

Usage: python tools/check_cfbd_names.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.cfbd_priors import load_records  # noqa: E402
from cfbrank.games import load_season, team_divisions  # noqa: E402


def main() -> None:
    for season in (2016, 2021, 2025):
        fbs = {t for t, d in team_divisions(load_season(season)).items()
               if d == "fbs"}
        for name, key in (("talent", "team"),
                          ("returning_production", "team"),
                          ("recruiting_teams", "team")):
            theirs = {r[key] for r in load_records(season, name)}
            missing = sorted(fbs - theirs)
            print(f"{season} {name:<22} FBS unmatched: {len(missing)} {missing}")


if __name__ == "__main__":
    main()
