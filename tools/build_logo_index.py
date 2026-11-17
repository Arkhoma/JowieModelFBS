"""Write data/team_logos.json: team name -> CollegeFootballData team id.

    python tools/build_logo_index.py

The logo itself is never stored. CFBD serves every size at a predictable
URL (see cfbrank.logos), so all we need per team is its numeric id. The
index is committed because data/raw/ is not: the weekly GitHub build only
has teams_all.json.gz when a CFBD key is configured, and a site that
quietly loses its logos for want of one file would be a bad surprise.

Re-run after a team joins FBS (needs data/raw/teams_all.json.gz, from
`python tools/fetch_cfbd.py --only meta` or a full pull).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "teams_all.json.gz"
OUT = ROOT / "data" / "team_logos.json"


def build_index(teams: list[dict]) -> dict[str, int]:
    """{school: id} for every team CFBD has a logo for (first record wins)."""
    ids: dict[str, int] = {}
    for team in teams:
        if team.get("logos") and team.get("school") and team["school"] not in ids:
            ids[team["school"]] = int(team["id"])
    return dict(sorted(ids.items()))


def main() -> None:
    if not SOURCE.exists():
        sys.exit(f"{SOURCE} not found. Fetch it first (tools/fetch_cfbd.py).")
    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        ids = build_index(json.load(handle))
    OUT.write_text(json.dumps({"source": "CFBD /teams", "ids": ids},
                              indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(ids)} teams)")


if __name__ == "__main__":
    main()
