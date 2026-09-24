"""Are rosters complete enough to measure returning production?

For each FBS team, what share of last season's offensive yards came from
players found on ANY roster this season? Seniors leave, so this should
sit around 50-75%. A team near 0% has a broken roster, not a rebuild.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.returning import player_production, load_roster  # noqa: E402
from cfbrank.games import load_season, team_divisions  # noqa: E402

for season in range(2015, 2027):
    prev = player_production(season - 1)
    roster = load_roster(season)
    fbs = {t for t, d in team_divisions(load_season(season - 1)).items()
           if d == "fbs"}
    on_any = prev.player_id.isin(set(roster.athlete_id))
    prev = prev.assign(back=on_any)
    by_team = prev[prev.team.isin(fbs)].groupby("team").apply(
        lambda f: (f.off_yds * f.back).sum() / max(f.off_yds.sum(), 1),
        include_groups=False)
    size = roster[roster.team.isin(fbs)].groupby("team").size()
    print(f"{season}: FBS roster median {int(size.median()):>3} "
          f"min {int(size.min()):>3} | returning-anywhere median "
          f"{by_team.median():.0%}, teams under 20%: "
          f"{int((by_team < 0.2).sum())}")
