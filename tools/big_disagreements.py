"""What do the games where we disagree with Vegas by 7+ points have in common?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from gap_anatomy import BENCH, attach_game_info  # noqa: E402
from cfbrank.games import load_season  # noqa: E402

df = attach_game_info(pd.read_csv(BENCH, dtype={"game_id": str}))
names = {}
for s in df.season.unique():
    for g in load_season(int(s)):
        names[str(g.game_id)] = (g.home_team, g.away_team)
df["home"] = [names[g][0] for g in df.game_id]
df["away"] = [names[g][1] for g in df.game_id]
df["diff"] = df.ours - df.line
big = df[df["diff"].abs() >= 7]

print("Who we're wrong about (7+ disagreements), team appearances:")
teams = pd.concat([big.home, big.away]).value_counts()
print(teams.head(20).to_string())
print(f"\nshare in week 4-6: {(big.week <= 6).mean():.0%} vs all games "
      f"{(df.week <= 6).mean():.0%}")
print(f"share involving FCS: {big.fcs.mean():.0%} vs {df.fcs.mean():.0%}")
print(f"we are closer than Vegas in {(abs(big.actual - big.ours) < abs(big.actual - big.line)).mean():.0%}")

print("\nSIGNED: mean actual by our predicted margin (nonlinearity check)")
b = pd.cut(df.ours, [-99, -28, -21, -14, -7, 0, 7, 14, 21, 28, 35, 99])
print(df.groupby(b, observed=True).agg(
    n=("actual", "size"), ours=("ours", "mean"), actual=("actual", "mean"),
    line=("line", "mean")).round(1).to_string())

print("\nWeek 4-5 by game type: gap and bias")
early = df[df.week <= 5]
kind = np.select([early.fcs, early.neutral, early.conf_game],
                 ["vs FCS", "neutral", "conference"], "non-conf FBS")
early = early.assign(kind=kind, gap=abs(early.actual - early.ours)
                     - abs(early.actual - early.line))
print(early.groupby("kind").agg(n=("gap", "size"), gap=("gap", "mean"),
      bias=("ours", lambda s: (early.actual[s.index] - s).mean())).round(2))

print("\nWorst 25:")
cols = ["season", "week", "home", "away", "ours", "line", "actual"]
print(big.reindex(big["diff"].abs().sort_values(ascending=False).index)[cols]
      .head(25).round(1).to_string(index=False))
