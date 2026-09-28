"""Split the FCS part of the gap: FBS-vs-FCS vs FCS-vs-FCS, by week."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from gap_anatomy import BENCH  # noqa: E402
from cfbrank.games import load_season  # noqa: E402

df = pd.read_csv(BENCH, dtype={"game_id": str})
kind = {}
for s in df.season.unique():
    for g in load_season(int(s)):
        n_fcs = (g.home_division == "fcs") + (g.away_division == "fcs")
        kind[str(g.game_id)] = ["FBS-FBS", "FBS-FCS", "FCS-FCS"][n_fcs]
df["kind"] = [kind[g] for g in df.game_id]
df["gap"] = abs(df.actual - df.ours) - abs(df.actual - df.line)
df["wk"] = pd.cut(df.week, [0, 5, 8, 30], labels=["4-5", "6-8", "9+"])
total = df.gap.sum()
out = df.groupby(["kind", "wk"], observed=True).agg(
    n=("gap", "size"), gap=("gap", "mean"),
    bias=("gap", lambda s: (df.actual[s.index] - df.ours[s.index]).mean()),
    line_bias=("gap", lambda s: (df.actual[s.index] - df.line[s.index]).mean()))
out["share"] = df.groupby(["kind", "wk"], observed=True).gap.sum() / total
print(out.round(2).to_string())
print("\nFCS-FCS with a line, by season:", df[df.kind == "FCS-FCS"]
      .groupby("season").size().to_dict())
