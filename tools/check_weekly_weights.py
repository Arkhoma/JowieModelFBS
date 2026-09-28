"""Should the margin / success-rate blend weights change with the week?

Early in the season success rate has had fewer games to stabilise than
margin -- or maybe more, since per-play stats settle faster. The shipped
weights are one pair for all weeks. Using the cached walk-forward
components (data/improvement_components.csv, roster-prior versions),
fit weights per week bucket leave-one-season-out and compare to one
global pair fit the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "improvement_components.csv")
df = df.dropna(subset=["m_roster", "s_roster"])
BUCKETS = pd.cut(df.week, [0, 5, 6, 8, 11, 99],
                 labels=["4-5", "6", "7-8", "9-11", "12+"]).astype(str)
COLS = ["m_roster", "s_roster"]


def loso(groups: pd.Series | None) -> np.ndarray:
    pred = np.zeros(len(df))
    for season in df.season.unique():
        test = (df.season == season).to_numpy()
        keys = [None] if groups is None else groups.unique()
        for k in keys:
            g = np.ones(len(df), bool) if k is None else (groups == k).to_numpy()
            tr, te = ~test & g, test & g
            w, *_ = np.linalg.lstsq(df.loc[tr, COLS], df.actual[tr], rcond=None)
            pred[te] = df.loc[te, COLS] @ w
    return pred


base = loso(None)
week = loso(BUCKETS)
err_b, err_w = np.abs(df.actual - base), np.abs(df.actual - week)
d = err_w - err_b
print(f"{len(df)} games. global-weight MAE {err_b.mean():.3f}, "
      f"per-week {err_w.mean():.3f}, delta {d.mean():+.3f} "
      f"+/-{1.96 * d.std() / np.sqrt(len(d)):.3f}")
for k in ["4-5", "6", "7-8", "9-11", "12+"]:
    m = BUCKETS == k
    w, *_ = np.linalg.lstsq(df.loc[m, COLS], df.actual[m], rcond=None)
    print(f"  wk {k:5s} n={m.sum():4d} weights m={w[0]:.3f} s={w[1]:.3f}  "
          f"delta {d[m].mean():+.3f}")
