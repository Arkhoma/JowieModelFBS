"""Can FCS teams get a better preseason prior than flat carryover?

FBS teams get the roster model (prior_model.py). FCS teams fall back to
prior.build_prior: 0.63 x 1.117 x last season, same for everyone. That
slice (FCS-vs-FCS, weeks 4-5) is ~22% of the gap to the closing line.

Screen, leave-one-season-out, target = final within-FCS ridge rating:
  flat     : 0.63 * 1.117 * r1           (production)
  history  : regression on r1, r2, r3
  +ret     : plus returning production from our roster files, if any

RESULT 2026-09-28: screen looked good (RMSE 4.17 -> 4.07), but wired into
prior_model and run through the full walk-forward benchmark it was noise:
gap -0.002 +/-0.019 (wk<=8 -0.016, wk>8 +0.010). Not shipped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season, team_divisions  # noqa: E402
from cfbrank.prior import DEFAULT_CARRYOVER  # noqa: E402
from cfbrank.prior_model import _final  # noqa: E402
from cfbrank.returning import returning_production  # noqa: E402

SEASONS = range(2016, 2026)


def table() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        target, div = _final(season)
        r1, d1 = _final(season - 1)
        r2, _ = _final(season - 2)
        r3, _ = _final(season - 3)
        try:
            ret = returning_production(season)
        except FileNotFoundError:
            ret = {}
        for team, y in target.items():
            if div.get(team) != "fcs" or d1.get(team) != "fcs":
                continue
            a = r1[team]
            b = r2.get(team, a)
            c = r3.get(team, b)
            info = ret.get(team)
            rows.append(dict(season=season, team=team, y=y, r1=a, r2=b, r3=c,
                             ret_off=getattr(info, "ret_off", np.nan),
                             ret_def=getattr(info, "ret_def", np.nan),
                             ret_qb=getattr(info, "ret_qb", np.nan)))
    df = pd.DataFrame(rows)
    for k in ("ret_off", "ret_def", "ret_qb"):
        df[k] = (df[k] - df.groupby("season")[k].transform("mean")).fillna(0)
        df[f"r1_{k}"] = df.r1 * df[k]
    return df


def loso(df, cols):
    pred = np.zeros(len(df))
    for s in SEASONS:
        tr, te = df.season != s, df.season == s
        X = df.loc[tr, cols].to_numpy()
        w = np.linalg.solve(X.T @ X + np.eye(len(cols)), X.T @ df.y[tr])
        pred[te.to_numpy()] = df.loc[te, cols].to_numpy() @ w
    return pred


def main():
    df = table()
    print(f"{len(df)} FCS team-seasons; returning data present for "
          f"{(df.ret_off != 0).mean():.0%}")
    # The target is itself ridge-shrunk, so the flat prior in target units
    # is carryover * r1 (the unshrink factor is applied after, in prod).
    flat_t = DEFAULT_CARRYOVER * df.r1
    sets = {"history": ["r1", "r2", "r3"],
            "+ret": ["r1", "r2", "r3", "ret_off", "ret_def", "ret_qb",
                     "r1_ret_off", "r1_ret_def", "r1_ret_qb"]}
    rmse = lambda p: float(np.sqrt(((df.y - p) ** 2).mean()))
    print(f"  flat carryover     RMSE {rmse(flat_t):.3f}")
    for name, cols in sets.items():
        p = loso(df, cols)
        w = np.linalg.solve(df[cols].T @ df[cols] + np.eye(len(cols)), df[cols].T @ df.y)
        print(f"  {name:18s} RMSE {rmse(p):.3f}   weights {np.round(w, 3)}")
    # Same thing on FBS for scale
    print("(FBS roster model improvement over flat was ~0.1-0.2 RMSE for reference)")


if __name__ == "__main__":
    main()
