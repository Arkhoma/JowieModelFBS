"""Where does the gap to the closing line come from?

Slices data/benchmark_vs_market.csv (walk-forward production predictions)
every way that could point at a fix, and reports each slice's SHARE of
the total gap -- a big gap on 30 games matters less than a small gap on
3,000.

Also measures the two cheapest structural fixes directly:
  * a linear recalibration (ours * a + b) fit leave-one-season-out,
    in case the whole model is systematically over/under-confident;
  * the same per week bucket, in case early-season predictions need
    a different stretch than late ones.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402

BENCH = ROOT / "data" / "benchmark_vs_market.csv"


def attach_game_info(df: pd.DataFrame) -> pd.DataFrame:
    info = {}
    for season in sorted(df.season.unique()):
        for g in load_season(int(season)):
            info[str(g.game_id)] = (g.neutral_site, g.is_fbs_only,
                                    g.involves_fcs, g.conference_game)
    cols = ["neutral", "fbs_only", "fcs", "conf_game"]
    extra = pd.DataFrame([info.get(gid, (None,) * 4) for gid in df.game_id],
                         columns=cols, index=df.index)
    return pd.concat([df, extra], axis=1)


def slice_table(df: pd.DataFrame, key: pd.Series, title: str) -> None:
    total_gap = df.gap.sum()
    print(f"\n{title}")
    print(f"  {'slice':22s} {'n':>5} {'gap/g':>7} {'+/-':>5} "
          f"{'share':>6} {'bias ours':>9} {'bias line':>9}")
    for name, g in df.groupby(key, observed=True):
        ci = 1.96 * g.gap.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan
        print(f"  {str(name):22s} {len(g):>5} {g.gap.mean():>+7.3f} "
              f"{ci:>5.2f} {g.gap.sum() / total_gap:>6.0%} "
              f"{(g.actual - g.ours).mean():>+9.2f} "
              f"{(g.actual - g.line).mean():>+9.2f}")


def loso_recalibration(df: pd.DataFrame, groups: pd.Series | None) -> float:
    """MAE gain from ours*a+b, fit on other seasons (per group if given)."""
    gain = 0.0
    for season in df.season.unique():
        train, test = df[df.season != season], df[df.season == season]
        keys = groups.unique() if groups is not None else [None]
        for k in keys:
            tr = train if k is None else train[groups[train.index] == k]
            te = test if k is None else test[groups[test.index] == k]
            if len(tr) < 50 or te.empty:
                continue
            X = np.column_stack([tr.ours, np.ones(len(tr))])
            (a, b), *_ = np.linalg.lstsq(X, tr.actual, rcond=None)
            gain += (np.abs(te.actual - te.ours)
                     - np.abs(te.actual - (a * te.ours + b))).sum()
    return gain / len(df)


def main() -> None:
    df = pd.read_csv(BENCH, dtype={"game_id": str})
    df = attach_game_info(df)
    df["gap"] = np.abs(df.actual - df.ours) - np.abs(df.actual - df.line)
    print(f"{len(df)} games, gap {df.gap.mean():+.3f}/game")

    week = pd.cut(df.week, [0, 5, 6, 8, 10, 12, 30],
                  labels=["wk4-5", "wk6", "wk7-8", "wk9-10", "wk11-12", "wk13+"])
    week = week.astype(str).where(df.season_type != "postseason", "post")
    slice_table(df, week, "BY WEEK")

    kind = np.select([df.fcs, df.neutral, df.conf_game],
                     ["vs FCS", "neutral", "conference"], "non-conf FBS")
    slice_table(df, pd.Series(kind, index=df.index), "BY GAME TYPE")

    spread = pd.cut(df.line.abs(), [-1, 3, 7, 14, 21, 99],
                    labels=["0-3", "3.5-7", "7.5-14", "14.5-21", "21+"])
    slice_table(df, spread, "BY SIZE OF VEGAS SPREAD")

    disagree = pd.cut((df.ours - df.line).abs(), [-1, 2, 4, 7, 10, 99],
                      labels=["<2", "2-4", "4-7", "7-10", "10+"])
    slice_table(df, disagree, "BY HOW FAR WE DISAGREE WITH VEGAS")

    # Direction of disagreement: do we overrate favourites or underdogs?
    side = np.where(np.sign(df.ours - df.line) == np.sign(df.line),
                    "we like the fav MORE", "we like the dog MORE")
    slice_table(df, pd.Series(side, index=df.index), "WHICH WAY WE DISAGREE")

    print("\nRECALIBRATION (ours*a+b), leave-one-season-out MAE gain:")
    print(f"  one global line       {loso_recalibration(df, None):+.3f}")
    print(f"  per week bucket       {loso_recalibration(df, week):+.3f}")
    X = np.column_stack([df.ours, np.ones(len(df))])
    (a, b), *_ = np.linalg.lstsq(X, df.actual, rcond=None)
    print(f"  pooled fit: actual = {a:.3f} * ours {b:+.2f}")
    for name, g in df.groupby(week):
        (a, b), *_ = np.linalg.lstsq(
            np.column_stack([g.ours, np.ones(len(g))]), g.actual, rcond=None)
        (la, lb), *_ = np.linalg.lstsq(
            np.column_stack([g.line, np.ones(len(g))]), g.actual, rcond=None)
        print(f"    {name:8s} ours slope {a:.3f} int {b:+.2f}   "
              f"line slope {la:.3f} int {lb:+.2f}")


if __name__ == "__main__":
    main()
