"""Is 3 points a realistic "THE SCRIPT IS IN" threshold?

Stress-tests the alert cut-off on the walk-forward sample:
  * how often it fires (per season, per week) -- an alarm on every game
    is noise, one a year is useless
  * whether "script closer to the result" holds SEASON BY SEASON, with a
    bootstrap interval, not just pooled
  * whether the closing line itself sides with the script when it fires
    -- the cleanest check: if Vegas' final number lands nearer the
    script than us, the script really was reading the market
  * the same for the "flip" alert (different winner)

    python tools/check_script_threshold.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from probe_market_rating import stacked_frame  # noqa: E402

THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
RNG = np.random.default_rng(7)


def boot_ci(x: np.ndarray, n: int = 2000) -> tuple[float, float]:
    if len(x) < 5:
        return (np.nan, np.nan)
    means = RNG.choice(x, size=(n, len(x))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main() -> None:
    df = stacked_frame()
    split = (df.script - df.ours).abs()
    saved = (df.actual - df.ours).abs() - (df.actual - df.script).abs()
    closer = saved > 0
    line_sides = (df.line - df.script).abs() < (df.line - df.ours).abs()
    n_weeks = df.groupby(["season", "week"]).ngroups
    n_seasons = df.season.nunique()
    print(f"{len(df)} games, {n_seasons} seasons, {n_weeks} weeks "
          f"(FBS+FCS; the Predict page is FBS-only)\n")

    print("thr  fires  /week  /season  script closer  pts saved [95% CI]      "
          "line sides w/ script  seasons script closer >50%")
    for t in THRESHOLDS:
        m = split >= t
        lo, hi = boot_ci(saved[m].to_numpy())
        per_season = df[m].assign(c=closer[m]).groupby("season").c.mean()
        print(f"{t:>3}  {m.sum():>5}  {m.sum() / n_weeks:>5.1f}  "
              f"{m.sum() / n_seasons:>7.0f}  {closer[m].mean():>12.0%}  "
              f"{saved[m].mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  "
              f"{line_sides[m].mean():>18.0%}  "
              f"{(per_season > .5).sum()}/{n_seasons}  "
              f"({', '.join(f'{v:.0%}' for v in per_season)})")

    fbs = (df.home_division == "fbs") & (df.away_division == "fbs") \
        if "home_division" in df else None
    if fbs is not None:
        print(f"\nFBS-vs-FBS only (what the Predict page shows): "
              f"{(fbs & (split >= 3)).sum() / n_seasons:.0f}/season at 3+")

    print("\nAt 3+, by part of the season")
    print("  part         fire rate  pts saved  line sides w/ script")
    part = np.select([df.week <= 5, df.week <= 8, df.week <= 12],
                     ["wk 4-5", "wk 6-8", "wk 9-12"], "wk 13+/post")
    for name in ("wk 4-5", "wk 6-8", "wk 9-12", "wk 13+/post"):
        p = part == name
        m = p & (split >= 3).to_numpy()
        print(f"  {name:<12} {m.sum() / p.sum():>9.0%}  {saved[m].mean():>+9.2f}  "
              f"{line_sides[m].mean():>20.0%}")

    flip = np.sign(df.script) != np.sign(df.ours)
    right = (df.script > 0) == (df.actual > 0)
    line_right_side = np.sign(df.line) == np.sign(df.script)
    print("\nFLIP (different winner)")
    print("thr  games  /season  script's winner won [95% CI]  "
          "closing line agrees w/ script")
    for t in (0.0, 1.0, 2.0, 3.0, 4.0):
        m = flip & (split >= t)
        lo, hi = boot_ci(right[m].astype(float).to_numpy())
        print(f"{t:>3}  {m.sum():>5}  {m.sum() / n_seasons:>7.0f}  "
              f"{right[m].mean():>18.0%} [{lo:.0%}, {hi:.0%}]  "
              f"{line_right_side[m].mean():>18.0%}")


if __name__ == "__main__":
    main()
