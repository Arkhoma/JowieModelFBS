"""Which preseason signals predict a team's final rating?

A cheap screen before the expensive walk-forward benchmark. For every
offseason 2016-2025, each FBS team gets the current prior features plus
the CFBD candidates; the target is that season's final ridge rating.
Every feature set is scored LEAVE-ONE-SEASON-OUT, so no season is
predicted with weights that saw it.

Usage: python tools/screen_priors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank import cfbd_priors  # noqa: E402
from cfbrank.prior_model import FEATURES, _features, _final  # noqa: E402

SEASONS = range(2016, 2026)
RIDGE = 1.0
CANDIDATES = ("talent", "recruiting", "portal_net", "cfbd_ret", "new_coach")


def build_table() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        target, _ = _final(season)
        extra = {name: getattr(cfbd_priors, name if name != "cfbd_ret"
                               else "cfbd_returning")(season)
                 for name in CANDIDATES}
        for team, base in _features(season).items():
            if team not in target:
                continue
            row = dict(zip(FEATURES, base))
            row.update(season=season, team=team, y=target[team])
            for name, values in extra.items():
                row[name] = values.get(team, np.nan)
            rows.append(row)
    table = pd.DataFrame(rows)
    # Centre each candidate within its season (scales drift year to year),
    # then fill gaps with the season average, i.e. zero.
    for name in CANDIDATES:
        by_season = table.groupby("season")[name]
        table[name] = (table[name] - by_season.transform("mean")).fillna(0.0)
    table["r1_cfbd_ret"] = table.r1 * table.cfbd_ret
    table["r1_new_coach"] = table.r1 * table.new_coach
    return table


def loso_rmse(table: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    """Held-out RMSE overall and for 2021+ (the benchmark seasons)."""
    pred = np.zeros(len(table))
    for season in SEASONS:
        train, test = table.season != season, table.season == season
        x, y = table.loc[train, cols].values, table.loc[train, "y"].values
        mu, sd = x.mean(0), x.std(0) + 1e-9
        xs = (x - mu) / sd
        w = np.linalg.solve(xs.T @ xs + RIDGE * np.eye(len(cols)),
                            xs.T @ (y - y.mean()))
        pred[test.values] = ((table.loc[test, cols].values - mu) / sd) @ w \
            + y.mean()
    err = (table.y - pred) ** 2
    late = table.season >= 2021
    return float(np.sqrt(err.mean())), float(np.sqrt(err[late].mean()))


def main() -> None:
    table = build_table()
    base = list(FEATURES)
    sets = {
        "current prior": base,
        "+ talent": base + ["talent"],
        "+ recruiting": base + ["recruiting"],
        "+ portal_net": base + ["portal_net"],
        "+ cfbd_ret": base + ["cfbd_ret", "r1_cfbd_ret"],
        "+ new_coach": base + ["new_coach", "r1_new_coach"],
        "+ talent + recruiting": base + ["talent", "recruiting"],
        "+ talent + coach": base + ["talent", "new_coach",
                                     "r1_new_coach"],
        "+ tal + rec + coach": base + ["talent", "recruiting",
                                        "new_coach", "r1_new_coach"],
        "+ all": base + ["talent", "recruiting", "portal_net", "cfbd_ret",
                         "r1_cfbd_ret", "new_coach", "r1_new_coach"],
    }
    print(f"{len(table)} team-seasons. Held-out RMSE of final rating "
          f"(points; lower is better)\n")
    print(f"{'features':<24} {'all':>7} {'2021+':>7} {'vs cur':>7}")
    ref = None
    for name, cols in sets.items():
        overall, late = loso_rmse(table, cols)
        ref = ref if ref is not None else late
        print(f"{name:<24} {overall:>7.3f} {late:>7.3f} {late - ref:>+7.3f}")
    print("\nCorrelation with residual of the current prior is in the "
          "weights; see the benchmark for the real scoreboard.")


if __name__ == "__main__":
    main()
