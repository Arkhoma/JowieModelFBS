"""Score the production ensemble against the closing betting line.

Howie's challenge: "we are not beating ESPN FPI or SP+". Before agreeing
or arguing, measure on the SAME games. Published SP+/FPI error figures
use their own game sets, start weeks and FCS handling, so comparing a
headline MAE against ours is apples to oranges.

The closing line is the right yardstick: it is the sharpest public
predictor, and historically it beats SP+ and FPI in most seasons. If we
are within a few tenths of the market, we are in SP+/FPI territory.

Line source: cfbrank.lines (sportsdataverse mirror, CFBD fills gaps).
ESPN game ids, same as our schedules. We use the home side and prefer the
'consensus' book, falling back to the median across books.

Also reports:
  ATS    how often our model picks the right side of the line -- 52.4%
         is the break-even at standard -110 juice
  stack  regress actual margin on (ours, line): if our coefficient is
         meaningfully > 0, the model carries information the market
         has not priced in
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

# MIN_TRAINING and START_WEEK are unused here now, but check_ranking_order.py
# imports them from this module, so they stay re-exported.
from blend_epa_margin import (  # noqa: E402, F401
    MARGIN_PARAMS, MIN_TRAINING, RELATIVE_LAMBDA, START_WEEK,
)
from cfbrank import season_record  # noqa: E402
from cfbrank.epa_ridge import get_ep_model  # noqa: E402
from cfbrank.games import POSTSEASON  # noqa: E402
from cfbrank.lines import home_lines, total_lines  # noqa: E402
from cfbrank.scorecard import (  # noqa: E402
    EARLY_WEEK_CUTOFF, TIERS, grade, grade_totals,
)

# The web app's accuracy view reads this file -- rerun this tool after any
# model change so the site reports the model it actually serves.
OUTPUT_PATH = ROOT / "data" / "benchmark_vs_market.csv"
SEASONS = (2021, 2022, 2023, 2024, 2025)
BREAK_EVEN_ATS = 0.524

# Line loading lives in cfbrank.lines now (the app needs it for the Vegas
# Script). These names are kept so the experiment tools keep working.
load_home_lines = home_lines
load_total_lines = total_lines


def walk_season(season: int, market: dict[str, float],
                totals: dict[str, float]) -> pd.DataFrame:
    """Walk-forward PRODUCTION predictions joined to the closing lines.

    The logic lives in cfbrank.season_record (the web app grades the live
    season with it too). This wrapper passes THIS module's MARGIN_PARAMS and
    RELATIVE_LAMBDA at call time, because run_variant.py and
    tune_prior_strength.py re-assign them to test a variant.
    """
    return season_record.walk_season(
        season, market, totals,
        margin_params=MARGIN_PARAMS, relative_lambda=RELATIVE_LAMBDA)


def summarise(frame: pd.DataFrame, label: str) -> None:
    g = grade(frame)
    print(f"{label:>7} {g['n']:>5} {g['mae']:>7.2f} {g['line_mae']:>7.2f} "
          f"{g['gap']:>+6.2f} +/-{g['gap_ci']:.2f} {g['su']:>6.1%} "
          f"{g['line_su']:>6.1%} {g['ats']:>6.1%} {g['brier']:>6.3f}")


def main() -> None:
    print("Loading lines and fitting expected-points model ...")
    market, totals = load_home_lines(), load_total_lines()
    get_ep_model()

    frames = [walk_season(s, market, totals) for s in SEASONS]
    frames = [f for f in frames if not f.empty]
    everything = pd.concat(frames, ignore_index=True)

    print(f"\n{'season':>7} {'n':>5} {'ourMAE':>7} {'lineMAE':>7} "
          f"{'gap':>6} {'95%CI':>6} {'ourSU':>6} {'lineSU':>6} {'ATS':>6} "
          f"{'Brier':>6}")
    print("-" * 79)
    for frame in frames:
        summarise(frame, str(frame.season.iloc[0]))
    print("-" * 79)
    summarise(everything, "pooled")

    early = everything.week <= EARLY_WEEK_CUTOFF
    post = everything.season_type == POSTSEASON
    print()
    summarise(everything[early], f"wk<={EARLY_WEEK_CUTOFF}")
    summarise(everything[~early & ~post], f"wk>{EARLY_WEEK_CUTOFF}")
    summarise(everything[post], "post")
    for tier in TIERS:
        rows = everything[everything.tier == tier]
        if len(rows) > 1:
            summarise(rows, tier[:7])

    t = grade_totals(everything)
    print(f"\nTotals ({t['n']} games): ours {t['mae']:.2f}, "
          f"line {t['line_mae']:.2f}, league average {t['naive_mae']:.2f}")

    stacked = np.column_stack([everything.ours, everything.line])
    coef, *_ = np.linalg.lstsq(stacked, everything.actual, rcond=None)
    print(f"\nStack (actual ~ ours + line): ours {coef[0]:+.3f}, "
          f"line {coef[1]:+.3f}")
    print(f"ATS break-even at -110 juice: {BREAK_EVEN_ATS:.1%}")

    everything.to_csv(OUTPUT_PATH, index=False)
    print(f"Per-game detail written to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
