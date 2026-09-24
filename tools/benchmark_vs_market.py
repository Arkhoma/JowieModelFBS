"""Score the production ensemble against the closing betting line.

Howie's challenge: "we are not beating ESPN FPI or SP+". Before agreeing
or arguing, measure on the SAME games. Published SP+/FPI error figures
use their own game sets, start weeks and FCS handling, so comparing a
headline MAE against ours is apples to oranges.

The closing line is the right yardstick: it is the sharpest public
predictor, and historically it beats SP+ and FPI in most seasons. If we
are within a few tenths of the market, we are in SP+/FPI territory.

Line source: sportsdataverse cfbfastR-data betting/csv/cfb_line_odds.csv.gz
(ESPN game ids, same as our schedules). A spread line is quoted per team;
negative means favoured. We use the home side and prefer the 'consensus'
book, falling back to the median across books.

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

from blend_epa_margin import (  # noqa: E402
    MARGIN_PARAMS, MIN_TRAINING, RELATIVE_LAMBDA, START_WEEK,
)
from cfbrank.ensemble import EFFICIENCY_METRIC, EnsembleModel  # noqa: E402
from cfbrank.epa_ridge import (  # noqa: E402
    build_epa_games, fit_epa_ratings_primed, get_ep_model,
)
from cfbrank.games import POSTSEASON, load_season  # noqa: E402
from cfbrank.offdef_prior import fit_points_model  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.prior import home_field_prior  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit as fit_margin  # noqa: E402
from cfbrank.scorecard import (  # noqa: E402
    EARLY_WEEK_CUTOFF, TIERS, grade, grade_totals, postseason_tier,
)

LINES_PATH = ROOT / "data" / "raw" / "betting" / "cfb_line_odds.csv.gz"
# The web app's accuracy view reads this file -- rerun this tool after any
# model change so the site reports the model it actually serves.
OUTPUT_PATH = ROOT / "data" / "benchmark_vs_market.csv"
SEASONS = (2021, 2022, 2023, 2024, 2025)
BREAK_EVEN_ATS = 0.524


def _consensus_or_median(lines: pd.DataFrame) -> pd.Series:
    """game_id -> the 'consensus' book's line, else the median of books."""
    lines = lines.assign(game_id=lines.game_id.astype("int64").astype(str))
    consensus = lines[lines.book == "consensus"].groupby("game_id").lines.first()
    return consensus.combine_first(lines.groupby("game_id").lines.median())


def _market(kind: str) -> pd.DataFrame:
    lines = pd.read_csv(LINES_PATH)
    return lines[(lines.market_type == kind) & lines.lines.notna()
                 & lines.game_id.notna()]


def load_home_lines() -> dict[str, float]:
    """game_id -> market's predicted HOME margin (i.e. minus the spread)."""
    lines = _market("spread")
    lines = lines[lines.abbr == lines.game_desc.str.split("@").str[1]]
    return (-_consensus_or_median(lines)).to_dict()


def load_total_lines() -> dict[str, float]:
    """game_id -> market's over/under (combined points)."""
    lines = _market("total")
    return _consensus_or_median(lines[lines.abbr == "over"]).to_dict()


def walk_season(season: int, market: dict[str, float],
                totals: dict[str, float]) -> pd.DataFrame:
    """Walk-forward PRODUCTION predictions joined to the closing lines.

    Goes through Predictor exactly as the app does, so the margin, total
    and win probability graded here are the ones the site would have shown.
    """
    games = sorted(load_season(season), key=lambda g: g.week)
    epa_by_id = {g.game_id: g for g in build_epa_games(
        season, games, get_ep_model(), metric=EFFICIENCY_METRIC)}
    prior = build_roster_prior_or_empty(season) or None
    hfa_prior = home_field_prior(season)
    rows = []

    for week in sorted({g.week for g in games}):
        if week < START_WEEK:
            continue
        history = [g for g in games if g.week < week]
        upcoming = [g for g in games if g.week == week
                    and str(g.game_id) in market]
        if len(history) < MIN_TRAINING or not upcoming:
            continue
        epa_history = [epa_by_id[str(g.game_id)] for g in history
                       if str(g.game_id) in epa_by_id]
        if len(epa_history) < MIN_TRAINING:
            continue

        model = EnsembleModel(
            margin_model=fit_margin(history, prior=prior,
                                    home_field_prior=hfa_prior,
                                    **MARGIN_PARAMS),
            epa_model=fit_epa_ratings_primed(
                epa_history, prior, relative_lambda=RELATIVE_LAMBDA),
        )
        predictor = Predictor.from_model(
            model, history,
            total_model=fit_points_model(season, history, hfa_prior))
        for game in upcoming:
            if not (model.knows(game.home_team)
                    and model.knows(game.away_team)):
                continue
            p = predictor.predict(
                game.home_team, game.away_team, game.neutral_site)
            gid = str(game.game_id)
            rows.append({
                "season": season,
                "week": week,
                "season_type": game.season_type,
                "tier": (postseason_tier(game.notes)
                         if game.season_type == POSTSEASON else ""),
                "game_id": gid,
                "ours": p.predicted_margin,
                "line": market[gid],
                "actual": float(game.margin),
                "home_win_prob": p.home_win_probability,
                "ours_total": p.total,
                "total_line": totals.get(gid),
                "actual_total": float(game.home_points + game.away_points),
            })
    return pd.DataFrame(rows)


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
