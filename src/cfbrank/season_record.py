"""How the production model has done: walk-forward picks, graded by week.

`walk_season` replays one season week by week through the production
Predictor, exactly as the app would have shown it: each week's games are
predicted from a model fit on EARLIER weeks only, then joined to the
closing line. Nothing about a game's own result leaks into its pick.

Two callers share it:
  * tools/benchmark_vs_market.py, which writes the multi-season CSV the
    About page reads (and which experiment tools re-parametrize), and
  * the This Week page, for the season in progress.

The tuning constants below must equal the app's (app.main.FITTED_PARAMS)
and the experiment tools' (tools/blend_epa_margin.py); tests/
test_season_record.py fails if they ever drift apart.
"""

from __future__ import annotations

import pandas as pd

from cfbrank.ensemble import EFFICIENCY_METRIC, EnsembleModel
from cfbrank.epa_ridge import (
    build_epa_games, fit_epa_ratings_primed, get_ep_model,
)
from cfbrank.games import POSTSEASON, load_season
from cfbrank.offdef_prior import fit_points_model
from cfbrank.predict import Predictor
from cfbrank.prior import home_field_prior
from cfbrank.prior_model import build_roster_prior_or_empty
from cfbrank.ridge import fit as fit_margin
from cfbrank.scorecard import EARLY_WEEK_CUTOFF, grade, postseason_tier

START_WEEK = 4            # earlier weeks are mostly preseason-prior guesses
MIN_TRAINING = 50         # games needed before a fit means anything
RELATIVE_LAMBDA = 4.0
MARGIN_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}

MIN_WEEK_GAMES = 5        # a week with fewer games isn't worth a row
MIN_LINED_GAMES = 10      # fewer lined games than this: a miss of a few
                          # points is one blowout, not a comparison


def walk_season(season: int, market: dict[str, float],
                totals: dict[str, float], *,
                margin_params: dict | None = None,
                relative_lambda: float | None = None,
                include_unlined: bool = False) -> pd.DataFrame:
    """Walk-forward PRODUCTION predictions joined to the closing lines.

    Goes through Predictor exactly as the app does, so the margin, total
    and win probability graded here are the ones the site would have shown.
    `margin_params` / `relative_lambda` default to production; experiment
    tools pass their own to test a variant.

    By default only games WITH a closing line are returned (what the
    benchmark grades). `include_unlined=True` also returns the rest with
    `line` = NaN: picking a winner needs no betting line, and a season's
    lines can be patchy.
    """
    margin_params = MARGIN_PARAMS if margin_params is None else margin_params
    relative_lambda = (RELATIVE_LAMBDA if relative_lambda is None
                       else relative_lambda)
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
                    and (include_unlined or str(g.game_id) in market)]
        if len(history) < MIN_TRAINING or not upcoming:
            continue
        epa_history = [epa_by_id[str(g.game_id)] for g in history
                       if str(g.game_id) in epa_by_id]
        if len(epa_history) < MIN_TRAINING:
            continue

        model = EnsembleModel(
            margin_model=fit_margin(history, prior=prior,
                                    home_field_prior=hfa_prior,
                                    **margin_params),
            epa_model=fit_epa_ratings_primed(
                epa_history, prior, relative_lambda=relative_lambda),
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
                "line": market.get(gid, float("nan")),
                "actual": float(game.margin),
                "home_win_prob": p.home_win_probability,
                "ours_total": p.total,
                "total_line": totals.get(gid),
                "actual_total": float(game.home_points + game.away_points),
            })
    return pd.DataFrame(rows)


def _summary(rows: pd.DataFrame, **label) -> dict:
    """Winner accuracy over every game; the line comparison only over the
    games that have a line (None when there are too few to mean anything)."""
    lined = rows[rows.line.notna()]
    return {
        **label,
        "n": int(len(rows)),
        "lined": int(len(lined)),
        "su": float(((rows.ours > 0) == (rows.actual > 0)).mean()),
        "vs_line": grade(lined) if len(lined) >= MIN_LINED_GAMES else None,
    }


def weekly_record(frame: pd.DataFrame) -> dict | None:
    """Overall and week-by-week grades for one season's walk-forward picks.

    Picking winners is graded on EVERY game; the comparison with the closing
    line (scorecard.grade, the About page's own metrics) only on games that
    have a line, because a season's lines can be patchy and grading just
    those would quietly overweight whichever weeks happen to have them.
    `overall["early_lined"]` is the share of lined games from early weeks,
    where our gap to the line is usually widest, so the page can say so.
    Postseason games are pooled into one row (their `week` numbers restart).
    Returns None with no games.
    """
    if frame.empty:
        return None
    post = frame.season_type == POSTSEASON
    regular = frame[~post]
    weeks = [_summary(rows, label=f"Week {week}", week=int(week))
             for week, rows in regular.groupby("week")
             if len(rows) >= MIN_WEEK_GAMES]
    if post.sum() >= MIN_WEEK_GAMES:
        weeks.append(_summary(frame[post], label="Postseason", week=None))
    lined_regular = regular[regular.line.notna()]
    return {
        "overall": {**_summary(frame), "early_lined": (
            float((lined_regular.week <= EARLY_WEEK_CUTOFF).mean())
            if len(lined_regular) else 0.0)},
        "weeks": weeks,
        "min_lined": MIN_LINED_GAMES,
        "through": int(regular.week.max()) if len(regular) else None,
    }
