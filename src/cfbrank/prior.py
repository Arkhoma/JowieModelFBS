"""Preseason prior -- what we knew before a snap was played.

MODEL_SPEC.md line 156 says it plainly: "Without a prior, early-season
ridge output is noise." This module is that prior.

The problem it solves: through week 3 there are ~260 games across 228
teams, three apiece. Plain ridge shrinks every unknown team toward ZERO
(a league-average team), so three lucky wins against nobody can float a
team to #1 while a one-loss blueblood sits at #40. The regression is not
wrong -- it genuinely has no evidence. It just has no memory either.

So give it one. Instead of shrinking toward zero, shrink toward last
season's rating, mean-reverted. Teams regress hard year over year
(graduation, portal, coaching), so last year is a weak prior, not a
verdict -- and the strength of that pull is MEASURED here rather than
chosen, by asking how well prior-season ratings actually predicted the
next season's opening weeks.

The decay is free. The penalty stays fixed while evidence accumulates,
so the prior's influence falls away on its own as games pile up. No
hand-built "decays by week 6" schedule -- the algebra already does it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .games import load_season, team_divisions
from .ridge import curve_margin, fit

# Weeks that count as "early" when measuring carryover. Past this point
# the current season speaks for itself.
EARLY_WEEKS = 4


def measure_carryover(
    seasons,
    max_week: int = EARLY_WEEKS,
    lambda_: float = 5.0,
    margin_scale: float = 28.0,
) -> dict:
    """Fit how much of a rating survives the offseason.

    Regresses early-season margins on the previous season's rating edge:

        curve(margin) = rho * (prior_home - prior_away) + hfa

    `rho` is the answer -- 1.0 would mean teams are identical year to
    year, 0.0 that last season tells us nothing. Reality is in between,
    and we let least squares say where.

    Measured on FBS-vs-FBS games only, where the rating scale needs no
    division offset to be comparable.
    """
    rows: list[list[float]] = []
    targets: list[float] = []

    for season in seasons:
        try:
            previous_games = load_season(season - 1)
            current_games = load_season(season)
        except FileNotFoundError:
            continue

        model = fit(
            previous_games, lambda_=lambda_,
            margin_scale=margin_scale, halflife=1e6,
        )

        for game in current_games:
            if game.week > max_week or not game.is_fbs_only:
                continue
            if (game.home_team not in model.ratings
                    or game.away_team not in model.ratings):
                continue
            edge = (model.ratings[game.home_team]
                    - model.ratings[game.away_team])
            rows.append([edge, 0.0 if game.neutral_site else 1.0])
            targets.append(
                float(curve_margin(np.array([game.margin]), margin_scale)[0]))

    if not rows:
        return {"rho": 0.0, "home_field": 0.0, "n_games": 0}

    design = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)

    predicted = design @ coefficients
    residuals = target - predicted
    total = target - target.mean()
    r_squared = 1.0 - float(residuals @ residuals) / float(total @ total)

    return {
        "rho": float(coefficients[0]),
        "home_field": float(coefficients[1]),
        "n_games": len(target),
        "r_squared": r_squared,
    }


# True year-over-year carryover, measured directly: regress each season's
# final ratings on the previous season's and take the slope. Across ten
# transitions (2015->2016 .. 2024->2025) it averages 0.63 and never
# leaves the 0.49-0.74 band. A team keeps roughly 63% of its edge; 37%
# is gone by kickoff to graduation, the portal, and coaching turnover.
#
# Reproduce with: python tools/carryover_reality.py
DEFAULT_CARRYOVER = 0.63

# Ridge deliberately shrinks ratings toward the mean -- that is the
# penalty doing its job, and it is correct when rating a season in
# progress. But it leaves fitted ratings UNDER-dispersed relative to
# true team strength, so they must be expanded before being reused as a
# prior for the next season.
#
# Measured INDEPENDENTLY of carryover, by split-half reliability within
# each season: fit two disjoint random halves of one season's games and
# compare across-half covariance (true signal) to within-half variance
# (signal plus noise). Mean 1.117 across 2015-2025, range 1.086-1.147.
#
# An earlier version defined this as 1/carryover, which made the two
# cancel to exactly 1.0 -- circular by construction, and no better than
# the rho=1.073 it was meant to replace. Two parameters that cancel are
# one parameter wearing a disguise.
#
# Reproduce with: python tools/measure_shrinkage.py
DEFAULT_UNSHRINK = 1.117


def measure_persistence(
    seasons, lambda_: float = 5.0, margin_scale: float = 28.0
) -> dict:
    """Regress each season's final ratings on the previous season's.

    The slope is the honest carryover: 1.0 would mean nothing changes,
    0.5 that half the edge evaporates. Unlike measure_carryover(), this
    compares like with like -- final ratings against final ratings -- so
    no shrinkage correction gets tangled into the answer.
    """
    ratings: dict[int, dict[str, float]] = {}
    for season in seasons:
        try:
            model = fit(load_season(season), lambda_=lambda_,
                        margin_scale=margin_scale, halflife=1e6)
        except (FileNotFoundError, ValueError):
            continue
        ratings[season] = model.fbs_ratings()

    slopes: list[float] = []
    for season in sorted(ratings):
        if season + 1 not in ratings:
            continue
        shared = sorted(set(ratings[season]) & set(ratings[season + 1]))
        if len(shared) < 50:
            continue
        previous = np.array([ratings[season][t] for t in shared])
        current = np.array([ratings[season + 1][t] for t in shared])
        centred_previous = previous - previous.mean()
        variance = float((centred_previous ** 2).sum())
        if variance == 0.0:
            continue
        covariance = float(
            (centred_previous * (current - current.mean())).sum())
        slopes.append(covariance / variance)

    if not slopes:
        return {"carryover": DEFAULT_CARRYOVER, "n_transitions": 0}
    return {
        "carryover": float(np.mean(slopes)),
        "n_transitions": len(slopes),
        "min": float(np.min(slopes)),
        "max": float(np.max(slopes)),
    }


def build_prior(
    season: int,
    carryover: float = DEFAULT_CARRYOVER,
    lambda_: float = 5.0,
    margin_scale: float = 28.0,
    unshrink: float = DEFAULT_UNSHRINK,
    current_divisions: dict[str, str] | None = None,
) -> dict[str, float]:
    """Prior ratings for `season`, taken from the season before it.

    Two steps, deliberately kept separate so each can be verified:

      1. `unshrink` expands last season's ridge-shrunk ratings back
         toward their true spread.
      2. `carryover` then applies the real offseason decay.

    Ratings are supplied on the WITHIN-DIVISION scale, translated to
    whichever division each team occupies NOW. Two successive bugs came
    from getting this wrong.

    Passing raw within-division values breaks teams that MOVE: North
    Dakota State rated +15.14 within FCS in 2025 and was promoted to
    FBS in 2026, so that figure read as an elite FBS rating and put them
    #2 in the country.

    But converting everything to the common scale is also wrong. The
    ridge design already fits a column for the FBS/FCS gap, so a
    common-scale prior counts that gap twice -- which collapsed the
    fitted offset from -13.96 to -3.40.

    Correct treatment: a team that stays put keeps its within-division
    rating; a team that moves carries its quality across, re-expressed
    relative to its new peers. The offset column keeps owning the gap.

    Teams with no prior season -- genuine newcomers -- are simply
    absent, so they shrink toward zero (league average). An unknown team
    should be treated as average, not punished.
    """
    previous = load_season(season - 1)
    model = fit(
        previous, lambda_=lambda_, margin_scale=margin_scale, halflife=1e6)
    scale = carryover * unshrink

    if current_divisions is None:
        try:
            current_divisions = team_divisions(load_season(season))
        except FileNotFoundError:
            current_divisions = {}

    built: dict[str, float] = {}
    for team, rating in model.ratings.items():
        was = model.divisions.get(team)
        now = current_divisions.get(team, was)
        if was == "fcs" and now == "fbs":
            # Promoted. The offset is negative, so this correctly turns
            # an FCS powerhouse into a below-average FBS team.
            rating = rating + model.fcs_offset
        elif was == "fbs" and now == "fcs":
            rating = rating - model.fcs_offset
        built[team] = scale * rating

    return built


def build_prior_or_empty(
    season: int, carryover: float = DEFAULT_CARRYOVER, **kwargs
) -> dict[str, float]:
    """Prior if the previous season is on disk, empty dict otherwise.

    The earliest season we hold has nothing before it, and that is an
    expected absence rather than an error.
    """
    try:
        return build_prior(season, carryover, **kwargs)
    except FileNotFoundError:
        return {}


# Home-field prior: shrink the early-season home-field estimate toward
# the average full-season value of the previous five seasons. Strength is
# in equivalent games. Walk-forward MAE is flat across 50-200 (all within
# 0.002 of each other, tools/tune_home_field_prior.py), so accuracy does
# not pick the value; the middle of the flat region was taken. What it
# does fix is the week-4 estimate: 3.08 -> 2.71 against a true
# full-season value of ~2.2, which matters for site-adjusted resume
# credit even though it barely moves MAE.
HOME_FIELD_PRIOR_STRENGTH = 100.0
HOME_FIELD_LOOKBACK = 5


@lru_cache(maxsize=None)
def home_field_prior(
    season: int, lambda_: float = 5.0, margin_scale: float = 28.0
) -> tuple[float, float] | None:
    """(centre, strength) from earlier seasons only, or None if none."""
    values = []
    for past in range(season - HOME_FIELD_LOOKBACK, season):
        try:
            values.append(fit(load_season(past), lambda_=lambda_,
                              margin_scale=margin_scale,
                              halflife=1e6).home_field)
        except (FileNotFoundError, ValueError):
            continue
    if not values:
        return None
    return float(np.mean(values)), HOME_FIELD_PRIOR_STRENGTH
