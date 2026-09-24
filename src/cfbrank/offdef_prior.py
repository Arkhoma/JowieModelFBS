"""Preseason prior for offense and defense SEPARATELY.

This is the main reason a split can out-predict the net rating. The
roster prior (prior_model.py) feeds returning offense, returning
defense and portal yards into ONE number, so losing your quarterback
drags down your defense's prior too. Here each side gets its own
regression, fit on past offseasons only:

    next off  ~  off r1/r2/r3, r1_off x ret_off/ret_qb, ret_*, portal,
    next def  ~  def r1/r2/r3, r1_def x ret_def, ret_*, portal

Both sides see every level feature; least squares decides what matters
(we do not assume a defense ignores its offense's returning production).

Teams that changed division, and newcomers, get no prior and shrink to
average -- deliberately simple; they are a handful of teams a season.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import offdef
from .epa_games import season_epa
from .epa_ridge import get_ep_model
from .games import load_season, team_divisions
from .returning import returning_production

FIRST_TARGET_SEASON = 2016
RIDGE = 1.0
# Penalty for the in-season points model that feeds predicted totals and
# the Off/Def columns. Walk-forward 2021-2025 (tools/offdef_experiment.py):
# total-points MAE 13.80 / 13.50 / 13.19 / 12.94 / 12.95 at lambda
# 0.5 / 1 / 2 / 5 / 12 -- flat from 5 up, so 5.
TOTALS_LAMBDA = 5.0
# Ridge penalty for the full-season fits used as prior inputs and
# targets, in GAMES' WORTH of evidence. Deliberately NOT scaled by the
# target's variance (as epa_ridge.scaled_lambda does): a ridge solution
# is equivariant to rescaling y, so shrinkage depends on lambda relative
# to X'X (games played), never on the units of the statistic.
FINAL_LAMBDA = 2.0
RETURNING_KEYS = ("ret_off", "ret_def", "ret_qb", "portal")


@lru_cache(maxsize=None)
def _season_records(season: int):
    """Per-team game efficiency; ~13s a season, so computed once."""
    return tuple(season_epa(season, get_ep_model()))


def season_sides(season: int, kind: str) -> list[offdef.SideGame]:
    """Every side observation for a season, points or success rate."""
    games = load_season(season)
    if kind == "points":
        return offdef.sides_from_games(games)
    return offdef.sides_from_efficiency(
        games, list(_season_records(season)), kind)


@lru_cache(maxsize=None)
def final_ratings(season: int, kind: str) -> offdef.OffDefResult | None:
    """Full-season off/def ratings, or None if the season is missing."""
    try:
        sides = season_sides(season, kind)
    except FileNotFoundError:
        return None
    if not sides:
        return None
    return offdef.fit(sides, FINAL_LAMBDA, is_points=(kind == "points"))


def _returning(season: int) -> dict[str, dict[str, float]]:
    """Returning features, centred on the league mean; {} if unavailable."""
    try:
        raw = returning_production(season)
    except FileNotFoundError:
        return {}
    means = {k: float(np.mean([getattr(r, k) for r in raw.values()]))
             for k in RETURNING_KEYS}
    return {team: {k: getattr(r, k) - means[k] for k in RETURNING_KEYS}
            for team, r in raw.items()}


def _history(season: int, kind: str, lag: int):
    model = final_ratings(season - lag, kind)
    if model is None:
        return {}, {}, {}
    return model.offense, model.defense, model.divisions


def features(season: int, kind: str) -> dict[str, dict[str, np.ndarray]]:
    """{team: {"off": row, "def": row}} for FBS->FBS teams."""
    off1, def1, div1 = _history(season, kind, 1)
    off2, def2, _ = _history(season, kind, 2)
    off3, def3, _ = _history(season, kind, 3)
    try:
        now = team_divisions(load_season(season))
    except FileNotFoundError:
        now = {}
    returning = _returning(season)
    zero = dict.fromkeys(RETURNING_KEYS, 0.0)

    rows: dict[str, dict[str, np.ndarray]] = {}
    for team in off1:
        if div1.get(team) != "fbs" or now.get(team, "fbs") != "fbs":
            continue
        r = returning.get(team, zero)
        levels = [r["ret_off"], r["ret_def"], r["ret_qb"], r["portal"]]
        o1, d1 = off1[team], def1[team]
        o2, d2 = off2.get(team, o1), def2.get(team, d1)
        o3, d3 = off3.get(team, o2), def3.get(team, d2)
        rows[team] = {
            "off": np.array([o1, o2, o3, o1 * r["ret_off"],
                             o1 * r["ret_qb"], *levels]),
            "def": np.array([d1, d2, d3, d1 * r["ret_def"], *levels]),
        }
    return rows


@lru_cache(maxsize=None)
def fit_weights(before_season: int, kind: str) -> dict[str, np.ndarray]:
    """Per-side weights from every offseason whose target is < season."""
    xs = {"off": [], "def": []}
    ys = {"off": [], "def": []}
    for target in range(FIRST_TARGET_SEASON, before_season):
        truth = final_ratings(target, kind)
        if truth is None:
            continue
        side_truth = {"off": truth.offense, "def": truth.defense}
        for team, row in features(target, kind).items():
            if team not in truth.offense:
                continue
            for side in xs:
                xs[side].append(row[side])
                ys[side].append(side_truth[side][team])
    if not ys["off"]:
        raise ValueError(f"No training offseasons before {before_season}")
    weights = {}
    for side in xs:
        x, y = np.array(xs[side]), np.array(ys[side])
        weights[side] = np.linalg.solve(
            x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ y)
    return weights


def build_offdef_prior(
    season: int, kind: str, scale: float = 1.0
) -> tuple[dict[str, float], dict[str, float]] | None:
    """(offense_prior, defense_prior) for `season`, or None.

    `scale` multiplies both sides -- the analogue of the margin prior's
    unshrink factor, since the targets are themselves ridge-shrunk.
    """
    try:
        weights = fit_weights(season, kind)
    except ValueError:
        return None
    offense, defense = {}, {}
    for team, row in features(season, kind).items():
        offense[team] = scale * float(row["off"] @ weights["off"])
        defense[team] = scale * float(row["def"] @ weights["def"])
    return offense, defense


def fit_points_model(
    season: int, games, home_field_prior=None
) -> offdef.OffDefResult:
    """Production off/def points model for a season in progress."""
    return offdef.fit(offdef.sides_from_games(games), TOTALS_LAMBDA,
                      build_offdef_prior(season, "points"),
                      home_field_prior=home_field_prior)
