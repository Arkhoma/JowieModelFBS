"""Team-specific preseason prior, replacing the flat 0.63 carryover.

The old prior treated every team identically: last season x 0.63 x the
unshrink factor. This one predicts each team's coming season from:

  r1, r2, r3        final ratings one, two and three seasons back. Older
                    seasons capture program level -- recruiting, facilities,
                    coaching depth -- which is what SP+ gets from recruiting
                    rankings and we cannot download.
  r1 x ret_off      last season's edge, scaled by how much of the offence
  r1 x ret_def      and defence came back. A great team that lost everyone
  r1 x ret_qb       should regress hard; one that returned everyone should
                    not.
  ret_* , portal    level effects: returning experience and incoming
                    transfers help regardless of last year's rating.
  talent            247 team talent composite (CFBD), per 100 points --
                    the recruiting stars of this season's actual roster.
  new_coach,        a head coach who did not return for week 1, alone and
  r1 x new_coach    scaled by last season's rating (a new coach inherits
                    less of the old edge). tools/screen_priors.py: these
                    two beat recruiting classes, CFBD returning PPA and
                    portal ratings, which added nothing held out.

Target: next season's final ridge rating. Output is multiplied by the
same measured UNSHRINK as before, because the target itself is shrunk.

No leakage: the weights used for season S are fit only on offseasons
whose target season is before S.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import cfbd_priors
from .games import load_season, team_divisions
from .prior import DEFAULT_UNSHRINK, build_prior
from .returning import returning_production
from .ridge import fit

FIT_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
FEATURES = ("r1", "r2", "r3", "r1_ret_off", "r1_ret_def", "r1_ret_qb",
            "ret_off", "ret_def", "ret_qb", "portal",
            "talent", "new_coach", "r1_new_coach")
TALENT_UNIT = 100.0
FIRST_TARGET_SEASON = 2016
RIDGE = 1.0   # light, just to keep correlated history terms stable


@lru_cache(maxsize=None)
def _final(season: int) -> tuple[dict[str, float], dict[str, str]]:
    try:
        model = fit(load_season(season), **FIT_PARAMS)
    except (FileNotFoundError, ValueError):
        return {}, {}
    return dict(model.ratings), dict(model.divisions)


def _centred(values: dict[str, float], teams) -> dict[str, float]:
    """Values minus their mean over `teams`; missing teams get 0."""
    known = [values[t] for t in teams if t in values]
    mean = float(np.mean(known)) if known else 0.0
    return {t: values[t] - mean for t in teams if t in values}


def _features(season: int) -> dict[str, np.ndarray]:
    """Feature rows for teams that were FBS last season and still are."""
    r1, div1 = _final(season - 1)
    r2, _ = _final(season - 2)
    r3, _ = _final(season - 3)
    try:
        now = team_divisions(load_season(season))
    except FileNotFoundError:
        now = {}
    try:
        returning = returning_production(season)
    except FileNotFoundError:
        returning = {}

    ret = list(returning.values())
    means = {k: float(np.mean([getattr(x, k) for x in ret])) if ret else 0.0
             for k in ("ret_off", "ret_def", "ret_qb", "portal")}

    rows: dict[str, np.ndarray] = {}
    eligible = [t for t in r1
                if div1.get(t) == "fbs" and now.get(t, "fbs") == "fbs"]
    talent = _centred({t: v / TALENT_UNIT for t, v in
                       cfbd_priors.talent(season).items()}, eligible)
    coach = _centred(cfbd_priors.new_coach(season), eligible)
    for team in eligible:
        rating = r1[team]
        back2 = r2.get(team, rating)
        back3 = r3.get(team, back2)
        info = returning.get(team)
        centred = {k: (getattr(info, k) - means[k]) if info else 0.0
                   for k in means}
        rows[team] = np.array([
            rating, back2, back3,
            rating * centred["ret_off"], rating * centred["ret_def"],
            rating * centred["ret_qb"],
            centred["ret_off"], centred["ret_def"], centred["ret_qb"],
            centred["portal"],
            talent.get(team, 0.0),
            coach.get(team, 0.0), rating * coach.get(team, 0.0),
        ])
    return rows


@lru_cache(maxsize=None)
def _training_rows(season: int) -> tuple[np.ndarray, np.ndarray]:
    features = _features(season)
    target, _ = _final(season)
    teams = [t for t in features if t in target]
    if not teams:
        return np.empty((0, len(FEATURES))), np.empty(0)
    return (np.array([features[t] for t in teams]),
            np.array([target[t] for t in teams]))


@lru_cache(maxsize=None)
def fit_weights(before_season: int) -> np.ndarray:
    """Weights learned from every offseason whose target is < season."""
    xs, ys = [], []
    for target_season in range(FIRST_TARGET_SEASON, before_season):
        x, y = _training_rows(target_season)
        if len(y):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"No training offseasons before {before_season}")
    x, y = np.vstack(xs), np.concatenate(ys)
    gram = x.T @ x + RIDGE * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def build_roster_prior(season: int) -> dict[str, float]:
    """Prior for `season`: the roster model where it applies, the flat
    carryover prior everywhere else (FCS teams, division movers,
    newcomers) so nothing is silently dropped."""
    fallback = build_prior(season)
    try:
        weights = fit_weights(season)
    except ValueError:
        return fallback
    built = dict(fallback)
    for team, row in _features(season).items():
        built[team] = DEFAULT_UNSHRINK * float(row @ weights)
    return built


def build_roster_prior_or_empty(season: int) -> dict[str, float]:
    try:
        return build_roster_prior(season)
    except FileNotFoundError:
        return {}
