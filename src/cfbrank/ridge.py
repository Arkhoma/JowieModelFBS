"""Opponent adjustment by ridge regression -- the core rating engine.

The idea: every game is one equation.

    curve(margin) = rating[home] - rating[away] + home_field + error

Stack every game into a big sparse system and solve for all ~265 team
ratings at once. Strength of schedule is never computed as a separate
weighted input -- it falls out of the algebra, because the only way to
explain "Team A beat Team B by 10" is to know how good Team B is, and
that value is itself being solved from B's own games.

No hand-tuned weights. Every free parameter here (ridge penalty, margin
curve shape, recency decay, home field) is fit by cross-validation
against out-of-sample prediction error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .games import Game

# Margin transform. Winning by 50 should not count much more than winning
# by 35: the extra points are noise against a broken opponent. A concave
# curve encodes diminishing returns without the information loss of a
# hard cap. Scale is fit; see fit_margin_scale().
DEFAULT_MARGIN_SCALE = 14.0

# Recency: a November result is stronger evidence than a September one.
# Half-life in weeks; fit by cross-validation.
DEFAULT_RECENCY_HALFLIFE = 8.0

# The FBS and FCS subgraphs are joined by only ~121 games a season, and
# 33 FCS teams never face an FBS opponent at all. Without an explicit
# division term the regression cannot tell that the FCS pool sits on a
# lower level -- it just sees North Dakota State going 14-2 by 19.5 a
# game and rates them above Texas.
#
# The fix is one more fitted parameter, not a hand-set penalty: a
# division offset estimated from those crossover games. It stays
# unpenalized so ridge shrinkage cannot bias it toward zero.


def curve_margin(margin: np.ndarray, scale: float = DEFAULT_MARGIN_SCALE) -> np.ndarray:
    """Concave transform of scoring margin.

    `scale * tanh(margin / scale)` is near-linear for close games and
    flattens for blowouts, so a 42-point win counts meaningfully more
    than a 21-point win but nowhere near twice as much.
    """
    return scale * np.tanh(np.asarray(margin, dtype=float) / scale)


def recency_weights(
    weeks: np.ndarray,
    current_week: float,
    halflife: float = DEFAULT_RECENCY_HALFLIFE,
) -> np.ndarray:
    """Exponential decay on game age, in weeks."""
    age = np.maximum(0.0, current_week - np.asarray(weeks, dtype=float))
    return 0.5 ** (age / halflife)


# Calibration slope shrinkage. The slope is fit IN SAMPLE on the same
# games the ratings came from, which inflates it when data is thin:
# measured at 2.37 through week 3 of 2024 but 1.82 by season's end, and
# the honest out-of-sample value at week 3 was 2.09. So the early
# estimate is mostly real but overstated by ~0.3.
#
# An inflated slope multiplies every predicted margin, making the model
# loudest exactly when it knows least. Shrink toward the long-run value
# with weight that grows as games accumulate -- by midseason the data
# dominates and the prior stops mattering.
#
# Reproduce with: python tools/check_calibration_stability.py
CALIBRATION_PRIOR = 1.85
CALIBRATION_PRIOR_GAMES = 400.0


def _shrink_calibration(raw: float, n_games: int) -> float:
    """Pull a thin-sample slope toward the long-run value.

    Equivalent to a weighted average of the fitted slope and the prior,
    where the prior carries the weight of CALIBRATION_PRIOR_GAMES games.
    """
    weight = n_games / (n_games + CALIBRATION_PRIOR_GAMES)
    return weight * raw + (1.0 - weight) * CALIBRATION_PRIOR


@dataclass(slots=True)
class RatingResult:
    """Fitted team ratings plus the parameters that produced them."""

    ratings: dict[str, float]
    home_field: float
    fcs_offset: float
    divisions: dict[str, str]
    lambda_: float
    margin_scale: float
    halflife: float
    n_games: int
    n_teams: int
    residual_std: float
    teams: list[str] = field(default_factory=list)
    # Ridge shrinks ratings toward the prior, so rating DIFFERENCES come
    # out systematically too small and predicted margins are too timid --
    # measured at slope 1.69 out of sample with no prior, i.e. roughly
    # 40% under-confident. Scaling predictions by the fitted slope fixes
    # that in one visible place.
    #
    # Without this, the compression leaks into whatever knob is free to
    # absorb it: the preseason prior was silently inflated to 1.07-1.2
    # (against a true carryover of 0.63) purely because over-weighting
    # last season happened to stretch the ratings back out.
    #
    # Reproduce with: python tools/check_calibration.py
    calibration: float = 1.0

    def rating(self, team: str) -> float:
        """Rating on the common scale, with the division offset applied.

        Stored ratings are relative to a team's own division pool; this
        adds the fitted FCS penalty so FBS and FCS teams are directly
        comparable.
        """
        base = self.ratings.get(team, 0.0)
        if self.divisions.get(team) == "fcs":
            return base + self.fcs_offset
        return base

    def fbs_ratings(self) -> dict[str, float]:
        """Only FBS teams, on the common scale. This is the published list."""
        return {
            team: self.rating(team)
            for team, division in self.divisions.items()
            if division == "fbs" and team in self.ratings
        }

    def predict_margin(
        self, home: str, away: str, neutral_site: bool = False
    ) -> float:
        """Expected home margin. This is the prediction engine."""
        edge = (self.rating(home) - self.rating(away)) * self.calibration
        return edge + (0.0 if neutral_site else self.home_field)

    def calibrated_rating(self, team: str) -> float:
        """Rating with the calibration slope baked in.

        `predict_margin` applies `calibration` to the rating DIFFERENCE,
        not to either team individually -- but since it is the same
        constant for every matchup, applying it here first gives the
        identity `calibrated_rating(home) - calibrated_rating(away) +
        home_field == predict_margin(home, away)`. The static export
        needs that identity to reproduce predictions client-side without
        re-deriving the ensemble math in JavaScript."""
        return self.rating(team) * self.calibration


def _build_design_matrix(
    games: list[Game],
    teams: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, neutral_flags) for the regression.

    X has one column per team, then home-field, then the FCS offset.
    Each team row is +1 for home, -1 for away. The FCS column is +1 when
    the home team is FCS and the away team is not, -1 for the reverse,
    and 0 when both sides share a division -- so it is identified purely
    by crossover games.
    """
    index = {team: position for position, team in enumerate(teams)}
    n_columns = len(teams) + 2

    design = np.zeros((len(games), n_columns), dtype=float)
    target = np.zeros(len(games), dtype=float)
    neutral = np.zeros(len(games), dtype=bool)

    for row, game in enumerate(games):
        design[row, index[game.home_team]] = 1.0
        design[row, index[game.away_team]] = -1.0
        if not game.neutral_site:
            design[row, -2] = 1.0

        home_is_fcs = game.home_division == "fcs"
        away_is_fcs = game.away_division == "fcs"
        if home_is_fcs != away_is_fcs:
            design[row, -1] = 1.0 if home_is_fcs else -1.0

        target[row] = game.margin
        neutral[row] = game.neutral_site

    return design, target, neutral


def fit(
    games: list[Game],
    lambda_: float = 40.0,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    halflife: float = DEFAULT_RECENCY_HALFLIFE,
    current_week: float | None = None,
    prior: dict[str, float] | None = None,
    home_field_prior: tuple[float, float] | None = None,
) -> RatingResult:
    """Solve for team ratings by weighted ridge regression.

    The ridge penalty is what keeps a 3-0 team from topping the table in
    September: with little evidence, ratings are pulled toward a default.
    As games accumulate the data overwhelms the penalty.

    That default is zero unless `prior` says otherwise. Shrinking toward
    zero assumes every unknown team is exactly average, which in week 3
    is a strictly worse guess than "roughly what they were last year."
    Passing a prior minimises

        ||W^.5 (Xb - y)||^2 + lambda ||b - prior||^2

    which costs one extra term in the normal equations. The prior's grip
    loosens automatically as real games accumulate -- no decay schedule
    to hand-tune, the algebra handles it.

    `home_field_prior` = (centre, strength) does the same for home field.
    Weeks 1-3 are mostly home games against FCS and low-major opponents,
    and the early fit overshoots every year (2.7-3.6 vs 1.7-2.5 full
    season; 5.2 in week 3 of 2026). Shrinking toward the historical
    value fixes that, and fades out as the season fills in.
    """
    if not games:
        raise ValueError("No games supplied to the rating engine.")

    teams = sorted({g.home_team for g in games} | {g.away_team for g in games})
    divisions: dict[str, str] = {}
    for game in games:
        divisions[game.home_team] = game.home_division
        divisions[game.away_team] = game.away_division

    design, raw_margin, _ = _build_design_matrix(games, teams)

    target = curve_margin(raw_margin, margin_scale)

    weeks = np.array([g.week for g in games], dtype=float)
    if current_week is None:
        current_week = float(weeks.max())
    weights = recency_weights(weeks, current_week, halflife)

    # Weighted ridge: minimize ||W^.5 (Xb - y)||^2 + lambda ||b||^2
    sqrt_weights = np.sqrt(weights)[:, None]
    weighted_design = design * sqrt_weights
    weighted_target = target * np.sqrt(weights)

    n_columns = design.shape[1]
    penalty = np.eye(n_columns) * lambda_
    # Home-field advantage and the division offset are structural effects,
    # not team quality. Shrinking them toward zero would bias them, so
    # both stay unpenalized.
    penalty[-1, -1] = 0.0
    penalty[-2, -2] = 0.0

    gram = weighted_design.T @ weighted_design + penalty
    moment = weighted_design.T @ weighted_target

    # Shrink toward the prior rather than toward zero. Teams absent from
    # the prior keep a target of zero, i.e. league average -- the right
    # default for a newcomer we know nothing about.
    if prior:
        centre = np.zeros(n_columns, dtype=float)
        for position, team in enumerate(teams):
            centre[position] = prior.get(team, 0.0)
        moment = moment + penalty @ centre

    if home_field_prior is not None:
        hfa_centre, hfa_strength = home_field_prior
        gram[-2, -2] += hfa_strength
        moment[-2] += hfa_strength * hfa_centre

    solution = np.linalg.solve(gram, moment)

    team_ratings = solution[:-2].copy()
    home_field = float(solution[-2])
    fcs_offset = float(solution[-1])

    # Centre each division separately. The offset column now carries the
    # level difference between the pools, so centring them together would
    # double-count it.
    for division in ("fbs", "fcs"):
        mask = np.array([divisions.get(t) == division for t in teams])
        if mask.any():
            team_ratings[mask] -= team_ratings[mask].mean()

    predicted = design @ np.concatenate(
        [team_ratings, [home_field, fcs_offset]])
    residuals = target - predicted

    # Fit the calibration slope: regress ACTUAL margin on the rating
    # edge, through the origin. An even matchup must predict zero, so
    # there is no intercept to fit. Home field is already additive and
    # separately estimated, so it is removed before measuring the slope
    # rather than being stretched along with team quality.
    edges = design[:, :-2] @ team_ratings
    location = design[:, -2] * home_field + design[:, -1] * fcs_offset
    adjusted = raw_margin - location
    denominator = float(edges @ edges)
    raw_calibration = (float(edges @ adjusted) / denominator
                       if denominator else 1.0)
    calibration = _shrink_calibration(raw_calibration, len(games))

    return RatingResult(
        ratings=dict(zip(teams, team_ratings)),
        home_field=home_field,
        fcs_offset=fcs_offset,
        divisions=divisions,
        lambda_=lambda_,
        margin_scale=margin_scale,
        halflife=halflife,
        n_games=len(games),
        n_teams=len(teams),
        residual_std=float(residuals.std()),
        teams=teams,
        calibration=calibration,
    )


def cross_validate(
    games: list[Game],
    lambdas=(5, 10, 20, 40, 80, 160, 320),
    margin_scales=(10.0, 14.0, 20.0, 28.0),
    halflives=(4.0, 8.0, 16.0, 1e6),
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Grid-search the free parameters against held-out prediction error.

    This is the step that removes subjectivity: we do not choose the
    ridge penalty or the margin curve, we measure which values predict
    unseen games most accurately.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(games))
    folds = np.array_split(order, n_folds)

    best = {"mae": float("inf")}
    results = []

    for lambda_ in lambdas:
        for scale in margin_scales:
            for halflife in halflives:
                errors = []
                for fold in folds:
                    holdout = set(fold.tolist())
                    train = [g for i, g in enumerate(games) if i not in holdout]
                    test = [games[i] for i in fold]
                    if not train or not test:
                        continue
                    try:
                        model = fit(train, lambda_, scale, halflife)
                    except np.linalg.LinAlgError:
                        continue
                    for game in test:
                        # Only score teams the model has actually seen.
                        if (game.home_team not in model.ratings
                                or game.away_team not in model.ratings):
                            continue
                        predicted = model.predict_margin(
                            game.home_team, game.away_team, game.neutral_site
                        )
                        errors.append(abs(predicted - game.margin))
                if not errors:
                    continue
                mae = float(np.mean(errors))
                results.append({
                    "lambda": lambda_, "margin_scale": scale,
                    "halflife": halflife, "mae": mae,
                })
                if mae < best["mae"]:
                    best = {
                        "mae": mae, "lambda": lambda_,
                        "margin_scale": scale, "halflife": halflife,
                    }

    best["all_results"] = sorted(results, key=lambda r: r["mae"])
    return best
