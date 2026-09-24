"""Rate teams on per-play efficiency instead of final score.

This is the upgrade that brings the model in line with SP+ and FEI. The
ridge machinery is unchanged -- same simultaneous solve, same opponent
adjustment, same fitted home field. The only difference is what each
game contributes as evidence.

Final score is a lossy summary of ~150 plays, and a noisy one: two
pick-sixes can flip a 24-21 game into a 35-3 game without either team
playing differently. Net EPA per play uses every snap, so a team that
consistently gains ground rates well even when the scoreboard lies.

The observation for each game is the home team's net EPA-per-play edge:

    (home off EPA - home def EPA) - (away off EPA - away def EPA)

which is positive when the home team was the more efficient side. That
plays the same structural role as margin did, so everything downstream
-- opponent adjustment, home field, the FCS offset, the preseason prior,
the calibration slope -- carries over untouched.

Ratings come out on an EPA-per-play scale (~ -0.4 to +0.4), which is not
points. A conversion factor is fitted against actual margins so
predictions stay in points, where they are interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import expected_points as ep
from .epa_games import TeamGameEPA, season_epa
from .games import Game
from .plays import load_plays

# Seasons used to fit the expected-points table. Held apart from the
# season being rated so the EP model is not fit on the games it scores.
EP_TRAINING_SEASONS = (2022, 2023)

_ep_model_cache: dict[tuple, ep.ExpectedPointsModel] = {}


def get_ep_model(
    seasons: tuple = EP_TRAINING_SEASONS,
) -> ep.ExpectedPointsModel:
    """Fit (and cache) the expected-points table.

    Fitting scans several hundred thousand plays, so it is cached: the
    table depends only on the training seasons, never on what we rate.
    """
    if seasons not in _ep_model_cache:
        frames = [load_plays(season) for season in seasons]
        _ep_model_cache[seasons] = ep.fit(
            pd.concat(frames, ignore_index=True))
    return _ep_model_cache[seasons]


# Ridge penalty, expressed RELATIVE to the variance of what is being
# explained. This matters: a raw lambda is not portable between scales.
#
# Final margins have sd ~20.7; EPA edges have sd ~0.78 -- margin varies
# 26x more. Reusing lambda=5 (tuned for margins) on EPA is therefore
# ~26^2 times too harsh, and it showed: EPA rating spread collapsed from
# 0.49 to 0.27 and EPA lost to plain margin by 1.8 points of MAE.
#
# Scaling by the observed variance makes the penalty mean the same thing
# on either scale, so the number below is a real modelling choice rather
# than an artifact of the units.
#
# Re-measured after fixing the prior double-counting bug (both this and
# the margin engine now share ONE historical prior, see
# fit_epa_ratings_primed): swept 1/4/8/16/40 walk-forward on 2024-2025,
# 4.0 won (13.163 MAE) over the previously-used 8.0 (13.524 MAE). The
# old value was chosen under a biased comparison and is retired, not
# hand-adjusted -- see tools/compare_epa_fair.py.
RELATIVE_LAMBDA = 4.0


def scaled_lambda(target: np.ndarray, relative: float = RELATIVE_LAMBDA) -> float:
    """Convert a scale-free penalty into one matched to this target."""
    variance = float(np.var(target))
    return relative * variance if variance > 0 else relative


@dataclass(frozen=True, slots=True)
class EPAGame:
    """One game reduced to an efficiency edge, ready for the regression.

    Deliberately mirrors the shape the ridge engine already expects, so
    the same solver handles both margin-based and EPA-based ratings.
    """

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_division: str
    away_division: str
    neutral_site: bool
    epa_edge: float
    margin: int

    @property
    def is_fbs_only(self) -> bool:
        return self.home_division == "fbs" and self.away_division == "fbs"


def _net_edge(home: TeamGameEPA, away: TeamGameEPA, metric: str) -> float:
    """Home-minus-away net edge on the chosen per-play metric.

    'success' uses success rate, which stabilises faster than EPA (it
    rewards consistency, not explosiveness) and so carries more signal
    in September when every team has three games.
    """
    if metric == "epa":
        return home.net_epa_per_play - away.net_epa_per_play
    if metric == "success":
        return ((home.offense_success_rate - home.defense_success_rate)
                - (away.offense_success_rate - away.defense_success_rate))
    raise ValueError(f"Unknown metric {metric!r}")


def build_epa_games(
    season: int,
    games: list[Game],
    model: ep.ExpectedPointsModel | None = None,
    metric: str = "epa",
    drop_garbage_time: bool = True,
) -> list[EPAGame]:
    """Join per-play efficiency onto the schedule.

    The schedule file is authoritative for who played whom, where, and
    the final score -- play data is used only for in-game state, because
    overtime scoring is missing from it. Games without usable play data
    are dropped rather than guessed at.
    """
    model = model or get_ep_model()
    records = season_epa(season, model, drop_garbage_time=drop_garbage_time)

    by_game: dict[str, dict[str, TeamGameEPA]] = {}
    for record in records:
        by_game.setdefault(record.game_id, {})[record.team] = record

    built: list[EPAGame] = []
    for game in games:
        sides = by_game.get(str(game.game_id))
        if not sides or len(sides) != 2:
            continue
        home = sides.get(game.home_team)
        away = sides.get(game.away_team)
        if home is None or away is None:
            continue

        built.append(EPAGame(
            game_id=str(game.game_id),
            season=game.season,
            week=game.week,
            home_team=game.home_team,
            away_team=game.away_team,
            home_division=game.home_division,
            away_division=game.away_division,
            neutral_site=game.neutral_site,
            epa_edge=_net_edge(home, away, metric),
            margin=game.margin,
        ))

    return built


@dataclass(slots=True)
class EPARatingResult:
    """Team efficiency ratings, opponent-adjusted.

    Mirrors RatingResult's interface so the app and predictor can treat
    the two interchangeably.
    """

    ratings: dict[str, float]
    home_field: float
    fcs_offset: float
    divisions: dict[str, str]
    lambda_: float
    n_games: int
    n_teams: int
    residual_std: float
    points_per_epa: float
    calibration: float = 1.0

    def rating(self, team: str) -> float:
        base = self.ratings.get(team, 0.0)
        if self.divisions.get(team) == "fcs":
            return base + self.fcs_offset
        return base

    def fbs_ratings(self) -> dict[str, float]:
        return {
            team: self.rating(team)
            for team, division in self.divisions.items()
            if division == "fbs" and team in self.ratings
        }

    def rating_in_points(self, team: str) -> float:
        """Rating converted to points, for display alongside margins."""
        return self.rating(team) * self.points_per_epa

    def predict_margin(
        self, home: str, away: str, neutral_site: bool = False
    ) -> float:
        """Expected home margin IN POINTS.

        The efficiency edge is converted using a fitted points-per-EPA
        factor. Home field is estimated directly in points and added
        afterwards, so it is never scaled by the conversion.
        """
        edge = (self.rating(home) - self.rating(away)) * self.calibration
        points = edge * self.points_per_epa
        return points + (0.0 if neutral_site else self.home_field)

    def calibrated_rating(self, team: str) -> float:
        """Points-scale rating with the calibration slope applied.

        Same identity as RatingResult.calibrated_rating: the difference
        of these, plus home_field, equals predict_margin(). See there.
        """
        return self.rating_in_points(team) * self.calibration


def translate_margin_prior(
    margin_prior: dict[str, float], points_per_epa: float
) -> dict[str, float]:
    """Express the margin-scale prior in EPA-per-play units.

    Deliberately NOT a second measured constant. The margin engine
    already answers "how much of a team's quality survives the
    offseason" via its carryover/unshrink pipeline (prior.py). EPA and
    margin describe the same underlying team quality in different
    units, so the honest move is to convert one measured prior, not fit
    a second one from scratch on a noisier, shorter-history signal.

    `points_per_epa` is the receiving model's own fitted conversion
    factor -- a straight unit change, not a new opinion.
    """
    if not points_per_epa:
        return {}
    return {team: value / points_per_epa
            for team, value in margin_prior.items()}


def fit_epa_ratings_primed(
    epa_games: list[EPAGame],
    margin_prior: dict[str, float] | None = None,
    relative_lambda: float = RELATIVE_LAMBDA,
) -> EPARatingResult:
    """Fit EPA ratings, shrinking toward the SAME historical prior the
    margin engine uses -- translated into EPA units.

    Why this exists: without a prior, three games is not enough evidence
    to keep a blowout over a weak or FCS opponent from dominating a
    team's rating. New Mexico beat Central Michigan and an FCS team by
    an EPA margin close to Notre Dame's margin over Rice, and with
    nothing to weigh that against, plain ridge took it at face value and
    rated New Mexico above Ole Miss and near LSU. The margin engine does
    not have this failure mode, because it already shrinks toward last
    season. This makes both engines behave the same way, on purpose,
    instead of each inventing its own rule for how much to trust three
    games.

    Two passes: fit unprimed once purely to learn this season's
    points-per-EPA conversion (needed to translate the prior into EPA
    units), then refit for real with that translated prior applied.
    """
    unprimed = fit_epa_ratings(epa_games, relative_lambda=relative_lambda)
    if not margin_prior:
        return unprimed
    epa_prior = translate_margin_prior(margin_prior, unprimed.points_per_epa)
    return fit_epa_ratings(
        epa_games, prior=epa_prior, relative_lambda=relative_lambda)


def fit_epa_ratings(
    epa_games: list[EPAGame],
    lambda_: float | None = None,
    prior: dict[str, float] | None = None,
    relative_lambda: float = RELATIVE_LAMBDA,
) -> EPARatingResult:
    """Opponent-adjust efficiency by ridge regression.

    Structurally identical to ridge.fit(): one equation per game, one
    column per team, plus home-field and division-offset columns. Only
    the observation differs -- efficiency edge rather than scoring
    margin.

    `lambda_` defaults to a penalty scaled to the variance of the EPA
    edges. Passing a raw value overrides that, which is useful for
    sweeps but easy to get wrong by an order of magnitude.
    """
    if not epa_games:
        raise ValueError("No EPA games supplied.")

    teams = sorted({g.home_team for g in epa_games}
                   | {g.away_team for g in epa_games})
    index = {team: position for position, team in enumerate(teams)}
    divisions: dict[str, str] = {}
    for game in epa_games:
        divisions[game.home_team] = game.home_division
        divisions[game.away_team] = game.away_division

    n_columns = len(teams) + 2
    design = np.zeros((len(epa_games), n_columns), dtype=float)
    target = np.zeros(len(epa_games), dtype=float)

    for row, game in enumerate(epa_games):
        design[row, index[game.home_team]] = 1.0
        design[row, index[game.away_team]] = -1.0
        if not game.neutral_site:
            design[row, -2] = 1.0

        home_is_fcs = game.home_division == "fcs"
        away_is_fcs = game.away_division == "fcs"
        if home_is_fcs != away_is_fcs:
            design[row, -1] = 1.0 if home_is_fcs else -1.0

        target[row] = game.epa_edge

    if lambda_ is None:
        lambda_ = scaled_lambda(target, relative_lambda)

    penalty = np.eye(n_columns) * lambda_
    penalty[-1, -1] = 0.0
    penalty[-2, -2] = 0.0

    gram = design.T @ design + penalty
    moment = design.T @ target

    if prior:
        centre = np.zeros(n_columns, dtype=float)
        for position, team in enumerate(teams):
            centre[position] = prior.get(team, 0.0)
        moment = moment + penalty @ centre

    solution = np.linalg.solve(gram, moment)

    team_ratings = solution[:-2].copy()
    home_field_epa = float(solution[-2])
    fcs_offset = float(solution[-1])

    for division in ("fbs", "fcs"):
        mask = np.array([divisions.get(t) == division for t in teams])
        if mask.any():
            team_ratings[mask] -= team_ratings[mask].mean()

    predicted = design @ np.concatenate(
        [team_ratings, [home_field_epa, fcs_offset]])
    residuals = target - predicted

    # Convert the efficiency scale into points by regressing actual
    # margins on the rating edge. Fitted, never assumed -- and it also
    # absorbs the ridge compression, so no separate calibration term is
    # needed on this path.
    #
    # Home field is taken from the EFFICIENCY regression (converted to
    # points) rather than re-fit here. Fitting both jointly lets least
    # squares dump unexplained scoring into the home-field column when
    # the rating spread is small: through week 3 of 2026 that produced a
    # home-field advantage of 21 points, which is nonsense. The
    # efficiency estimate is identified by thousands of plays rather
    # than a few hundred final scores, so it stays believable when data
    # is thin.
    edges = design[:, :-2] @ team_ratings
    margins = np.array([g.margin for g in epa_games], dtype=float)
    location = np.array(
        [0.0 if g.neutral_site else 1.0 for g in epa_games])

    denominator = float(edges @ edges)
    points_per_epa = (float(edges @ (margins - margins.mean()))
                      / denominator) if denominator else 0.0
    # Guard against a degenerate or sign-flipped conversion on thin data.
    if not 5.0 < points_per_epa < 80.0:
        points_per_epa = 25.0

    home_field_points = home_field_epa * points_per_epa

    return EPARatingResult(
        ratings=dict(zip(teams, team_ratings)),
        home_field=home_field_points,
        fcs_offset=fcs_offset,
        divisions=divisions,
        lambda_=lambda_,
        n_games=len(epa_games),
        n_teams=len(teams),
        residual_std=float(residuals.std()),
        points_per_epa=points_per_epa,
    )
