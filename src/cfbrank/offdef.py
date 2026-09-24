"""Separate offense and defense ratings, opponent-adjusted together.

The net rating answers "who wins"; it cannot say HOW. A 38-35 team and a
17-14 team can share a net rating and play completely different games.
This engine splits every game into two observations -- each offense
against the opposing defense -- and solves for both sides at once:

    value(side) = mu + off[offense] - def[defense] + venue * home_field/2
                  + division term + error

`value` is any per-side statistic: points scored (schedule only, every
season) or offensive success rate (play-by-play). Higher `off` is a
better offense; higher `def` is a better defense (it SUBTRACTS from what
opponents produce), so both columns read "bigger is better".

Why a split could beat the net rating at all: in a linear model the
predicted margin only depends on off + def, so the split adds nothing
UNLESS the two sides are treated differently. They are, in three ways:
each side shrinks toward its own prior (returning offense predicts next
year's offense, not its defense), points for and against are separate
evidence where a margin throws half of it away, and off/def predict the
game TOTAL, which a single net number cannot do at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .epa_games import TeamGameEPA
from .games import Game

# Home field is estimated on the MARGIN scale: the home offense gets +half
# and the away offense -half, so the fitted coefficient is directly
# comparable with the margin engine's home_field.
_HALF = 0.5


@dataclass(frozen=True, slots=True)
class SideGame:
    """One offense against one defense in one game."""

    game_id: str
    season: int
    week: int
    offense: str
    defense: str
    offense_division: str
    defense_division: str
    venue: float          # +1 offense at home, -1 on the road, 0 neutral
    value: float          # the per-side statistic being explained
    margin: float         # offense points minus defense points (for scaling)


def sides_from_games(games: list[Game]) -> list[SideGame]:
    """Points scored per side -- works on any season, no play data needed."""
    sides: list[SideGame] = []
    for g in games:
        venue = 0.0 if g.neutral_site else 1.0
        sides.append(SideGame(str(g.game_id), g.season, g.week, g.home_team,
                              g.away_team, g.home_division, g.away_division,
                              venue, float(g.home_points), float(g.margin)))
        sides.append(SideGame(str(g.game_id), g.season, g.week, g.away_team,
                              g.home_team, g.away_division, g.home_division,
                              -venue, float(g.away_points), float(-g.margin)))
    return sides


def sides_from_efficiency(
    games: list[Game], records: list[TeamGameEPA], metric: str = "success"
) -> list[SideGame]:
    """Per-side offensive efficiency, joined onto the schedule.

    The schedule stays authoritative for who, where and the final score;
    games without play data for both teams are dropped, not guessed.
    """
    attribute = {"success": "offense_success_rate",
                 "epa": "offense_epa_per_play"}[metric]
    by_key = {(r.game_id, r.team): getattr(r, attribute) for r in records}
    sides: list[SideGame] = []
    for side in sides_from_games(games):
        value = by_key.get((side.game_id, side.offense))
        if value is None or (side.game_id, side.defense) not in by_key:
            continue
        sides.append(SideGame(side.game_id, side.season, side.week,
                              side.offense, side.defense,
                              side.offense_division, side.defense_division,
                              side.venue, float(value), side.margin))
    return sides


@dataclass(slots=True)
class OffDefResult:
    """Fitted offense and defense ratings on the statistic's own scale."""

    offense: dict[str, float]
    defense: dict[str, float]
    mean: float
    home_field: float
    fcs_offset: float
    divisions: dict[str, str]
    lambda_: float
    n_sides: int
    residual_std: float
    # Predicted margin in points = edge_scale * net edge + home field.
    # For points this stretch undoes ridge compression; for success rate
    # it is also the unit conversion. See _fit_scale().
    edge_scale: float = 1.0
    home_field_points: float = 0.0
    teams: list[str] = field(default_factory=list)

    @property
    def ratings(self) -> dict[str, float]:
        """Net rating (off + def), so this quacks like the other models."""
        return {t: self.net_rating(t) for t in self.offense}

    def net_rating(self, team: str) -> float:
        return self.offense_rating(team) + self.defense_rating(team)

    def rating(self, team: str) -> float:
        """Same contract as RatingResult.rating (common scale)."""
        return self.net_rating(team)

    @property
    def calibration(self) -> float:
        return self.edge_scale

    def rating_in_points(self, team: str) -> float:
        return self.net_rating(team) * self.edge_scale

    def _division_shift(self, team: str) -> float:
        return self.fcs_offset if self.divisions.get(team) == "fcs" else 0.0

    def offense_rating(self, team: str) -> float:
        return self.offense.get(team, 0.0) + self._division_shift(team)

    def defense_rating(self, team: str) -> float:
        return self.defense.get(team, 0.0) + self._division_shift(team)

    def expected_value(
        self, offense: str, defense: str, venue: float
    ) -> float:
        """Raw predicted statistic for one side (points, or success rate)."""
        return (self.mean + self.offense_rating(offense)
                - self.defense_rating(defense)
                + venue * _HALF * self.home_field)

    def predict_scores(
        self, home: str, away: str, neutral_site: bool = False
    ) -> tuple[float, float]:
        """(home, away) on the statistic's scale -- uncalibrated."""
        venue = 0.0 if neutral_site else 1.0
        return (self.expected_value(home, away, venue),
                self.expected_value(away, home, -venue))

    def predict_margin(
        self, home: str, away: str, neutral_site: bool = False
    ) -> float:
        """Expected home margin IN POINTS, same contract as RatingResult."""
        edge = self.net_rating(home) - self.net_rating(away)
        location = 0.0 if neutral_site else self.home_field_points
        return edge * self.edge_scale + location

    def predict_total(
        self, home: str, away: str, neutral_site: bool = False
    ) -> float:
        """Expected combined score. Only meaningful for the points model."""
        home_value, away_value = self.predict_scores(home, away, neutral_site)
        return home_value + away_value

    def fbs_ratings(self) -> dict[str, float]:
        return {t: r for t, r in self.ratings.items()
                if self.divisions.get(t) == "fbs"}


def fit(
    sides: list[SideGame],
    lambda_: float,
    prior: tuple[dict[str, float], dict[str, float]] | None = None,
    home_field_prior: tuple[float, float] | None = None,
    is_points: bool = True,
) -> OffDefResult:
    """Ridge-solve offense and defense for every team simultaneously.

    `prior` is (offense_prior, defense_prior); each side shrinks toward
    its own centre, which is the whole point of splitting. Teams absent
    from a prior shrink toward zero (average), as in the margin engine.

    `is_points` says whether the statistic is already points. If not
    (success rate), the edge-to-points conversion is fitted against
    actual margins.
    """
    if not sides:
        raise ValueError("No side observations supplied.")

    teams = sorted({s.offense for s in sides} | {s.defense for s in sides})
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    divisions: dict[str, str] = {}
    for s in sides:
        divisions[s.offense] = s.offense_division
        divisions[s.defense] = s.defense_division

    # Columns: offense[n], defense[n], home field, FCS offset, mean.
    columns = 2 * n + 3
    design = np.zeros((len(sides), columns))
    target = np.array([s.value for s in sides], dtype=float)
    for row, s in enumerate(sides):
        design[row, index[s.offense]] = 1.0
        design[row, n + index[s.defense]] = -1.0
        design[row, -3] = s.venue * _HALF
        off_fcs = s.offense_division == "fcs"
        def_fcs = s.defense_division == "fcs"
        if off_fcs != def_fcs:
            # An FCS offense scores less against FBS (+f, f < 0); an FCS
            # defense allows more (-f). One offset shifts both sides.
            design[row, -2] = 1.0 if off_fcs else -1.0
        design[row, -1] = 1.0

    penalty = np.zeros(columns)
    penalty[:2 * n] = lambda_
    # Structural columns stay unpenalized -- unless a slice has no data
    # for them (no FCS crossovers, every game neutral), in which case the
    # column is all zeros and a unit penalty pins it at zero instead of
    # making the system singular.
    for col in (-3, -2):
        if not design[:, col].any():
            penalty[col] = 1.0
    gram = design.T @ design + np.diag(penalty)
    moment = design.T @ target

    if prior:
        centre = np.zeros(columns)
        offense_prior, defense_prior = prior
        for team, i in index.items():
            centre[i] = offense_prior.get(team, 0.0)
            centre[n + i] = defense_prior.get(team, 0.0)
        moment = moment + penalty * centre

    if home_field_prior is not None:
        hfa_centre, strength = home_field_prior
        gram[-3, -3] += strength
        moment[-3] += strength * hfa_centre

    solution = np.linalg.solve(gram, moment)
    offense = solution[:n].copy()
    defense = solution[n:2 * n].copy()
    home_field, fcs_offset, mean = (float(v) for v in solution[-3:])

    # Centre each side within each division. For FBS the shift moves into
    # the mean (offense up, defense down cancel in every prediction);
    # FCS stays relative to its own pool with the offset owning the gap.
    for division in ("fbs", "fcs"):
        mask = np.array([divisions.get(t) == division for t in teams])
        if not mask.any():
            continue
        off_shift = offense[mask].mean()
        def_shift = defense[mask].mean()
        offense[mask] -= off_shift
        defense[mask] -= def_shift
        if division == "fbs":
            mean += off_shift - def_shift

    fitted = design @ np.concatenate(
        [offense, defense, [home_field, fcs_offset, mean]])
    residuals = target - fitted

    result = OffDefResult(
        offense=dict(zip(teams, offense)), defense=dict(zip(teams, defense)),
        mean=mean, home_field=home_field, fcs_offset=fcs_offset,
        divisions=divisions, lambda_=lambda_, n_sides=len(sides),
        residual_std=float(residuals.std()), teams=teams)
    _fit_scale(result, sides, is_points)
    return result


def _fit_scale(
    result: OffDefResult, sides: list[SideGame], is_points: bool
) -> None:
    """Map the net edge onto actual points by a through-origin slope.

    Ridge compresses rating differences, so predicted margins come out
    timid (the margin engine measured ~1.7x); the slope undoes that. For
    success rate the same slope is also the unit conversion.

    Home field: on the points scale it is already in points and is taken
    off the margin before measuring the slope, so it is never stretched.
    On the success scale it converts with the slope -- mirroring
    epa_ridge, and for the same reason: re-fitting it jointly against
    final scores lets thin early-season data invent absurd values.
    """
    one_per_game = {s.game_id: s for s in sides if s.venue >= 0}.values()
    edges = np.array([result.net_rating(s.offense)
                      - result.net_rating(s.defense) for s in one_per_game])
    margins = np.array([s.margin for s in one_per_game])
    venues = np.array([s.venue for s in one_per_game])

    target = (margins - venues * result.home_field if is_points
              else margins - margins.mean())
    denominator = float(edges @ edges)
    slope = float(edges @ target) / denominator if denominator else 0.0

    if is_points:
        result.edge_scale = slope if 0.5 < slope < 3.0 else 1.0
        result.home_field_points = result.home_field
    else:
        result.edge_scale = slope if slope > 0 else 1.0
        result.home_field_points = result.home_field * result.edge_scale
