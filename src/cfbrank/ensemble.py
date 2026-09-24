"""Combine the margin and efficiency models into one prediction.

The finding that produced this module: EPA alone LOSES to final margin
(13.16 vs 12.45 MAE pooled 2024-2025), but stacking the two beats either
(12.18 on 2025, held out). Fitted coefficients are margin +0.679, EPA
+0.211 -- both clearly positive, so each carries information the other
does not.

Why that makes sense. Points are the objective, so a model trained
directly on margin is aimed at the right target. But margin is a noisy
readout of team quality: two turnovers can swing it 14 points without
either team playing differently. EPA measures the process that generates
points, and is therefore steadier but one step removed from what we are
predicting. Neither dominates, so use both.

My original plan -- replace margin with EPA -- was simply wrong, and
three rounds of trying to make EPA win on its own were wasted effort
chasing the wrong question.

The coefficients sum to ~0.89 rather than 1.0, which is not a bug: it is
shrinkage. Both models are individually over-confident, and least
squares discounts them accordingly.

Re-measured (tools/blend_epa_margin.py) after fixing a prior-scale bug:
the EPA engine used to shrink toward zero with no historical prior at
all, while the margin engine shrank toward last season. That let a
three-game blowout streak (New Mexico, week 3 2026) dominate the EPA
rating with nothing to weigh it against. Both engines now share ONE
measured prior (see epa_ridge.fit_epa_ratings_primed), and these weights
reflect the corrected fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .epa_ridge import EPARatingResult
from .ridge import RatingResult

# Fitted on 2021-2025 walk-forward predictions, validated leave-one-
# season-out (tools/improvement_experiment.py). The efficiency component
# is now SUCCESS RATE, not EPA: with the roster prior in place, stacking
# margin + success rate matched margin + EPA + success rate to within
# 0.002 pts/game (tools/check_epa_redundancy.py), so EPA was dropped as
# redundant. The `epa_*` names are kept for interface stability; the
# component is whichever per-play metric EFFICIENCY_METRIC names.
EFFICIENCY_METRIC = "success"
DEFAULT_MARGIN_WEIGHT = 0.731
DEFAULT_EPA_WEIGHT = 0.187


@dataclass(slots=True)
class EnsembleModel:
    """Two rating models and the weights that combine them."""

    margin_model: RatingResult
    epa_model: EPARatingResult
    margin_weight: float = DEFAULT_MARGIN_WEIGHT
    epa_weight: float = DEFAULT_EPA_WEIGHT

    def knows(self, team: str) -> bool:
        """Whether both models have seen this team."""
        return (team in self.margin_model.ratings
                and team in self.epa_model.ratings)

    def predict_margin(
        self, home: str, away: str, neutral_site: bool = False
    ) -> float:
        """Weighted combination of both models' point predictions.

        If either model has not seen a team -- an FCS newcomer, or a game
        with no play data -- fall back to whichever model has. A partial
        answer beats refusing to answer.
        """
        has_margin = (home in self.margin_model.ratings
                      and away in self.margin_model.ratings)
        has_epa = (home in self.epa_model.ratings
                   and away in self.epa_model.ratings)

        if has_margin and has_epa:
            from_margin = self.margin_model.predict_margin(
                home, away, neutral_site)
            from_epa = self.epa_model.predict_margin(home, away, neutral_site)
            return (self.margin_weight * from_margin
                    + self.epa_weight * from_epa)
        if has_margin:
            return self.margin_model.predict_margin(home, away, neutral_site)
        if has_epa:
            return self.epa_model.predict_margin(home, away, neutral_site)
        return 0.0

    def rating(self, team: str) -> float:
        """Combined rating in points, on the same scale as predictions.

        Built from the same weights as predict_margin() so the published
        table and the predictions cannot disagree.
        """
        total = 0.0
        if team in self.margin_model.ratings:
            total += self.margin_weight * self.margin_model.rating(team)
        if team in self.epa_model.ratings:
            total += self.epa_weight * self.epa_model.rating_in_points(team)
        return total

    def calibrated_rating(self, team: str) -> float:
        """Weighted, calibrated rating -- see RatingResult.calibrated_rating.

        Mirrors predict_margin()'s happy path (both submodels know the
        team). If only one submodel knows a team, predict_margin() falls
        back to giving that submodel 100% of the say for that MATCHUP,
        which a per-team number cannot represent -- an edge case for
        newcomers with no play data, not for any real FBS matchup.
        """
        total = 0.0
        if team in self.margin_model.ratings:
            total += self.margin_weight * self.margin_model.calibrated_rating(team)
        if team in self.epa_model.ratings:
            total += self.epa_weight * self.epa_model.calibrated_rating(team)
        return total

    def fbs_ratings(self) -> dict[str, float]:
        """Published table: every FBS team either model knows."""
        teams = set(self.margin_model.fbs_ratings())
        teams |= set(self.epa_model.fbs_ratings())
        return {team: self.rating(team) for team in teams}

    @property
    def home_field(self) -> float:
        """Blended home-field advantage, in points."""
        return (self.margin_weight * self.margin_model.home_field
                + self.epa_weight * self.epa_model.home_field)

    @property
    def ratings(self) -> dict[str, float]:
        """All rated teams, for compatibility with the single models."""
        teams = set(self.margin_model.ratings) | set(self.epa_model.ratings)
        return {team: self.rating(team) for team in teams}

    @property
    def divisions(self) -> dict[str, str]:
        combined = dict(self.epa_model.divisions)
        combined.update(self.margin_model.divisions)
        return combined

    # --- Interface parity with RatingResult -------------------------
    # Predictor and the templates accept either model, so the ensemble
    # must answer the same questions. These are reported rather than
    # recombined: the underlying models already fitted them, and the
    # blend weights apply to PREDICTIONS, not to fit diagnostics.

    @property
    def calibration(self) -> float:
        """Always 1.0 -- the blend weights already absorb compression.

        Each component applies its own calibration internally, and the
        stacked coefficients (0.679 + 0.211 = 0.89, deliberately under
        1.0) shrink the combination further. Applying another factor on
        top would double-count.
        """
        return 1.0

    @property
    def fcs_offset(self) -> float:
        return self.margin_model.fcs_offset

    @property
    def residual_std(self) -> float:
        return self.margin_model.residual_std

    @property
    def n_games(self) -> int:
        return self.margin_model.n_games

    @property
    def n_teams(self) -> int:
        return len(self.ratings)

    @property
    def lambda_(self) -> float:
        return self.margin_model.lambda_

    @property
    def margin_scale(self) -> float:
        return self.margin_model.margin_scale

    @property
    def halflife(self) -> float:
        return self.margin_model.halflife


def fit_weights(
    margin_predictions: np.ndarray,
    epa_predictions: np.ndarray,
    actual_margins: np.ndarray,
) -> tuple[float, float]:
    """Least-squares weights for combining two models.

    No intercept: an even matchup on a neutral field must predict zero,
    and fitting a constant would let the model claim a baseline edge for
    nobody in particular.
    """
    stacked = np.column_stack([margin_predictions, epa_predictions])
    coefficients, *_ = np.linalg.lstsq(stacked, actual_margins, rcond=None)
    return float(coefficients[0]), float(coefficients[1])
