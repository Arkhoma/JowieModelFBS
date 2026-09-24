"""Game prediction -- the model evaluated forward instead of backward.

Nothing here is generated or invented. A prediction is arithmetic on
parameters the ridge regression already fit against thousands of real
games:

    margin = rating[home] - rating[away] + home_field

Win probability comes from the EMPIRICAL distribution of this model's
own historical errors, not an assumed bell curve. If the model has
historically been off by 10 points on average, a 3-point predicted
margin is close to a coin flip, and the number says so.

Every prediction carries its own explanation and an error bar. A point
estimate without a confidence interval invites false precision, and
college football is genuinely high-variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .games import Game
from .ridge import RatingResult

# Scoring environment. Fitted from the games supplied, not assumed.
DEFAULT_TOTAL_POINTS = 52.0


@dataclass(slots=True)
class Prediction:
    """One predicted game, with the reasoning that produced it."""

    home_team: str
    away_team: str
    neutral_site: bool
    predicted_margin: float
    home_score: float
    away_score: float
    home_win_probability: float
    error_std: float
    factors: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.home_score + self.away_score

    @property
    def favourite(self) -> str:
        return self.home_team if self.predicted_margin > 0 else self.away_team

    @property
    def spread(self) -> float:
        """Absolute margin, the way a spread is normally quoted."""
        return abs(self.predicted_margin)

    def explain(self) -> str:
        """Human-readable breakdown. Every prediction shows its work."""
        location = "neutral site" if self.neutral_site else (
            f"at {self.home_team}")
        lines = [
            f"{self.away_team} at {self.home_team} ({location})",
            "",
            f"  Predicted:  {self.home_team} {self.home_score:.1f} - "
            f"{self.away_score:.1f} {self.away_team}",
            f"  Favourite:  {self.favourite} by {self.spread:.1f}",
            f"  Win prob:   {self.home_team} "
            f"{self.home_win_probability:.0%} / {self.away_team} "
            f"{1 - self.home_win_probability:.0%}",
            f"  Error bar:  +/- {self.error_std:.1f} points (1 sigma)",
            "",
            "  Why:",
        ]
        for label, value in self.factors:
            lines.append(f"    {label:<28} {value:+7.2f}")
        return "\n".join(lines)


def _empirical_win_probability(
    margin: float, errors: np.ndarray
) -> float:
    """P(home wins) using the model's own historical error distribution.

    The home team wins when the actual margin exceeds zero, i.e. when the
    prediction error is greater than -margin. Counting how often that
    happened historically beats assuming a normal distribution.
    """
    if errors.size == 0:
        return 0.5
    return float(np.mean(errors > -margin))


def _fit_scoring_environment(games: list[Game]) -> float:
    """Average combined points per game, for splitting margin into a score."""
    if not games:
        return DEFAULT_TOTAL_POINTS
    totals = [g.home_points + g.away_points for g in games]
    return float(np.mean(totals))


@dataclass(slots=True)
class Predictor:
    """Turns a fitted rating model into game predictions.

    `total_model`, if given, is an offense/defense model (cfbrank.offdef)
    used ONLY for the combined score. The margin still comes from
    `model`: in walk-forward testing the off/def split did not improve
    spreads, but it cut total-points error from 13.71 (league-average
    total) to 12.94 vs the market's 12.63. tools/offdef_experiment.py
    """

    model: RatingResult
    errors: np.ndarray
    average_total: float
    total_model: object | None = None

    @classmethod
    def from_model(
        cls, model: RatingResult, games: list[Game],
        total_model: object | None = None,
    ) -> "Predictor":
        """Calibrate against the games the model was fit on.

        The residuals here are in-sample, so the error bar is mildly
        optimistic. The backtest reports the honest out-of-sample number.
        """
        errors = []
        for game in games:
            if (game.home_team not in model.ratings
                    or game.away_team not in model.ratings):
                continue
            predicted = model.predict_margin(
                game.home_team, game.away_team, game.neutral_site)
            errors.append(game.margin - predicted)
        # Centre the residuals. Only their SPREAD should set the win
        # probability; the point estimate already carries the location.
        # Across 2021-2025 walk-forward the mean residual is ~0, so this
        # changes nothing historically (Brier -0.0001). But at week 3 of
        # 2026 the in-sample mean was +4.2 points -- an inflated early
        # home-field fit -- which pushed every probability ~14 points
        # toward the home side, resume ratings included.
        # tools/check_error_centering.py
        residuals = np.asarray(errors, dtype=float)
        if residuals.size:
            residuals = residuals - residuals.mean()
        return cls(
            model=model,
            errors=residuals,
            average_total=_fit_scoring_environment(games),
            total_model=total_model,
        )

    def predict_total(
        self, home_team: str, away_team: str, neutral_site: bool = False
    ) -> float:
        """Combined points: matchup-specific if we can, league average if not."""
        tm = self.total_model
        if (tm is not None and home_team in tm.ratings
                and away_team in tm.ratings):
            return tm.predict_total(home_team, away_team, neutral_site)
        return self.average_total

    def predict(
        self, home_team: str, away_team: str, neutral_site: bool = False
    ) -> Prediction:
        """Predict one game."""
        model = self.model
        home_rating = model.rating(home_team)
        away_rating = model.rating(away_team)
        home_field = 0.0 if neutral_site else model.home_field

        # Delegate to the model rather than re-deriving the formula. An
        # earlier version recomputed the margin inline here, so when the
        # calibration slope was added to predict_margin() this path
        # silently kept returning uncalibrated numbers -- the backtest
        # did not move at all, which is how it was caught. One formula,
        # one home.
        margin = model.predict_margin(home_team, away_team, neutral_site)

        total = self.predict_total(home_team, away_team, neutral_site)
        home_score = (total + margin) / 2.0
        away_score = (total - margin) / 2.0

        factors = [
            (f"{home_team} rating", home_rating),
            (f"{away_team} rating", away_rating),
            ("Rating edge", home_rating - away_rating),
        ]
        if model.calibration != 1.0:
            factors.append(
                ("Calibration", (home_rating - away_rating)
                 * (model.calibration - 1.0)))
        if neutral_site:
            factors.append(("Home field (neutral)", 0.0))
        else:
            factors.append(("Home field", home_field))
        factors.append(("Predicted margin", margin))

        return Prediction(
            home_team=home_team,
            away_team=away_team,
            neutral_site=neutral_site,
            predicted_margin=margin,
            home_score=home_score,
            away_score=away_score,
            home_win_probability=_empirical_win_probability(
                margin, self.errors),
            error_std=float(np.std(self.errors)) if self.errors.size else 0.0,
            factors=factors,
        )

    def known_teams(self) -> list[str]:
        return sorted(self.model.ratings)
