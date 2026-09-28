"""The Vegas Script: what the bookies have "decided" will happen.

A second, clearly-labelled prediction that sits next to JowieModel's own
number on the Predict page. The rankings never use it.

How it works. Before each week, fit the same ridge engine on the closing
LINES of every game already played -- not the results. That reads off how
the market rates each team as of last week, which quietly bakes in the
things we have no data for (injuries, depth charts, QB changes). Then
blend it with our own prediction.

No leak: only lines for games already played are used, never the line
for the game being predicted.

Measured walk-forward, 5,284 games 2021-25, weights fit on four seasons
and scored on the fifth (tools/probe_market_rating.py):
    gap to the closing line  +0.338 -> +0.244   (weeks 4-8: +0.452 -> +0.265)
Lambda swept 0.02 / 0.1 / 0.5 / 2 / 8: 0.1 best, flat below it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .games import Game
from .ridge import RatingResult, fit

# Lines are already on a sensible scale: no blowout curve, no recency
# decay, a light penalty (the market's opinions are not noisy), and the
# raw calibration slope -- ridge.fit's 1.85 slope prior is for noisy
# scores and inflated market ratings ~1.5x in thin early-season fits.
MARKET_PARAMS = {"lambda_": 0.1, "margin_scale": 1e4, "halflife": 1e6,
                 "shrink_calibration": False}
# Fitted on the full 2021-25 walk-forward sample; leave-one-season-out
# weights stayed within +/-0.02 of these.
OURS_WEIGHT = 0.682
MARKET_WEIGHT = 0.365
# Below this many priced games the market ratings are too thin to fit
# (the backtest started a season's Vegas Script at 100 too).
MIN_PRICED_GAMES = 100

# "The script is in." When the Vegas Script and our number split by this
# many points, the script has historically been the better call
# (tools/probe_market_rating.py, 5,284 games, script fit out of season):
#   split 0-3 pts : script closer 51-52%, saves 0.02-0.13 pts   -> no alert
#   split 3+ pts  : script closer 54-56%, saves 0.31-0.59 pts   -> alert
#   3+ AND it picks a different winner: script's winner won 61% (75 games)
SCRIPT_IN_POINTS = 3.0


def script_alert(ours: float, script: float | None) -> str | None:
    """None, "in" (3+ point split) or "flip" (3+ and a different winner)."""
    if script is None or abs(script - ours) < SCRIPT_IN_POINTS:
        return None
    return "flip" if (script > 0) != (ours > 0) else "in"


def priced_games(games: list[Game], lines: dict[str, float]) -> list[Game]:
    """Games with a closing line, the line standing in for the score."""
    return [replace(g, home_points=lines[str(g.game_id)], away_points=0)
            for g in games if str(g.game_id) in lines]


def fit_market_ratings(games: list[Game],
                       lines: dict[str, float]) -> RatingResult | None:
    """Ridge ratings implied by past closing lines, or None if too few."""
    priced = priced_games(games, lines)
    if len(priced) < MIN_PRICED_GAMES:
        return None
    return fit(priced, **MARKET_PARAMS)


@dataclass(slots=True)
class VegasScript:
    """Blends our margin with the market-implied one."""

    market: RatingResult

    def knows(self, home: str, away: str) -> bool:
        return home in self.market.ratings and away in self.market.ratings

    def market_margin(self, home: str, away: str, neutral: bool) -> float:
        return self.market.predict_margin(home, away, neutral)

    def blend(self, ours: float, home: str, away: str, neutral: bool) -> float:
        return (OURS_WEIGHT * ours
                + MARKET_WEIGHT * self.market_margin(home, away, neutral))
