"""JSON payload for the client-side predictor used by the static site.

The static export has no backend, so `app/static/predict.js` has to
reproduce `Predictor.predict()` in the browser. Rather than re-deriving
the math in two places, every rating model exposes `calibrated_rating`
such that

    margin = calibrated_rating[home] - calibrated_rating[away]
             + (0 if neutral else model.home_field)

is identical to `model.predict_margin(home, away, neutral)`. This module
only reads numbers off an already-fitted model; it computes nothing new.

Win probability needs the model's own historical error distribution
(see cfbrank.predict), so the sorted residuals ship too -- small (one
float per game in the season) and lets the browser do the same
"how often did an error this size flip the result" count predict.py does.
"""

from __future__ import annotations

from .predict import Predictor


def predictor_payload(predictor: Predictor, teams: list[str]) -> dict:
    """Everything predict.js needs to predict games among `teams`.

    `teams` should already be filtered to teams the app is willing to
    offer in the picker (see app.main._predict_teams) -- this function
    just skips anyone the model itself doesn't know, the same guard
    Predictor.predict() relies on implicitly via the caller.
    """
    model = predictor.model
    total_model = predictor.total_model

    payload = {
        "home_field": model.home_field,
        "average_total": predictor.average_total,
        "errors": sorted(float(e) for e in predictor.errors),
        "rating": {t: model.rating(t) for t in teams if t in model.ratings},
        "calibrated": {
            t: model.calibrated_rating(t) for t in teams if t in model.ratings
        },
    }
    if total_model is not None:
        payload["offense"] = {
            t: total_model.offense_rating(t)
            for t in teams if t in total_model.ratings
        }
        payload["defense"] = {
            t: total_model.defense_rating(t)
            for t in teams if t in total_model.ratings
        }
        payload["total_mean"] = total_model.mean
    return payload
