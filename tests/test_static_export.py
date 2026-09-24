"""The static predictor payload must reproduce Predictor.predict() exactly.

app/static/predict.js has no access to the Python model -- it only gets
this payload. If these identities ever drift, the offline site would
silently show different numbers than the live one.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfbrank.ensemble import EnsembleModel
from cfbrank.epa_ridge import build_epa_games, fit_epa_ratings_primed, get_ep_model
from cfbrank.games import load_season
from cfbrank.offdef_prior import fit_points_model
from cfbrank.predict import Predictor
from cfbrank.ridge import fit
from cfbrank.static_export import predictor_payload

FITTED = {"lambda_": 5, "margin_scale": 28.0, "halflife": 1e6}
MATCHUPS = [
    ("Ohio State", "Michigan"), ("Georgia", "Alabama"),
    ("Texas", "Oklahoma"), ("Oregon", "Washington"),
]


@pytest.fixture(scope="module")
def games_2024():
    return load_season(2024)


@pytest.fixture(scope="module")
def margin_model(games_2024):
    return fit(games_2024, **FITTED)


@pytest.fixture(scope="module")
def ensemble_model(games_2024, margin_model):
    """Same shape as production: margin + EPA, when play data is there."""
    try:
        epa_games = build_epa_games(2024, games_2024, get_ep_model())
        epa_model = fit_epa_ratings_primed(epa_games, None)
    except (FileNotFoundError, ValueError, KeyError, np.linalg.LinAlgError):
        pytest.skip("No play-by-play data for 2024 on this machine.")
    return EnsembleModel(margin_model, epa_model)


@pytest.mark.parametrize("home,away", MATCHUPS)
def test_calibrated_rating_reproduces_predict_margin(margin_model, home, away):
    """The identity the whole export depends on, for the plain model."""
    lhs = (margin_model.calibrated_rating(home)
           - margin_model.calibrated_rating(away) + margin_model.home_field)
    assert lhs == pytest.approx(margin_model.predict_margin(home, away))


@pytest.mark.parametrize("home,away", MATCHUPS)
def test_calibrated_rating_reproduces_predict_margin_ensemble(
    ensemble_model, home, away
):
    """Same identity, through the production margin+EPA ensemble."""
    lhs = (ensemble_model.calibrated_rating(home)
           - ensemble_model.calibrated_rating(away)
           + ensemble_model.home_field)
    assert lhs == pytest.approx(ensemble_model.predict_margin(home, away))


def test_payload_omits_total_model_when_absent(games_2024, margin_model):
    predictor = Predictor.from_model(margin_model, games_2024)
    payload = predictor_payload(predictor, ["Ohio State", "Michigan"])
    assert "offense" not in payload
    assert "total_mean" not in payload


def test_payload_reconstructs_a_full_prediction(games_2024, margin_model):
    """Rebuild margin, total, and win probability from the JSON payload
    alone, matching Predictor.predict() -- the exact job predict.js does."""
    total_model = fit_points_model(2024, games_2024)
    predictor = Predictor.from_model(
        margin_model, games_2024, total_model=total_model)
    home, away = "Georgia", "Alabama"
    payload = predictor_payload(predictor, [home, away])
    live = predictor.predict(home, away)

    margin = (payload["calibrated"][home] - payload["calibrated"][away]
              + payload["home_field"])
    assert margin == pytest.approx(live.predicted_margin)

    total = (2 * payload["total_mean"]
              + payload["offense"][home] + payload["offense"][away]
              - payload["defense"][home] - payload["defense"][away])
    assert total == pytest.approx(live.total)

    errors = np.array(payload["errors"])
    win_prob = float(np.mean(errors > -margin))
    assert win_prob == pytest.approx(live.home_win_probability)


def test_payload_errors_are_sorted(games_2024, margin_model):
    predictor = Predictor.from_model(margin_model, games_2024)
    payload = predictor_payload(predictor, [])
    assert payload["errors"] == sorted(payload["errors"])
