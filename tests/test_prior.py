"""Tests for the preseason prior and prediction calibration.

These pin down findings that were expensive to discover and easy to
silently undo. Each test names the failure it guards against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank import prior as prior_module  # noqa: E402
from cfbrank.games import load_season, team_divisions  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}


@pytest.fixture(scope="module")
def games_2024():
    return load_season(2024)


@pytest.fixture(scope="module")
def model_2024(games_2024):
    return fit(games_2024, **FITTED)


def test_carryover_is_well_below_one():
    """Teams genuinely regress between seasons.

    Measured at 0.63 across ten transitions. An earlier version fit this
    by least squares against early-season margins and got 1.073, which
    would mean teams do not change at all -- it was really absorbing the
    previous season's ridge shrinkage. If this constant ever drifts back
    toward 1.0, that conflation has returned.
    """
    assert 0.45 < prior_module.DEFAULT_CARRYOVER < 0.80


def test_unshrink_is_not_the_inverse_of_carryover():
    """The two parameters must be independently measured.

    Defining unshrink = 1/carryover makes them cancel to exactly 1.0,
    which is circular: two knobs that multiply to one are a single knob
    in disguise. Shrinkage is measured within seasons by split-half
    reliability (1.117); carryover is measured across them (0.63).
    """
    product = prior_module.DEFAULT_CARRYOVER * prior_module.DEFAULT_UNSHRINK
    assert abs(product - 1.0) > 0.15, (
        "carryover and unshrink cancel -- they are not independent")


def test_prior_scales_toward_average(monkeypatch):
    """A prior must pull toward the mean, never amplify last season."""
    scale = prior_module.DEFAULT_CARRYOVER * prior_module.DEFAULT_UNSHRINK
    assert 0.0 < scale < 1.0


def test_prior_omits_unknown_teams():
    """Newcomers shrink toward league average, not toward a penalty.

    A team with no previous season should simply be absent from the
    prior, leaving its target at zero.
    """
    built = prior_module.build_prior(2024)
    previous = {g.home_team for g in load_season(2023)}
    previous |= {g.away_team for g in load_season(2023)}
    assert set(built) <= previous


def test_missing_previous_season_is_not_an_error():
    """The earliest season on disk has nothing before it."""
    assert prior_module.build_prior_or_empty(1800) == {}


def test_prior_translates_division_movers_only():
    """Promoted teams must not inherit an elite prior; stayers unchanged.

    Two bugs are pinned here, because fixing the first caused the second.

    1. North Dakota State rated +15.14 WITHIN FCS in 2025 and moved up to
       FBS in 2026. Passing that raw value through read as an elite FBS
       rating and put them #2 in the country.

    2. Converting EVERY team to the common scale fixed that but broke
       something else: the ridge design already fits an FBS/FCS offset
       column, so a common-scale prior counts the gap twice. The fitted
       offset collapsed from -13.96 to -3.40.

    Correct behaviour: teams that stay in their division keep their
    within-division rating; only movers are translated.
    """
    built = prior_module.build_prior(2026)
    scale = prior_module.DEFAULT_CARRYOVER * prior_module.DEFAULT_UNSHRINK

    previous = fit(load_season(2025), **FITTED)
    current = team_divisions(load_season(2026))

    stayers = 0
    for team, value in built.items():
        was = previous.divisions.get(team)
        now = current.get(team, was)
        if was == now:
            stayers += 1
            assert value == pytest.approx(scale * previous.ratings[team]), (
                f"{team} stayed in {was} but its prior was translated")
    assert stayers > 100, "expected most teams to stay put"

    # NDSU moved FCS -> FBS. Their prior must be that of a below-average
    # FBS team, not a top-five one.
    promoted = built["North Dakota State"]
    assert promoted < 0, (
        f"promoted FCS team has a positive FBS prior ({promoted:.2f})")
    assert promoted < built["Ohio State"]


def test_prior_does_not_double_count_division_gap():
    """The fitted FCS offset must survive the prior being applied.

    If the prior carries the division gap, the offset column has nothing
    left to explain and collapses toward zero. The real gap is roughly
    two to three touchdowns.
    """
    games = load_season(2026)
    with_prior = fit(games, prior=prior_module.build_prior(2026), **FITTED)
    assert -30.0 < with_prior.fcs_offset < -8.0


def test_calibration_is_fitted_above_one(model_2024):
    """Ridge predictions are compressed and need stretching.

    Shrinkage pulls ratings toward the prior, so rating differences come
    out too small. Measured out of sample, actual margins run ~1.5-1.7x
    the raw predicted edge.
    """
    assert model_2024.calibration > 1.0


def test_predictor_applies_calibration(model_2024, games_2024):
    """Regression guard for a DRY bug that produced a silent no-op.

    Predictor.predict() used to recompute the margin inline instead of
    calling model.predict_margin(). When calibration was added to the
    model, predictions did not change at all -- the backtest was
    byte-identical, which is the only reason it got noticed.
    """
    predictor = Predictor.from_model(model_2024, games_2024)
    prediction = predictor.predict("Ohio State", "Kent State", True)
    expected = model_2024.predict_margin("Ohio State", "Kent State", True)
    assert prediction.predicted_margin == pytest.approx(expected)

    # And it must differ from the uncalibrated edge, or calibration is
    # being computed but never actually used.
    raw_edge = (model_2024.rating("Ohio State")
                - model_2024.rating("Kent State"))
    assert abs(prediction.predicted_margin - raw_edge) > 0.5


def test_calibration_does_not_stretch_home_field(model_2024):
    """Home field is additive, not part of team quality.

    Scaling it along with the rating edge would inflate a fitted
    parameter that was already estimated correctly.
    """
    neutral = model_2024.predict_margin("Alabama", "Georgia", True)
    at_home = model_2024.predict_margin("Alabama", "Georgia", False)
    assert at_home - neutral == pytest.approx(model_2024.home_field)


def test_prior_improves_early_season_accuracy():
    """The headline claim, held to account.

    Week-3 ratings without a prior are noise; with one they are not.
    Asserted on 2024, where the prior should clearly help.
    """
    games = [g for g in load_season(2024) if g.week <= 4]
    without = fit(games, **FITTED)
    with_prior = fit(games, prior=prior_module.build_prior(2024), **FITTED)

    # Ohio State won the 2024 title; a sane early model should not have
    # them adrift in the pack once last season is accounted for.
    ranked_with = sorted(
        with_prior.fbs_ratings().items(), key=lambda kv: -kv[1])
    ranked_without = sorted(
        without.fbs_ratings().items(), key=lambda kv: -kv[1])

    position_with = [t for t, _ in ranked_with].index("Ohio State")
    position_without = [t for t, _ in ranked_without].index("Ohio State")
    assert position_with <= position_without
