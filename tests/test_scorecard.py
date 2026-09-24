"""Scorecard: one grading path for the CLI benchmark and the web page."""

from __future__ import annotations

import pandas as pd
import pytest

from cfbrank.scorecard import grade, load_scorecard, scorecard


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2024, 2024, 2025, 2025],
        "week": [4, 10, 5, 12],
        "ours": [7.0, -3.0, 10.0, 1.0],
        "line": [3.0, -1.0, 14.0, 2.0],
        "actual": [10.0, 4.0, 14.0, -3.0],
        "home_win_prob": [0.7, 0.4, 0.8, 0.55],
        "ours_total": [50.0, 60.0, 45.0, 55.0],
        "total_line": [52.0, None, 44.0, 57.0],
        "actual_total": [48.0, 62.0, 41.0, 60.0],
    })


def test_grade_matches_hand_computation():
    g = grade(_frame())
    assert g["mae"] == pytest.approx((3 + 7 + 4 + 4) / 4)
    assert g["line_mae"] == pytest.approx((7 + 5 + 0 + 5) / 4)
    assert g["gap"] == pytest.approx(g["mae"] - g["line_mae"])
    assert g["su"] == pytest.approx(2 / 4)
    assert g["brier"] == pytest.approx(
        (0.09 + 0.36 + 0.04 + 0.3025) / 4)


def test_scorecard_splits_and_totals():
    card = scorecard(_frame())
    assert card["early"]["n"] + card["late"]["n"] == 4
    assert set(card["per_season"]) == {2024, 2025}
    assert card["totals"]["n"] == 3          # the game with no O/U is skipped
    assert card["totals"]["mae"] == pytest.approx((2 + 4 + 5) / 3)


def test_missing_benchmark_is_none_not_a_crash(tmp_path):
    assert load_scorecard(tmp_path / "nope.csv") is None
