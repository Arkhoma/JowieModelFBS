"""Tests for cfbrank.season_record and the This Week performance record."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from app.scenario_view import record_view  # noqa: E402
from cfbrank import season_record  # noqa: E402
from cfbrank.scorecard import grade  # noqa: E402


def _frame(weeks: dict[int, int], postseason: int = 0, unlined_weeks=(),
           seed: int = 1) -> pd.DataFrame:
    """Walk-forward-shaped picks: `weeks` maps week -> number of games.
    Games in `unlined_weeks` have no closing line (NaN), like a patchy season."""
    rng = np.random.default_rng(seed)
    rows = []
    for week, n in weeks.items():
        rows += [("regular", week)] * n
    rows += [("postseason", 1)] * postseason
    n = len(rows)
    line = rng.normal(0, 10, n)
    frame = pd.DataFrame({
        "season": 2026,
        "season_type": [r[0] for r in rows],
        "week": [r[1] for r in rows],
        "ours": line + rng.normal(0, 2, n),
        "line": line,
        "actual": line + rng.normal(0, 12, n),
        "home_win_prob": rng.uniform(0.1, 0.9, n),
    })
    frame.loc[frame.week.isin(unlined_weeks) & (frame.season_type == "regular"), "line"] = np.nan
    return frame


# --- weekly_record ---------------------------------------------------------

def test_weeks_with_too_few_games_are_left_out_of_the_table_but_not_the_total():
    record = season_record.weekly_record(_frame({4: 8, 5: 8, 6: 2}))
    assert [w["label"] for w in record["weeks"]] == ["Week 4", "Week 5"]
    assert record["overall"]["n"] == 18          # week 6's two games still count
    assert record["through"] == 6


def test_postseason_games_are_pooled_into_one_row_after_the_regular_weeks():
    record = season_record.weekly_record(_frame({4: 6, 5: 6}, postseason=7))
    assert [w["label"] for w in record["weeks"]] == ["Week 4", "Week 5", "Postseason"]
    assert record["weeks"][-1]["n"] == 7
    assert record["through"] == 5                # bowls don't move "through week N"


def test_the_line_comparison_is_exactly_the_about_pages_grade():
    """One definition of accuracy: scorecard.grade, so pages can't disagree."""
    frame = _frame({4: 10, 5: 10, 6: 10})
    assert season_record.weekly_record(frame)["overall"]["vs_line"] == grade(frame)


def test_unlined_games_count_for_winners_but_not_for_the_line():
    """Weeks 6-7 have no lines, as in a patchy season. Picking winners needs
    no line, so those games still count there; the line comparison must not
    quietly be built from only the lucky weeks that have one."""
    frame = _frame({4: 10, 5: 10, 6: 10, 7: 10}, unlined_weeks=(6, 7))
    record = season_record.weekly_record(frame)
    o = record["overall"]
    assert (o["n"], o["lined"]) == (40, 20)
    assert o["su"] == pytest.approx(((frame.ours > 0) == (frame.actual > 0)).mean())
    assert o["vs_line"] == grade(frame[frame.line.notna()])
    by_week = {w["week"]: w for w in record["weeks"]}
    assert by_week[6]["n"] == 10 and by_week[6]["lined"] == 0
    assert by_week[6]["vs_line"] is None         # nothing to compare, but the week is shown
    assert 0 <= by_week[6]["su"] <= 1
    assert by_week[4]["vs_line"] is not None


def test_a_few_lined_games_are_not_enough_to_compare_against():
    frame = _frame({4: 20})
    keep = season_record.MIN_LINED_GAMES - 1
    frame.loc[frame.index[keep:], "line"] = np.nan          # only 9 games keep a line
    record = season_record.weekly_record(frame)
    assert record["overall"]["lined"] == keep
    assert record["overall"]["vs_line"] is None             # 9 games: one blowout decides it
    assert record["weeks"][0]["su"] > 0                      # winners are still graded


def test_no_games_means_no_record():
    assert season_record.weekly_record(_frame({})) is None


# --- the verdict and caveat are honest -------------------------------------

def _record(gap, ci, lined=100, n=100, early=0.0):
    line = {"gap": gap, "gap_ci": ci} if gap is not None else None
    return {"overall": {"vs_line": line, "lined": lined, "n": n, "early_lined": early,
                        "su": 0.7}, "weeks": [], "through": 9}


@pytest.mark.parametrize("gap, ci, expected", [
    (0.10, 0.30, "a statistical tie"),           # inside its own interval
    (-0.10, 0.30, "a statistical tie"),
    (0.50, 0.20, "the closing line is ahead by 0.50 points a game"),
    (-0.50, 0.20, "we are ahead of the closing line by 0.50 points a game"),
])
def test_verdict_calls_a_gap_inside_its_interval_a_tie(gap, ci, expected):
    assert expected in record_view(_record(gap, ci))["verdict"]


def test_full_coverage_has_no_caveat():
    assert record_view(_record(0.5, 0.2, lined=90, n=100))["caveat"] is None


def test_thin_coverage_is_flagged_and_early_season_skew_is_named():
    skewed = record_view(_record(0.9, 0.7, lined=30, n=100, early=0.8))["caveat"]
    assert "Only 30 of 100 games have a closing line" in skewed
    assert "early-season" in skewed
    even = record_view(_record(0.9, 0.7, lined=30, n=100, early=0.2))["caveat"]
    assert "Only 30 of 100" in even and "early-season" not in even


def test_with_nothing_to_compare_there_is_no_verdict_and_no_caveat():
    view = record_view(_record(None, None, lined=2, n=100))
    assert view["verdict"] is None and view["caveat"] is None


def test_no_record_gives_no_view():
    assert record_view(None) is None


def test_weeks_without_a_line_render_as_dashes_in_every_column_not_blank_cells():
    import re
    from app.pages import base_context
    from build_static_site import STATIC_URLS, _env
    record = record_view(season_record.weekly_record(
        _frame({4: 12, 5: 12, 6: 12}, unlined_weeks=(6,))))
    html = _env().get_template("slate.html").render(
        base_context("slate", 2026, urls=STATIC_URLS, slate=None, record=record))
    week6 = re.search(r"Week 6</th>(.*?)</tr>", html, re.S).group(1)
    assert week6.count("&mdash;") == 3 and "colspan" not in week6   # phone hides one column
    assert f"fewer than {record['min_lined']} games" in " ".join(html.split())   # ignore line breaks


# --- guards against the three copies of the tuning constants drifting ------

def test_production_constants_match_the_app_and_the_experiment_tools():
    import blend_epa_margin as tools
    from app.main import FITTED_PARAMS
    assert season_record.MARGIN_PARAMS == FITTED_PARAMS == tools.MARGIN_PARAMS
    assert season_record.RELATIVE_LAMBDA == tools.RELATIVE_LAMBDA
    assert season_record.START_WEEK == tools.START_WEEK
    assert season_record.MIN_TRAINING == tools.MIN_TRAINING


def test_the_benchmark_tool_still_honors_experiment_tools_patching(monkeypatch):
    """run_variant.py and tune_prior_strength.py re-assign these module
    globals to test a variant; the wrapper must read them at call time."""
    import benchmark_vs_market as bench
    seen = {}
    monkeypatch.setattr(season_record, "walk_season",
                        lambda season, market, totals, **kw: seen.update(kw))
    monkeypatch.setattr(bench, "MARGIN_PARAMS", {"lambda_": 99.0})
    monkeypatch.setattr(bench, "RELATIVE_LAMBDA", 7.5)
    bench.walk_season(2025, {}, {})
    assert seen == {"margin_params": {"lambda_": 99.0}, "relative_lambda": 7.5}
