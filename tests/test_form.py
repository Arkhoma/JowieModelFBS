"""Tests for cfbrank.form and app.form_view: flagging and chart geometry.

Synthetic data throughout -- the real-data render check lives in
test_pages.py. What these pin: a step change is found where it happened,
steady noise is left alone, a defense's miss is the opposing offense's
miss with the sign flipped, and the chart can never put a bar outside
its frame or call an ordinary team "moving".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.form_view import (  # noqa: E402
    HEIGHT, MIN_GAMES, WIDTH, describe_shift, team_form_context,
)
from cfbrank.form import (  # noqa: E402
    FLAG_Z, MIN_SEGMENT, FormGame, SeasonForm, Shift, TeamForm,
    _team_shifts, add_defense_misses, scan_split, season_form,
)

SIGMA = 10.0


def test_scan_split_finds_a_step_where_it_happened():
    x = np.array([-10.0] * 5 + [10.0] * 5)
    z, at = scan_split(x, SIGMA)
    assert at == 5
    assert z > 0
    # +20 swing, 5 v 5 games: 20 / (10 * sqrt(2/5)) = 3.16
    assert abs(z - 20 / (SIGMA * np.sqrt(0.4))) < 1e-9


def test_scan_split_sign_follows_direction():
    assert scan_split(np.array([8.0] * 4 + [-8.0] * 4), SIGMA)[0] < 0


def test_scan_split_flat_series_is_zero():
    assert scan_split(np.zeros(8), SIGMA) == (0.0, MIN_SEGMENT)


def test_scan_split_never_leaves_a_stub_segment():
    """A break after game 1 would be one lucky game, not a shift."""
    x = np.array([50.0] + [0.0] * 9)
    _, at = scan_split(x, SIGMA)
    assert MIN_SEGMENT <= at <= len(x) - MIN_SEGMENT


def test_defense_miss_is_the_opponents_offense_miss_flipped():
    frame = pd.DataFrame([
        {"game_id": "1", "team": "A", "opponent": "B", "miss": 7.0},
        {"game_id": "1", "team": "B", "opponent": "A", "miss": -3.0},
        {"game_id": "2", "team": "A", "opponent": "C", "miss": 1.0},  # C unseen
    ])
    out = add_defense_misses(frame).set_index(["game_id", "team"])
    assert out.loc[("1", "A"), "def_miss"] == 3.0     # held B 3 below expectation
    assert out.loc[("1", "B"), "def_miss"] == -7.0    # let A 7 above
    assert np.isnan(out.loc[("2", "A"), "def_miss"])


def _grouped(offense_misses, kind="points"):
    n = len(offense_misses)
    frame = pd.DataFrame({
        "week": range(3, 3 + n), "miss": offense_misses,
        "def_miss": np.zeros(n)})
    return {kind: {"T": frame}}


SIGMAS = {(k, s): SIGMA for k in ("points", "success")
          for s in ("offense", "defense")}


def test_a_big_sustained_break_is_flagged_with_its_week():
    shifts = _team_shifts("T", _grouped([-15.0] * 5 + [15.0] * 5), SIGMAS)
    assert len(shifts) == 1
    s = shifts[0]
    assert (s.side, s.kind) == ("offense", "points")
    assert s.from_week == 8 and s.before == -15.0 and s.after == 15.0
    assert s.z >= FLAG_Z


def test_ordinary_noise_is_not_flagged():
    rng = np.random.default_rng(0)
    flagged = sum(bool(_team_shifts("T", _grouped(list(rng.normal(0, SIGMA, 8))),
                                    SIGMAS)) for _ in range(500))
    assert flagged / 500 < 0.03          # z >= 3 is rare by construction


def test_too_few_games_never_flag():
    assert _team_shifts("T", _grouped([-50.0] * 2 + [50.0] * 3), SIGMAS) == []


def test_seasons_before_the_priors_exist_have_no_form():
    assert season_form(2017) is None


# --- chart ---------------------------------------------------------------

def _game(week, off, dfn=0.0, off_success=None, def_success=None):
    return FormGame(week=week, opponent=f"Opp{week}", scored=24.0,
                    expected=24.0 - off, off_miss=off, def_miss=dfn,
                    off_success=off_success, def_success=def_success)


def _form(games, shifts=()):
    sigma = {("points", "offense"): 11.5, ("points", "defense"): 11.4,
             ("success", "offense"): 0.079, ("success", "defense"): 0.079}
    return SeasonForm(sigma=sigma, teams={"T": TeamForm(tuple(games), tuple(shifts))})


def test_hidden_without_enough_games_or_data():
    assert team_form_context(None, "T") is None
    assert team_form_context(_form([_game(w, 1.0) for w in range(MIN_GAMES - 1)]), "T") is None
    assert team_form_context(_form([_game(w, 1.0) for w in range(6)]), "Nobody") is None


def test_steady_team_gets_the_steady_headline_and_no_bullets():
    ctx = team_form_context(_form([_game(w, 3.0) for w in range(3, 9)]), "T")
    assert not ctx["moving"]
    assert ctx["shifts"] == []
    assert "Nothing here" in ctx["headline"]


def test_noise_is_quoted_in_both_units_when_success_rate_exists():
    ctx = team_form_context(_form([_game(w, 1.0) for w in range(3, 9)]), "T")
    assert ctx["noise"] == "12"            # 11.5 pts
    assert ctx["noise_success"] == "8"      # 0.079 -> 7.9 success-rate pts


def test_noise_omits_success_rate_for_a_points_only_season():
    form = _form([_game(w, 1.0) for w in range(3, 9)])
    points_only = SeasonForm(
        sigma={k: v for k, v in form.sigma.items() if k[0] == "points"},
        teams=form.teams)
    assert team_form_context(points_only, "T")["noise_success"] is None


def test_flagged_team_gets_the_moving_headline_and_a_sentence():
    shift = Shift("offense", "success", 3.9, 7, -0.074, 0.142)
    games = [_game(w, 3.0, off_success=-0.07 if w < 7 else 0.14) for w in range(3, 9)]
    ctx = team_form_context(_form(games, [shift]), "T")
    assert ctx["moving"]
    assert "Worth a look" in ctx["headline"]
    assert ctx["shifts"] == [describe_shift(shift)]
    assert "-7.4 success-rate pts" in ctx["shifts"][0]
    assert "+14.2 success-rate pts" in ctx["shifts"][0]
    assert "week 7" in ctx["shifts"][0]


def test_bars_stay_in_frame_and_sit_on_the_correct_side_of_zero():
    games = [_game(3, 60.0, -45.0), _game(4, -60.0, 45.0)] + [
        _game(w, 0.5) for w in range(5, 9)]
    for chart in team_form_context(_form(games), "T")["charts"]:
        for bar in chart["bars"]:
            assert 0 <= bar["y"] and bar["y"] + bar["h"] <= HEIGHT
            assert 0 <= bar["x"] and bar["x"] + bar["w"] <= WIDTH
            if bar["up"]:
                assert bar["y"] + bar["h"] <= chart["zero_y"] + 0.1
            else:
                assert bar["y"] >= chart["zero_y"] - 0.1


def test_both_charts_share_one_scale_so_they_compare_by_eye():
    ctx = team_form_context(_form([_game(w, 40.0, 1.0) for w in range(3, 9)]), "T")
    offense, defense = ctx["charts"]
    assert offense["ticks"] == defense["ticks"]
    assert offense["zero_y"] == defense["zero_y"]


def test_a_missing_defense_miss_leaves_a_gap_not_a_zero():
    games = [_game(w, 2.0) for w in range(3, 9)]
    games[2] = FormGame(5, "Opp5", 24.0, 22.0, 2.0, None)
    offense, defense = team_form_context(_form(games), "T")["charts"]
    assert len(offense["bars"]) == 6
    assert len(defense["bars"]) == 5
    assert "over 5 games" in defense["summary"]


def test_summary_reports_the_season_total_and_per_game_average():
    ctx = team_form_context(_form([_game(w, -2.0) for w in range(3, 7)]), "T")
    assert "-8.0 pts vs. expectation over 4 games (-2.0 per game)" in ctx["charts"][0]["summary"]


# --- charting the statistic that tripped the flag ---------------------------

def _wsu_like(shift_week=7):
    """Points look like noise; success rate has a clear break at `shift_week`."""
    games = [_game(w, miss, 0.0, off_success=succ, def_success=0.0)
             for w, miss, succ in [(3, 4.0, -0.10), (4, -5.0, -0.08), (5, 3.0, -0.06),
                                   (6, -2.0, -0.07), (7, 5.0, 0.12), (8, -4.0, 0.15),
                                   (9, 2.0, 0.14), (11, 6.0, 0.16)]]
    return _form(games, [Shift("offense", "success", 3.9, shift_week, -0.078, 0.143)])


def test_a_flagged_side_is_charted_in_the_statistic_that_flagged_it():
    offense, defense = team_form_context(_wsu_like(), "T")["charts"]
    assert offense["stat"] == "success rate"
    assert defense["stat"] == "points"           # nothing flagged: the default
    assert "success-rate pts" in offense["summary"]
    assert "success-rate pts" in offense["bars"][0]["tip"]
    assert "pts vs." in defense["summary"] and "success" not in defense["summary"]


def test_success_bars_are_scaled_to_percentage_points():
    offense, _ = team_form_context(_wsu_like(), "T")["charts"]
    assert "-10.0 success-rate pts" in offense["bars"][0]["tip"]   # stored as -0.10
    assert "+16.0 success-rate pts" in offense["bars"][-1]["tip"]


def test_only_the_flagged_chart_gets_a_break_marker_at_the_right_game():
    offense, defense = team_form_context(_wsu_like(), "T")["charts"]
    assert defense["marker"] is None
    marker = offense["marker"]
    assert marker["label"] == "from wk 7"
    before, after = offense["bars"][3], offense["bars"][4]    # wk 6 | wk 7
    assert before["x"] + before["w"] < marker["x"] < after["x"]
    assert marker["y1"] < marker["y2"] <= HEIGHT


def test_no_marker_when_the_break_is_at_or_before_the_first_game():
    assert team_form_context(_wsu_like(shift_week=3), "T")["charts"][0]["marker"] is None
    assert team_form_context(_wsu_like(shift_week=1), "T")["charts"][0]["marker"] is None


def test_break_label_flips_left_near_the_right_edge():
    late = team_form_context(_wsu_like(shift_week=11), "T")["charts"][0]["marker"]
    early = team_form_context(_wsu_like(shift_week=5), "T")["charts"][0]["marker"]
    assert late["anchor"] == "end" and late["label_x"] < late["x"]
    assert early["anchor"] == "start" and early["label_x"] > early["x"]


def test_the_strongest_break_on_a_side_picks_the_statistic():
    games = [_game(w, 1.0, off_success=0.01) for w in range(3, 9)]
    weak = Shift("offense", "points", -3.1, 6, 5.0, -5.0)
    strong = Shift("offense", "success", 4.2, 7, -0.08, 0.14)
    offense, _ = team_form_context(_form(games, [weak, strong]), "T")["charts"]
    assert offense["stat"] == "success rate"
    assert offense["marker"]["label"] == "from wk 7"


def test_charts_on_different_statistics_get_their_own_scale():
    games = [_game(w, 30.0, 30.0, off_success=0.05, def_success=0.0) for w in range(3, 9)]
    flagged = _form(games, [Shift("offense", "success", 3.5, 6, -0.05, 0.1)])
    offense, defense = team_form_context(flagged, "T")["charts"]
    assert offense["ticks"][0]["label"] != defense["ticks"][0]["label"]
    # ...and the points chart keeps its own range (30 -> 30 axis).
    assert defense["ticks"][0]["label"] == "+30"


def test_missing_success_values_leave_gaps_in_a_success_chart():
    games = [_game(w, 1.0, off_success=0.05) for w in range(3, 9)]
    games[1] = _game(4, 1.0, off_success=None)
    flagged = _form(games, [Shift("offense", "success", 3.5, 6, -0.05, 0.1)])
    offense, _ = team_form_context(flagged, "T")["charts"]
    assert len(offense["bars"]) == 5


def test_table_rows_carry_success_in_display_units():
    ctx = team_form_context(_wsu_like(), "T")
    assert ctx["has_success"]
    assert ctx["rows"][0]["off_success"] == -10.0
    assert ctx["rows"][0]["off_miss"] == 4.0          # points untouched
    assert not team_form_context(_form([_game(w, 1.0) for w in range(3, 9)]),
                                 "T")["has_success"]
