"""Tests for the team page's "Why #N" lines (cfbrank.explain).

Fakes, not a real season fit: these check the wording rules and the
rank arithmetic, which don't need 30 seconds of ridge regression.
"""

from __future__ import annotations

from types import SimpleNamespace

from cfbrank.explain import SPLIT_GAP, explain_rankings, game_swings


class FakeRatings(SimpleNamespace):
    def rating(self, team):
        return self.ratings[team]

    def rating_in_points(self, team):
        return self.ratings[team]


class FakePredictor:
    """Predicts the home team by a fixed margin, every game."""

    def __init__(self, margin=0.0):
        self.margin = margin

    def predict(self, home, away, neutral):
        return SimpleNamespace(predicted_margin=self.margin)


def game(home, away, hp, ap, neutral=False):
    return SimpleNamespace(home_team=home, away_team=away, home_points=hp,
                           away_points=ap, neutral_site=neutral)


def rows(*teams, off=None, deff=None):
    off = off or {}
    deff = deff or {}
    return [{"team": t, "rank": i, "sos": -i,
             "offense_rank": off.get(t, i), "defense_rank": deff.get(t, i)}
            for i, t in enumerate(teams, 1)]


def model(scores, plays):
    return SimpleNamespace(margin_model=FakeRatings(ratings=scores),
                           epa_model=FakeRatings(ratings=plays))


def why(team, table, mdl=None, prior=None, games=(), predictor=None):
    mdl = mdl or SimpleNamespace()
    return explain_rankings(table, mdl, prior or {}, list(games),
                            predictor or FakePredictor())[team]


def test_preseason_movement_counts_spots_in_the_right_direction():
    table = rows(*"ABCDEFGHIJ")
    prior = {t: -i for i, t in enumerate("JABCDEFGHI")}  # J was #1 preseason
    assert "moved them down 9 spots" in why("J", table, prior=prior)[0]
    assert "and the games have backed that up" in why("A", table, prior=prior)[0]


def test_luck_flag_only_when_results_outrun_plays_by_the_threshold():
    teams = [f"T{i}" for i in range(SPLIT_GAP + 2)]
    scores = {t: -i for i, t in enumerate(teams)}           # T0 #1 on scores
    plays = {t: -i for i, t in enumerate(teams[1:] + teams[:1])}  # T0 last
    lines = why("T0", rows(*teams), mdl=model(scores, plays))
    assert any("may be luck" in line for line in lines)
    # T1 is #2 on scores, #1 on plays: they agree.
    lines = why("T1", rows(*teams), mdl=model(scores, plays))
    assert any("agree (#2 and #1)" in line for line in lines)


def test_single_model_skips_the_split_line():
    lines = why("A", rows("A", "B"))
    assert not any("per-play" in line for line in lines)


def test_offense_defense_lean():
    table = rows("A", "B", off={"A": 1}, deff={"A": 1 + SPLIT_GAP})
    assert "Built on offense (#1) more than defense" in " ".join(why("A", table))
    plain = rows("A", "B")
    assert "Offense #1, defense #1." in why("A", plain)


def test_game_swings_use_the_team_perspective():
    # Model says home by 7 every game. A wins at home by 20 (+13 vs the
    # number), loses away by 3 (it was expected to lose by 7: +4), and
    # wins on a neutral field by 1 (expected home-listed A by 7: -6).
    games = [game("A", "B", 27, 7), game("C", "A", 24, 21),
             game("A", "D", 21, 20, neutral=True)]
    swings = game_swings("A", games, FakePredictor(margin=7))
    assert [round(s.vs_model) for s in swings] == [13, 4, -6]
    assert swings[0].where == "vs B" and swings[1].where == "at C"
    assert swings[2].where == "vs D"  # neutral reads "vs", never "at"


def test_best_and_worst_game_lines_need_two_games():
    games = [game("A", "B", 27, 7), game("C", "A", 24, 21)]
    lines = why("A", rows("A", "B", "C"), games=games,
                predictor=FakePredictor(margin=7))
    assert any(line.startswith("Best game: W 27-7 vs B, 13 pts") for line in lines)
    assert not any(line.startswith("Worst game") for line in lines)  # both beat it
    one = why("A", rows("A", "B"), games=games[:1])
    assert not any(line.startswith("Best game") for line in one)


def test_schedule_line_uses_ordinals():
    table = rows(*[f"T{i}" for i in range(23)])
    assert why("T0", table)[-1] == "Toughest schedule in FBS."
    assert why("T1", table)[-1] == "2nd-toughest schedule in FBS."
    assert why("T10", table)[-1] == "11th-toughest schedule in FBS."
    assert why("T22", table)[-1] == "23rd-toughest schedule in FBS."
