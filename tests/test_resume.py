"""Resume rating tests.

The headline cases are Howie's, from week 3 of 2026, and they are the
reason this module exists. They are pinned here so that a future change
that reintroduces the inversion fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfbrank.resume import (
    BENCHMARK_RANK, ResumeGame, benchmark_rating, build_resumes, rank_resumes,
)
from cfbrank.games import Game


def make_game(home, away, home_points, away_points, week=1, neutral=False):
    return Game(
        game_id=f"{home}-{away}-{week}", season=2026, week=week,
        season_type="regular", kickoff=None,
        home_team=home, away_team=away,
        home_division="fbs", away_division="fbs",
        home_points=home_points, away_points=away_points,
        neutral_site=neutral, conference_game=False,
        home_conference="X", away_conference="Y",
    )


class FakeModel:
    """Minimal stand-in for a fitted rating model."""

    def __init__(self, ratings, home_field=2.0):
        self._ratings = ratings
        self.home_field = home_field

    def rating(self, team):
        return self._ratings.get(team, 0.0)

    def fbs_ratings(self):
        return dict(self._ratings)


ERRORS = np.linspace(-30, 30, 601)


class TestCredit:
    def test_credit_is_result_minus_expectation(self):
        game = ResumeGame("Foe", 1, won=True, win_probability=0.75)
        assert game.credit == pytest.approx(0.25)

    def test_losing_to_an_elite_team_costs_little(self):
        """The benchmark was unlikely to win either, so it barely stings."""
        game = ResumeGame("Elite", 1, won=False, win_probability=0.10)
        assert game.credit == pytest.approx(-0.10)

    def test_losing_to_a_cupcake_is_a_disaster(self):
        game = ResumeGame("Cupcake", 1, won=False, win_probability=0.97)
        assert game.credit == pytest.approx(-0.97)

    def test_beating_a_cupcake_earns_almost_nothing(self):
        game = ResumeGame("Cupcake", 1, won=True, win_probability=0.97)
        assert game.credit == pytest.approx(0.03)


class TestBenchmark:
    def test_benchmark_is_the_nth_best_team(self):
        model = FakeModel({f"T{i}": float(100 - i) for i in range(50)})
        # 25th best of a descending list starting at 100.
        assert benchmark_rating(model, 25) == pytest.approx(76.0)

    def test_benchmark_defaults_to_top_25(self):
        assert BENCHMARK_RANK == 25

    def test_empty_model_is_an_error_not_a_silent_zero(self):
        with pytest.raises(ValueError):
            benchmark_rating(FakeModel({}))


class TestHeadToHead:
    """The Ole Miss / LSU case, reduced to its essentials."""

    def test_beating_a_peer_outranks_losing_to_them(self):
        model = FakeModel({"Winner": 5.0, "Loser": 7.0, "Filler": 0.0})
        games = [
            make_game("Winner", "Loser", 32, 24, week=3),
            make_game("Winner", "Filler", 41, 9, week=2),
            make_game("Loser", "Filler", 45, 14, week=2),
        ]
        resumes = build_resumes(games, model, ERRORS)
        assert resumes["Winner"].score > resumes["Loser"].score

    def test_margin_is_irrelevant_to_resume(self):
        """A one-point win and a fifty-point win are worth the same."""
        model = FakeModel({"A": 5.0, "B": 5.0})
        narrow = build_resumes([make_game("A", "B", 21, 20)], model, ERRORS)
        blowout = build_resumes([make_game("A", "B", 70, 20)], model, ERRORS)
        assert narrow["A"].score == pytest.approx(blowout["A"].score)


class TestBadLossesStillHurt:
    """The Oklahoma State case: beating a good team does NOT wipe out
    losing to a bad one."""

    def test_good_win_plus_bad_loss_ranks_below_clean_record(self):
        model = FakeModel({"Messy": 0.0, "Clean": 0.0,
                           "Good": 8.0, "Bad": -8.0})
        games = [
            make_game("Messy", "Good", 39, 31, week=2),
            make_game("Bad", "Messy", 24, 10, week=1),
            make_game("Clean", "Good", 20, 17, week=2),
            make_game("Clean", "Bad", 30, 10, week=1),
        ]
        resumes = build_resumes(games, model, ERRORS)
        assert resumes["Clean"].score > resumes["Messy"].score


class TestReporting:
    def test_best_win_is_the_least_likely_one(self):
        model = FakeModel({"A": 0.0, "Tough": 20.0, "Easy": -20.0})
        games = [
            make_game("A", "Tough", 21, 20, week=1),
            make_game("A", "Easy", 50, 0, week=2),
        ]
        resume = build_resumes(games, model, ERRORS)["A"]
        assert resume.best_win.opponent == "Tough"
        assert resume.worst_loss is None

    def test_best_win_tie_goes_to_stronger_opponent_not_earlier_week(self):
        """Ole Miss 2026: Louisville (neutral, wk1) and LSU (home, wk3)
        were equally hard for the benchmark. LSU is the better team."""
        model = FakeModel({"A": 0.0, "Good": 6.0, "Better": 10.0},
                          home_field=4.0)
        # Better at home: 10 - 4 = 6 edge; Good neutral: 6 edge. Equal.
        games = [
            make_game("A", "Good", 30, 20, week=1, neutral=True),
            make_game("A", "Better", 30, 20, week=3),
        ]
        resume = build_resumes(games, model, ERRORS)["A"]
        assert resume.best_win.opponent == "Better"

    def test_win_probabilities_ignore_residual_bias(self):
        """An even game must be ~50/50 even if the model's residuals
        happen to be off-centre -- Predictor centres them."""
        from cfbrank.predict import Predictor

        class Biased(FakeModel):
            ratings = {"A": 0.0, "B": 0.0}

            def predict_margin(self, home, away, neutral_site=False):
                return 0.0

        games = [make_game("A", "B", 24 + i % 9, 20, week=i + 1)
                 for i in range(40)]
        predictor = Predictor.from_model(Biased({"A": 0.0, "B": 0.0}),
                                         games)
        assert abs(predictor.errors.mean()) < 1e-9

    def test_record_is_counted_correctly(self):
        model = FakeModel({"A": 0.0, "B": 0.0, "C": 0.0})
        games = [
            make_game("A", "B", 21, 20, week=1),
            make_game("C", "A", 30, 10, week=2),
        ]
        resume = build_resumes(games, model, ERRORS)["A"]
        assert (resume.wins, resume.losses) == (1, 1)

    def test_ranking_excludes_non_fbs_teams(self):
        model = FakeModel({"A": 0.0})  # only A is FBS
        games = [make_game("A", "NotRated", 21, 20)]
        resumes = build_resumes(games, model, ERRORS)
        ranked = rank_resumes(resumes, model)
        assert [r.team for _, r in ranked] == ["A"]
