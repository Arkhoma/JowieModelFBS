"""Resume rating: how hard was it to earn this exact record?

The published table until now was a PREDICTIVE rating -- it answers "who
would win on a neutral field," and it is allowed to rank a team that lost
above the team that beat it, because margin carries information that a
single result does not.

That is the correct answer to the wrong question for a poll. Howie caught
it with Ole Miss and LSU in 2026: Ole Miss went 3-0 including a head-to-head
win over LSU, and still sat seven spots below 2-1 LSU. Contrast that with
Oklahoma State, who also beat a higher-ranked team (Oregon) but who own a
loss to Tulsa -- there the gap is earned. The difference between those two
cases is exactly the difference between predictive and resume.

The method: STRENGTH OF RECORD.

    For each game, ask how likely a benchmark team would have been to win
    that same game, at that same location, against that same opponent.
    Sum those probabilities to get expected wins. Resume score is actual
    wins minus expected wins.

Properties that fall out of the definition rather than being patched in:

- Margin is ignored entirely. A win is a win. This is the whole point of
  keeping it separate from the predictive rating.
- Beating a team you should beat earns near zero. Beating an elite team
  earns a lot.
- Losing to an elite team costs little. Losing to Campbell is a disaster.
- An undefeated team that beat a top-10 opponent cannot fall below a team
  whose only differences are a worse best-win and an extra loss -- the
  arithmetic forbids it. Head-to-head is never special-cased; it is
  respected because beating a team both raises your score and lowers theirs.

The benchmark is NOT a tuned parameter. It is the definition of the
question -- "how would a top-25-caliber team have fared against this
schedule?" -- and it is read straight off the ratings as the 25th-best FBS
team. No hand-set number.

Opponent strength and win probability both come from the already-fitted
predictive model, so this module introduces no new free parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .games import Game
from .predict import _empirical_win_probability

# "Top-25 caliber" -- the rank whose rating defines the benchmark team.
# This is the question being asked, not a knob to tune for nicer output.
BENCHMARK_RANK = 25


@dataclass(frozen=True, slots=True)
class ResumeGame:
    """One game's contribution to a team's resume."""

    opponent: str
    week: int
    won: bool
    win_probability: float
    """Chance the BENCHMARK team would have won this game."""
    opponent_rating: float = 0.0
    """Opponent's rating, used only to break ties between equally hard
    games (e.g. a better team at home vs a lesser one on a neutral site)."""

    @property
    def credit(self) -> float:
        """Result minus expectation. Positive means better than benchmark."""
        return (1.0 if self.won else 0.0) - self.win_probability


@dataclass(slots=True)
class TeamResume:
    """A team's full resume, with every game that produced it."""

    team: str
    wins: int
    losses: int
    expected_wins: float
    games: list[ResumeGame] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Wins above what a top-25-caliber team would have managed."""
        return (self.wins - self.expected_wins)

    @property
    def best_win(self) -> ResumeGame | None:
        """Least likely win -- the marquee result on the resume.

        Ties go to the stronger opponent. Without this, min() silently
        returned the EARLIEST game: Ole Miss 2026 beat Louisville
        (neutral) and LSU (home) at identical benchmark odds, and the
        week-1 game won purely on schedule order.
        """
        wins = [g for g in self.games if g.won]
        return (min(wins, key=lambda g: (g.win_probability,
                                         -g.opponent_rating))
                if wins else None)

    @property
    def worst_loss(self) -> ResumeGame | None:
        """Most likely win that became a loss -- the blemish. Ties go to
        the weaker opponent."""
        losses = [g for g in self.games if not g.won]
        return (max(losses, key=lambda g: (g.win_probability,
                                           -g.opponent_rating))
                if losses else None)


def benchmark_rating(model, rank: int = BENCHMARK_RANK) -> float:
    """Rating of the Nth-best FBS team -- the yardstick every resume is
    measured against.

    Read from the data rather than hard-coded, so the benchmark tracks the
    league instead of drifting out of date.
    """
    ratings = sorted(model.fbs_ratings().values(), reverse=True)
    if not ratings:
        raise ValueError("Model has no FBS ratings; cannot set a benchmark.")
    return ratings[min(rank, len(ratings)) - 1]


def build_resumes(
    games: list[Game],
    model,
    errors: np.ndarray,
    rank: int = BENCHMARK_RANK,
) -> dict[str, TeamResume]:
    """Score every team's resume against the benchmark team.

    `errors` is the model's own historical prediction-error distribution,
    reused from the predictor so win probabilities here and on the predict
    page come from one source.
    """
    benchmark = benchmark_rating(model, rank)
    resumes: dict[str, TeamResume] = {}

    for game in games:
        for team, opponent, is_home in (
            (game.home_team, game.away_team, True),
            (game.away_team, game.home_team, False),
        ):
            # The benchmark team stands in for `team`, playing `opponent`
            # at the same site. Margin never enters.
            edge = benchmark - model.rating(opponent)
            if not game.neutral_site:
                edge += model.home_field if is_home else -model.home_field
            probability = _empirical_win_probability(edge, errors)

            won = game.winner == team
            record = resumes.setdefault(
                team, TeamResume(team=team, wins=0, losses=0,
                                 expected_wins=0.0))
            record.wins += int(won)
            record.losses += int(not won)
            record.expected_wins += probability
            record.games.append(ResumeGame(
                opponent=opponent, week=game.week, won=won,
                win_probability=probability,
                opponent_rating=model.rating(opponent)))

    return resumes


def rank_resumes(
    resumes: dict[str, TeamResume], model
) -> list[tuple[int, TeamResume]]:
    """Rank FBS teams by resume score, best first."""
    fbs = set(model.fbs_ratings())
    ordered = sorted(
        (r for team, r in resumes.items() if team in fbs),
        key=lambda r: -r.score,
    )
    return list(enumerate(ordered, 1))
