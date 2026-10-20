"""Rest-of-season projections: final record, bowl odds, conference races.

Nothing here is a second model. Every number is read off the same
Predictor that already drives the Predict page's spreads and win
probabilities -- a projected record is just a team's current record plus
what that predictor expects from the games still on the schedule.

All three outputs (team records, bowl/undefeated odds, conference-race
odds) are read from ONE shared Monte Carlo draw (`simulate_remaining`),
not three separate simulations. That matters: two conference rivals'
remaining games are correlated (they play each other), so projecting
each team independently and then asking "who leads the conference" would
silently assume games are more independent than they are.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .games import Game, ScheduledGame
from .predict import Predictor

BOWL_ELIGIBLE_WINS = 6
DEFAULT_SIMS = 2000
# Fixed by default rather than random: the static site is built once and
# published, so a reader comparing two page loads should see the same
# odds, not simulation noise masquerading as the schedule changing.
DEFAULT_SEED = 20260101


@dataclass(frozen=True, slots=True)
class SimGame:
    """One remaining game plus the model's opinion of it."""

    game: ScheduledGame
    home_win_probability: float


@dataclass(frozen=True, slots=True)
class SeasonSimulation:
    """The shared Monte Carlo draw everything else in this module reads."""

    games: list[SimGame]
    # n_sims x len(games) bool matrix; True = home team won that trial.
    home_wins: np.ndarray
    n_sims: int


@dataclass(slots=True)
class TeamProjection:
    """One team's rest-of-season outlook."""

    team: str
    wins: int
    losses: int
    games_remaining: int
    expected_wins: float
    expected_losses: float
    bowl_eligible_pct: float
    undefeated_pct: float
    most_likely_record: tuple[int, int]

    @property
    def record_so_far(self) -> str:
        return f"{self.wins}-{self.losses}"

    @property
    def projected_record(self) -> str:
        """Most-likely final record, the headline number for the UI."""
        w, l = self.most_likely_record
        return f"{w}-{l}"


def simulate_remaining(
    remaining: list[ScheduledGame], predictor: Predictor,
    n_sims: int = DEFAULT_SIMS, seed: int = DEFAULT_SEED,
) -> SeasonSimulation:
    """Draw `n_sims` possible outcomes for every remaining rated game.

    A game is skipped (not a coin flip) when either side isn't in the
    model's ratings -- an FCS fill-in added mid-season, say. Guessing
    50/50 for a game the model has no opinion on would quietly become an
    assumption baked into every number downstream.
    """
    ratings = predictor.model.ratings
    games = []
    for g in remaining:
        if g.home_team not in ratings or g.away_team not in ratings:
            continue
        prediction = predictor.predict(g.home_team, g.away_team, g.neutral_site)
        games.append(SimGame(g, prediction.home_win_probability))

    rng = np.random.default_rng(seed)
    probs = np.array([sg.home_win_probability for sg in games])
    draws = (rng.random((n_sims, len(games))) < probs) if games else np.empty((n_sims, 0), dtype=bool)
    return SeasonSimulation(games=games, home_wins=draws, n_sims=n_sims)


def _team_game_columns(team: str, sim: SeasonSimulation) -> list[tuple[int, bool]]:
    """(column index, is_home) for every simulated game this team plays."""
    return [
        (i, sg.game.home_team == team)
        for i, sg in enumerate(sim.games)
        if team in (sg.game.home_team, sg.game.away_team)
    ]

def project_team(
    team: str, wins: int, losses: int, sim: SeasonSimulation,
) -> TeamProjection:
    """One team's projection, read off the shared simulation."""
    columns = _team_game_columns(team, sim)
    games_remaining = len(columns)

    if games_remaining == 0:
        return TeamProjection(
            team=team, wins=wins, losses=losses, games_remaining=0,
            expected_wins=float(wins), expected_losses=float(losses),
            bowl_eligible_pct=100.0 if wins >= BOWL_ELIGIBLE_WINS else 0.0,
            undefeated_pct=100.0 if losses == 0 else 0.0,
            most_likely_record=(wins, losses),
        )

    team_won = np.column_stack([
        sim.home_wins[:, i] if is_home else ~sim.home_wins[:, i]
        for i, is_home in columns
    ])
    sim_wins = wins + team_won.sum(axis=1)
    sim_losses = losses + games_remaining - team_won.sum(axis=1)

    counts = Counter(zip(sim_wins.tolist(), sim_losses.tolist()))
    most_likely = counts.most_common(1)[0][0]

    return TeamProjection(
        team=team, wins=wins, losses=losses, games_remaining=games_remaining,
        expected_wins=float(sim_wins.mean()),
        expected_losses=float(sim_losses.mean()),
        bowl_eligible_pct=float(np.mean(sim_wins >= BOWL_ELIGIBLE_WINS)) * 100,
        undefeated_pct=float(np.mean(sim_losses == 0)) * 100,
        most_likely_record=(int(most_likely[0]), int(most_likely[1])),
    )


def project_all(
    teams: list[str], records: dict[str, tuple[int, int]],
    sim: SeasonSimulation,
) -> dict[str, TeamProjection]:
    """Project every team against the one shared simulation."""
    return {
        team: project_team(team, *records.get(team, (0, 0)), sim)
        for team in teams
    }


def _current_conference_wins(played: list[Game]) -> dict[str, int]:
    """In-conference wins so far, from completed conference games only."""
    wins: dict[str, int] = {}
    for g in played:
        if g.conference_game:
            wins[g.winner] = wins.get(g.winner, 0) + 1
    return wins


def project_conference_races(
    conference_of: dict[str, str], played: list[Game], sim: SeasonSimulation,
) -> dict[str, list[tuple[str, float]]]:
    """Per conference: each team's odds of finishing with the most wins
    IN CONFERENCE games.

    This is deliberately not "wins the conference championship" -- actual
    tie-breakers, divisions and title games vary by conference and aren't
    modelled here. "Best conference record" is the honest, calculable
    proxy. Ties split the win credit evenly across the tied teams so each
    conference's odds sum to 100%.
    """
    by_conference: dict[str, list[str]] = {}
    for team, conf in conference_of.items():
        if conf:
            by_conference.setdefault(conf, []).append(team)

    current_wins = _current_conference_wins(played)

    results: dict[str, list[tuple[str, float]]] = {}
    for conf, teams in by_conference.items():
        if len(teams) < 2:
            continue
        conf_wins = np.zeros((sim.n_sims, len(teams)), dtype=float)
        for col, team in enumerate(teams):
            baseline = current_wins.get(team, 0)
            columns = [
                (i, is_home) for i, is_home in _team_game_columns(team, sim)
                if sim.games[i].game.conference_game
                and sim.games[i].game.home_team in teams
                and sim.games[i].game.away_team in teams
            ]
            if columns:
                won = np.column_stack([
                    sim.home_wins[:, i] if is_home else ~sim.home_wins[:, i]
                    for i, is_home in columns
                ])
                conf_wins[:, col] = baseline + won.sum(axis=1)
            else:
                conf_wins[:, col] = baseline

        best = conf_wins.max(axis=1, keepdims=True)
        is_best = conf_wins >= best
        credit = is_best / is_best.sum(axis=1, keepdims=True)
        odds = credit.mean(axis=0) * 100
        results[conf] = sorted(zip(teams, odds.tolist()),
                               key=lambda kv: -kv[1])
    return results
