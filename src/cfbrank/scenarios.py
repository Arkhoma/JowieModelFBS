"""What-if analysis for the playoff odds, read off the shared simulation.

No new model. The playoff odds come from thousands of simulated seasons,
each with its own game results and its own 12-team field. "What if
Oklahoma wins out?" is therefore just: keep the seasons where it did, and
count how many of *those* had Oklahoma in the field. Every number here is
a filter over the draw `app.main.get_season` already made, so it can never
disagree with the odds printed beside it.

Three questions, one table (`made`: season x team, True = in the field):

    by_final_record   how often a team ends 9-3, and its odds when it does
    team_games        its odds if it wins / loses each remaining game
    game_impacts      which games move the most odds across the whole field
    rooting_guide     which games move ONE team's odds the most

Honesty rules:

* A filter that leaves fewer than MIN_TRIALS seasons is noise, so it
  returns None rather than a confident-looking number.
* With a few thousand seasons, comparing two halves always shows *some*
  difference, even between games that cannot matter. A swing only counts
  when it is NOISE_Z standard errors bigger than sampling alone would
  produce; everything smaller is treated as no effect, so nobody is told
  to root for a game that merely wobbled their average.
* A game's impact is weighted by how uncertain it is. Raw swing would
  crown foregone conclusions (a 97% favourite's loss would flip a lot, but
  almost never happens). The weight is 2*p*(1-p): the expected size of the
  move in a team's odds once the result is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .projection import SeasonSimulation, SimGame, team_game_results

MIN_TRIALS = 30
# A swing must beat sampling noise by this many standard errors to count.
NOISE_Z = 3.0
# Don't call out a swing smaller than this many percentage points.
MIN_SWING = 1.0
MAX_MOVERS = 3


@dataclass(frozen=True, slots=True)
class RecordOutcome:
    """One possible final record for a team."""

    wins: int
    losses: int
    share: float             # % of simulated seasons that end this way
    playoff: float | None    # % of THOSE seasons in the field; None = too few
    n: int


@dataclass(frozen=True, slots=True)
class GameOdds:
    """A team's playoff odds hinging on one of its own remaining games."""

    game: SimGame
    team_win_probability: float
    if_win: float | None
    if_lose: float | None


@dataclass(frozen=True, slots=True)
class Mover:
    """A team whose odds hinge on someone else's result."""

    team: str
    if_home_wins: float
    if_away_wins: float


@dataclass(frozen=True, slots=True)
class GameImpact:
    game: SimGame
    impact: float                    # summed expected move, percentage points
    movers: tuple[Mover, ...]        # biggest swings, largest first


@dataclass(frozen=True, slots=True)
class RootFor:
    """One game worth watching for one team."""

    game: SimGame
    root_for: str                    # the side that helps the team more
    odds_if_it_wins: float
    odds_if_it_loses: float
    swing: float                     # expected move, percentage points


@dataclass(frozen=True, eq=False)
class PlayoffScenarios:
    sim: SeasonSimulation
    teams: list[str]
    made: np.ndarray                 # n_sims x n_teams bool
    _home_odds: np.ndarray           # games x teams: odds if home wins (NaN = too few)
    _away_odds: np.ndarray
    _delta: np.ndarray               # home_odds - away_odds where it beats the noise, else 0
    _weight: np.ndarray              # per game: 2*p*(1-p)

    @classmethod
    def from_fields(cls, sim: SeasonSimulation, fields: list) -> PlayoffScenarios:
        """`fields[i]` is the 12-team field of simulated season i (the same
        draw `sim` came from); each entry needs only a `.team`."""
        if len(fields) != sim.n_sims:
            raise ValueError(f"{len(fields)} fields for {sim.n_sims} simulated seasons")
        teams = sorted({t.team for field in fields for t in field})
        index = {team: i for i, team in enumerate(teams)}
        made = np.zeros((len(fields), len(teams)), dtype=bool)
        for i, field in enumerate(fields):
            for t in field:
                made[i, index[t.team]] = True

        percent = made.astype(float) * 100
        n_games = len(sim.games)
        home_odds = np.full((n_games, len(teams)), np.nan)
        away_odds = np.full((n_games, len(teams)), np.nan)
        delta = np.zeros((n_games, len(teams)))
        weight = np.zeros(n_games)
        for g in range(n_games):
            home_won = sim.home_wins[:, g]
            p = home_won.mean()
            weight[g] = 2 * p * (1 - p)
            n_home, n_away = int(home_won.sum()), int((~home_won).sum())
            if n_home >= MIN_TRIALS and n_away >= MIN_TRIALS:
                home_odds[g] = percent[home_won].mean(axis=0)
                away_odds[g] = percent[~home_won].mean(axis=0)
                h, a = home_odds[g] / 100, away_odds[g] / 100
                noise = 100 * np.sqrt(h * (1 - h) / n_home + a * (1 - a) / n_away)
                diff = home_odds[g] - away_odds[g]
                delta[g] = np.where(np.abs(diff) > NOISE_Z * noise, diff, 0.0)
        return cls(sim, teams, made, home_odds, away_odds, delta, weight)

    # --- one team ---------------------------------------------------------

    def made_odds(self, team: str) -> float:
        """% of all simulated seasons with `team` in the field."""
        return self._odds(team, np.ones(self.sim.n_sims, dtype=bool)) or 0.0

    def _odds(self, team: str, seasons: np.ndarray) -> float | None:
        """Field odds within the simulated seasons where `seasons` is True."""
        if seasons.sum() < MIN_TRIALS:
            return None
        if team not in self.teams:
            return 0.0                   # never in any field
        return float(self.made[seasons, self.teams.index(team)].mean() * 100)

    def by_final_record(self, team: str, wins: int, losses: int) -> list[RecordOutcome]:
        """Every record `team` can still finish with, best first."""
        _, won = team_game_results(team, self.sim)
        games = won.shape[1]
        if games == 0:
            return []
        extra = won.sum(axis=1)
        return [
            RecordOutcome(wins + e, losses + games - e,
                          share=float((extra == e).mean() * 100),
                          playoff=self._odds(team, extra == e),
                          n=int((extra == e).sum()))
            for e in sorted(set(extra.tolist()), reverse=True)
        ]

    def team_games(self, team: str) -> list[GameOdds]:
        """`team`'s remaining games in schedule order, with odds if it wins / loses."""
        columns, won = team_game_results(team, self.sim)
        games = []
        for j, column in enumerate(columns):
            sg = self.sim.games[column]
            p = (sg.home_win_probability if sg.game.home_team == team
                 else 1 - sg.home_win_probability)
            games.append(GameOdds(sg, p, self._odds(team, won[:, j]),
                                  self._odds(team, ~won[:, j])))
        return sorted(games, key=lambda g: g.game.game.week)

    def rooting_guide(self, team: str, limit: int = 5,
                      include_own: bool = True) -> list[RootFor]:
        """The games that move `team`'s odds the most. `include_own=False`
        leaves out the team's own games (those are in `team_games`) so what's
        left is a true rooting guide: games someone else plays."""
        if team not in self.teams:
            return []
        t = self.teams.index(team)
        delta = self._delta[:, t]
        swing = self._weight * np.abs(delta)
        guide = []
        for g in np.argsort(-swing):
            if len(guide) == limit or swing[g] < MIN_SWING:
                break
            game = self.sim.games[g]
            if not include_own and team in (game.game.home_team, game.game.away_team):
                continue
            home_helps = delta[g] > 0
            guide.append(RootFor(
                game=game,
                root_for=game.game.home_team if home_helps else game.game.away_team,
                odds_if_it_wins=float(max(self._home_odds[g, t], self._away_odds[g, t])),
                odds_if_it_loses=float(min(self._home_odds[g, t], self._away_odds[g, t])),
                swing=float(swing[g])))
        return guide

    # --- the whole field --------------------------------------------------

    def game_impacts(self) -> list[GameImpact]:
        """Every remaining game, biggest effect on the field first."""
        swing = self._weight[:, None] * np.abs(self._delta)
        impacts = []
        for g, game in enumerate(self.sim.games):
            order = np.argsort(-np.abs(self._delta[g]))
            movers = tuple(
                Mover(self.teams[t], float(self._home_odds[g, t]), float(self._away_odds[g, t]))
                for t in order if abs(self._delta[g, t]) >= MIN_SWING
            )[:MAX_MOVERS]
            impacts.append(GameImpact(game, float(swing[g].sum()), movers))
        return sorted(impacts, key=lambda i: -i.impact)
