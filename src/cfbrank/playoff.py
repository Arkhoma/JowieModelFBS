"""Playoff projection: 12-team CFP field, seeding, and bracket simulation.

Same rule as every other projection in this codebase: no new model. Field
selection and seeding read off the RESUME ranking, not predictive -- a
real selection committee claims to reward record and quality wins, which
is the question Resume answers ("who has earned it"), not the question
Predictive answers ("who is the best team on a neutral field"). This
project never averages the two (see cfbrank.resume); picking one,
clearly, for one well-defined purpose is different from blending them.

Format matches the 2024-season 12-team CFP:
    5 auto bids: the five highest-(resume)-ranked conference champions.
    7 at-large: the next-best teams by resume rank, any conference.
    Seeds 1-4: the four highest-ranked of those five champions get a
        bye to the quarterfinals.
    Seeds 5-12: everyone else, slotted straight through by resume rank.
        Round of 12 is played at the higher seed's home field; every
        round after that is a neutral site.

"Conference champion" is read from the shared season simulation
(cfbrank.projection.project_conference_races) as whichever team is most
likely to finish with the best in-conference record -- a point estimate
like every other number on this site, not a guarantee. Call sites are
expected to have already dropped non-championship groupings (FBS
Independents and anything else that isn't a real conference) before
this module ever sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .predict import Predictor

AUTO_BIDS = 5
BYE_SEEDS = 4
FIELD_SIZE = 12

# Round of 12: seed 5 hosts 12, 6 hosts 11, 7 hosts 10, 8 hosts 9.
FIRST_ROUND = [(5, 12), (6, 11), (7, 10), (8, 9)]

# Quarterfinal draw: which bye seed (1-4) plays which first-round winner.
# Seed 1 draws the 8/9 winner, 2 draws 7/10, 3 draws 6/11, 4 draws 5/12 --
# the actual 12-team CFP bracket pairing, not a simplified reseed.
QUARTERFINAL_DRAW = {1: (8, 9), 2: (7, 10), 3: (6, 11), 4: (5, 12)}

# Semifinals: seed 1's quarter plays seed 4's, seed 2's plays seed 3's.
SEMIFINAL_PAIRS = [(1, 4), (2, 3)]


@dataclass(frozen=True, slots=True)
class PlayoffTeam:
    """One team's spot in the 12-team field."""

    seed: int
    team: str
    conference: str
    rank: int
    champion: bool


def select_field(
    resume_rank: dict[str, int],
    conference_of: dict[str, str],
    champion_of_conference: dict[str, str],
) -> list[PlayoffTeam]:
    """Pick and seed the 12-team field from already-computed rankings.

    Raises ValueError rather than silently shipping a short bracket if
    there aren't 12 ranked FBS teams to fill it -- a season this thin
    isn't a "projection," it's a different problem.
    """
    champions = sorted(
        champion_of_conference.values(),
        key=lambda team: resume_rank.get(team, 10**9),
    )[:AUTO_BIDS]
    champion_set = set(champions)

    at_large_pool = [team for team in resume_rank if team not in champion_set]
    at_large = sorted(at_large_pool, key=lambda t: resume_rank[t])[
        :FIELD_SIZE - len(champions)]

    bye_seeds = champions[:BYE_SEEDS]
    remaining_seeds = sorted(champions[BYE_SEEDS:] + at_large,
                             key=lambda t: resume_rank[t])
    ordered = bye_seeds + remaining_seeds

    if len(ordered) < FIELD_SIZE:
        raise ValueError(
            f"Only {len(ordered)} teams available for a {FIELD_SIZE}-team "
            "field -- too few ranked conferences/teams to project a bracket."
        )

    return [
        PlayoffTeam(seed=i, team=team, conference=conference_of.get(team, ""),
                   rank=resume_rank[team], champion=team in champion_set)
        for i, team in enumerate(ordered, 1)
    ]


@dataclass(frozen=True, slots=True)
class BracketOdds:
    """Per-team odds of reaching each round, from the shared simulation."""

    field: list[PlayoffTeam]
    reached_quarterfinal: dict[str, float]
    reached_semifinal: dict[str, float]
    reached_championship: dict[str, float]
    won_it_all: dict[str, float]


def simulate_bracket(
    field: list[PlayoffTeam], predictor: Predictor,
    n_sims: int = 2000, seed: int = 20260101,
) -> BracketOdds:
    """Monte Carlo the single-elimination bracket, game probabilities read
    straight off this same Predictor -- exactly how every other matchup
    on the site is scored, hypothetical or not.
    """
    by_seed = {t.seed: t.team for t in field}
    rng = np.random.default_rng(seed)
    ratings = predictor.model.ratings

    def win_probability(home: str, away: str, neutral: bool) -> float:
        if home not in ratings or away not in ratings:
            return 0.5  # shouldn't happen -- the field is built from rated teams
        return predictor.predict(home, away, neutral).home_win_probability

    reached_qf = {t.team: 0 for t in field}
    reached_sf = {t.team: 0 for t in field}
    reached_champ = {t.team: 0 for t in field}
    won_title = {t.team: 0 for t in field}

    for _ in range(n_sims):
        round1 = {}
        for hi, lo in FIRST_ROUND:
            home, away = by_seed[hi], by_seed[lo]
            p = win_probability(home, away, neutral=False)
            round1[hi] = home if rng.random() < p else away

        qf_winner = {}
        for bye_seed, (hi, _lo) in QUARTERFINAL_DRAW.items():
            bye_team = by_seed[bye_seed]
            opponent = round1[hi]
            reached_qf[bye_team] += 1
            reached_qf[opponent] += 1
            p = win_probability(bye_team, opponent, neutral=True)
            qf_winner[bye_seed] = bye_team if rng.random() < p else opponent

        sf_winner = []
        for a, b in SEMIFINAL_PAIRS:
            team_a, team_b = qf_winner[a], qf_winner[b]
            reached_sf[team_a] += 1
            reached_sf[team_b] += 1
            p = win_probability(team_a, team_b, neutral=True)
            sf_winner.append(team_a if rng.random() < p else team_b)

        champ_a, champ_b = sf_winner
        reached_champ[champ_a] += 1
        reached_champ[champ_b] += 1
        p = win_probability(champ_a, champ_b, neutral=True)
        champion = champ_a if rng.random() < p else champ_b
        won_title[champion] += 1

    n = float(n_sims)
    return BracketOdds(
        field=field,
        reached_quarterfinal={t: c / n * 100 for t, c in reached_qf.items()},
        reached_semifinal={t: c / n * 100 for t, c in reached_sf.items()},
        reached_championship={t: c / n * 100 for t, c in reached_champ.items()},
        won_it_all={t: c / n * 100 for t, c in won_title.items()},
    )
