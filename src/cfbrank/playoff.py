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
(cfbrank.projection.simulate_conference_champions): whoever finishes
with the best in-conference record IN THAT TRIAL. The field is rebuilt
for every trial, so a diffuse race (the SEC, say, split four ways) sends
different teams -- with different resumes -- to different trials, instead
of one "likeliest" team standing in for the whole conference. Odds of
making the field, getting a bye and winning each round are all counted
across those trials; the single bracket shown on the page is read back
off those odds (`consensus_field`).

One simplification: Resume rank is the CURRENT one in every trial. A
simulated future doesn't get its own re-ranking (that would be a second
model), so a team's seed doesn't move with its simulated wins and losses.

Call sites are expected to have already dropped non-championship
groupings (FBS Independents and anything else that isn't a real
conference) before this module ever sees them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .predict import Predictor
from .projection import DEFAULT_SEED

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


def simulate_fields(
    resume_rank: dict[str, int],
    conference_of: dict[str, str],
    champions_by_conference: dict[str, list[str]],
    n_sims: int,
) -> list[list[PlayoffTeam]]:
    """One 12-team field per simulated season, each built from that
    trial's own conference champions via the same `select_field` rules.

    `champions_by_conference` is what
    `projection.simulate_conference_champions` returns: conference ->
    one champion per trial.
    """
    return [
        select_field(resume_rank, conference_of,
                     {conf: champs[i] for conf, champs in champions_by_conference.items()})
        for i in range(n_sims)
    ]


@dataclass(frozen=True, slots=True)
class BracketOdds:
    """Per-team odds (percent of simulated seasons), keyed by every team
    that appeared in ANY simulated field -- not just the ones on the
    page's projected bracket. Round odds are unconditional: a team that
    misses the field in 70% of trials has at most a 30% shot at anything.
    """

    made_field: dict[str, float]
    auto_bid: dict[str, float]
    bye: dict[str, float]
    reached_quarterfinal: dict[str, float]
    reached_semifinal: dict[str, float]
    reached_championship: dict[str, float]
    won_it_all: dict[str, float]


def simulate_bracket(
    fields: list[list[PlayoffTeam]], predictor: Predictor,
    seed: int = DEFAULT_SEED,
) -> BracketOdds:
    """Play out one single-elimination bracket per field in `fields`
    (one per simulated season), game probabilities read straight off this
    same Predictor -- exactly how every other matchup on the site is
    scored, hypothetical or not.
    """
    rng = np.random.default_rng(seed)
    ratings = predictor.model.ratings
    cache: dict[tuple[str, str, bool], float] = {}

    def win_probability(home: str, away: str, neutral: bool) -> float:
        key = (home, away, neutral)
        if key not in cache:
            if home not in ratings or away not in ratings:
                cache[key] = 0.5  # shouldn't happen -- fields are built from rated teams
            else:
                cache[key] = predictor.predict(home, away, neutral).home_win_probability
        return cache[key]

    made, auto, bye = Counter(), Counter(), Counter()
    reached_qf, reached_sf = Counter(), Counter()
    reached_champ, won_title = Counter(), Counter()

    for field in fields:
        by_seed = {t.seed: t.team for t in field}
        for t in field:
            made[t.team] += 1
            auto[t.team] += t.champion
            bye[t.team] += t.seed <= BYE_SEEDS

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
        won_title[champ_a if rng.random() < p else champ_b] += 1

    scale = 100.0 / len(fields)

    def pct(counts: Counter) -> dict[str, float]:
        return {team: counts[team] * scale for team in made}

    return BracketOdds(
        made_field=pct(made), auto_bid=pct(auto), bye=pct(bye),
        reached_quarterfinal=pct(reached_qf), reached_semifinal=pct(reached_sf),
        reached_championship=pct(reached_champ), won_it_all=pct(won_title),
    )


def consensus_field(
    odds: BracketOdds, resume_rank: dict[str, int],
    conference_of: dict[str, str],
) -> list[PlayoffTeam]:
    """The one concrete bracket the page shows, read back off the odds.

    No individual simulated field is "the" projection -- they differ
    trial to trial. So: the 12 teams most often in the field, the four of
    those most often holding a bye take seeds 1-4, everyone else is slotted
    by Resume rank (as in `select_field`). `champion` marks teams that
    were in the field as a conference champion at least as often as as an
    at-large pick.

    Ties on percentages fall back to Resume rank, so the result is stable.
    """
    in_field = sorted(odds.made_field,
                      key=lambda t: (-odds.made_field[t], resume_rank[t]))[:FIELD_SIZE]
    byes = sorted(sorted(in_field, key=lambda t: -odds.bye[t])[:BYE_SEEDS],
                  key=lambda t: resume_rank[t])
    rest = sorted((t for t in in_field if t not in byes), key=lambda t: resume_rank[t])
    return [
        PlayoffTeam(seed=i, team=team, conference=conference_of.get(team, ""),
                    rank=resume_rank[team],
                    champion=odds.auto_bid[team] * 2 >= odds.made_field[team])
        for i, team in enumerate(byes + rest, 1)
    ]
