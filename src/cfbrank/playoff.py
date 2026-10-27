"""Playoff projection: 12-team CFP field, seeding, and bracket simulation.

Same rule as every other projection in this codebase: no new model. Field
selection and seeding read off the RESUME ranking, not predictive -- a
real selection committee claims to reward record and quality wins, which
is the question Resume answers ("who has earned it"), not the question
Predictive answers ("who is the best team on a neutral field"). This
project never averages the two (see cfbrank.resume); picking one,
clearly, for one well-defined purpose is different from blending them.

Format: the 2026-27 CFP rules (collegefootballplayoff.com, "2026-27 CFP
Bracket" and the Selection Committee Protocol; NCAA.com, Aug 31 2026):

    4 auto bids: the ACC, Big Ten, Big 12 and SEC champions -- however
        low they're ranked.
    1 auto bid: the highest-ranked team from the American, Conference
        USA, MAC, Mountain West, Pac-12 or Sun Belt. A TEAM, not a
        champion: it needn't have won its conference.
    1 auto bid: Notre Dame, but only if it is ranked in the top 12. A
        Notre Dame that qualifies this way leaves 6 at-large bids
        instead of 7 -- the field is always 12.
    At-large: the best remaining teams, any conference.
    Seeding: straight by committee ranking. The four highest-ranked
        teams (champion or not) are seeds 1-4 and get byes; everyone
        else is seeded 5-12 by ranking, so an auto bid ranked outside the
        top 12 lands at the bottom, in rank order. No reseeding.
        The round of 12 is at the higher seed's home field; every round
        after that is a neutral site.

That is a different format from 2024 (five highest-ranked champions, byes
to the four highest-ranked champions) and from 2025 (same five
champions, but straight seeding). Older seasons shown on the site are
what-ifs under today's rules.

"Conference champion" is read from the shared season simulation
(cfbrank.projection.simulate_conference_champions): whoever finishes
with the best in-conference record IN THAT TRIAL. Resume rank is the
trial's own too (cfbrank.resume.simulate_resume_ranks), so the field is
rebuilt for every trial: a wide-open conference sends different teams to
different trials, and a team that wins out climbs the board in the trials
where it does. Odds of making the field, getting a bye and winning each
round are all counted across those trials; the single bracket shown on
the page is read back off those odds (`consensus_field`).

Call sites are expected to have already dropped non-championship
groupings (FBS Independents and anything else that isn't a real
conference) from `conference_of` before this module ever sees them.
Notre Dame is recognised by name, not by conference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .predict import Predictor
from .projection import DEFAULT_SEED

BYE_SEEDS = 4
FIELD_SIZE = 12

# The four conferences whose champions are in, ranked or not.
POWER_CONFERENCES = frozenset({"ACC", "Big Ten", "Big 12", "SEC"})
# The conferences competing for the single highest-ranked-team bid.
GROUP_OF_SIX = frozenset({
    "American Athletic", "Conference USA", "Mid-American",
    "Mountain West", "Pac-12", "Sun Belt",
})
NOTRE_DAME = "Notre Dame"
NOTRE_DAME_AUTO_RANK = 12

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
    auto_bid: bool
    """In by rule (Power 4 champion, top Group-of-6 team, or Notre Dame
    in the top 12) rather than as an at-large pick."""


def automatic_qualifiers(
    resume_rank: dict[str, int],
    conference_of: dict[str, str],
    champion_of_conference: dict[str, str],
) -> set[str]:
    """Everyone who is in by rule, before any at-large pick."""
    qualifiers = {
        champion for conf, champion in champion_of_conference.items()
        if conf in POWER_CONFERENCES
    }

    group_of_six = [team for team in resume_rank
                    if conference_of.get(team) in GROUP_OF_SIX]
    if group_of_six:
        qualifiers.add(min(group_of_six, key=resume_rank.get))

    if resume_rank.get(NOTRE_DAME, 10**9) <= NOTRE_DAME_AUTO_RANK:
        qualifiers.add(NOTRE_DAME)
    return qualifiers


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
    automatic = automatic_qualifiers(resume_rank, conference_of,
                                     champion_of_conference)
    at_large_pool = [team for team in resume_rank if team not in automatic]
    at_large = sorted(at_large_pool, key=resume_rank.get)[
        :FIELD_SIZE - len(automatic)]

    # Straight seeding by ranking: an automatic qualifier ranked outside
    # the top 12 sorts to the bottom by itself, in rank order.
    ordered = sorted(list(automatic) + at_large, key=resume_rank.get)

    if len(ordered) < FIELD_SIZE:
        raise ValueError(
            f"Only {len(ordered)} teams available for a {FIELD_SIZE}-team "
            "field -- too few ranked conferences/teams to project a bracket."
        )

    return [
        PlayoffTeam(seed=i, team=team, conference=conference_of.get(team, ""),
                    rank=resume_rank[team], auto_bid=team in automatic)
        for i, team in enumerate(ordered, 1)
    ]


def simulate_fields(
    teams: list[str],
    ranks: np.ndarray,
    conference_of: dict[str, str],
    champions_by_conference: dict[str, list[str]],
) -> list[list[PlayoffTeam]]:
    """One 12-team field per simulated season, each built from that
    trial's own resume ranking and conference champions via the same
    `select_field` rules.

    `teams`/`ranks` are what `resume.simulate_resume_ranks` returns
    (`ranks[i, j]` = rank of `teams[j]` in trial i);
    `champions_by_conference` is what
    `projection.simulate_conference_champions` returns: conference ->
    one champion per trial.
    """
    return [
        select_field(
            dict(zip(teams, trial_ranks.tolist())), conference_of,
            {conf: champs[i] for conf, champs in champions_by_conference.items()})
        for i, trial_ranks in enumerate(ranks)
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
    mean_seed: dict[str, float]
    """Average seed in the trials where the team made the field."""
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

    made, auto, bye, seed_total = Counter(), Counter(), Counter(), Counter()
    reached_qf, reached_sf = Counter(), Counter()
    reached_champ, won_title = Counter(), Counter()

    for field in fields:
        by_seed = {t.seed: t.team for t in field}
        for t in field:
            made[t.team] += 1
            auto[t.team] += t.auto_bid
            bye[t.team] += t.seed <= BYE_SEEDS
            seed_total[t.team] += t.seed

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
        mean_seed={team: seed_total[team] / made[team] for team in made},
        reached_quarterfinal=pct(reached_qf), reached_semifinal=pct(reached_sf),
        reached_championship=pct(reached_champ), won_it_all=pct(won_title),
    )


def consensus_field(
    odds: BracketOdds, resume_rank: dict[str, int],
    conference_of: dict[str, str],
) -> list[PlayoffTeam]:
    """The one concrete bracket the page shows, read back off the odds.

    No individual simulated field is "the" projection -- they differ
    trial to trial. So: the 12 teams most often in the field; seeds 1-4
    go to the four of those most often holding a bye, and 5-12 to the rest,
    each group ordered by the average seed the team got when it was in
    the field. (Bye odds, not average seed, pick the byes: a team that
    only makes it in the seasons it wins its conference has a flattering
    average seed but rarely a bye.) `auto_bid` marks teams that were in by
    rule at least as often as as an at-large pick.

    Ties on percentages fall back to today's Resume rank, so the result
    is stable.
    """
    in_field = sorted(odds.made_field,
                      key=lambda t: (-odds.made_field[t], resume_rank[t]))[:FIELD_SIZE]
    by_bye = sorted(in_field, key=lambda t: (-odds.bye[t], resume_rank[t]))
    byes = set(by_bye[:BYE_SEEDS])
    seeded = sorted(in_field, key=lambda t: (t not in byes, odds.mean_seed[t],
                                             resume_rank[t]))
    return [
        PlayoffTeam(seed=i, team=team, conference=conference_of.get(team, ""),
                    rank=resume_rank[team],
                    auto_bid=odds.auto_bid[team] * 2 >= odds.made_field[team])
        for i, team in enumerate(seeded, 1)
    ]
