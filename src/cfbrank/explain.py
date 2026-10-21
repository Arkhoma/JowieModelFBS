"""Why a team ranks where it does, in a few plain sentences.

Nothing new is modelled here. Every line reads numbers the rating model
already produced and puts them side by side:

  preseason   where the roster prior had the team vs where it is now
  split       rank on the scoreboard half of the rating vs the per-play
              success half (cfbrank.ensemble). A big gap is the closest
              thing the model has to a luck flag.
  off / def   which side of the ball the team is built on
  schedule    strength-of-schedule rank
  games       the results that most beat / missed the model's number

Ranks are among the teams passed in (the FBS table), so every "#N" here
means the same thing as the "#N" on the rankings page.
"""

from __future__ import annotations

from dataclasses import dataclass

# A rank gap this big (or bigger) is worth pointing out; smaller ones are
# just two noisy numbers that roughly agree.
SPLIT_GAP = 15
# Within this many spots of the preseason rank counts as "held steady".
STEADY = 3


@dataclass(frozen=True, slots=True)
class GameSwing:
    """One played game, measured against the model's current number."""

    result: str        # "W 42-10"
    where: str         # "vs Texas" / "at Texas"
    vs_model: float    # actual margin minus predicted, team's perspective


def _ranks(values: dict[str, float]) -> dict[str, int]:
    """1 = biggest value."""
    ordered = sorted(values, key=values.get, reverse=True)
    return {team: i for i, team in enumerate(ordered, 1)}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def game_swings(team: str, games, predictor) -> list[GameSwing]:
    """Every played game for `team`, best performance vs the model first."""
    swings = []
    for g in games:
        if team not in (g.home_team, g.away_team):
            continue
        home = g.home_team == team
        scored, allowed = ((g.home_points, g.away_points) if home
                           else (g.away_points, g.home_points))
        opponent = g.away_team if home else g.home_team
        predicted = predictor.predict(
            g.home_team, g.away_team, g.neutral_site).predicted_margin
        expected = predicted if home else -predicted
        swings.append(GameSwing(
            result=f"{'W' if scored > allowed else 'L'} {scored}-{allowed}",
            where=f"{'at' if not home and not g.neutral_site else 'vs'} {opponent}",
            vs_model=(scored - allowed) - expected,
        ))
    return sorted(swings, key=lambda s: s.vs_model, reverse=True)


def _preseason_line(rank: int, preseason: int | None) -> str | None:
    if preseason is None:
        return None
    moved = preseason - rank
    if abs(moved) <= STEADY:
        return f"Preseason #{preseason}, and the games have backed that up."
    direction = "up" if moved > 0 else "down"
    return f"Preseason #{preseason}. Games have moved them {direction} {abs(moved)} spots."


def _split_line(scores: int | None, plays: int | None) -> str | None:
    if scores is None or plays is None:
        return None
    if plays - scores >= SPLIT_GAP:
        return (f"Results (#{scores}) run ahead of per-play success "
                f"(#{plays}), so some of that may be luck.")
    if scores - plays >= SPLIT_GAP:
        return (f"Per-play success (#{plays}) is better than the "
                f"results (#{scores}) show.")
    return f"Results and per-play success agree (#{scores} and #{plays})."


def _side_line(offense: int | None, defense: int | None) -> str | None:
    if offense is None or defense is None:
        return None
    if offense - defense >= SPLIT_GAP:
        return f"Built on defense (#{defense}) more than offense (#{offense})."
    if defense - offense >= SPLIT_GAP:
        return f"Built on offense (#{offense}) more than defense (#{defense})."
    return f"Offense #{offense}, defense #{defense}."


def _schedule_line(sos_rank: int) -> str:
    if sos_rank == 1:
        return "Toughest schedule in FBS."
    return f"{_ordinal(sos_rank)}-toughest schedule in FBS."


def _game_lines(swings: list[GameSwing]) -> list[str]:
    if len(swings) < 2:
        return []
    lines = []
    best, worst = swings[0], swings[-1]
    if best.vs_model > 0:
        lines.append(f"Best game: {best.result} {best.where}, "
                     f"{best.vs_model:.0f} pts better than the model's number.")
    if worst.vs_model < 0:
        lines.append(f"Worst game: {worst.result} {worst.where}, "
                     f"{-worst.vs_model:.0f} pts worse than the model's number.")
    return lines


def explain_rankings(
    rows: list[dict], model, prior: dict[str, float], games, predictor,
) -> dict[str, list[str]]:
    """team -> a few short sentences on why it ranks where it does.

    `rows` are the rankings-table rows (need rank, sos, offense_rank,
    defense_rank). `model` may be an EnsembleModel (gets the scoreboard vs
    play-by-play line) or a single RatingResult (doesn't).
    """
    teams = [r["team"] for r in rows]
    preseason = _ranks({t: prior[t] for t in teams if t in prior})
    sos = _ranks({r["team"]: r["sos"] for r in rows})

    margin_model = getattr(model, "margin_model", None)
    plays_model = getattr(model, "epa_model", None)
    scores_rank: dict[str, int] = {}
    plays_rank: dict[str, int] = {}
    if margin_model is not None and plays_model is not None:
        scores_rank = _ranks({t: margin_model.rating(t) for t in teams
                              if t in margin_model.ratings})
        plays_rank = _ranks({t: plays_model.rating_in_points(t) for t in teams
                             if t in plays_model.ratings})

    out = {}
    for r in rows:
        team = r["team"]
        lines = [
            _preseason_line(r["rank"], preseason.get(team)),
            _split_line(scores_rank.get(team), plays_rank.get(team)),
            _side_line(r.get("offense_rank"), r.get("defense_rank")),
            _schedule_line(sos[team]),
            *_game_lines(game_swings(team, games, predictor)),
        ]
        out[team] = [line for line in lines if line]
    return out
