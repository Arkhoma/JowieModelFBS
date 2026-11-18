"""Presentation for the "what if" features: a team's path to the playoff,
the games that matter most, and the weekly slate.

Turns cfbrank.scenarios' numbers into display-ready rows, so the Jinja
templates only loop. Shared by the live app and the static builder (both
go through app.pages). Nothing here is a new calculation: every odds figure
is read from the same simulated seasons as the Playoff page.
"""

from __future__ import annotations

from cfbrank.scenarios import GameImpact, RootFor

KEY_GAMES = 10          # rows in "games that matter most"


def _opponent(game, team: str) -> str:
    return game.away_team if game.home_team == team else game.home_team


def _venue(game, team: str) -> str:
    """'vs' at home or on a neutral field, 'at' on the road."""
    return "vs" if game.neutral_site or game.home_team == team else "at"


def n_sims(data) -> int:
    """How many simulated seasons the what-if numbers are read from (0 = none)."""
    return data.scenarios.sim.n_sims if data.scenarios else 0


def team_path(data, team: str, wins: int, losses: int) -> dict | None:
    """One team's route to the 12-team field, or None when there is
    nothing to say (no scenarios for the season, or no games left)."""
    scenarios = data.scenarios
    if scenarios is None:
        return None
    games = scenarios.team_games(team)
    if not games:
        return None
    records = scenarios.by_final_record(team, wins, losses)
    baseline = scenarios.made_odds(team)
    best = records[0]
    return {
        "baseline": baseline,
        "n_sims": n_sims(data),
        "records": records,
        "best_record": f"{best.wins}-{best.losses}",
        # Zero in every simulated season, however the schedule ends.
        "out_of_reach": baseline == 0 and all(not r.playoff for r in records),
        "games": [{
            "opponent": _opponent(g.game.game, team),
            "venue": _venue(g.game.game, team),
            "neutral": g.game.game.neutral_site,
            "week": g.game.game.week,
            "win_probability": g.team_win_probability * 100,
            "if_win": g.if_win,
            "if_lose": g.if_lose,
        } for g in games],
        "rooting": ([_root_row(r) for r in
                      scenarios.rooting_guide(team, include_own=False)]
                     if baseline > 0 else []),
    }


def _root_row(root: RootFor) -> dict:
    game = root.game.game
    return {
        "week": game.week, "away": game.away_team, "home": game.home_team,
        "root_for": root.root_for,
        "if_it_wins": root.odds_if_it_wins, "if_it_loses": root.odds_if_it_loses,
    }


def _kickoff(game) -> str:
    k = game.kickoff
    return f"{k:%a %b} {k.day}" if k else ""


def game_row(data, game, impact: GameImpact | None, rank_of: dict) -> dict:
    """One upcoming game: our pick, the market's, and how much it matters."""
    p = data.predictor.predict(game.home_team, game.away_team, game.neutral_site)
    home_favoured = p.predicted_margin > 0
    vegas = p.vegas_margin
    return {
        "week": game.week,
        "kickoff": _kickoff(game),
        "home": game.home_team, "away": game.away_team,
        "home_rank": rank_of.get(game.home_team),
        "away_rank": rank_of.get(game.away_team),
        "neutral": game.neutral_site,
        "conference_game": game.conference_game,
        "favourite": p.favourite,
        "favourite_pct": 100 * (p.home_win_probability if home_favoured
                                else 1 - p.home_win_probability),
        "spread": p.spread,
        "vegas": (None if vegas is None else
                  {"favourite": p.vegas_favourite, "margin": abs(vegas)}),
        "script_alert": p.script_alert,
        "script_gap": None if vegas is None else abs(vegas - p.predicted_margin),
        "impact": impact.impact if impact else 0.0,
        "movers": [{"team": m.team, "if_home_wins": m.if_home_wins,
                    "if_away_wins": m.if_away_wins}
                   for m in (impact.movers if impact else ())],
    }


def _rank_of(data) -> dict[str, int]:
    return {r["team"]: r["rank"] for r in data.rankings}


def key_games(data, limit: int = KEY_GAMES) -> list[dict]:
    """The remaining games that move the most playoff odds, biggest first."""
    if data.scenarios is None:
        return []
    ranks = _rank_of(data)
    return [game_row(data, i.game.game, i, ranks)
            for i in data.scenarios.game_impacts()[:limit] if i.impact > 0]


def slate(data) -> dict | None:
    """The next week's games with picks and playoff impact, most important
    first; None when nothing is left to play (offseason, finished season)."""
    rated = data.predictor.model.ratings
    games = [g for g in data.remaining
             if g.home_team in rated and g.away_team in rated]
    if not games:
        return None
    week = min(g.week for g in games)
    impacts = ({(i.game.game.week, i.game.game.home_team, i.game.game.away_team): i
                for i in data.scenarios.game_impacts()}
               if data.scenarios else {})
    ranks = _rank_of(data)
    rows = [game_row(data, g, impacts.get((g.week, g.home_team, g.away_team)), ranks)
            for g in games if g.week == week]
    # Most playoff-relevant first; ties (and the many zero-impact games)
    # fall back to the better-ranked team, so marquee games lead.
    rows.sort(key=lambda r: (-r["impact"],
                             min(r["home_rank"] or 999, r["away_rank"] or 999)))
    return {"week": week, "rows": rows, "n_sims": n_sims(data),
            "max_impact": max((r["impact"] for r in rows), default=0.0) or 1.0}
