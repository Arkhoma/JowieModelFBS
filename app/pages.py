"""What each page shows, independent of how it's served.

The live FastAPI app (app.main) and the GitHub Pages export
(tools/build_static_site.py) render the same templates with the same
context, built here. The only thing that differs between the two sites
is the URL scheme, carried by one `SiteUrls` value per site -- so a new
page type is one new field there, not a new keyword threaded through
every call (which is how the old `team_url=None` sentinel happened).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote

from cfbrank.games import available_seasons, fbs_seasons
from cfbrank.playoff import FIRST_ROUND
from cfbrank.time_machine import season_snapshot


@dataclass(frozen=True, slots=True)
class SiteUrls:
    """Where every page lives on one site. Templates see these as
    nav / rank_url / team_url / predict_url / playoff_url."""

    nav: dict                              # home, about, static, time_machine
    rank: Callable[[int, str], str]        # (season, sort)
    team: Callable[[str, int], str]        # (team, season)
    predict: Callable[[int], str]          # (season)
    playoff: Callable[[int], str]          # (season)


LIVE_URLS = SiteUrls(
    nav={"home": "/", "about": "/about", "static": "/static",
         "time_machine": "/time-machine"},
    rank=lambda season, sort: f"/?season={season}&sort={sort}",
    team=lambda team, season: f"/team/{quote(team)}?season={season}",
    predict=lambda season: f"/predict?season={season}",
    playoff=lambda season: f"/playoff?season={season}",
)


def _rank_key(column: str):
    """Sort by a rank column; unranked teams go last instead of crashing."""
    return lambda row: (row[column] is None, row[column] or 0)


SORTS = {"predictive": _rank_key("rank"), "resume": _rank_key("resume_rank"),
         "offense": _rank_key("offense_rank"),
         "defense": _rank_key("defense_rank")}


def base_context(active_tab: str, season: int, urls: SiteUrls = LIVE_URLS,
                 **extra) -> dict:
    """Context every template needs, plus the page's own `extra`."""
    return {
        "active_tab": active_tab,
        "season": season,
        "seasons": available_seasons(),
        "nav": urls.nav,
        "rank_url": urls.rank,
        "team_url": urls.team,
        "predict_url": urls.predict,
        "playoff_url": urls.playoff,
        **extra,
    }


# --- Per-page context. `data` is an app.main.SeasonModel. ---------------

def rankings_context(data, sort: str) -> dict:
    return {
        "rankings": sorted(data.rankings, key=SORTS[sort]),
        "sort": sort,
        "model": data.model,
        "n_games": data.n_games,
        "max_week": data.max_week,
        "has_postseason": data.has_postseason,
        "conference_races": data.conference_races,
    }


def team_context(data, team: str) -> dict:
    row = next((r for r in data.rankings if r["team"] == team), None)
    return {
        "team": team,
        "row": row,
        "team_seasons": fbs_seasons(team),
        "game_log": (team_game_log(team, data.games, data.remaining,
                                   data.predictor) if row else []),
    }


def predict_context(data) -> dict:
    return {"teams": predict_teams(data), "prediction": None}


def time_machine_context(get_season) -> dict:
    """Every season's snapshot, for any-team-any-year matchups.

    `get_season` is passed in (app.main's cached fitter) rather than
    imported, so this module never imports app.main. Fits every season,
    so the first load on a fresh local server takes a few minutes.
    """
    snapshots, conference = {}, {}
    for season in available_seasons():
        data = get_season(season)
        label = (str(season) if not data.remaining
                 else f"{season} (through week {data.max_week})")
        snapshots[str(season)] = season_snapshot(data.rankings, data.predictor, label)
        conference.update({r["team"]: r["conference"] for r in data.rankings})
    return {
        "snapshots": snapshots,
        # Everyone who was ever FBS; latest conference wins.
        "teams": [{"name": t, "conference": c}
                  for t, c in sorted(conference.items())],
    }


def playoff_context(data) -> dict:
    return {
        "field": data.playoff_field,
        "odds": data.playoff_odds,
        "first_round": (first_round_matchups(data.playoff_field, data.predictor)
                        if data.playoff_field else []),
        # Nothing left to play: the bracket is a what-if on final ratings.
        "season_complete": not data.remaining,
    }


# --- Builders behind the contexts above. --------------------------------

def predict_teams(data) -> list[dict]:
    """Team picker entries for the predict page: name, rank, conference,
    record. Shared with the static export so the offline page can never
    quietly list a different set of teams than the live one."""
    rankings = {r["team"]: r for r in data.rankings}
    return [
        {"name": t, "rank": rankings[t]["rank"],
         "conference": rankings[t]["conference"],
         "record": f"{rankings[t]['wins']}-{rankings[t]['losses']}"}
        for t in sorted(data.model.fbs_ratings()) if t in rankings
    ]


def _log_entry(team: str, g, predictor, played: bool) -> dict:
    is_home = g.home_team == team
    prediction = predictor.predict(g.home_team, g.away_team, g.neutral_site)
    win_probability = (prediction.home_win_probability if is_home
                       else 1 - prediction.home_win_probability)
    entry = {
        "week": g.week,
        "opponent": g.away_team if is_home else g.home_team,
        "location": "neutral" if g.neutral_site else ("vs" if is_home else "at"),
        "played": played,
        "result": None,
        "win_probability": win_probability,
        "upset": None,
    }
    if played:
        scored, allowed = ((g.home_points, g.away_points) if is_home
                           else (g.away_points, g.home_points))
        won = scored > allowed
        entry["result"] = f"{'W' if won else 'L'} {scored}-{allowed}"
        entry["upset"] = won != (win_probability >= 0.5)
    return entry


def team_game_log(team: str, games: list, remaining: list, predictor) -> list[dict]:
    """One team's full schedule: played results plus upcoming predictions,
    in week order.

    The "model said" column for a PLAYED game is this exact predictor
    evaluated on that matchup -- not a separate backtest metric -- so a
    reader comparing it against the actual score is comparing the model
    to itself honestly, calibration warts included.
    """
    ratings = predictor.model.ratings
    entries = [_log_entry(team, g, predictor, played=True)
               for g in games if team in (g.home_team, g.away_team)]
    entries += [
        _log_entry(team, g, predictor, played=False)
        for g in remaining
        if team in (g.home_team, g.away_team)
        # same "no opinion, no guess" rule as simulate_remaining
        and g.home_team in ratings and g.away_team in ratings
    ]
    entries.sort(key=lambda e: e["week"])
    return entries


def first_round_matchups(field: list, predictor) -> list[dict]:
    """The Round of 12's four concrete games.

    Seeding is already locked in from the current standings, so these are
    real matchups with a real spread -- unlike the deeper rounds, which
    depend on who wins here and only exist as simulated odds.
    """
    by_seed = {t.seed: t for t in field}
    matchups = []
    for hi, lo in FIRST_ROUND:
        home, away = by_seed[hi], by_seed[lo]
        prediction = predictor.predict(home.team, away.team, neutral_site=False)
        matchups.append({
            "home": home, "away": away,
            "home_win_probability": prediction.home_win_probability,
        })
    return matchups
