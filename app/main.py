"""FastAPI app: rankings, predictions, backtest, and methodology.

Ratings are computed once at startup and cached -- the ridge solve takes
a couple of seconds and the underlying data only changes weekly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cfbrank.ensemble import EFFICIENCY_METRIC, EnsembleModel  # noqa: E402
from cfbrank.epa_ridge import (  # noqa: E402
    build_epa_games, fit_epa_ratings_primed, get_ep_model,
)
from cfbrank.games import (  # noqa: E402
    POSTSEASON, available_seasons, load_remaining_schedule, load_season,
)
from cfbrank.lines import home_lines  # noqa: E402
from cfbrank.offdef_prior import fit_points_model  # noqa: E402
from cfbrank.explain import explain_rankings  # noqa: E402
from cfbrank.playoff import FIRST_ROUND, select_field, simulate_bracket  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.prior import home_field_prior  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.projection import (  # noqa: E402
    TeamProjection, project_all, project_conference_races, simulate_remaining,
)
from cfbrank.resume import build_resumes, rank_resumes  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402
from cfbrank.scorecard import load_scorecard  # noqa: E402
from cfbrank.vegas_script import (  # noqa: E402
    VegasScript, alerts_active, fit_market_ratings,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))

# Cross-validated on 2024: lambda=5, scale=28, no recency decay.
# See docs/MODEL_SPEC.md -- these were measured, not chosen.
FITTED_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}

# Written by tools/benchmark_vs_market.py: walk-forward PRODUCTION
# predictions next to the closing line. Rerun it after any model change.
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_vs_market.csv"

app = FastAPI(title="JowieModel FBS")
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")),
          name="static")


@dataclass(slots=True)
class SeasonModel:
    """Everything the UI needs for one season."""

    season: int
    model: object
    predictor: Predictor
    rankings: list[dict]
    n_games: int
    max_week: int
    has_postseason: bool = False
    uses_epa: bool = False
    conference_races: dict[str, list[tuple[str, float]]] = None
    games: list = None
    remaining: list = None
    playoff_field: list = None
    playoff_odds: object = None


def _build_epa_model(season: int, games, margin_prior):
    """Efficiency ratings for one season, or None if unavailable.

    Play data does not cover every season, and a missing parquet file is
    an expected absence rather than an error -- the margin model alone
    still produces a perfectly good ranking.

    Shares the margin model's preseason prior (translated into EPA
    units) rather than shrinking toward zero. Without it, three games of
    blowouts over weak opponents can out-rate a real contender -- see
    epa_ridge.fit_epa_ratings_primed for the New Mexico case that found
    this.
    """
    try:
        epa_games = build_epa_games(season, games, get_ep_model(),
                                    metric=EFFICIENCY_METRIC)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    if len(epa_games) < 50:
        return None
    try:
        return fit_epa_ratings_primed(epa_games, margin_prior)
    except (ValueError, np.linalg.LinAlgError):
        return None


@lru_cache(maxsize=16)
def get_season(season: int) -> SeasonModel:
    """Fit and cache the rating model for one season."""
    games = load_season(season)
    # Shrink toward a roster-aware preseason estimate rather than league
    # average: last three seasons plus returning production and portal
    # moves (cfbrank.prior_model). Without a prior, week-3 ratings are
    # noise -- 260 games across 228 teams once had Mississippi State #1.
    prior = build_roster_prior_or_empty(season)
    margin_model = fit(games, prior=prior or None,
                       home_field_prior=home_field_prior(season),
                       **FITTED_PARAMS)

    # Stack per-play success rate on top of scoring margin. Each carries
    # signal the other lacks; see cfbrank.ensemble for the measurements.
    epa_model = _build_epa_model(season, games, prior or None)
    model = (EnsembleModel(margin_model, epa_model) if epa_model is not None
             else margin_model)

    # Offense and defense in points. Used for DISPLAY and for predicted
    # totals only -- it did not improve spreads (the split priors bought
    # -0.002 pts/game over the control), so it stays out of the ranking.
    offdef_model = fit_points_model(season, games, home_field_prior(season))

    predictor = Predictor.from_model(model, games, total_model=offdef_model,
                                     vegas=_vegas_script(games))

    # Two ratings, never averaged. Predictive answers "who is best";
    # resume answers "who has earned the most." They genuinely disagree,
    # and forcing them into one number is the exact incoherence this
    # project exists to avoid. See docs/MODEL_SPEC.md.
    resumes = build_resumes(games, model, predictor.errors)
    resume_rank = {r.team: rank for rank, r in rank_resumes(resumes, model)}

    records = _compute_records(games)
    ranked = sorted(model.fbs_ratings().items(), key=lambda kv: -kv[1])
    off_rank = _rank_fbs(offdef_model, offdef_model.offense_rating)
    def_rank = _rank_fbs(offdef_model, offdef_model.defense_rating)

    # FCS teams are rated (see cfbrank.games for why) but never shown --
    # the rankings table, Predict page and resume board all filter down
    # to FBS alone, so the conference map does too. Without this filter,
    # an FBS team's handful of FCS opponents pulled whole FCS conferences
    # (Big Sky, Ivy League, ...) into the Conference Races section.
    #
    # "FBS Independents" is dropped too -- it isn't a real conference, its
    # members never play a "conference game" by definition, so every team
    # in it would sit at a permanent, meaningless 3-way tie that never
    # updates with results. A playoff auto-bid for "winning" that would be
    # nonsense, not a projection.
    conference_of = {
        team: conf for team, conf in _conference_map(games).items()
        if model.divisions.get(team) == "fbs" and "independent" not in conf.lower()
    }

    # Rest-of-season outlook: one shared Monte Carlo draw feeds both the
    # per-team projected record and the conference-race odds, so they
    # can't quietly disagree about how a shared game turned out.
    # Seasons with nothing left to play (remaining == []) still work --
    # every team just projects to its final record with 0 games left.
    remaining = load_remaining_schedule(season)
    sim = simulate_remaining(remaining, predictor)
    projections = project_all([team for team, _ in ranked], records, sim)
    conference_races = project_conference_races(conference_of, games, sim)

    # Playoff field: seeded from Resume rank (not Predictive -- see
    # cfbrank.playoff for why), champions taken as each conference's
    # likeliest finisher from the same conference-race odds above.
    champion_of_conference = {
        conf: standings[0][0] for conf, standings in conference_races.items()
        if standings
    }
    try:
        playoff_field = select_field(resume_rank, conference_of, champion_of_conference)
        playoff_odds = simulate_bracket(playoff_field, predictor)
    except ValueError:
        # Too few ranked conferences/teams for a 12-team field (a thin
        # fixture season, say) -- no playoff page rather than a broken one.
        playoff_field, playoff_odds = None, None

    rankings = []
    for rank, (team, rating) in enumerate(ranked, 1):
        wins, losses = records.get(team, (0, 0))
        resume = resumes.get(team)
        projection = projections.get(team)
        rankings.append({
            "rank": rank,
            "team": team,
            "rating": rating,
            "wins": wins,
            "losses": losses,
            "conference": conference_of.get(team, ""),
            "sos": _strength_of_schedule(games, team, model),
            "resume_rank": resume_rank.get(team),
            "resume_score": resume.score if resume else None,
            "best_win": _describe(resume.best_win) if resume else None,
            "worst_loss": _describe(resume.worst_loss) if resume else None,
            "offense": offdef_model.offense_rating(team),
            "defense": offdef_model.defense_rating(team),
            "offense_rank": off_rank.get(team),
            "defense_rank": def_rank.get(team),
            "projection": projection,
        })

    # "Why #N" lines for the team page, from numbers already fitted above.
    why = explain_rankings(rankings, model, prior, games, predictor)
    for row in rankings:
        row["why"] = why[row["team"]]

    return SeasonModel(
        season=season,
        model=model,
        predictor=predictor,
        rankings=rankings,
        n_games=len(games),
        max_week=max(g.week for g in games),
        has_postseason=any(g.season_type == POSTSEASON for g in games),
        conference_races=conference_races,
        uses_epa=epa_model is not None,
        games=games,
        remaining=remaining,
        playoff_field=playoff_field,
        playoff_odds=playoff_odds,
    )


def _vegas_script(games) -> VegasScript | None:
    """The Predict page's second number. Never touches the rankings."""
    market = fit_market_ratings(games, home_lines())
    return (VegasScript(market, alerts_on=alerts_active(games))
            if market is not None else None)


def _rank_fbs(model, score) -> dict[str, int]:
    """1 = best FBS team by `score` (both sides read bigger-is-better)."""
    fbs = [t for t in model.teams if model.divisions.get(t) == "fbs"]
    return {team: i for i, team in enumerate(
        sorted(fbs, key=score, reverse=True), 1)}


def _describe(game) -> str | None:
    """Short label for a marquee win or a bad loss."""
    if game is None:
        return None
    return f"{game.opponent} ({game.win_probability:.0%})"


def _compute_records(games) -> dict[str, tuple[int, int]]:
    records: dict[str, list[int]] = {}
    for game in games:
        for team in (game.home_team, game.away_team):
            records.setdefault(team, [0, 0])
        records[game.winner][0] += 1
        records[game.loser][1] += 1
    return {team: (w, l) for team, (w, l) in records.items()}


def _conference_map(games) -> dict[str, str]:
    """team -> conference, built in one pass over the season's games.

    Replaces a team-by-team scan (one full pass over every game, per
    team) that got expensive once conference-race odds needed the same
    mapping for every FBS team at once.
    """
    conference_of: dict[str, str] = {}
    for game in games:
        if game.home_conference:
            conference_of.setdefault(game.home_team, game.home_conference)
        if game.away_conference:
            conference_of.setdefault(game.away_team, game.away_conference)
    return conference_of


def _team_game_log(
    team: str, games: list, remaining: list, predictor: Predictor,
) -> list[dict]:
    """One team's full schedule: played results plus upcoming predictions,
    in week order.

    The "model said" column for a PLAYED game is this exact predictor
    evaluated on that matchup -- not a separate backtest metric -- so a
    reader comparing it against the actual score is comparing the model
    to itself honestly, calibration warts included.
    """
    entries = []
    for g in games:
        if team not in (g.home_team, g.away_team):
            continue
        is_home = g.home_team == team
        opponent = g.away_team if is_home else g.home_team
        team_points = g.home_points if is_home else g.away_points
        opp_points = g.away_points if is_home else g.home_points
        won = team_points > opp_points
        prediction = predictor.predict(g.home_team, g.away_team, g.neutral_site)
        win_probability = (prediction.home_win_probability if is_home
                           else 1 - prediction.home_win_probability)
        entries.append({
            "week": g.week,
            "opponent": opponent,
            "location": "neutral" if g.neutral_site else ("vs" if is_home else "at"),
            "played": True,
            "result": f"{'W' if won else 'L'} {team_points}-{opp_points}",
            "win_probability": win_probability,
            "upset": won != (win_probability >= 0.5),
        })

    ratings = predictor.model.ratings
    for g in remaining:
        if team not in (g.home_team, g.away_team):
            continue
        if g.home_team not in ratings or g.away_team not in ratings:
            continue  # same "no opinion, no guess" rule as simulate_remaining
        is_home = g.home_team == team
        opponent = g.away_team if is_home else g.home_team
        prediction = predictor.predict(g.home_team, g.away_team, g.neutral_site)
        win_probability = (prediction.home_win_probability if is_home
                           else 1 - prediction.home_win_probability)
        entries.append({
            "week": g.week,
            "opponent": opponent,
            "location": "neutral" if g.neutral_site else ("vs" if is_home else "at"),
            "played": False,
            "result": None,
            "win_probability": win_probability,
            "upset": None,
        })

    entries.sort(key=lambda e: e["week"])
    return entries


def _first_round_matchups(field: list, predictor: Predictor) -> list[dict]:
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


def _strength_of_schedule(games, team: str, model) -> float:
    """Average rating of every opponent faced."""
    opponents = [
        game.away_team if game.home_team == team else game.home_team
        for game in games
        if team in (game.home_team, game.away_team)
    ]
    if not opponents:
        return 0.0
    return sum(model.rating(o) for o in opponents) / len(opponents)


def _latest_season() -> int:
    seasons = available_seasons()
    if not seasons:
        raise RuntimeError("No schedule data. Run tools/fetch_mirror.py")
    return max(seasons)


def _rank_key(column: str):
    """Sort by a rank column; unranked teams go last instead of crashing."""
    return lambda row: (row[column] is None, row[column] or 0)


SORTS = {"predictive": _rank_key("rank"), "resume": _rank_key("resume_rank"),
         "offense": _rank_key("offense_rank"),
         "defense": _rank_key("defense_rank")}

# Absolute paths for the live app. tools/build_static_site.py renders the
# same templates with a different `nav` dict (relative filenames) so a
# GitHub Pages export needs no server -- one set of templates either way.
NAV = {"home": "/", "predict": "/predict", "about": "/about", "static": "/static",
       "playoff": "/playoff"}


def _rank_url(season: int, sort: str) -> str:
    return f"/?season={season}&sort={sort}"


def _team_url(team: str, season: int) -> str:
    return f"/team/{quote(team)}?season={season}"


# Sentinel so _base_context can tell "team_url wasn't passed, default to
# the live route" apart from "team_url was explicitly passed as None,
# meaning no link at all." build_static_site.py relies on the second
# case for seasons that don't get a team page (every season but the
# latest) -- a plain `team_url=None` falling back to the live site's
# /team/ path there would silently ship a dead link into the static
# export.
_UNSET = object()


def _base_context(
    active_tab: str, season: int,
    nav: dict | None = None, rank_url=None, team_url=_UNSET, **extra,
) -> dict:
    """Shared page context. `nav`/`rank_url`/`team_url` default to the live
    routes' absolute paths; tools/build_static_site.py overrides them with
    relative filenames so the exact same templates render a standalone
    static export."""
    return {
        "active_tab": active_tab,
        "season": season,
        "seasons": available_seasons(),
        "nav": nav or NAV,
        "rank_url": rank_url or _rank_url,
        "team_url": _team_url if team_url is _UNSET else team_url,
        **extra,
    }


def _predict_teams(data: SeasonModel) -> list[dict]:
    """Team picker entries for the predict page: name, rank, conference,
    record. Shared with tools/build_static_site.py so the offline export
    can never quietly list a different set of teams than the live page."""
    rankings = {r["team"]: r for r in data.rankings}
    return [
        {"name": t, "rank": rankings[t]["rank"],
         "conference": rankings[t]["conference"],
         "record": f"{rankings[t]['wins']}-{rankings[t]['losses']}"}
        for t in sorted(data.model.fbs_ratings()) if t in rankings
    ]


@app.get("/", response_class=HTMLResponse)
def rankings_page(
    request: Request,
    season: int | None = None,
    sort: str = "predictive",
):
    """The rankings table, sortable by either rating.

    Sorting re-orders one table rather than serving two pages, so the
    other rating always stays visible next to it. Seeing the disagreement
    is the point.
    """
    season = season or _latest_season()
    data = get_season(season)

    sort = sort if sort in SORTS else "predictive"
    rows = sorted(data.rankings, key=SORTS[sort])

    return TEMPLATES.TemplateResponse(request, "rankings.html", _base_context(
        "rankings", season,
        rankings=rows,
        sort=sort,
        model=data.model,
        n_games=data.n_games,
        max_week=data.max_week,
        has_postseason=data.has_postseason,
        conference_races=data.conference_races,
    ))


@app.get("/team/{team_name}", response_class=HTMLResponse)
def team_page(request: Request, team_name: str, season: int | None = None):
    """One team's card: record, projection, and full game log.

    `team_name` comes in URL-decoded by FastAPI already, so this matches
    directly against the rankings dict built from the same team names.
    """
    season = season or _latest_season()
    data = get_season(season)
    row = next((r for r in data.rankings if r["team"] == team_name), None)
    game_log = (_team_game_log(team_name, data.games, data.remaining, data.predictor)
                if row else [])
    return TEMPLATES.TemplateResponse(request, "team.html", _base_context(
        "rankings", season,
        team=team_name,
        row=row,
        game_log=game_log,
    ))


@app.get("/playoff", response_class=HTMLResponse)
def playoff_page(request: Request, season: int | None = None):
    """Projected 12-team CFP field, seeded, with per-team bracket odds.

    None when the season is too thin for a 12-team field (get_season
    already tried and gave up) -- the template shows a plain message
    instead of a broken bracket.
    """
    season = season or _latest_season()
    data = get_season(season)
    first_round = (_first_round_matchups(data.playoff_field, data.predictor)
                  if data.playoff_field else [])
    return TEMPLATES.TemplateResponse(request, "playoff.html", _base_context(
        "playoff", season,
        field=data.playoff_field,
        odds=data.playoff_odds,
        first_round=first_round,
    ))


@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request, season: int | None = None):
    season = season or _latest_season()
    data = get_season(season)
    return TEMPLATES.TemplateResponse(request, "predict.html", _base_context(
        "predict", season,
        teams=_predict_teams(data),
        prediction=None,
    ))


@app.post("/predict", response_class=HTMLResponse)
def run_prediction(
    request: Request,
    home_team: str = Form(...),
    away_team: str = Form(...),
    season: int = Form(...),
    neutral_site: str = Form("off"),
):
    data = get_season(season)
    is_neutral = neutral_site == "on"
    home_team, away_team = home_team.strip(), away_team.strip()

    # The pickers are free-text search boxes now, so a typo can reach us.
    known = data.model.fbs_ratings()
    unknown = [t for t in (away_team, home_team) if t not in known]
    if unknown:
        return TEMPLATES.TemplateResponse(request, "_prediction_result.html", {
            "error": f"Unknown team: {', '.join(unknown)}. "
                     "Pick one from the list.",
            "prediction": None,
        })

    if home_team == away_team:
        return TEMPLATES.TemplateResponse(request, "_prediction_result.html", {
            "error": "A team cannot play itself. Pick two different teams.",
            "prediction": None,
        })

    prediction = data.predictor.predict(home_team, away_team, is_neutral)
    return TEMPLATES.TemplateResponse(request, "_prediction_result.html", {
        "prediction": prediction,
        "error": None,
    })


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    """Accuracy and methodology, on one page.

    Accuracy is graded from the saved production benchmark rather than
    recomputed here. The old tab re-ran a margin-only backtest on every
    load, which was slow and, worse, graded a model the site no longer
    served once the ensemble shipped.
    """
    season = _latest_season()
    return TEMPLATES.TemplateResponse(request, "about.html", _base_context(
        "about", season,
        model=get_season(season).model,
        card=load_scorecard(BENCHMARK_PATH),
    ))


@app.get("/accuracy")
@app.get("/method")
def legacy_about():
    """Old tab URLs, kept alive for bookmarks."""
    return RedirectResponse("/about", status_code=301)


@app.get("/health")
def health():
    return {"status": "ok", "seasons": available_seasons()}
