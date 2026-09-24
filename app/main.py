"""FastAPI app: rankings, predictions, backtest, and methodology.

Ratings are computed once at startup and cached -- the ridge solve takes
a couple of seconds and the underlying data only changes weekly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
from cfbrank.games import POSTSEASON, available_seasons, load_season  # noqa: E402
from cfbrank.offdef_prior import fit_points_model  # noqa: E402
from cfbrank.predict import Predictor  # noqa: E402
from cfbrank.prior import home_field_prior  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.resume import build_resumes, rank_resumes  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402
from cfbrank.scorecard import load_scorecard  # noqa: E402

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

    predictor = Predictor.from_model(model, games, total_model=offdef_model)

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

    rankings = []
    for rank, (team, rating) in enumerate(ranked, 1):
        wins, losses = records.get(team, (0, 0))
        resume = resumes.get(team)
        rankings.append({
            "rank": rank,
            "team": team,
            "rating": rating,
            "wins": wins,
            "losses": losses,
            "conference": _team_conference(games, team),
            "sos": _strength_of_schedule(games, team, model),
            "resume_rank": resume_rank.get(team),
            "resume_score": resume.score if resume else None,
            "best_win": _describe(resume.best_win) if resume else None,
            "worst_loss": _describe(resume.worst_loss) if resume else None,
            "offense": offdef_model.offense_rating(team),
            "defense": offdef_model.defense_rating(team),
            "offense_rank": off_rank.get(team),
            "defense_rank": def_rank.get(team),
        })

    return SeasonModel(
        season=season,
        model=model,
        predictor=predictor,
        rankings=rankings,
        n_games=len(games),
        max_week=max(g.week for g in games),
        has_postseason=any(g.season_type == POSTSEASON for g in games),
        uses_epa=epa_model is not None,
    )


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


def _team_conference(games, team: str) -> str:
    for game in games:
        if game.home_team == team and game.home_conference:
            return game.home_conference
        if game.away_team == team and game.away_conference:
            return game.away_conference
    return ""


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
NAV = {"home": "/", "predict": "/predict", "about": "/about", "static": "/static"}


def _rank_url(season: int, sort: str) -> str:
    return f"/?season={season}&sort={sort}"


def _base_context(
    active_tab: str, season: int,
    nav: dict | None = None, rank_url=None, **extra,
) -> dict:
    """Shared page context. `nav`/`rank_url` default to the live routes'
    absolute paths; tools/build_static_site.py overrides both with
    relative filenames so the exact same templates render a standalone
    static export."""
    return {
        "active_tab": active_tab,
        "season": season,
        "seasons": available_seasons(),
        "nav": nav or NAV,
        "rank_url": rank_url or _rank_url,
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
