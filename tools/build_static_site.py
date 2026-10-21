"""Render the Jinja templates to flat static HTML for GitHub Pages.

No server needed once this runs: every page becomes a plain file, and
the one page with real backend logic -- Predict a Game -- ships its
math as JSON (cfbrank.static_export.predictor_payload) so
app/static/predict.js can compute it in the browser instead.

Page content comes from app.pages, the same context builders the live
app uses, so the two sites can only differ in URL scheme (STATIC_URLS
below vs app.pages.LIVE_URLS). tests/test_static_export.py pins the
predictor math; tests/test_pages.py pins the URLs.

Usage:
    python tools/build_static_site.py [--out _site] [--season 2026]

Every available season gets every season-specific page: rankings (all
sorts), each FBS team, Predict and Playoff. The latest season keeps the
plain filenames (predict.html, team-ohio-state.html) so existing links
never break; older seasons add the year (predict-2024.html,
team-ohio-state-2024.html). About and the Time Machine (every season on
one page) are season-independent: one copy each.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import BENCHMARK_PATH, get_season  # noqa: E402
from app.pages import (  # noqa: E402
    SORTS, SiteUrls, base_context, playoff_context, predict_context,
    rankings_context, team_context, time_machine_context,
)
from cfbrank.games import available_seasons  # noqa: E402
from cfbrank.scorecard import load_scorecard  # noqa: E402
from cfbrank.static_export import predictor_payload  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

TEMPLATES_DIR = ROOT / "app" / "templates"
STATIC_SRC = ROOT / "app" / "static"


@lru_cache(maxsize=None)
def _latest() -> int:
    return max(available_seasons())


def _suffix(season: int) -> str:
    """'' for the latest season (stable URLs), '-2024' for older ones."""
    return "" if season == _latest() else f"-{season}"


def _team_slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower()).strip("-")


# Every page lives flat in the output directory, so relative names with
# no subfolders are enough -- see the "index.html" alias in build().
STATIC_URLS = SiteUrls(
    nav={"home": "index.html", "about": "about.html", "static": "static",
         "time_machine": "time-machine.html"},
    rank=lambda season, sort: f"rankings-{season}-{sort}.html",
    team=lambda team, season: f"team-{_team_slug(team)}{_suffix(season)}.html",
    predict=lambda season: f"predict{_suffix(season)}.html",
    playoff=lambda season: f"playoff{_suffix(season)}.html",
)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _render(env: Environment, out_dir: Path, filename: str, template: str,
            active_tab: str, season: int, **extra) -> None:
    html = env.get_template(template).render(
        base_context(active_tab, season, urls=STATIC_URLS, **extra))
    (out_dir / filename).write_text(html, encoding="utf-8")


def build_season(env: Environment, out_dir: Path, season: int) -> None:
    """Every season-specific page for one season."""
    data = get_season(season)
    urls = STATIC_URLS
    for sort in SORTS:
        _render(env, out_dir, urls.rank(season, sort), "rankings.html",
                "rankings", season, **rankings_context(data, sort))
    for row in data.rankings:
        team = row["team"]
        _render(env, out_dir, urls.team(team, season), "team.html",
                "rankings", season, **team_context(data, team))
    context = predict_context(data)
    payload = predictor_payload(data.predictor,
                                [t["name"] for t in context["teams"]])
    _render(env, out_dir, urls.predict(season), "predict.html", "predict",
            season, static_build=True, model_json=payload, **context)
    _render(env, out_dir, urls.playoff(season), "playoff.html", "playoff",
            season, **playoff_context(data))


def build(out_dir: Path, seasons: list[int] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _env()
    seasons = seasons or available_seasons()
    latest = _latest()

    for season in seasons:
        build_season(env, out_dir, season)
    _render(env, out_dir, "about.html", "about.html", "about", latest,
            model=get_season(latest).model,
            card=load_scorecard(BENCHMARK_PATH))
    _render(env, out_dir, "time-machine.html", "time_machine.html", "predict",
            latest, **time_machine_context(get_season))

    # Root landing page = latest season, predictive sort. Only makes
    # sense if that page actually got built.
    if latest in seasons:
        shutil.copy(out_dir / STATIC_URLS.rank(latest, "predictive"),
                    out_dir / "index.html")

    dest_static = out_dir / "static"
    if dest_static.exists():
        shutil.rmtree(dest_static)
    shutil.copytree(STATIC_SRC, dest_static)

    n_files = sum(1 for _ in out_dir.rglob("*") if _.is_file())
    print(f"Built {n_files} files into {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="_site", help="Output directory")
    parser.add_argument("--season", type=int, action="append",
                        help="Limit to this season (repeatable). "
                             "Default: every available season.")
    args = parser.parse_args()
    build(ROOT / args.out, args.season)


if __name__ == "__main__":
    main()
