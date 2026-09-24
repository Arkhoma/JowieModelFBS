"""Render the Jinja templates to flat static HTML for GitHub Pages.

No server needed once this runs: every page becomes a plain file, and
the one page with real backend logic -- Predict a Game -- ships its
math as JSON (cfbrank.static_export.predictor_payload) so
app/static/predict.js can compute it in the browser instead.

This calls app.main's own context builders (_base_context,
_predict_teams, get_season) rather than re-implementing them, so the
static export can only drift from the live app if someone edits one
side and not the other -- and the identity tests in
tests/test_static_export.py catch the math half of that.

Usage:
    python tools/build_static_site.py [--out _site] [--season 2026]

Without --season, every available season gets a rankings page (that's
what the live site's season switcher expects to link to); Predict and
About stay pinned to the latest season, same as the live defaults.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import (  # noqa: E402
    BENCHMARK_PATH, SORTS, _base_context, _predict_teams, get_season,
)
from cfbrank.games import available_seasons  # noqa: E402
from cfbrank.scorecard import load_scorecard  # noqa: E402
from cfbrank.static_export import predictor_payload  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

TEMPLATES_DIR = ROOT / "app" / "templates"
STATIC_SRC = ROOT / "app" / "static"

# Every page lives flat in the output directory, so relative names with
# no subfolders are enough -- see the "index.html" alias below.
NAV = {"home": "index.html", "predict": "predict.html",
       "about": "about.html", "static": "static"}


def _rank_url(season: int, sort: str) -> str:
    return f"rankings-{season}-{sort}.html"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _write(out_dir: Path, name: str, html: str) -> None:
    (out_dir / name).write_text(html, encoding="utf-8")


def build_rankings(env: Environment, out_dir: Path, seasons: list[int]) -> None:
    for season in seasons:
        data = get_season(season)
        for sort_key, key_fn in SORTS.items():
            rows = sorted(data.rankings, key=key_fn)
            html = env.get_template("rankings.html").render(_base_context(
                "rankings", season, nav=NAV, rank_url=_rank_url,
                rankings=rows, sort=sort_key, model=data.model,
                n_games=data.n_games, max_week=data.max_week,
                has_postseason=data.has_postseason,
            ))
            _write(out_dir, _rank_url(season, sort_key), html)


def build_about(env: Environment, out_dir: Path, season: int) -> None:
    data = get_season(season)
    html = env.get_template("about.html").render(_base_context(
        "about", season, nav=NAV,
        model=data.model, card=load_scorecard(BENCHMARK_PATH),
    ))
    _write(out_dir, "about.html", html)


def build_predict(env: Environment, out_dir: Path, season: int) -> None:
    """Latest season only -- ratings are season-specific and the picker
    already defaults here on the live site. Multi-season predict would
    mean one payload per season; not worth it until someone asks."""
    data = get_season(season)
    teams = _predict_teams(data)
    payload = predictor_payload(data.predictor, [t["name"] for t in teams])
    html = env.get_template("predict.html").render(_base_context(
        "predict", season, nav=NAV, static_build=True,
        teams=teams, prediction=None, model_json=payload,
    ))
    _write(out_dir, "predict.html", html)


def build(out_dir: Path, seasons: list[int] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _env()
    all_seasons = available_seasons()
    seasons = seasons or all_seasons
    latest = max(all_seasons)

    build_rankings(env, out_dir, seasons)
    build_about(env, out_dir, latest)
    build_predict(env, out_dir, latest)

    # Root landing page = latest season, predictive sort. Only makes
    # sense if that page actually got built.
    if latest in seasons:
        shutil.copy(out_dir / _rank_url(latest, "predictive"),
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
