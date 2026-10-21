# CFB Rankings

College football ratings with two separate answers, never averaged:

- **Predictive**: who wins on a neutral field (uses margin).
- **Resume**: who has earned the most (wins only, vs a top-25 benchmark).

No hand-tuned weights; every parameter is fit to out-of-sample error.

## Quick start

Windows: double-click `run.bat`. macOS/Linux: `chmod +x run.sh && ./run.sh`.

Creates a venv, downloads ~45 MB of public data, opens <http://127.0.0.1:8421>.

Manual (Python 3.11+):

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/fetch_mirror.py
python -m uvicorn app.main:app --port 8421
```

## Pages

Every page works for every season since 2014 (season menu top right; the
nav keeps your season). The header search finds any team, any season
(try "LSU 2019"), and every team name on the site links to its page.

- **Rankings**: predictive, resume, offense and defense, sortable by each.
- **Team pages**: record, projection, "Why #N", full schedule vs the model.
- **Conferences**: conference-race odds plus a box plot of each league's ranks.
- **Predict**: score, spread, total and win probability for any matchup,
  plus the **Time Machine** (any team-season vs any other, no Vegas line).
- **Playoff**: projected 12-team field and bracket odds.
- **How It Works**: accuracy vs the closing line, method, what was tested.

## Model

- Ridge regression over all FBS + FCS teams: `margin = home - away + home field`.
- Blend of scoring margin and play-by-play success rate.
- Preseason prior from the last three seasons, returning production and transfers.
- Offense/defense split drives predicted totals, not the rank.

## Accuracy (2021-2025, 5,215 games incl. 146 postseason, walk-forward)

| | Ours | Vegas |
|---|---|---|
| Spread miss | 12.34 | 11.97 |
| Winner picked | 72.7% | 73.5% |
| Total miss | 12.97 | 12.65 |

Gap per game: weeks 4-8 +0.52, week 9+ +0.21, postseason +1.08 (bowls are
noisy; opt-outs aren't in the data).

## Development

```bash
python -m pytest tests -q                 # full suite
python tools/benchmark_vs_market.py       # regrades vs Vegas; feeds How It Works
python tools/fetch_mirror.py --current    # in-season refresh, then restart
python tools/build_css.py                 # after adding Tailwind classes
```

CSS is Tailwind compiled ahead of time into `app/static/site.css` (no
in-browser CDN). `build_css.py` needs the v3.4.17 standalone CLI in
`tools/bin/` (no Node); see its docstring.

Full reasoning and history: [`docs/MODEL_SPEC.md`](docs/MODEL_SPEC.md).

## Publishing (GitHub Pages)

`python tools/build_static_site.py` renders every page for every season
to flat HTML in `_site/` (~1,800 files, ~4 min). The latest season keeps
plain filenames (`team-ohio-state.html`); older ones add the year
(`team-ohio-state-2019.html`). Predict and the Time Machine run in the
browser from embedded JSON, so there's no server.
`python tools/check_static_links.py _site` fails on any dead link.

`.github/workflows/publish.yml` does this **every Sunday at 14:00 UTC**
(9am CDT / 8am CST): fetch new games, run tests, build CSS and site,
check links, deploy. If any step fails, nothing gets published. One-time setup: push to GitHub, then
**Settings > Pages > Source: GitHub Actions**. Use **Actions > Run
workflow** to refresh by hand.

## Data

[sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data),
a public CollegeFootballData mirror. No key needed. Postseason before 2023
comes from the mirror's `cfb_games_info.csv` (2007-2020); 2021-22 bowls
need `tools/fetch_cfbd.py`, which uses `CFBD_API_KEY` from `.env`.

MIT license.
