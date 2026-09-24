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

- **Rankings**: predictive, resume, offense and defense, sortable by each.
- **Predict**: score, spread, total and win probability for any matchup.
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
```

Full reasoning and history: [`docs/MODEL_SPEC.md`](docs/MODEL_SPEC.md).

## Publishing (GitHub Pages)

`python tools/build_static_site.py` renders every page to flat HTML in
`_site/` (~2 min). Predict a Game runs in the browser from an embedded
JSON copy of the model, so there's no server.

`.github/workflows/publish.yml` does this **every Sunday at 14:00 UTC**
(9am CDT / 8am CST): fetch new games, run tests, build, deploy. If the
tests fail, nothing gets published. One-time setup: push to GitHub, then
**Settings > Pages > Source: GitHub Actions**. Use **Actions > Run
workflow** to refresh by hand.

## Data

[sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data),
a public CollegeFootballData mirror. No key needed. Postseason before 2023
comes from the mirror's `cfb_games_info.csv` (2007-2020); 2021-22 bowls
need `tools/fetch_cfbd.py`, which uses `CFBD_API_KEY` from `.env`.

MIT license.
