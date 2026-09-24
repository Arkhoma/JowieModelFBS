# Optional: fetching from the CFBD API

The site runs entirely on the public mirror (`tools/fetch_mirror.py`, no
key). `tools/fetch_cfbd.py` pulls extras straight from
[CollegeFootballData](https://collegefootballdata.com): 2021-22 bowl
results (missing from the mirror) and recruiting/talent data.

## Setup

- Python 3.9+. The script uses only the standard library.
- A free CFBD key: https://collegefootballdata.com/key
- Copy `.env.example` to `.env` and paste the key in. `.env` is
  gitignored. Or set `CFBD_API_KEY` in your environment.

## Run

Smoke test first. It's one cheap call that checks the key and the network:

    python tools/fetch_cfbd.py --smoke

Expected: `OK -- 134 FBS teams returned.` A `401` means the key is wrong
(check for typos or stray quotes). A network error means your network
blocks the API; try another connection.

Then the full pull:

    python tools/fetch_cfbd.py --years 2021 2022 2023 2024 2025 2026

This takes 15-25 minutes and waits one second between calls to be polite
to a free API. If it's interrupted, re-run the same command: anything
already on disk is skipped.

Files land in `data/raw/<year>/`. `cfbrank.games` picks up CFBD postseason
files automatically, so there's nothing to change in the code.

## Different machine?

If the machine running the site can't reach the API, run the script
somewhere that can, then copy `data/raw/` across. `manifest.json` lists
every file and its size, so you can check nothing got truncated.
