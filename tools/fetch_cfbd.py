"""Standalone CFBD bulk fetcher.

Run this on a machine that can reach api.collegefootballdata.com, then zip
up the `data/raw/` folder and move it to the workspace machine.

Deliberately STDLIB ONLY -- no pip install, no venv. Just Python 3.9+.

Usage:
    set CFBD_API_KEY=your_key_here        (Windows cmd)
    $env:CFBD_API_KEY="your_key_here"     (PowerShell)
    export CFBD_API_KEY=your_key_here     (mac/Linux)

    python fetch_cfbd.py --smoke                  # 1 cheap call, verify access
    python fetch_cfbd.py --years 2022 2023 2024 2025 2026
    python fetch_cfbd.py --only priors --years 2014 2015  # roster data only

Behind a corporate proxy, point urllib at it first, e.g.
    set HTTPS_PROXY=http://your.proxy:8080

The free tier is 1,000 calls/month. A full season costs ~32 calls (most of
it play-by-play); `--only priors` costs ~5.

Resumable: already-downloaded files are skipped, so a crash or Ctrl-C costs
you nothing. Re-run the same command to pick up where it left off.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("CFBD_BASE_URL", "https://api.collegefootballdata.com")
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# Be a good citizen: the free tier is rate limited.
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 4

# Regular-season weeks to sweep. Empty weeks are harmless (we just get []).
REGULAR_WEEKS = range(1, 17)


class FetchError(RuntimeError):
    """Raised when an endpoint cannot be retrieved after retries."""


def _dotenv_key(path: Path = OUT_ROOT.parent.parent / ".env") -> str:
    """CFBD_API_KEY from the project's gitignored .env, if present."""
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "CFBD_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def _api_key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip() or _dotenv_key()
    if not key:
        sys.exit(
            "ERROR: CFBD_API_KEY is not set.\n"
            "  Windows cmd:  set CFBD_API_KEY=your_key_here\n"
            "  PowerShell:   $env:CFBD_API_KEY=\"your_key_here\"\n"
            "  mac/Linux:    export CFBD_API_KEY=your_key_here"
        )
    return key


def fetch_json(path: str, params: dict | None = None) -> list | dict:
    """GET an endpoint and return parsed JSON, with retry/backoff."""
    query = urllib.parse.urlencode(params or {})
    url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Accept": "application/json",
            "User-Agent": "cfb-rankings/0.1 (personal research)",
        },
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise FetchError(
                    "401 Unauthorized -- the API key was rejected. "
                    "Check for stray quotes/spaces, or reissue the key."
                ) from exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                backoff = 2**attempt
                print(f"    HTTP {exc.code}, retrying in {backoff}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(backoff)
                continue
            raise FetchError(f"HTTP {exc.code} on {url}") from exc
        except urllib.error.URLError as exc:
            # This is the failure mode on a network that blocks the domain.
            if attempt < MAX_RETRIES:
                backoff = 2**attempt
                print(f"    Network error ({exc.reason}), retrying in {backoff}s")
                time.sleep(backoff)
                continue
            raise FetchError(
                f"Could not reach {BASE_URL} -- {exc.reason}. "
                "Is this machine's network blocking the domain?"
            ) from exc

    raise FetchError(f"Exhausted retries for {url}")


def save(payload: list | dict, *parts: str) -> int:
    """Write gzipped JSON to data/raw/<parts...>. Returns record count."""
    destination = OUT_ROOT.joinpath(*parts).with_suffix(".json.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return len(payload) if isinstance(payload, list) else 1


def already_have(*parts: str) -> bool:
    return OUT_ROOT.joinpath(*parts).with_suffix(".json.gz").exists()


def grab(label: str, path: str, params: dict, *parts: str) -> None:
    """Fetch one endpoint and persist it, skipping if already on disk."""
    if already_have(*parts):
        print(f"  [skip] {label}")
        return
    print(f"  [get ] {label} ...", end=" ", flush=True)
    try:
        payload = fetch_json(path, params)
    except FetchError as exc:
        print(f"FAILED\n         {exc}")
        return
    count = save(payload, *parts)
    print(f"{count} records")
    time.sleep(SLEEP_BETWEEN_CALLS)


# The transfer portal endpoint has no data before this season.
FIRST_PORTAL_SEASON = 2021
FIRST_COACH_YEAR = 2008


def fetch_priors(year: int) -> None:
    """Everything known about a roster before kickoff."""
    y = str(year)
    # Returning production = preseason prior. Talent = recruiting baseline.
    grab("returning production", "/player/returning", {"year": year},
         y, "returning_production")
    grab("team talent", "/talent", {"year": year}, y, "talent")
    # Individual recruits join to rosters via recruit_ids.
    grab("recruits", "/recruiting/players", {"year": year}, y, "recruits")
    grab("recruiting classes", "/recruiting/teams", {"year": year},
         y, "recruiting_teams")
    if year >= FIRST_PORTAL_SEASON:
        grab("transfer portal", "/player/portal", {"year": year}, y, "portal")


def fetch_games(year: int) -> None:
    """Just the results (1 call). The mirror lacks 2021-22 bowls."""
    grab("games", "/games", {"year": year, "seasonType": "both"},
         str(year), "games")


def fetch_context(year: int) -> None:
    """Schedules, results, polls, lines and published ratings."""
    y = str(year)
    grab("teams (FBS)", "/teams/fbs", {"year": year}, y, "teams")
    grab("calendar", "/calendar", {"year": year}, y, "calendar")
    fetch_games(year)
    grab("team box scores", "/games/teams",
         {"year": year, "seasonType": "both"}, y, "games_teams")
    grab("drives", "/drives",
         {"year": year, "seasonType": "both"}, y, "drives")

    # AP / Coaches polls -- this is what we benchmark the model against.
    grab("polls", "/rankings", {"year": year, "seasonType": "both"}, y, "polls")
    # Betting lines: the sharpest public predictor, our accuracy yardstick.
    grab("betting lines", "/lines",
         {"year": year, "seasonType": "both"}, y, "lines")
    # Existing published ratings, for sanity-checking our output.
    grab("SP+ ratings", "/ratings/sp", {"year": year}, y, "ratings_sp")
    grab("SRS ratings", "/ratings/srs", {"year": year}, y, "ratings_srs")


def fetch_plays(year: int) -> None:
    """Play-by-play: the big one, paginated by week (~21 calls)."""
    y = str(year)
    for week in REGULAR_WEEKS:
        grab(f"plays wk{week:02d}", "/plays",
             {"year": year, "week": week, "seasonType": "regular"},
             y, "plays", f"regular_{week:02d}")

    for week in (1, 2, 3, 4, 5):
        grab(f"plays post wk{week}", "/plays",
             {"year": year, "week": week, "seasonType": "postseason"},
             y, "plays", f"postseason_{week:02d}")


GROUPS = {"priors": fetch_priors, "context": fetch_context, "plays": fetch_plays,
          "games": fetch_games}


def fetch_season(year: int, groups: list[str]) -> None:
    print(f"\n=== {year} ===")
    for name in groups:
        GROUPS[name](year)


def write_manifest(years: list[int]) -> None:
    """Record what actually landed on disk, so the other machine can verify."""
    files = sorted(OUT_ROOT.rglob("*.json.gz"))
    manifest = {
        "years_requested": years,
        "base_url": BASE_URL,
        "file_count": len(files),
        "total_bytes": sum(f.stat().st_size for f in files),
        "files": [
            {
                "path": str(f.relative_to(OUT_ROOT)).replace("\\", "/"),
                "bytes": f.stat().st_size,
            }
            for f in files
        ],
    }
    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    megabytes = manifest["total_bytes"] / 1_000_000
    print(f"\nWrote manifest: {manifest_path}")
    print(f"{len(files)} files, {megabytes:.1f} MB total")
    print(f"\nNow zip this folder and move it over:\n  {OUT_ROOT}")


def smoke_test() -> None:
    """One cheap call to prove the key and the network both work."""
    print(f"Smoke test against {BASE_URL} ...")
    try:
        teams = fetch_json("/teams/fbs", {"year": 2024})
    except FetchError as exc:
        sys.exit(f"\nSMOKE TEST FAILED\n  {exc}")
    print(f"OK -- {len(teams)} FBS teams returned.")
    if teams:
        sample = teams[0]
        print(f"Sample record: {sample.get('school')} "
              f"({sample.get('conference')})")
    print("\nKey and network are good. Now run the full pull:")
    print("  python fetch_cfbd.py --years 2022 2023 2024 2025 2026")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025, 2026],
                        help="Seasons to download.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a single test call and exit.")
    parser.add_argument("--only", nargs="+", choices=list(GROUPS),
                        default=list(GROUPS),
                        help="Endpoint groups to fetch (default: all).")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    print(f"Downloading to: {OUT_ROOT}")
    print("Safe to Ctrl-C -- re-running skips what is already saved.")
    for year in args.years:
        fetch_season(year, args.only)
    if "priors" in args.only:
        # Every coach's history in ONE call. Delete data/raw/coaches.json.gz
        # after the season starts so new hires are picked up.
        grab("coaches", "/coaches",
             {"minYear": FIRST_COACH_YEAR, "maxYear": max(args.years)},
             "coaches")
    write_manifest(args.years)


if __name__ == "__main__":
    main()
