"""Download CFB data from the public sportsdataverse GitHub mirror.

No API key, no auth -- these are static files on raw.githubusercontent.com.
This is all the live site needs; tools/fetch_cfbd.py is optional extras.

Coverage, verified 2026-09-22:
    schedules/scores   2001-2026  (CSV)
    play-level data    2014-2026  (parquet, `player_stats` family, 41 MB)
    legacy pbp         2002-2021  (parquet, 927 MB, optional)

The `player_stats` naming is a trap: those files are one-row-per-PLAY with
full down/distance/field-position state, and they run through the live
season. That is what makes current-season EPA possible.

Usage:
    python tools/fetch_mirror.py                      # schedules + plays
    python tools/fetch_mirror.py --legacy-pbp 2019    # add legacy pbp

Resumable: existing files are skipped unless --force.
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

MIRROR = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Through the current calendar year, so the scheduled refresh picks up a
# new season on its own. Not-yet-published files 404 and are skipped.
THIS_YEAR = date.today().year
# From 2014, matching cfbrank.games.FIRST_SEASON: older schedules exist
# on the mirror but the model can't rate them (no play data).
SCHEDULE_YEARS = range(2014, THIS_YEAR + 1)

# The `player_stats` family is misleadingly named: it is PLAY-LEVEL data
# (one row per play, with down/distance/yards_to_goal/clock), not season
# aggregates. Crucially it runs 2014-2026 -- through the LIVE season --
# where the legacy `pbp/` directory stops at 2021. This is what makes
# current-season EPA possible. Verified 2026-09-22: 0% nulls on every
# EPA-critical field.
PLAY_YEARS = range(2014, THIS_YEAR + 1)

# Legacy play-by-play. Richer columns, but frozen at 2021. Kept only for
# deep historical work; not needed for the live pipeline.
LEGACY_PBP_YEARS = range(2002, 2022)

MAX_RETRIES = 3


def download(url: str, destination: Path, force: bool = False) -> bool:
    """Fetch one file. Returns True if bytes were written."""
    if destination.exists() and not force:
        size_mb = destination.stat().st_size / 1e6
        print(f"  [skip] {destination.name} ({size_mb:.1f} MB)")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {destination.name} ...", end=" ", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "cfb-rankings/0.1"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            # Write to a temp path first so an interrupted download never
            # leaves a truncated file that a later run would happily skip.
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            print(f"{len(payload) / 1e6:.1f} MB")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print("not published")
                return False
            if attempt < MAX_RETRIES:
                print(f"HTTP {exc.code}, retry {attempt}", end=" ", flush=True)
                time.sleep(2**attempt)
                continue
            print(f"FAILED (HTTP {exc.code})")
            return False
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < MAX_RETRIES:
                print(f"net error, retry {attempt}", end=" ", flush=True)
                time.sleep(2**attempt)
                continue
            print(f"FAILED ({exc})")
            return False
    return False


def fetch_schedules(years, force: bool) -> None:
    print(f"\nSchedules / results ({min(years)}-{max(years)})")
    # Per-season files lack the postseason before 2023; this one has the
    # 2007-2020 bowls. cfbrank.games merges it in automatically.
    download(f"{MIRROR}/schedules/cfb_games_info.csv",
             RAW_DIR / "schedules" / "cfb_games_info.csv", force)
    for year in years:
        download(
            f"{MIRROR}/schedules/csv/cfb_schedules_{year}.csv",
            RAW_DIR / "schedules" / f"schedules_{year}.csv",
            force,
        )


def fetch_plays(years, force: bool) -> None:
    """Download play-level data (the `player_stats` family, 2014-2026)."""
    if not years:
        return
    print(f"\nPlay-level data ({min(years)}-{max(years)})")
    for year in years:
        if year not in PLAY_YEARS:
            print(f"  [warn] {year} outside available range "
                  f"({min(PLAY_YEARS)}-{max(PLAY_YEARS)})")
            continue
        download(
            f"{MIRROR}/player_stats/parquet/player_stats_{year}.parquet",
            RAW_DIR / "plays" / f"plays_{year}.parquet",
            force,
        )


def fetch_rosters(years, force: bool) -> None:
    """Rosters feed the preseason prior (returning production, portal)."""
    print(f"\nRosters ({min(years)}-{max(years)})")
    for year in years:
        download(
            f"{MIRROR}/rosters/parquet/rosters_{year}.parquet",
            RAW_DIR / "rosters" / f"rosters_{year}.parquet",
            force,
        )


def fetch_lines(force: bool) -> None:
    """Closing lines: only the accuracy benchmark reads these."""
    print("\nBetting lines")
    download(f"{MIRROR}/betting/csv/cfb_line_odds.csv.gz",
             RAW_DIR / "betting" / "cfb_line_odds.csv.gz", force)


def fetch_legacy_pbp(years, force: bool) -> None:
    """Download the frozen 2002-2021 play-by-play. Large; rarely needed."""
    if not years:
        return
    print(f"\nLegacy play-by-play ({min(years)}-{max(years)}) -- large files")
    for year in years:
        if year not in LEGACY_PBP_YEARS:
            print(f"  [warn] {year} has no legacy play-by-play "
                  f"(available: {min(LEGACY_PBP_YEARS)}-{max(LEGACY_PBP_YEARS)})")
            continue
        download(
            f"{MIRROR}/pbp/parquet/play_by_play_{year}.parquet",
            RAW_DIR / "pbp_legacy" / f"pbp_{year}.parquet",
            force,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=list(SCHEDULE_YEARS),
                        help="Schedule seasons to download.")
    parser.add_argument("--plays", type=int, nargs="*", default=list(PLAY_YEARS),
                        help="Play-level seasons (2014-2026). Default: all.")
    parser.add_argument("--legacy-pbp", type=int, nargs="*", default=[],
                        help="Legacy play-by-play seasons (2002-2021). Large.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file exists.")
    parser.add_argument("--current", action="store_true",
                        help="Weekly refresh: re-download the two most "
                             "recent seasons (and the lines), fetch any "
                             "other missing files.")
    args = parser.parse_args()

    print(f"Mirror : {MIRROR}")
    print(f"Target : {RAW_DIR}")

    if args.current:
        # Last two seasons, not one: in January the bowls and title game
        # belong to LAST year's season, and this year's file doesn't
        # exist until August.
        recent = {THIS_YEAR - 1, THIS_YEAR}
        fetch_schedules([y for y in args.years if y not in recent], force=False)
        fetch_schedules(sorted(recent), force=True)
        fetch_plays([y for y in PLAY_YEARS if y not in recent], force=False)
        fetch_plays(sorted(recent), force=True)
        fetch_rosters([y for y in PLAY_YEARS if y not in recent], force=False)
        fetch_rosters(sorted(recent), force=True)
        fetch_lines(force=True)
    else:
        fetch_schedules(args.years, args.force)
        fetch_plays(args.plays, args.force)
        fetch_rosters(list(PLAY_YEARS), args.force)
        fetch_lines(args.force)
        fetch_legacy_pbp(args.legacy_pbp, args.force)

    files = sorted(RAW_DIR.rglob("*.csv")) + sorted(RAW_DIR.rglob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nOn disk: {len(files)} files, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
