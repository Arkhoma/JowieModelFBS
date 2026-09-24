"""Postseason loading: supplements merge in, and bowls sort AFTER the season."""

from __future__ import annotations

import csv
import gzip
import json

from cfbrank.games import POSTSEASON, load_season

COLUMNS = ["game_id", "season", "week", "season_type", "start_date",
           "neutral_site", "conference_game", "home_team", "away_team",
           "home_division", "away_division", "home_points", "away_points",
           "home_conference", "away_conference"]


def _row(gid, week, date, home, away, hp, ap, kind="regular", div="fbs"):
    return {"game_id": gid, "season": "2019", "week": str(week),
            "season_type": kind, "start_date": f"{date}T20:00:00.000Z",
            "neutral_site": "FALSE", "conference_game": "FALSE",
            "home_team": home, "away_team": away, "home_division": div,
            "away_division": div, "home_points": str(hp),
            "away_points": str(ap), "home_conference": "",
            "away_conference": ""}


def _write(path, rows, columns=COLUMNS, encoding="utf-8"):
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _regular_season(directory):
    _write(directory / "schedules_2019.csv", [
        _row("1", 1, "2019-08-31", "A", "B", 30, 10),
        _row("2", 14, "2019-11-30", "C", "D", 21, 20),
    ])


def test_games_info_fills_missing_postseason_with_divisions(tmp_path):
    _regular_season(tmp_path)
    bowl = _row("9", 1, "2019-12-28", "A", "C", 24, 17, kind=POSTSEASON)
    no_division = [c for c in COLUMNS if not c.endswith("_division")]
    _write(tmp_path / "cfb_games_info.csv", [bowl], no_division, "latin-1")

    games = load_season(2019, tmp_path)
    post = [g for g in games if g.season_type == POSTSEASON]
    assert len(post) == 1
    assert post[0].home_division == "fbs"       # filled from the regular season
    assert post[0].week > 14                    # not upstream's "week 1"


def test_cfbd_dump_preferred_over_games_info(tmp_path):
    schedules = tmp_path / "schedules"
    schedules.mkdir()
    _regular_season(schedules)
    (tmp_path / "2019").mkdir()
    with gzip.open(tmp_path / "2019" / "games.json.gz", "wt") as handle:
        json.dump([{"id": 7, "season": 2019, "week": 1,
                    "seasonType": POSTSEASON,
                    "startDate": "2020-01-13T20:00:00.000Z",
                    "neutralSite": True, "homeTeam": "A", "awayTeam": "C",
                    "homeClassification": "fbs",
                    "awayClassification": "fbs",
                    "homePoints": 42, "awayPoints": 25}], handle)

    post = [g for g in load_season(2019, schedules)
            if g.season_type == POSTSEASON]
    assert [(g.game_id, g.neutral_site, g.margin) for g in post] == [
        ("7", True, 17)]


def test_playoff_rounds_keep_their_order(tmp_path):
    _write(tmp_path / "schedules_2019.csv", [
        _row("1", 1, "2019-08-31", "A", "B", 30, 10),
        _row("2", 14, "2019-11-30", "C", "D", 21, 20),
        _row("3", 1, "2019-12-21", "A", "C", 20, 10, kind=POSTSEASON),
        _row("4", 1, "2020-01-01", "A", "D", 20, 10, kind=POSTSEASON),
        _row("5", 1, "2020-01-20", "A", "B", 20, 10, kind=POSTSEASON),
    ])
    weeks = {g.game_id: g.week for g in load_season(2019, tmp_path)}
    assert 14 < weeks["3"] < weeks["4"] < weeks["5"]


def test_postseason_during_regular_weeks_keeps_that_week(tmp_path):
    """FCS playoffs start before the FBS regular season ends."""
    _write(tmp_path / "schedules_2019.csv", [
        _row("1", 1, "2019-08-31", "A", "B", 30, 10),
        _row("2", 13, "2019-11-23", "C", "D", 21, 20),
        _row("3", 14, "2019-11-30", "A", "D", 21, 20),
        _row("4", 1, "2019-11-24", "B", "C", 20, 10, kind=POSTSEASON),
    ])
    weeks = {g.game_id: g.week for g in load_season(2019, tmp_path)}
    assert weeks["4"] == 13
