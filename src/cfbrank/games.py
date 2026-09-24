"""Load raw schedule CSVs into clean, typed game records.

One job: turn messy CSV rows into a `Game` list the model can trust.
Everything downstream assumes this module already handled the nulls,
the string/int coercion, and the FBS/FCS classification.

Design note -- FCS teams are NOT pooled. The mirror carries ~692
FCS-vs-FCS games per season across 129 FCS teams, so each one earns an
individual rating in the same regression. Beating Montana State (15-1)
and beating Campbell (3-9) are very different accomplishments, and the
graph is connected enough to prove it. See docs/MODEL_SPEC.md section 2.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

# games.py lives at <root>/src/cfbrank/, so the project root is three up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_SCHEDULES = PROJECT_ROOT / "data" / "raw" / "schedules"

# The mirror's per-season schedule files omit the postseason before 2023.
# Two supplements fill the gap, tried in order for any season lacking it:
#   cfb_games_info.csv          mirror, bowls 2007-2020, no division column
#   <raw>/<season>/games.json.gz  CFBD API via tools/fetch_cfbd.py (any year)
GAMES_INFO_FILE = "cfb_games_info.csv"
POSTSEASON = "postseason"

NULLISH = {"", "NA", "NULL", "None"}

# Divisions we rate. 'ii' and 'iii' opponents are too weakly connected to
# the FBS graph to carry signal, and including them invites instability.
RATED_DIVISIONS = {"fbs", "fcs"}


@dataclass(frozen=True, slots=True)
class Game:
    """One completed game between two rated teams."""

    game_id: str
    season: int
    week: int
    season_type: str
    kickoff: datetime | None
    home_team: str
    away_team: str
    home_division: str
    away_division: str
    home_points: int
    away_points: int
    neutral_site: bool
    conference_game: bool
    home_conference: str
    away_conference: str
    notes: str = ""   # e.g. "CFP Semifinal at the Rose Bowl"; mostly blank
    @property
    def margin(self) -> int:
        """Positive means the home team won. Neutral-site still uses this sign."""
        return self.home_points - self.away_points

    @property
    def winner(self) -> str:
        return self.home_team if self.margin > 0 else self.away_team

    @property
    def loser(self) -> str:
        return self.away_team if self.margin > 0 else self.home_team

    @property
    def is_fbs_only(self) -> bool:
        return self.home_division == "fbs" and self.away_division == "fbs"

    @property
    def involves_fcs(self) -> bool:
        return "fcs" in (self.home_division, self.away_division)


def _blank(value: str | None) -> bool:
    return value is None or value.strip() in NULLISH


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().upper() == "TRUE"


def _parse_kickoff(value: str | None) -> datetime | None:
    """CFBD emits ISO-8601 with a trailing Z. Return tz-aware UTC or None."""
    if _blank(value):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _row_to_game(row: dict[str, str]) -> Game | None:
    """Convert one CSV row to a Game, or None if it is unusable.

    Returning None (rather than raising) is deliberate: the raw files
    contain D-II/D-III filler and future unplayed games, and those are
    expected absences, not errors.
    """
    if _blank(row.get("home_points")) or _blank(row.get("away_points")):
        return None  # not yet played

    home_division = (row.get("home_division") or "").strip().lower()
    away_division = (row.get("away_division") or "").strip().lower()
    if home_division not in RATED_DIVISIONS or away_division not in RATED_DIVISIONS:
        return None

    try:
        season = int(row["season"])
        week = int(row["week"])
        home_points = int(row["home_points"])
        away_points = int(row["away_points"])
    except (KeyError, ValueError):
        return None

    # A 0-0 "result" is not a tie -- overtime has made genuine 0-0 games
    # impossible since 1996. Upstream records cancelled/abandoned games
    # this way instead of leaving the score null. Observed 5 times across
    # 2022-2024, all FCS-vs-FCS. Letting these through would feed the
    # regression phantom games where neither team could score.
    if home_points == 0 and away_points == 0:
        return None

    return Game(
        game_id=row.get("game_id", ""),
        season=season,
        week=week,
        season_type=(row.get("season_type") or "regular").strip(),
        kickoff=_parse_kickoff(row.get("start_date")),
        home_team=row["home_team"].strip(),
        away_team=row["away_team"].strip(),
        home_division=home_division,
        away_division=away_division,
        home_points=home_points,
        away_points=away_points,
        neutral_site=_parse_bool(row.get("neutral_site")),
        conference_game=_parse_bool(row.get("conference_game")),
        home_conference=(row.get("home_conference") or "").strip(),
        away_conference=(row.get("away_conference") or "").strip(),
        notes=(row.get("notes") or "").strip(),
    )


def _read_rows(path: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def _info_rows(season: int, directory: Path) -> list[dict[str, str]]:
    """Postseason rows from the mirror's games-info file (latin-1 encoded)."""
    path = directory / GAMES_INFO_FILE
    if not path.exists():
        return []
    return [r for r in _read_rows(path, encoding="latin-1")
            if r.get("season") == str(season)
            and r.get("season_type") == POSTSEASON]


def _cfbd_rows(season: int, directory: Path) -> list[dict[str, str]]:
    """Postseason rows from a CFBD /games dump, mapped to mirror columns."""
    path = directory.parent / str(season) / "games.json.gz"
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        games = json.load(handle)
    return [{
        "game_id": str(g.get("id", "")),
        "season": str(g.get("season", season)),
        "week": str(g.get("week", 1)),
        "season_type": POSTSEASON,
        "start_date": g.get("startDate") or "",
        "neutral_site": str(bool(g.get("neutralSite"))).upper(),
        "conference_game": str(bool(g.get("conferenceGame"))).upper(),
        "home_team": g.get("homeTeam") or "",
        "away_team": g.get("awayTeam") or "",
        "home_division": g.get("homeClassification") or "",
        "away_division": g.get("awayClassification") or "",
        "home_points": "" if g.get("homePoints") is None else str(g["homePoints"]),
        "away_points": "" if g.get("awayPoints") is None else str(g["awayPoints"]),
        "home_conference": g.get("homeConference") or "",
        "away_conference": g.get("awayConference") or "",
        "notes": g.get("notes") or "",
    } for g in games if g.get("seasonType") == POSTSEASON]


def _with_divisions(rows: list[dict[str, str]],
                    divisions: dict[str, str]) -> list[dict[str, str]]:
    """Fill blank divisions from the season's regular-season games."""
    return [{**r,
             "home_division": r.get("home_division")
             or divisions.get(r["home_team"].strip(), ""),
             "away_division": r.get("away_division")
             or divisions.get(r["away_team"].strip(), "")}
            for r in rows]


def _renumber_postseason(games: list[Game]) -> list[Game]:
    """Put postseason games on the regular season's week axis, by date.

    Upstream labels every postseason game week 1, which sorts bowls and
    playoff games before the opener. Anything that splits by week (walk-
    forward backtests, "early season" measurements) then treats the
    season's final results as already known in September.

    A game that kicks off during a regular-season week gets that week
    (FCS playoffs start in late November). Later games get one week per 7
    days after the last regular week, so each playoff round is predicted
    from the rounds before it, and nothing sees a game played after it.
    """
    post = [g for g in games if g.season_type == POSTSEASON]
    regular = [g for g in games if g.season_type != POSTSEASON]
    starts = {}
    for g in regular:
        if g.kickoff is not None:
            starts[g.week] = min(starts.get(g.week, g.kickoff), g.kickoff)
    if not post or not starts:
        return games
    last_week = max(starts)
    after = starts[last_week] + timedelta(days=7)

    def week_of(game: Game) -> int:
        if game.kickoff is None:
            return last_week + 1
        if game.kickoff >= after:
            return last_week + 1 + (game.kickoff - after).days // 7
        return max(w for w, s in starts.items()
                   if s <= game.kickoff) if game.kickoff >= min(
                       starts.values()) else min(starts)

    return regular + [replace(g, week=week_of(g)) for g in post]


def load_season(season: int, directory: Path = RAW_SCHEDULES) -> list[Game]:
    """Load all completed, rated games for one season, postseason included."""
    path = directory / f"schedules_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No schedule file for {season} at {path}. "
            "Run: python tools/fetch_mirror.py"
        )
    rows = _read_rows(path)
    if not any(r.get("season_type") == POSTSEASON for r in rows):
        seen = {r.get("game_id") for r in rows}
        supplement = _cfbd_rows(season, directory) or _info_rows(season, directory)
        divisions = {}
        for r in rows:
            divisions[r["home_team"].strip()] = (r.get("home_division") or "").strip().lower()
            divisions[r["away_team"].strip()] = (r.get("away_division") or "").strip().lower()
        rows += [r for r in _with_divisions(supplement, divisions)
                 if r.get("game_id") not in seen]
    games = [g for g in map(_row_to_game, rows) if g is not None]
    return _renumber_postseason(games)


def load_seasons(seasons, directory: Path = RAW_SCHEDULES) -> list[Game]:
    """Load several seasons at once, sorted chronologically."""
    collected: list[Game] = []
    for season in seasons:
        collected.extend(load_season(season, directory))
    collected.sort(key=lambda g: (g.season, g.week, g.kickoff or datetime.min.replace(
        tzinfo=timezone.utc)))
    return collected


def available_seasons(directory: Path = RAW_SCHEDULES) -> list[int]:
    """Which seasons are on disk."""
    seasons = []
    for path in directory.glob("schedules_*.csv"):
        try:
            seasons.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(seasons)


def team_divisions(games: list[Game]) -> dict[str, str]:
    """Map team -> division. Last observation wins, so a team that moves up
    from FCS to FBS mid-dataset is recorded at its most recent level."""
    divisions: dict[str, str] = {}
    for game in games:
        divisions[game.home_team] = game.home_division
        divisions[game.away_team] = game.away_division
    return divisions
