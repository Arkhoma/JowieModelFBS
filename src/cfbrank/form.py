"""Walk-forward offense/defense misses: how wrong were we, game by game?

For every side of every game this answers one question: given ONLY the
games played before that week, how much better or worse than expected
did this offense do against this defense?

    miss = actual - expected            (positive = beat expectations)

`expected` comes from the production off/def engine, refit every week on
earlier games alone, with the same preseason prior production uses. So a
miss is exactly the surprise a reader would have felt that week -- and a
run of same-signed misses means the ratings are stale for that team (a
new coordinator, a new quarterback, an injury) rather than the team
having bad luck.

A defense's miss is the opposing offense's miss with the sign flipped
(positive = held them below expectations), so one table serves both.

Whether a run of misses is a REAL shift is a statistics question, and
the answer measured in tools/form_shift_probe.py is "rarely provable":
four games against four can only prove a swing of ~24 points a game. So
`season_form` only flags a team when the best before/after split clears
a high bar (FLAG_Z) and says nothing otherwise. It never feeds the
ratings -- tracking form did not improve predictions (docs/MODEL_SPEC.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import offdef
from .games import load_season, team_divisions
from .offdef_prior import TOTALS_LAMBDA, build_offdef_prior, season_sides
from .prior import home_field_prior

KINDS = ("points", "success")
START_WEEK = 3        # need at least two weeks of evidence to fit anything
MAX_WEEK = 15         # regular season + conference title games; no bowls
SUCCESS_LAMBDA = 2.0  # shrinkage for the success-rate fit (see offdef_experiment)

# Earlier seasons lack the preseason priors the weekly fits lean on.
FIRST_FORM_SEASON = 2018
MIN_SEGMENT = 3       # games on each side of a candidate break
# |z| a best-split must reach to be flagged. Calibrated on 2018-2026 team-
# seasons: 0.1%-1.8% of them cross 3.0 per test (offense/defense x points/
# success), so a flag is rare but not unheard of.
FLAG_Z = 3.0

# Each (statistic, side) pair is tested separately.
SIDE_COLUMNS = (("offense", "miss"), ("defense", "def_miss"))


def _fit(kind: str, history: list[offdef.SideGame], season: int):
    prior = build_offdef_prior(season, kind)
    if kind == "points":
        return offdef.fit(history, TOTALS_LAMBDA, prior,
                          home_field_prior=home_field_prior(season))
    return offdef.fit(history, SUCCESS_LAMBDA, prior, is_points=False)


def walk_forward_misses(season: int, kind: str) -> pd.DataFrame:
    """One row per offense-side-of-a-game: actual, expected and miss.

    Columns: season, week, game_id, team, opponent, kind, actual,
    expected, miss. Sides whose teams the history has never seen are
    skipped ("no opinion, no guess", as in production).
    """
    sides = [s for s in season_sides(season, kind) if s.week <= MAX_WEEK]
    rows: list[dict] = []
    for week in sorted({s.week for s in sides}):
        history = [s for s in sides if s.week < week]
        upcoming = [s for s in sides if s.week == week]
        if week < START_WEEK or not history or not upcoming:
            continue
        model = _fit(kind, history, season)
        for side in upcoming:
            if side.offense not in model.offense or side.defense not in model.offense:
                continue
            expected = model.expected_value(side.offense, side.defense, side.venue)
            rows.append({
                "season": season, "week": week, "game_id": side.game_id,
                "team": side.offense, "opponent": side.defense, "kind": kind,
                "actual": side.value, "expected": expected,
                "miss": side.value - expected,
            })
    return pd.DataFrame(rows)


def add_defense_misses(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `def_miss`: the opposing offense's miss, sign flipped.

    NaN where the opponent has no row (an unseen team); callers that
    need both sides drop those rows.
    """
    theirs = frame[["game_id", "team", "miss"]].rename(
        columns={"team": "opponent", "miss": "their_miss"})
    merged = frame.merge(theirs, on=["game_id", "opponent"], how="left")
    merged["def_miss"] = -merged["their_miss"]
    return merged.drop(columns="their_miss")


def scan_split(x: np.ndarray, sigma: float) -> tuple[float, int]:
    """(z, split index) of the most extreme before/after difference.

    `sigma` is the league-wide spread of one game's miss, not this
    team's own: with 6-10 games a team's own spread is too unstable to
    divide by. Searching every split point is a multiple comparison, so
    callers that need a p-value must permute the whole scan (the probe
    does); the FLAG_Z bar is calibrated with the scan included.
    """
    best, at = 0.0, MIN_SEGMENT
    for j in range(MIN_SEGMENT, len(x) - MIN_SEGMENT + 1):
        before, after = x[:j], x[j:]
        z = (after.mean() - before.mean()) / (
            sigma * np.sqrt(1 / len(before) + 1 / len(after)))
        if abs(z) > abs(best):
            best, at = z, j
    return float(best), at


@dataclass(frozen=True, slots=True)
class Shift:
    """A before/after break big enough to flag."""

    side: str          # "offense" | "defense"
    kind: str          # "points" | "success"
    z: float
    from_week: int     # first week on the "after" side
    before: float      # mean miss before the break, in the statistic's units
    after: float


@dataclass(frozen=True, slots=True)
class FormGame:
    """One played game: points, plus success rate where play data exists.

    Success misses stay fractions (0.05 = five success-rate points); the
    view scales them. None = no fitted row for that statistic.
    """

    week: int
    opponent: str
    scored: float
    expected: float
    off_miss: float
    def_miss: float | None   # None when the opponent has no fitted row
    off_success: float | None = None
    def_success: float | None = None


@dataclass(frozen=True, slots=True)
class TeamForm:
    games: tuple[FormGame, ...]
    shifts: tuple[Shift, ...]


@dataclass(frozen=True, slots=True)
class SeasonForm:
    """Every FBS team's form for one season, plus the noise yardstick."""

    sigma: dict[tuple[str, str], float]   # (kind, side) -> one game's spread
    teams: dict[str, TeamForm]


def _tables(season: int) -> dict[str, pd.DataFrame]:
    """Points (required) and success rate (when play data exists)."""
    tables = {"points": add_defense_misses(walk_forward_misses(season, "points"))}
    try:
        success = walk_forward_misses(season, "success")
        if not success.empty:
            tables["success"] = add_defense_misses(success)
    except (FileNotFoundError, ValueError, KeyError):
        pass    # no play-by-play that year: points alone still works
    return tables


def _team_shifts(team: str, grouped: dict, sigma: dict) -> list[Shift]:
    shifts = []
    for kind, by_team in grouped.items():
        if team not in by_team:
            continue
        for side, column in SIDE_COLUMNS:
            part = by_team[team].dropna(subset=[column])
            if len(part) < 2 * MIN_SEGMENT:
                continue
            x = part[column].to_numpy()
            z, at = scan_split(x, sigma[(kind, side)])
            if abs(z) >= FLAG_Z:
                shifts.append(Shift(
                    side=side, kind=kind, z=z,
                    from_week=int(part["week"].iloc[at]),
                    before=float(x[:at].mean()), after=float(x[at:].mean())))
    return shifts


def _none_if_nan(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _success_lookup(by_team: dict | None) -> dict:
    """{(team, game_id): (offense miss, defense miss)} for success rate."""
    if not by_team:
        return {}
    return {(team, r.game_id): (_none_if_nan(r.miss), _none_if_nan(r.def_miss))
            for team, part in by_team.items() for r in part.itertuples()}


def season_form(season: int) -> SeasonForm | None:
    """Form for every FBS team, or None when the season can't support it."""
    if season < FIRST_FORM_SEASON:
        return None
    try:
        tables = _tables(season)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    fbs = {t for t, d in team_divisions(load_season(season)).items() if d == "fbs"}
    tables = {kind: frame[frame["team"].isin(fbs)] for kind, frame in tables.items()}
    sigma = {(kind, side): float(frame[column].std())
             for kind, frame in tables.items() for side, column in SIDE_COLUMNS}
    grouped = {kind: {t: g.sort_values("week") for t, g in frame.groupby("team")}
               for kind, frame in tables.items()}
    success = _success_lookup(grouped.get("success"))

    teams = {}
    for team, part in grouped["points"].items():
        games = []
        for r in part.itertuples():
            off_success, def_success = success.get((team, r.game_id), (None, None))
            games.append(FormGame(
                week=int(r.week), opponent=r.opponent, scored=float(r.actual),
                expected=float(r.expected), off_miss=float(r.miss),
                def_miss=_none_if_nan(r.def_miss),
                off_success=off_success, def_success=def_success))
        teams[team] = TeamForm(tuple(games), tuple(_team_shifts(team, grouped, sigma)))
    return SeasonForm(sigma=sigma, teams=teams)
