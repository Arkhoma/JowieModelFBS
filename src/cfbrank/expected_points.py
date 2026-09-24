"""Expected points model -- the foundation of EPA.

Answers one question for any game state: given down, distance, field
position and time, how many points will the team with the ball score,
net, before the half ends?

Built empirically. For every historical play we find the NEXT scoring
event in that half and credit it as positive (offense scored) or negative
(defense scored). Averaging those outcomes over similar states gives
expected points. No assumed values anywhere -- it is all counted from
1.69M real plays.

EPA for a play is then simply:

    EPA = EP(state after) - EP(state before)

which is the number of points that play was worth.

Why build our own instead of using a published one: we can explain and
audit every value, and the model is fit on the same era of football we
are ranking. Rules and scoring environments drift; a model trained on
2005 football would misprice 2026.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Point values of the scoring events we label.
TOUCHDOWN_POINTS = 7.0  # includes the expected PAT
FIELD_GOAL_POINTS = 3.0
SAFETY_POINTS = 2.0

# Field position is bucketed rather than modelled continuously: it keeps
# the model non-parametric and inspectable. 5-yard buckets give ~20 bins
# with tens of thousands of plays each.
FIELD_BUCKET_SIZE = 5

# Distance-to-go buckets. Beyond 15 yards the differences stop mattering.
DISTANCE_BUCKETS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 100]

MIN_SAMPLES_PER_CELL = 30


@dataclass(frozen=True, slots=True)
class ExpectedPointsModel:
    """A lookup table from game state to expected next-score value."""

    table: pd.DataFrame
    global_mean: float
    n_plays: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected points for each row of a play frame."""
        keys = _state_keys(frame)
        merged = keys.merge(
            self.table, on=["down", "distance_bucket", "field_bucket"],
            how="left",
        )
        return merged["expected_points"].fillna(self.global_mean).to_numpy()


def _state_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket raw game state into the model's cells."""
    down = frame["down"].astype(int).clip(1, 4)
    distance = frame["distance"].astype(float).clip(0, 99)
    yards_to_goal = frame["yards_to_goal"].astype(float).clip(1, 99)

    return pd.DataFrame({
        "down": down.to_numpy(),
        "distance_bucket": pd.cut(
            distance, bins=DISTANCE_BUCKETS, labels=False, include_lowest=True
        ),
        "field_bucket": (yards_to_goal // FIELD_BUCKET_SIZE).astype(int).to_numpy(),
    })


def label_next_score(frame: pd.DataFrame) -> pd.Series:
    """Net points scored next, from the perspective of the team with the ball.

    Walks each half backwards. A drive that ends in the offense scoring a
    touchdown labels every play of that drive +7; if the defense scores
    next, those plays get -7. Plays with no subsequent score in the half
    are labelled 0.
    """
    points = _points_on_play(frame)
    scoring_team = np.where(points > 0, frame["team"], frame["opponent"])
    scoring_team = np.where(points == 0, None, scoring_team)

    labels = np.zeros(len(frame), dtype=float)

    group_keys = (frame["game_id"].astype(str) + "_"
                  + frame["half"].astype(str)).to_numpy()
    teams = frame["team"].to_numpy()
    abs_points = np.abs(points)

    # Walk backwards so each play sees the next score already resolved.
    next_points = 0.0
    next_team = None
    previous_group = None

    for index in range(len(frame) - 1, -1, -1):
        group = group_keys[index]
        if group != previous_group:
            next_points = 0.0
            next_team = None
            previous_group = group

        if next_team is None:
            labels[index] = 0.0
        else:
            labels[index] = (next_points if teams[index] == next_team
                             else -next_points)

        if abs_points[index] > 0:
            next_points = abs_points[index]
            next_team = scoring_team[index]

    return pd.Series(labels, index=frame.index, name="next_score")


def _points_on_play(frame: pd.DataFrame) -> np.ndarray:
    """Signed points scored on each play, from the offense's perspective.

    Positive means the team with the ball scored. Defensive scores are
    detected by a pick-six or fumble return touchdown, which show up as a
    touchdown on a play where possession was lost.
    """
    points = np.zeros(len(frame), dtype=float)

    touchdown = frame.get("is_touchdown", pd.Series(False, index=frame.index))
    field_goal = frame.get("is_field_goal_made",
                           pd.Series(False, index=frame.index))
    interception = frame.get("is_interception",
                             pd.Series(False, index=frame.index))
    fumble_lost = frame.get("is_fumble_recovered",
                            pd.Series(False, index=frame.index))

    turnover = interception.to_numpy() | fumble_lost.to_numpy()
    touchdown_array = touchdown.to_numpy()

    # Offensive touchdown: scored without a turnover on the same play.
    points[touchdown_array & ~turnover] = TOUCHDOWN_POINTS
    # Defensive touchdown: a turnover returned for a score.
    points[touchdown_array & turnover] = -TOUCHDOWN_POINTS
    # Field goals are always the offense.
    points[field_goal.to_numpy() & ~touchdown_array] = FIELD_GOAL_POINTS

    return points


def fit(frame: pd.DataFrame) -> ExpectedPointsModel:
    """Build the expected-points lookup table from historical plays."""
    working = frame.copy()
    working["next_score"] = label_next_score(working)

    keys = _state_keys(working)
    working["down"] = keys["down"].to_numpy()
    working["distance_bucket"] = keys["distance_bucket"].to_numpy()
    working["field_bucket"] = keys["field_bucket"].to_numpy()

    grouped = working.groupby(
        ["down", "distance_bucket", "field_bucket"], dropna=True
    )["next_score"]
    table = grouped.agg(expected_points="mean", samples="size").reset_index()

    # Thin cells are noise; fall back to the coarser down/field average.
    coarse = working.groupby(["down", "field_bucket"])["next_score"].mean()
    coarse.name = "coarse_ep"
    table = table.merge(coarse, on=["down", "field_bucket"], how="left")
    thin = table["samples"] < MIN_SAMPLES_PER_CELL
    table.loc[thin, "expected_points"] = table.loc[thin, "coarse_ep"]
    table = table.drop(columns="coarse_ep")

    return ExpectedPointsModel(
        table=table,
        global_mean=float(working["next_score"].mean()),
        n_plays=len(working),
    )


def add_epa(frame: pd.DataFrame, model: ExpectedPointsModel) -> pd.DataFrame:
    """Attach expected points and EPA to a play frame.

    EPA is the change in expected points from this play to the next play
    in the same half. Possession changes flip the sign of the next state,
    because expected points are always from the ball-carrier's view.
    """
    result = frame.copy()
    result["expected_points"] = model.predict(result)

    same_half = (
        (result["game_id"].shift(-1) == result["game_id"])
        & (result["half"].shift(-1) == result["half"])
    )
    next_ep = result["expected_points"].shift(-1)
    possession_flipped = result["team"].shift(-1) != result["team"]
    next_ep = np.where(possession_flipped, -next_ep, next_ep)

    # On a scoring play the "next state" is the score itself, not a snap.
    points = _points_on_play(result)
    scored = points != 0

    epa = np.where(
        scored,
        points - result["expected_points"],
        np.where(same_half, next_ep - result["expected_points"], np.nan),
    )
    result["epa"] = epa
    return result
