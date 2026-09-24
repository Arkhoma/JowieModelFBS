"""Grade walk-forward predictions against the closing line.

One home for the metrics, so the CLI benchmark (tools/benchmark_vs_market.py)
and the web page cannot report different numbers for the same games.

Input is the per-game frame the benchmark writes: `ours`, `line`, `actual`,
and optionally `ours_total` / `total_line` / `actual_total` / `home_win_prob`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARLY_WEEK_CUTOFF = 8   # weeks <= this are "early"; the gap lives there
POSTSEASON = "postseason"

# Postseason tiers, from the schedule's `notes` text. CFP games are also
# played in NY6 bowls, so the playoff check must run first.
NY6_BOWLS = ("rose bowl", "sugar bowl", "orange bowl", "cotton bowl",
             "peach bowl", "fiesta bowl")
TIERS = ("CFP", "NY6", "Other bowl", "FCS playoff")


def postseason_tier(notes: str | None) -> str:
    """CFP / NY6 / Other bowl / FCS playoff, from a game's notes."""
    text = (notes or "").lower()
    if "fcs" in text:
        return "FCS playoff"
    if "playoff" in text or "cfp" in text:
        return "CFP"
    if any(bowl in text for bowl in NY6_BOWLS):
        return "NY6"
    return "Other bowl"


def grade(frame: pd.DataFrame) -> dict:
    """Spread metrics for any slice of games."""
    ours_err = np.abs(frame.actual - frame.ours)
    line_err = np.abs(frame.actual - frame.line)
    gap = ours_err - line_err
    decided = frame.actual != frame.line
    ats = (np.sign(frame.ours - frame.line)
           == np.sign(frame.actual - frame.line))[decided].mean()
    result = {
        "n": int(len(frame)),
        "mae": float(ours_err.mean()),
        "line_mae": float(line_err.mean()),
        "gap": float(gap.mean()),
        "gap_ci": float(1.96 * gap.std(ddof=1) / np.sqrt(len(gap))),
        "su": float(((frame.ours > 0) == (frame.actual > 0)).mean()),
        "line_su": float(((frame.line > 0) == (frame.actual > 0)).mean()),
        "ats": float(ats),
    }
    if "home_win_prob" in frame:
        won = (frame.actual > 0).astype(float)
        result["brier"] = float(((frame.home_win_prob - won) ** 2).mean())
    return result


def grade_totals(frame: pd.DataFrame) -> dict | None:
    """Total-points metrics, on games that have an over/under."""
    if "ours_total" not in frame:
        return None
    f = frame[frame.total_line.notna()]
    if f.empty:
        return None
    return {
        "n": int(len(f)),
        "mae": float(np.abs(f.actual_total - f.ours_total).mean()),
        "line_mae": float(np.abs(f.actual_total - f.total_line).mean()),
        "naive_mae": float(
            np.abs(f.actual_total - f.actual_total.mean()).mean()),
    }


def scorecard(frame: pd.DataFrame) -> dict:
    """Everything the accuracy view shows, from one frame."""
    post = (frame.season_type == POSTSEASON if "season_type" in frame
            else pd.Series(False, index=frame.index))
    early = (frame.week <= EARLY_WEEK_CUTOFF) & ~post
    return {
        "pooled": grade(frame),
        "early": grade(frame[early]),
        "late": grade(frame[~early & ~post]),
        "post": grade(frame[post]) if post.any() else None,
        "tiers": ({t: grade(g) for t, g in frame[post].groupby("tier")}
                  if "tier" in frame and post.any() else {}),
        "per_season": {int(s): grade(g)
                       for s, g in frame.groupby("season")},
        "totals": grade_totals(frame),
        "seasons": sorted(int(s) for s in frame.season.unique()),
    }


def load_scorecard(path: Path) -> dict | None:
    """Scorecard from a saved benchmark file, or None if not generated."""
    if not path.exists():
        return None
    return scorecard(pd.read_csv(path, dtype={"game_id": str}))
