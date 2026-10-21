"""Time Machine: any team from any season vs any team from any other.

Every season's ratings are "points better than an average FBS team THAT
season" (each season's FBS mean is 0 -- tests/test_time_machine.py pins
it). So a cross-season game is the same arithmetic as a regular neutral-
site prediction, with each side's numbers taken from its own season:

    margin = rating[A in its season] - rating[B in its season]

The one assumption, stated on the page: an average FBS team is equally
good every year. This compares how dominant each team was in its own
era, not whose roster would win a literal game.

Totals use each side's offense/defense against the other, around the
average of the two seasons' scoring levels. Win probability uses the
model's own historical misses from BOTH seasons, pooled. No Vegas
Script: nobody ever set a line on these games.

The page runs entirely in the browser (app/static/time_machine.js) from
`season_snapshot` payloads, live and static alike. `matchup` below is
the Python reference that JS mirrors; tests hold both to the regular
predictor.
"""

from __future__ import annotations

import numpy as np

# Each season's residuals ship as this many evenly spaced quantiles
# rather than one float per game: 13 seasons x ~800 games is too much to
# embed in one page, and 200 quantiles move a win % by well under 1 pt.
QUANTILES = 200
DIGITS = 2   # ratings rounded to the hundredth of a point


def _quantiles(errors: np.ndarray) -> list[float]:
    """Evenly spaced quantiles of the misses AND their mirror image.

    Residuals are recorded from the home side and are slightly skewed,
    so the raw distribution made "A vs B" and "B vs A" add up to ~102%.
    On a neutral field a +7 miss for one side is a -7 miss for the other,
    so mirroring is the honest distribution -- and it sums to 100%.
    """
    if errors.size == 0:
        return []
    mirrored = np.concatenate([errors, -errors])
    levels = (np.arange(QUANTILES) + 0.5) / QUANTILES
    return [round(float(q), DIGITS) for q in np.quantile(mirrored, levels)]


def season_snapshot(rankings: list[dict], predictor, label: str) -> dict:
    """One season's numbers for cross-season matchups (FBS teams only).

    Ratings and off/def are centred on the season's FBS mean. Off/def
    centring is folded into `points_mean`, so a same-season matchup's
    total is unchanged.
    """
    model, totals = predictor.model, predictor.total_model
    teams = [r["team"] for r in rankings]
    rating = {t: model.calibrated_rating(t) for t in teams}
    off = {t: totals.offense_rating(t) for t in teams}
    dfn = {t: totals.defense_rating(t) for t in teams}
    mean_r, mean_o, mean_d = (float(np.mean(list(v.values())))
                              for v in (rating, off, dfn))
    return {
        "label": label,
        "points_mean": round(totals.mean + mean_o - mean_d, DIGITS),
        "errors": _quantiles(np.asarray(predictor.errors, dtype=float)),
        "teams": {
            r["team"]: {
                "rating": round(rating[r["team"]] - mean_r, DIGITS),
                "off": round(off[r["team"]] - mean_o, DIGITS),
                "def": round(dfn[r["team"]] - mean_d, DIGITS),
                "rank": r["rank"],
                "record": f"{r['wins']}-{r['losses']}",
            }
            for r in rankings
        },
    }


def matchup(a: dict, team_a: str, b: dict, team_b: str) -> dict:
    """Neutral-site game between `team_a` (snapshot `a`) and `team_b`
    (snapshot `b`). Positive margin = A wins. Mirrored by time_machine.js."""
    ta, tb = a["teams"][team_a], b["teams"][team_b]
    margin = ta["rating"] - tb["rating"]
    total = (a["points_mean"] + b["points_mean"]
             + ta["off"] + tb["off"] - ta["def"] - tb["def"])
    errors = a["errors"] + b["errors"]
    win_a = (sum(e > -margin for e in errors) / len(errors)) if errors else 0.5
    return {
        "margin": margin,
        "total": total,
        "score_a": (total + margin) / 2,
        "score_b": (total - margin) / 2,
        "win_a": win_a,
    }
