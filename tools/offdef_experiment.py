"""Do offense/defense ratings improve predictions? Measured, not assumed.

Walk-forward on 2021-2025, identical protocol to improvement_experiment:
each week, fit on games already played, predict the coming week, stack
components with LEAVE-ONE-SEASON-OUT weights, grade against the close.

Components (all use the production home-field prior where applicable):
  m            margin ridge + roster prior              (production)
  s            success-rate ridge + roster prior         (production)
  p_split_L    off/def POINTS, split off/def prior, lambda L
  p_net_L      off/def points, the NET roster prior halved onto each
               side -- the control that isolates what separate priors buy
  p_none_L     off/def points, no prior
  sr_split_L   off/def SUCCESS RATE, split prior
  tot_L        predicted total from p_split_L, graded vs the over/under

    python tools/offdef_experiment.py [--refresh] [--lambdas 0.5,1,2]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_vs_market import load_home_lines, load_total_lines  # noqa: E402
from blend_epa_margin import (  # noqa: E402
    MARGIN_PARAMS, MIN_TRAINING, RELATIVE_LAMBDA, START_WEEK,
)
from cfbrank import offdef  # noqa: E402
from cfbrank.ensemble import EFFICIENCY_METRIC  # noqa: E402
from cfbrank.epa_ridge import build_epa_games, fit_epa_ratings_primed  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.offdef_prior import (  # noqa: E402
    _season_records, build_offdef_prior, season_sides,
)
from cfbrank.prior import home_field_prior  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit as fit_margin  # noqa: E402
from improvement_experiment import loso_stack, score  # noqa: E402

SEASONS = (2021, 2022, 2023, 2024, 2025)
# First sweep was (2, 5, 12): accuracy improved monotonically toward the
# smallest value, so the grid was extended down to find the optimum.
LAMBDAS = tuple(float(x) for x in (
    sys.argv[sys.argv.index("--lambdas") + 1].split(",")
    if "--lambdas" in sys.argv else (0.5, 1, 2, 5)))
CACHE = ROOT / "data" / "offdef_components.csv"


def halve(prior: dict[str, float] | None):
    """Net prior split evenly onto offense and defense (the control)."""
    if not prior:
        return None
    half = {team: value / 2 for team, value in prior.items()}
    return half, dict(half)


def walk_season(season: int, spreads: dict, totals: dict) -> pd.DataFrame:
    games = sorted(load_season(season), key=lambda g: g.week)
    ep_games = {g.game_id: g for g in build_epa_games(
        season, games, metric=EFFICIENCY_METRIC)}
    roster = build_roster_prior_or_empty(season) or None
    hfa = home_field_prior(season)
    split_pts = build_offdef_prior(season, "points")
    split_sr = build_offdef_prior(season, "success")
    net_pts = halve(roster)
    pts_sides = season_sides(season, "points")
    sr_sides = season_sides(season, "success")
    rows = []

    for week in sorted({g.week for g in games}):
        if week < START_WEEK:
            continue
        history = [g for g in games if g.week < week]
        ids = {str(g.game_id) for g in history}
        upcoming = [g for g in games
                    if g.week == week and str(g.game_id) in spreads]
        ep_hist = [ep_games[i] for i in ids if i in ep_games]
        if len(history) < MIN_TRAINING or len(ep_hist) < MIN_TRAINING \
                or not upcoming:
            continue
        pts_hist = [s for s in pts_sides if s.game_id in ids]
        sr_hist = [s for s in sr_sides if s.game_id in ids]

        models = {
            "m": fit_margin(history, prior=roster, home_field_prior=hfa,
                            **MARGIN_PARAMS),
            "s": fit_epa_ratings_primed(ep_hist, roster,
                                        relative_lambda=RELATIVE_LAMBDA),
        }
        for lam in LAMBDAS:
            tag = f"{lam:g}"
            models[f"p_split_{tag}"] = offdef.fit(
                pts_hist, lam, split_pts, home_field_prior=hfa)
            models[f"p_net_{tag}"] = offdef.fit(
                pts_hist, lam, net_pts, home_field_prior=hfa)
            models[f"p_none_{tag}"] = offdef.fit(
                pts_hist, lam, None, home_field_prior=hfa)
            models[f"sr_split_{tag}"] = offdef.fit(
                sr_hist, lam, split_sr, is_points=False)

        for game in upcoming:
            if not all(game.home_team in m.ratings
                       and game.away_team in m.ratings
                       for m in models.values()):
                continue
            gid = str(game.game_id)
            row = {"season": season, "week": week, "game_id": gid,
                   "line": spreads[gid], "total_line": totals.get(gid),
                   "actual": float(game.margin),
                   "actual_total": float(game.home_points + game.away_points)}
            for name, model in models.items():
                row[name] = model.predict_margin(
                    game.home_team, game.away_team, game.neutral_site)
            for lam in LAMBDAS:
                row[f"tot_{lam:g}"] = models[f"p_split_{lam:g}"].predict_total(
                    game.home_team, game.away_team, game.neutral_site)
            rows.append(row)
    _season_records.cache_clear()   # ~100 MB of plays per season
    print(f"  {season}: {len(rows)} games", flush=True)
    return pd.DataFrame(rows)


def report_spreads(frame: pd.DataFrame) -> None:
    variants = {"production (m + s)": ["m", "s"]}
    for lam in LAMBDAS:
        t = f"{lam:g}"
        variants.update({
            f"m + s + p_split_{t}": ["m", "s", f"p_split_{t}"],
            f"m + s + p_net_{t}  (control)": ["m", "s", f"p_net_{t}"],
            f"m + s + sr_split_{t}": ["m", "s", f"sr_split_{t}"],
            f"m + s + p_split + sr_split {t}":
                ["m", "s", f"p_split_{t}", f"sr_split_{t}"],
            f"p_split_{t} + sr_split_{t} only":
                [f"p_split_{t}", f"sr_split_{t}"],
        })
    for single in ("m", *(f"p_split_{l:g}" for l in LAMBDAS),
                   *(f"p_none_{l:g}" for l in LAMBDAS)):
        variants[f"{single} alone"] = [single]

    early = (frame.week <= 8).values
    base = loso_stack(frame, variants["production (m + s)"])
    base_err = np.abs(frame.actual - base)
    print(f"\nSPREADS -- {len(frame)} games. gap = our MAE - closing line MAE\n")
    print(f"{'variant':<36} {'MAE':>6} {'gap':>6} {'wk4-8':>6} {'wk9+':>6} "
          f"{'vs prod':>8} {'+/-':>5} {'SU':>6}")
    print("-" * 86)
    for name, cols in variants.items():
        pred = loso_stack(frame, cols)
        s = score(frame, pred)
        d = np.abs(frame.actual - pred) - base_err
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"{name:<36} {s['mae']:>6.3f} {s['gap']:>+6.3f} "
              f"{score(frame[early], pred[early])['gap']:>+6.3f} "
              f"{score(frame[~early], pred[~early])['gap']:>+6.3f} "
              f"{d.mean():>+8.3f} {ci:>5.3f} {s['su']:>6.1%}")


def report_totals(frame: pd.DataFrame) -> None:
    has = frame.total_line.notna()
    f = frame[has]
    line_err = np.abs(f.actual_total - f.total_line)
    print(f"\nTOTALS -- {len(f)} games with an over/under. "
          f"Line MAE {line_err.mean():.2f}; "
          f"naive (league-average total) MAE "
          f"{np.abs(f.actual_total - f.actual_total.mean()).mean():.2f}\n")
    print(f"{'model':<12} {'MAE':>6} {'gap':>7} {'bias':>6} {'O/U hit':>8}")
    tot_names = [f"tot_{lam:g}" for lam in LAMBDAS]
    for name in tot_names:
        if name not in f:
            continue
        pred = f[name]
        err = np.abs(f.actual_total - pred)
        decided = f.actual_total != f.total_line
        hit = (np.sign(pred - f.total_line)
               == np.sign(f.actual_total - f.total_line))[decided].mean()
        print(f"{name:<12} {err.mean():>6.2f} "
              f"{(err - line_err).mean():>+7.2f} "
              f"{(pred - f.actual_total).mean():>+6.2f} {hit:>8.1%}")


def main() -> None:
    if CACHE.exists() and "--refresh" not in sys.argv:
        frame = pd.read_csv(CACHE, dtype={"game_id": str})
    else:
        spreads, totals = load_home_lines(), load_total_lines()
        frame = pd.concat([walk_season(s, spreads, totals) for s in SEASONS],
                          ignore_index=True)
        frame.to_csv(CACHE, index=False)
    report_spreads(frame)
    report_totals(frame)


if __name__ == "__main__":
    main()
