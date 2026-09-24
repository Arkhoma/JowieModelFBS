"""Which upgrades close the gap to the closing line?

Walk-forward on 2021-2025. Each week, several component models are fit on
games already played, and their predictions for the coming week are
stored side by side. Variants are then scored offline.

Components:
  m_flat        margin ridge, flat carryover prior (production today)
  m_roster      margin ridge, returning-production prior
  m_gt          + garbage-time margins (margin when the game was decided)
  m_luck        + fumble-recovery luck removed (LUCK_POINTS per recovery)
  m_gt_luck     both
  e_flat/e_roster   EPA ridge, primed with the matching prior
  s_roster      success-rate ridge, primed with the roster prior

Stack weights are fit LEAVE-ONE-SEASON-OUT for every variant, baseline
included, so no variant is scored on weights that saw its own games.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_vs_market import load_home_lines  # noqa: E402
from cfbrank.epa_ridge import (  # noqa: E402
    build_epa_games, fit_epa_ratings_primed, get_ep_model,
)
from cfbrank.game_features import season_features  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit as fit_margin  # noqa: E402

SEASONS = (2021, 2022, 2023, 2024, 2025)
START_WEEK, MIN_TRAINING, RELATIVE_LAMBDA = 4, 50, 4.0
MARGIN_PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
LUCK_POINTS = 4.0   # standard value of one turnover-sized possession swing
CACHE = ROOT / "data" / "improvement_components.csv"


def adjusted(games, features, garbage: bool, luck: bool):
    out = []
    for g in games:
        f = features.get(str(g.game_id))
        margin = float(g.margin)
        if f is not None:
            if garbage and f.garbage_margin is not None:
                margin = f.garbage_margin
            if luck:
                margin -= LUCK_POINTS * f.home_fumble_luck
        out.append(replace(g, home_points=g.away_points + margin))
    return out


def walk_season(season: int, market: dict[str, float]) -> pd.DataFrame:
    games = sorted(load_season(season), key=lambda g: g.week)
    ep_model = get_ep_model()
    epa = {g.game_id: g for g in build_epa_games(season, games, ep_model)}
    sr = {g.game_id: g for g in
          build_epa_games(season, games, ep_model, metric="success")}
    epa_all = {g.game_id: g for g in build_epa_games(
        season, games, ep_model, drop_garbage_time=False)}
    sr_all = {g.game_id: g for g in build_epa_games(
        season, games, ep_model, metric="success", drop_garbage_time=False)}
    features = season_features(season, games)
    priors = {"flat": build_prior_or_empty(season) or None,
              "roster": build_roster_prior_or_empty(season) or None}
    rows = []

    for week in sorted({g.week for g in games}):
        if week < START_WEEK:
            continue
        history = [g for g in games if g.week < week]
        upcoming = [g for g in games
                    if g.week == week and str(g.game_id) in market]
        epa_hist = [epa[str(g.game_id)] for g in history
                    if str(g.game_id) in epa]
        sr_hist = [sr[str(g.game_id)] for g in history
                   if str(g.game_id) in sr]
        epa_all_hist = [epa_all[str(g.game_id)] for g in history
                        if str(g.game_id) in epa_all]
        sr_all_hist = [sr_all[str(g.game_id)] for g in history
                       if str(g.game_id) in sr_all]
        if len(history) < MIN_TRAINING or len(epa_hist) < MIN_TRAINING \
                or not upcoming:
            continue

        models = {
            "m_flat": fit_margin(history, prior=priors["flat"],
                                 **MARGIN_PARAMS),
            "e_flat": fit_epa_ratings_primed(
                epa_hist, priors["flat"], relative_lambda=RELATIVE_LAMBDA),
            "m_roster": fit_margin(history, prior=priors["roster"],
                                   **MARGIN_PARAMS),
            "e_roster": fit_epa_ratings_primed(
                epa_hist, priors["roster"], relative_lambda=RELATIVE_LAMBDA),
            "s_roster": fit_epa_ratings_primed(
                sr_hist, priors["roster"], relative_lambda=RELATIVE_LAMBDA),
            "e_all": fit_epa_ratings_primed(
                epa_all_hist, priors["roster"],
                relative_lambda=RELATIVE_LAMBDA),
            "s_all": fit_epa_ratings_primed(
                sr_all_hist, priors["roster"],
                relative_lambda=RELATIVE_LAMBDA),
        }
        for name, gt, lk in (("m_gt", True, False), ("m_luck", False, True),
                             ("m_gt_luck", True, True)):
            models[name] = fit_margin(adjusted(history, features, gt, lk),
                                      prior=priors["roster"],
                                      **MARGIN_PARAMS)

        for game in upcoming:
            if not all(game.home_team in m.ratings
                       and game.away_team in m.ratings
                       for m in models.values()):
                continue
            row = {"season": season, "week": week,
                   "game_id": str(game.game_id),
                   "line": market[str(game.game_id)],
                   "actual": float(game.margin)}
            for name, model in models.items():
                row[name] = model.predict_margin(
                    game.home_team, game.away_team, game.neutral_site)
            rows.append(row)
    print(f"  {season}: {len(rows)} games", flush=True)
    return pd.DataFrame(rows)


def loso_stack(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Leave-one-season-out least-squares stack, no intercept."""
    out = np.zeros(len(frame))
    for season in frame.season.unique():
        train, test = frame.season != season, frame.season == season
        coef, *_ = np.linalg.lstsq(frame.loc[train, columns].values,
                                   frame.loc[train, "actual"].values,
                                   rcond=None)
        out[test.values] = frame.loc[test, columns].values @ coef
    return out


VARIANTS = {
    "baseline (production)": ["m_flat", "e_flat"],
    "+ roster prior": ["m_roster", "e_roster"],
    "+ roster + garbage time": ["m_gt", "e_roster"],
    "+ roster + fumble luck": ["m_luck", "e_roster"],
    "+ roster + gt + luck": ["m_gt_luck", "e_roster"],
    "+ roster + success rate": ["m_roster", "e_roster", "s_roster"],
    "everything": ["m_gt_luck", "e_roster", "s_roster"],
    "roster + SR, no EPA garbage filter": ["m_roster", "e_all", "s_all"],
    "roster + SR + luck": ["m_luck", "e_roster", "s_roster"],
}


def score(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    err = np.abs(frame.actual - pred)
    line_err = np.abs(frame.actual - frame.line)
    gap = err - line_err
    decided = frame.actual != frame.line
    side = np.sign(pred - frame.line) == np.sign(frame.actual - frame.line)
    return {"mae": err.mean(), "gap": gap.mean(),
            "ci": 1.96 * gap.std(ddof=1) / np.sqrt(len(gap)),
            "su": ((pred > 0) == (frame.actual > 0)).mean(),
            "ats": side[decided].mean()}


def main() -> None:
    if CACHE.exists() and "--refresh" not in sys.argv:
        frame = pd.read_csv(CACHE)
    else:
        market = load_home_lines()
        frame = pd.concat([walk_season(s, market) for s in SEASONS],
                          ignore_index=True)
        frame.to_csv(CACHE, index=False)

    early = (frame.week <= 8).values
    base = None
    print(f"\n{len(frame)} games. Gap = our MAE minus closing-line MAE "
          f"(lower is better; 0 = matches Vegas)\n")
    print(f"{'variant':<36} {'MAE':>6} {'gap':>6} {'+/-':>5} "
          f"{'wk4-8':>6} {'wk9+':>6} {'vs base':>8} {'SU':>6} {'ATS':>6}")
    print("-" * 94)
    for name, cols in VARIANTS.items():
        pred = loso_stack(frame, cols)
        s = score(frame, pred)
        e = score(frame[early], pred[early])["gap"]
        l = score(frame[~early], pred[~early])["gap"]
        if base is None:
            base = pred
        d = np.abs(frame.actual - pred) - np.abs(frame.actual - base)
        vs = f"{d.mean():+.3f}" if name != "baseline (production)" else ""
        print(f"{name:<36} {s['mae']:>6.2f} {s['gap']:>+6.2f} "
              f"{s['ci']:>5.2f} {e:>+6.2f} {l:>+6.2f} {vs:>8} "
              f"{s['su']:>6.1%} {s['ats']:>6.1%}")
    print(f"\nPaired SE of a variant-vs-baseline diff is roughly "
          f"{(np.abs(frame.actual - loso_stack(frame, VARIANTS['everything'])) - np.abs(frame.actual - base)).std() / np.sqrt(len(frame)):.3f}")

    winner = VARIANTS["+ roster + success rate"]
    print("\nWinner vs baseline, by season (negative = improvement):")
    pred = loso_stack(frame, winner)
    for season in SEASONS:
        m = (frame.season == season).values
        d = (np.abs(frame.actual - pred) - np.abs(frame.actual - base))[m]
        print(f"  {season}: {d.mean():+.3f}")
    coef, *_ = np.linalg.lstsq(frame[winner].values, frame.actual.values,
                               rcond=None)
    print("Full-sample stack weights:",
          ", ".join(f"{c} {w:+.3f}" for c, w in zip(winner, coef)))


if __name__ == "__main__":
    main()
