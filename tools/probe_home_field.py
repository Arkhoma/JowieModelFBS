"""Does home field vary by crowd, stadium, altitude, travel or matchup?

Uses the walk-forward predictions in data/benchmark_vs_market.csv, so
every residual (actual - predicted home margin) is out of sample. Each
candidate term is scored leave-one-season-out: fit on four seasons,
apply to the fifth, and compare MAE against a flat home field fit on
the same four. A term ships only if it wins across seasons.

All features are known BEFORE kickoff (crowd = the home team's previous
season average, never the game's own gate).

Data: python tools/fetch_cfbd.py --only games --years 2019 ... 2026
      (needs HTTPS_PROXY=http://proxy.wal-mart.com:8080 on the work laptop)

RESULT 2026-09-28 (v2, full CFBD attendance/venues, 2,902 FBS-vs-FBS
home games 2021-25): in-sample, crowd fill (+0.44 per 10 pts of
capacity), travel (+0.54 per doubling of miles) and top-20 matchups
(+2.5) all lean the right way. Out of sample, the best combo (fill +
travel) gains +0.024 MAE, CI +/-0.044, winning 3 of 5 seasons. Every
term is inside its noise band; team-specific HFA r = 0.07. Flat HFA
stays. Rerun after 2026 finishes -- one more season tightens the CI.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.venue import season_context  # noqa: E402

SEASONS = range(2019, 2027)
SHRINKS = (1.0, 0.5, 0.25)


def game_table() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        ctx = season_context(season)
        for g in load_season(season):
            c = ctx.get(str(g.game_id))
            rows.append({
                "game_id": str(g.game_id), "season": season,
                "home_team": g.home_team, "away_team": g.away_team,
                "neutral": g.neutral_site, "conf_game": g.conference_game,
                "fbs_both": g.is_fbs_only,
                "attendance": c.attendance if c else None,
                "capacity": c.capacity if c else None,
                "elev": c.elevation_m if c else None,
                "travel": c.travel_miles if c else None,
                "elo_h": c.home_elo if c else None,
                "elo_a": c.away_elo if c else None,
            })
    return pd.DataFrame(rows)


def prior_crowd(games: pd.DataFrame) -> pd.Series:
    """(season, team) -> previous season's mean home attendance and fill.
    2020 (COVID) is skipped; we fall back to 2019."""
    home = games[~games.neutral & games.attendance.notna()
                 & (games.season != 2020)].copy()
    home["fill"] = (home.attendance / home.capacity).clip(upper=1.1)
    avg = home.groupby(["season", "home_team"])[["attendance", "fill"]].mean()
    out = {}
    for (season, team) in games[["season", "home_team"]].drop_duplicates().itertuples(index=False):
        for back in (1, 2):
            key = (season - back, team)
            if key in avg.index:
                out[(season, team)] = avg.loc[key]
                break
    return pd.DataFrame(out).T


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Centered features, each scaled to a readable unit (see comments)."""
    f = pd.DataFrame(index=df.index)
    f["crowd"] = np.log2(df.prior_att / 50_000)          # per doubling of crowd
    f["fill"] = (df.prior_fill - 0.8) * 10               # per 10 pts of capacity
    f["stadium"] = np.log2(df.capacity / 50_000)        # per doubling of stadium
    # 1,200 m ~ 4,000 ft: Air Force, Wyoming, Colorado, Colorado St, Utah,
    # BYU, New Mexico, UNLV-era Sam Boyd... CFBD elevations are metres.
    f["altitude"] = (df.elev.fillna(0) > 1200).astype(float)
    f["travel"] = np.log2(df.travel.clip(lower=25) / 500)  # per doubling of miles
    floor = df[["elo_h", "elo_a"]].min(axis=1)
    f["big_game"] = (floor - 1500) / 200                 # weaker team's Elo
    f["top_matchup"] = (floor > 1700).astype(float)      # both roughly top-20
    f["conf_game"] = df.conf_game.astype(float)          # rivals travel better
    # Prefix so feature names can never collide with raw columns.
    return f.add_prefix("x_")


def loso(df: pd.DataFrame, feats: list[str]) -> tuple[list[float], np.ndarray, float]:
    """Leave-one-season-out MAE gain vs a flat shift, at each shrink level.
    Returns (pooled gain per shrink level, per-season gains at 100%,
    95% CI half-width of the pooled 100% gain from per-game differences)."""
    total = np.zeros(len(SHRINKS))
    per_season, per_game = [], []
    for season in sorted(df.season.unique()):
        train, test = df[df.season != season], df[df.season == season]
        X = np.column_stack([np.ones(len(train))] + [train[f] for f in feats])
        coef, *_ = np.linalg.lstsq(X, train.resid, rcond=None)
        flat = np.abs(test.resid - train.resid.mean())
        Xt = np.column_stack([test[f] for f in feats])
        adjusted = [np.abs(test.resid - coef[0] - Xt @ (coef[1:] * s))
                    for s in SHRINKS]
        gains = [(flat - a).sum() for a in adjusted]
        per_game.append(flat - adjusted[0])
        total += gains
        per_season.append(gains[0] / len(test))
    diffs = np.concatenate(per_game)
    ci = 1.96 * diffs.std() / np.sqrt(len(diffs))
    return list(total / len(df)), np.array(per_season), ci


def main() -> None:
    bench = pd.read_csv(ROOT / "data" / "benchmark_vs_market.csv",
                        dtype={"game_id": str})
    games = game_table()
    crowd = prior_crowd(games)
    games["prior_att"] = [crowd.attendance.get((s, t)) if len(crowd) else None
                          for s, t in zip(games.season, games.home_team)]
    games["prior_fill"] = [crowd.fill.get((s, t)) if len(crowd) else None
                           for s, t in zip(games.season, games.home_team)]

    df = bench.merge(games, on=["game_id", "season"], how="inner")
    df = df[~df.neutral & df.fbs_both].copy()
    df["resid"] = df.actual - df.ours
    df["mkt_resid"] = df.actual - df.line
    feat = features(df)
    df = pd.concat([df, feat], axis=1).dropna(subset=list(feat.columns))
    print(f"FBS-vs-FBS home-site games with full context: {len(df)} "
          f"({df.season.min()}-{df.season.max()})")
    print(f"mean resid: ours {df.resid.mean():+.2f}, market {df.mkt_resid.mean():+.2f}")

    print("\nIN-SAMPLE slope per unit (ours / market), one feature at a time:")
    for f in feat.columns:
        x = df[f] - df[f].mean()
        slopes = [np.polyfit(x, df[c], 1)[0] for c in ("resid", "mkt_resid")]
        se = df.resid.std() / np.sqrt((x ** 2).sum())
        print(f"  {f:12s} {slopes[0]:+.2f} +/-{1.96 * se:.2f}   "
              f"market {slopes[1]:+.2f}")

    # Team-specific HFA placebo: does a team's extra home edge repeat?
    halves = []
    for seasons in ((2021, 2022, 2023), (2024, 2025)):
        part = df[df.season.isin(seasons)]
        h = part.groupby("home_team").resid.agg(["mean", "count"])
        a = part.groupby("away_team").resid.agg(["mean", "count"])
        j = h.join(a, lsuffix="_h", rsuffix="_a").dropna()
        j = j[(j.count_h >= 6) & (j.count_a >= 6)]
        halves.append((j.mean_h + j.mean_a) / 2)
    r = pd.concat(halves, axis=1).dropna().corr().iloc[0, 1]
    print(f"\nTeam-specific HFA repeatability, 2021-23 vs 2024-25: r = {r:+.3f}")

    print("\nLEAVE-ONE-SEASON-OUT MAE gain vs flat (positive = better), "
          "at 100/50/25% strength:")
    candidates = [[f] for f in feat.columns] + [
        ["x_crowd", "x_big_game"], ["x_fill", "x_travel"],
        ["x_crowd", "x_altitude", "x_travel", "x_big_game"],
        list(feat.columns)]
    for feats in candidates:
        pooled, seasons, ci = loso(df, feats)
        wins = int((seasons > 0).sum())
        print(f"  {'+'.join(x[2:] for x in feats):40s} "
              + " / ".join(f"{g:+.3f}" for g in pooled)
              + f"  +/-{ci:.3f}  seasons won {wins}/{len(seasons)}")

    if len(sys.argv) > 1:
        cols = ["season", "week", "away_team", "ours", "line", "actual",
                "prior_att", "travel", "elo_h", "elo_a"]
        print(df[df.home_team == sys.argv[1]][cols].to_string())


if __name__ == "__main__":
    main()
