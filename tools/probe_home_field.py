"""Does home field vary by venue, crowd, or matchup size?

Uses the walk-forward predictions in data/benchmark_vs_market.csv, so
every residual is out of sample. Three questions:

1. Team-specific HFA: is a team's (home resid - away resid) in 2021-23
   correlated with the same number in 2024-25? Noise won't repeat.
2. Crowd: do home residuals rise with the home team's PRIOR-season
   average attendance (a number known before kickoff)?
3. Big game: do home residuals rise when both teams are strong?
4. Holdout: fit crowd/big-game terms on one era, score the other.

RESULT (2026-09-28): all three look real in-sample and none survive the
holdout. Crowd and big-game terms make out-of-sample MAE WORSE. The
team-specific HFA split-half r is 0.03 (noise). Flat HFA stays.

Also reports the market's residual (actual - line) for each, which tells
us whether Vegas already prices it in.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCHED = ROOT / "data" / "raw" / "schedules"


def schedules() -> pd.DataFrame:
    frames = []
    for season in range(2019, 2027):
        with open(SCHED / f"schedules_{season}.csv", encoding="utf-8") as h:
            frames.append(pd.DataFrame(list(csv.DictReader(h))))
    df = pd.concat(frames)
    df["season"] = df["season"].astype(int)
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
    return df


def prior_attendance(sched: pd.DataFrame) -> dict[tuple[int, str], float]:
    """(season, team) -> mean home attendance of the most recent prior
    season with data (2020 skipped: COVID crowds)."""
    home = sched[(sched["neutral_site"] != "TRUE") & sched["attendance"].gt(0)
                 & (sched["season"] != 2020)]
    avg = home.groupby(["season", "home_team"])["attendance"].mean()
    out = {}
    for season in range(2021, 2027):
        for team in sched["home_team"].unique():
            for back in range(1, 6):
                key = (season - back, team)
                if key in avg.index:
                    out[(season, team)] = float(avg[key])
                    break
    return out


def main() -> None:
    bench = pd.read_csv(ROOT / "data" / "benchmark_vs_market.csv",
                        dtype={"game_id": str})
    sched = schedules()
    cols = ["game_id", "home_team", "away_team", "neutral_site",
            "home_division", "away_division", "attendance",
            "home_pregame_elo", "away_pregame_elo"]
    df = bench.merge(sched[cols], on="game_id", how="left")
    df = df[df["neutral_site"] != "TRUE"].copy()
    df["resid"] = df["actual"] - df["ours"]
    df["mkt_resid"] = df["actual"] - df["line"]
    print(f"home-site games: {len(df)}  mean resid ours {df.resid.mean():+.2f}"
          f"  market {df.mkt_resid.mean():+.2f}")

    # --- 1. team-specific HFA, split-half repeatability -----------------
    print("\n1. TEAM-SPECIFIC HFA (home resid - away resid, per team)")
    for label, col in (("ours", "resid"), ("market", "mkt_resid")):
        halves = []
        for seasons in ((2021, 2022, 2023), (2024, 2025, 2026)):
            part = df[df.season.isin(seasons)]
            h = part.groupby("home_team")[col].agg(["mean", "count"])
            a = part.groupby("away_team")[col].agg(["mean", "count"])
            j = h.join(a, lsuffix="_h", rsuffix="_a").dropna()
            j = j[(j.count_h >= 8) & (j.count_a >= 8)]
            halves.append((j.mean_h + j.mean_a) / 2)  # away resid is -ve of home view
        # Residuals are home-view. A team that is simply underrated shows
        # +d at home and -d away, so the sum cancels; a team with extra
        # home edge h shows +h at home and 0 away. Sum/2 isolates h.
        both = pd.concat(halves, axis=1).dropna()
        r = both.corr().iloc[0, 1]
        print(f"   {label:6s} split-half r = {r:+.3f}  (n={len(both)} teams)")

    # --- 2. crowd size ---------------------------------------------------
    print("\n2. CROWD: home resid by home team's PRIOR-season avg attendance")
    pa = prior_attendance(sched)
    df["prior_att"] = [pa.get((s, t)) for s, t in zip(df.season, df.home_team)]
    fbs = df[(df.home_division == "fbs") & (df.away_division == "fbs")
             & df.prior_att.notna()].copy()
    # Placebo: same stadium size, but that team is on the ROAD. If big
    # programs are just underrated, their road residual (flipped to their
    # view) rises with crowd size too. If it's the crowd, it won't.
    df["away_prior_att"] = [pa.get((s, t)) for s, t in zip(df.season, df.away_team)]
    road = df[(df.home_division == "fbs") & (df.away_division == "fbs")
              & df.away_prior_att.notna()]
    xr = np.log(road.away_prior_att)
    for col in ("resid", "mkt_resid"):
        slope = np.polyfit(xr - xr.mean(), -road[col], 1)[0]
        print(f"   PLACEBO road team's own stadium size, {col:9s}: "
              f"{slope*np.log(2):+.2f} pts per doubling")
    fbs["bucket"] = pd.qcut(fbs.prior_att, 5)
    print(fbs.groupby("bucket", observed=True)[["resid", "mkt_resid"]]
          .agg(["mean", "count"]).round(2).to_string())
    x = np.log(fbs.prior_att)
    for col in ("resid", "mkt_resid"):
        slope = np.polyfit(x - x.mean(), fbs[col], 1)[0]
        print(f"   {col:9s} slope per doubling of crowd: {slope*np.log(2):+.2f} pts")

    # --- 3. big game -----------------------------------------------------
    print("\n3. BIG GAME: home resid by the WEAKER team's pregame Elo")
    df["elo_h"] = pd.to_numeric(df.home_pregame_elo, errors="coerce")
    df["elo_a"] = pd.to_numeric(df.away_pregame_elo, errors="coerce")
    big = df[(df.home_division == "fbs") & (df.away_division == "fbs")
             & df.elo_h.notna() & df.elo_a.notna()].copy()
    big["floor"] = big[["elo_h", "elo_a"]].min(axis=1)
    big["bucket"] = pd.qcut(big.floor, 5)
    print(big.groupby("bucket", observed=True)[["resid", "mkt_resid"]]
          .agg(["mean", "count"]).round(2).to_string())
    top = big[(big.elo_h >= big.floor.quantile(.9))
              & (big.elo_a >= big.floor.quantile(.9))]
    print(f"   both top-10%-ish: n={len(top)} ours {top.resid.mean():+.2f}"
          f" +/-{1.96*top.resid.std()/np.sqrt(len(top)):.2f}"
          f"  market {top.mkt_resid.mean():+.2f}")
    # Big crowd x big game interaction
    both = big[big.prior_att.notna()]
    loud = both[(both.prior_att >= 80000)
                & (both.floor >= both.floor.quantile(.6))]
    print(f"   80k+ crowd AND good opponent: n={len(loud)}"
          f" ours {loud.resid.mean():+.2f}"
          f" +/-{1.96*loud.resid.std()/np.sqrt(len(loud)):.2f}"
          f"  market {loud.mkt_resid.mean():+.2f}")

    # --- 4. out-of-sample MAE: fit on 2021-23, score 2024-25 ------------
    print("\n4. HOLDOUT: fit on one era, score the other (both directions)")
    everything = df[(df.home_division == "fbs") & (df.away_division == "fbs")].copy()
    med = np.log(fbs.prior_att.median())
    everything["crowd"] = np.log(everything.prior_att) - med
    everything["big"] = (everything[["elo_h", "elo_a"]].min(axis=1) - 1400) / 200
    everything = everything.dropna(subset=["crowd", "big"])
    for tr, te in (((2021, 2022, 2023), (2024, 2025)),
                   ((2024, 2025), (2021, 2022, 2023))):
        train = everything[everything.season.isin(tr)]
        test = everything[everything.season.isin(te)]
        print(f"   train {tr} -> test {te}: n={len(test)}")
        print("     test-set median resid by crowd quintile:",
              test.groupby(pd.qcut(test.crowd, 5), observed=True)
              .resid.median().round(2).tolist())
        for feats in (["crowd"], ["big"], ["crowd", "big"]):
            X = np.column_stack([np.ones(len(train))] + [train[f] for f in feats])
            coef, *_ = np.linalg.lstsq(X, train.resid, rcond=None)
            Xt = np.column_stack([test[f] for f in feats])
            flat = np.abs(test.resid - coef[0]).mean()
            gains = []
            for shrink in (1.0, 0.5, 0.25):
                pred = coef[0] + Xt @ (coef[1:] * shrink)
                gains.append(flat - np.abs(test.resid - pred).mean())
            print(f"     {'+'.join(feats):11s} coef {np.round(coef[1:], 2)}"
                  f"  gain vs flat @100/50/25%: "
                  + " / ".join(f"{g:+.3f}" for g in gains))

    if len(sys.argv) > 1:
        print(fbs[fbs.home_team == sys.argv[1]][
            ["season", "week", "away_team", "ours", "line", "actual",
             "prior_att"]].to_string())


if __name__ == "__main__":
    main()
