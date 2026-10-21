"""Do rest days or quarterback changes explain what the model misses?

Same method as probe_home_field.py (and it reuses its `loso`): take the
walk-forward residuals in data/benchmark_vs_market.csv, which are all out
of sample, and test whether a candidate feature corrects them when fit
on four seasons and applied to the fifth.

Every feature is knowable BEFORE kickoff -- built only from each team's
earlier games in the same season:

  rest      days since each team's previous game (home minus away)
  bye       had 12+ days off (home minus away)
  short     played on 5 or fewer days' rest (home minus away)
  qb_share  share of the team's season-to-date dropbacks thrown by the QB
            who led its MOST RECENT game. Low = a new or backup QB has
            just taken over, which our ratings (built on the old QB's
            games) have not priced in.
  qb_new    most recent game's lead passer is not the season leader
  qb_hurt   in the most recent game the season leader threw under half
            the dropbacks (left early, benched or hurt)
  qb_fresh  the most recent game's lead QB has under 35% of the team's
            season dropbacks: a brand-new starter, not a settled change

Dropbacks = completions + incompletions + interceptions + sacks, by the
passer id on each play.

RESULT 2026-10-21 (5,299 games 2021-25; MAE gain vs flat, + = better):
rest +0.003 +/-0.011, bye +0.003 +/-0.011, short week -0.001 (12 games).
qb_share/new/hurt all -0.001..-0.002. qb_fresh is REAL in-sample (a
brand-new starter underperforms us by 1.13 +/-0.88 pts; the market also
misses by 0.62) but too rare to matter: +0.008 +/-0.015 out of sample,
3/5 seasons. Nothing ships. Passer detection verified on 2024 Georgia
(flags only the Sugar Bowl, Stockton for Beck). Same session: recency
half-life 10 / 16 weeks = -0.004 +/-0.022 / -0.007 +/-0.014 (noise;
tools/run_variant.py), so halflife stays off.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.plays import PLAYS_DIR  # noqa: E402
from probe_home_field import loso  # noqa: E402

SEASONS = range(2021, 2026)
PASSER_COLUMNS = ("completion_player_id", "incompletion_player_id",
                  "interception_thrown_player_id", "sack_taken_player_id")


def dropbacks(season: int) -> pd.DataFrame:
    """(game_id, team, passer) -> dropback count."""
    path = PLAYS_DIR / f"plays_{season}.parquet"
    frame = pd.read_parquet(path)
    cols = [c for c in PASSER_COLUMNS if c in frame.columns]
    frame = frame[["game_id", "team", *cols]].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    passer = pd.Series(pd.NA, index=frame.index, dtype="object")
    for c in cols:
        ids = frame[c].astype(str).str.strip()
        real = frame[c].notna() & ~ids.isin(("", "nan", "None", "<NA>"))
        passer = passer.where(passer.notna() | ~real, ids)
    frame["passer"] = passer
    frame = frame.dropna(subset=["passer"])
    return (frame.groupby(["game_id", "team", "passer"]).size()
            .rename("n").reset_index())


def team_features(season: int) -> dict[tuple[str, str], dict]:
    """(game_id, team) -> pre-kickoff features from that team's earlier games."""
    games = sorted((g for g in load_season(season) if g.kickoff),
                   key=lambda g: g.kickoff)
    db = dropbacks(season)
    by_game = {k: dict(zip(v.passer, v.n))
               for k, v in db.groupby(["game_id", "team"])}

    last_kick: dict[str, object] = {}
    season_counts: dict[str, defaultdict] = defaultdict(lambda: defaultdict(int))
    last_game_counts: dict[str, dict] = {}
    out = {}
    for g in games:
        gid = str(g.game_id)
        for team in (g.home_team, g.away_team):
            f = {}
            if team in last_kick:
                f["rest"] = (g.kickoff - last_kick[team]).total_seconds() / 86400
            totals, recent = season_counts[team], last_game_counts.get(team)
            if recent and sum(totals.values()) >= 20:
                lead = max(recent, key=recent.get)
                f["qb_share"] = totals[lead] / sum(totals.values())
                earlier = {p: totals[p] - recent.get(p, 0) for p in totals}
                if sum(earlier.values()) >= 20:
                    leader = max(earlier, key=earlier.get)
                    f["qb_new"] = float(lead != leader)
                    f["qb_hurt"] = float(
                        recent.get(leader, 0) < 0.5 * sum(recent.values()))
            out[(gid, team)] = f
        # Only now record this game, so it never informs itself.
        for team in (g.home_team, g.away_team):
            last_kick[team] = g.kickoff
            counts = by_game.get((gid, team))
            if counts:
                last_game_counts[team] = counts
                for p, n in counts.items():
                    season_counts[team][p] += n
    return out


def game_table() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        feats = team_features(season)
        for g in load_season(season):
            h = feats.get((str(g.game_id), g.home_team), {})
            a = feats.get((str(g.game_id), g.away_team), {})
            rows.append({"game_id": str(g.game_id), "season": season,
                         **{f"h_{k}": v for k, v in h.items()},
                         **{f"a_{k}": v for k, v in a.items()}})
    return pd.DataFrame(rows)


def features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    hr, ar = df.h_rest.clip(3, 21), df.a_rest.clip(3, 21)
    f["x_rest"] = (hr - ar) / 7                          # per week of rest edge
    f["x_bye"] = (hr >= 12).astype(float) - (ar >= 12).astype(float)
    f["x_short"] = (hr <= 5).astype(float) - (ar <= 5).astype(float)
    f["x_qb_share"] = df.h_qb_share - df.a_qb_share     # per 100% share
    # A genuinely new QB (a backup's first starts), not a settled change
    # the ratings have already absorbed.
    f["x_qb_fresh"] = ((df.h_qb_share < 0.35).astype(float)
                       - (df.a_qb_share < 0.35).astype(float))
    f["x_qb_new"] = df.h_qb_new - df.a_qb_new
    f["x_qb_hurt"] = df.h_qb_hurt - df.a_qb_hurt
    return f


def report(df: pd.DataFrame, cols: list[str], label: str) -> None:
    sub = df.dropna(subset=cols)
    print(f"\n{label}: {len(sub)} games")
    for c in cols:
        x = sub[c] - sub[c].mean()
        slope = [np.polyfit(x, sub[r], 1)[0] for r in ("resid", "mkt_resid")]
        se = sub.resid.std() / np.sqrt((x ** 2).sum())
        nonzero = int((sub[c] != 0).sum())
        print(f"  {c:12s} in-sample {slope[0]:+.2f} +/-{1.96 * se:.2f}"
              f"  (market {slope[1]:+.2f})  nonzero {nonzero}")
    for feats in [[c] for c in cols] + [cols]:
        pooled, seasons, ci = loso(sub, feats)
        print(f"  LOSO {'+'.join(x[2:] for x in feats):28s}"
              f" {pooled[0]:+.3f} +/-{ci:.3f}  (50%: {pooled[1]:+.3f})"
              f"  seasons won {int((seasons > 0).sum())}/{len(seasons)}")


def main() -> None:
    bench = pd.read_csv(ROOT / "data" / "benchmark_vs_market.csv",
                        dtype={"game_id": str})
    df = bench.merge(game_table(), on=["game_id", "season"], how="inner")
    df["resid"] = df.actual - df.ours
    df["mkt_resid"] = df.actual - df.line
    df = pd.concat([df, features(df)], axis=1)
    print(f"benchmark games matched: {len(df)} of {len(bench)}")
    print("LOSO = MAE gain vs flat, positive = better; "
          f"gap to line overall {(df.resid.abs() - df.mkt_resid.abs()).mean():+.3f}")
    report(df, ["x_rest", "x_bye", "x_short"], "REST")
    report(df, ["x_qb_share", "x_qb_new", "x_qb_hurt", "x_qb_fresh"],
           "QUARTERBACK")


if __name__ == "__main__":
    main()
