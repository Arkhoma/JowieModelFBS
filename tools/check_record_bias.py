"""Ad-hoc: does our model over-rate unbeaten teams / under-rate losers,
compared with the closing line?  Uses the cached walk-forward file, no refits.

For every priced game, take each team's view (record going in, margin from
its side) and compare our error to the line's error.  Positive bias means
the team did BETTER than the number said.

    python tools/check_record_bias.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = [("unbeaten/1 loss (>=80%)", 0.8, 1.01),
           ("winning (55-80%)", 0.55, 0.8),
           ("middling (40-55%)", 0.4, 0.55),
           ("losing (<40%)", 0.0, 0.4)]


def team_views() -> pd.DataFrame:
    bench = pd.read_csv(ROOT / "data" / "benchmark_vs_market.csv")
    frames = []
    for season in sorted(bench.season.unique()):
        path = ROOT / "data" / "raw" / "schedules" / f"schedules_{season}.csv"
        sched = pd.read_csv(path)
        sched = sched[sched.home_points.notna() & (sched.home_points != "NA")]
        sched["margin"] = (sched.home_points.astype(float)
                           - sched.away_points.astype(float))
        sched = sched.sort_values("week")
        # Each team's record BEFORE each game, over every game it played.
        long = pd.concat([
            pd.DataFrame({"game_id": sched.game_id, "team": sched.home_team,
                          "week": sched.week, "won": sched.margin > 0}),
            pd.DataFrame({"game_id": sched.game_id, "team": sched.away_team,
                          "week": sched.week, "won": sched.margin < 0}),
        ]).sort_values(["team", "week"])
        grouped = long.groupby("team")["won"]
        long["wins_before"] = grouped.cumsum() - long.won
        long["games_before"] = grouped.cumcount()
        frames.append(long.assign(season=season))
    records = pd.concat(frames)

    games = bench.merge(
        pd.concat(frames)[["game_id", "season", "team", "wins_before",
                           "games_before"]],
        on=["game_id", "season"])
    home = pd.concat([
        pd.read_csv(ROOT / "data" / "raw" / "schedules"
                    / f"schedules_{s}.csv")[["game_id", "home_team"]]
        for s in sorted(bench.season.unique())])
    games = games.merge(home, on="game_id")
    games["is_home"] = games.team == games.home_team
    sign = np.where(games.is_home, 1.0, -1.0)
    games["win_pct"] = games.wins_before / games.games_before.clip(lower=1)
    games["ours_v"] = games.ours * sign
    games["line_v"] = games.line * sign
    games["actual_v"] = games.actual * sign
    return games[games.games_before >= 3]


def report(games: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(f"{'bucket':<26}{'n':>6}{'ours bias':>11}{'line bias':>11}"
          f"{'ours MAE':>10}{'line MAE':>10}")
    for name, low, high in BUCKETS:
        g = games[(games.win_pct >= low) & (games.win_pct < high)]
        if g.empty:
            continue
        ours_err = g.actual_v - g.ours_v
        line_err = g.actual_v - g.line_v
        print(f"{name:<26}{len(g):>6}{ours_err.mean():>+11.2f}"
              f"{line_err.mean():>+11.2f}{ours_err.abs().mean():>10.2f}"
              f"{line_err.abs().mean():>10.2f}")


def main() -> None:
    games = team_views()
    report(games, "ALL weeks (3+ games played), 2021-2025")
    report(games[games.week >= 9], "Week 9+ only")


if __name__ == "__main__":
    main()
