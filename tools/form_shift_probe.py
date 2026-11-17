"""Is in-season form shift real, and can we track it across FBS?

Reads data/form_misses.csv (tools/build_form_misses.py): each game's
walk-forward offense/defense miss -- actual minus what the model
expected from earlier games alone. Four questions:

 1. Oklahoma 2026: is there a real break, or does it only look like one?
    Best-split scan with a permutation p-value that charges for having
    searched every split point.
 2. Persistence: across all FBS team-seasons, does the average miss of a
    team's last k games predict its NEXT game's miss? Zero would mean
    streaks are noise and the model is already adapting; a positive
    slope means ratings lag real change. Graded leave-one-season-out.
 3. Flag test: when the last 3 games sit far from the earlier ones, does
    the shift survive into the next 3?
 4. How unusual is Oklahoma among every FBS team-season, like-for-like
    (same number of games), and who else is moving right now?

    python tools/form_shift_probe.py [--team Oklahoma --season 2026]
"""

from __future__ import annotations

import argparse
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.form import MIN_SEGMENT, scan_split  # noqa: E402
from cfbrank.games import load_season, team_divisions  # noqa: E402

DATA = ROOT / "data" / "form_misses.csv"
RECENT = 3
FLAG_Z = 2.0
BOOTSTRAP = 1000
PERMUTATIONS = 20000
RNG = np.random.default_rng(7)
SIDES = {"off": "miss", "def": "def_miss"}


def load_fbs_misses() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    keep = []
    for season, part in frame.groupby("season"):
        fbs = {t for t, d in team_divisions(load_season(int(season))).items()
               if d == "fbs"}
        keep.append(part[part["team"].isin(fbs)])
    return pd.concat(keep).sort_values(["kind", "season", "team", "week"])


def permutation_p(x: np.ndarray, sigma: float) -> float:
    observed = abs(scan_split(x, sigma)[0])
    hits = sum(abs(scan_split(RNG.permutation(x), sigma)[0]) >= observed
               for _ in range(PERMUTATIONS))
    return (hits + 1) / (PERMUTATIONS + 1)


def sequences(frame: pd.DataFrame, kind: str, column: str) -> dict:
    """{(season, team): misses in game order}, complete rows only."""
    part = frame[frame["kind"] == kind].dropna(subset=[column])
    return {key: g[column].to_numpy()
            for key, g in part.groupby(["season", "team"])}


def sigma_of(frame: pd.DataFrame, kind: str, column: str) -> float:
    return float(frame.loc[frame["kind"] == kind, column].std())


# --- 1. Oklahoma ---------------------------------------------------------

def show_team(frame, team, season) -> None:
    print(f"\n=== 1. {team} {season}: game by game ===")
    for kind in ("points", "success"):
        part = frame[(frame.kind == kind) & (frame.team == team)
                     & (frame.season == season)].sort_values("week")
        unit = 1.0 if kind == "points" else 100.0
        print(f"\n {kind} (offense: actual / expected / miss;"
              f" defense miss, + = held opp below expectation)")
        for _, r in part.iterrows():
            print(f"  wk{int(r.week):>2} vs {r.opponent:<18} "
                  f"{r.actual * unit:6.1f} {r.expected * unit:6.1f} "
                  f"{r.miss * unit:+6.1f}   def {r.def_miss * unit:+6.1f}")
        for side, column in SIDES.items():
            x = part[column].dropna().to_numpy()
            sigma = sigma_of(frame, kind, column)
            z, j = scan_split(x, sigma)
            p = permutation_p(x, sigma)
            print(f"  {side} best split after game {j}: before {x[:j].mean() * unit:+.2f}"
                  f" -> after {x[j:].mean() * unit:+.2f}  z={z:+.2f}  "
                  f"scan-adjusted p={p:.4f}")


# --- 2. Persistence ------------------------------------------------------

def lagged_pairs(seqs: dict, window: int):
    """(cluster key, x = mean of previous `window` misses, y = this miss)."""
    for key, x in seqs.items():
        for k in range(window, len(x)):
            yield key, x[k - window:k].mean(), x[k]


def slope_stats(pairs: list) -> dict:
    keys = [p[0] for p in pairs]
    x = np.array([p[1] for p in pairs])
    y = np.array([p[2] for p in pairs])
    by_cluster: dict = {}
    for key, xi, yi in pairs:
        s = by_cluster.setdefault(key, [0.0, 0.0])
        s[0] += xi * yi
        s[1] += xi * xi
    sums = np.array(list(by_cluster.values()))
    draws = []
    for _ in range(BOOTSTRAP):
        pick = sums[RNG.integers(0, len(sums), len(sums))].sum(axis=0)
        draws.append(pick[0] / pick[1])
    seasons = np.array([k[0] for k in keys])
    base = adj = 0.0
    for s in np.unique(seasons):
        test, train = seasons == s, seasons != s
        b = (x[train] @ y[train]) / (x[train] @ x[train])
        base += float((y[test] ** 2).sum())
        adj += float(((y[test] - b * x[test]) ** 2).sum())
    return {"n": len(y), "slope": float(x @ y / (x @ x)),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "gain": 1 - adj / base}


def persistence(frame: pd.DataFrame) -> None:
    print("\n=== 2. Does recent form predict the next miss? (all FBS) ===")
    print(" slope 0 = streaks are noise; >0 = ratings lag real change."
          " gain = leave-one-season-out cut in squared error.\n")
    print(f" {'stat':<18}{'win':>4}{'n':>7}{'slope':>8}{'95% CI':>16}{'gain':>8}")
    for kind in ("points", "success"):
        for side, column in SIDES.items():
            seqs = sequences(frame, kind, column)
            for window in (2, 3, 4, 6):
                s = slope_stats(list(lagged_pairs(seqs, window)))
                print(f" {kind + ' ' + side:<18}{window:>4}{s['n']:>7}{s['slope']:>+8.3f}"
                      f"  [{s['lo']:+.3f},{s['hi']:+.3f}]{s['gain']:>8.2%}")
    net = frame[frame.kind == "points"].dropna(subset=["def_miss"]).copy()
    net["net"] = net["miss"] + net["def_miss"]
    seqs = sequences(net.assign(kind="points"), "points", "net")
    print()
    for window in (2, 3, 4, 6):
        s = slope_stats(list(lagged_pairs(seqs, window)))
        print(f" {'points NET margin':<18}{window:>4}{s['n']:>7}{s['slope']:>+8.3f}"
              f"  [{s['lo']:+.3f},{s['hi']:+.3f}]{s['gain']:>8.2%}")


# --- 3. Flag test --------------------------------------------------------

def cluster_mean_ci(rows: list) -> tuple[float, float, float]:
    """Mean and 95% CI of (cluster key, value) rows, resampling clusters."""
    by_cluster: dict = {}
    for key, value in rows:
        s = by_cluster.setdefault(key, [0.0, 0])
        s[0] += value
        s[1] += 1
    sums = np.array(list(by_cluster.values()))
    draws = []
    for _ in range(BOOTSTRAP):
        pick = sums[RNG.integers(0, len(sums), len(sums))].sum(axis=0)
        draws.append(pick[0] / pick[1])
    mean = sums[:, 0].sum() / sums[:, 1].sum()
    return float(mean), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def flag_test(frame: pd.DataFrame) -> None:
    chance = erfc(FLAG_Z / sqrt(2))   # two-sided P(|z| >= FLAG_Z) if all noise
    print(f"\n=== 3. Flag test: last {RECENT} vs earlier, |z| >= {FLAG_Z} ===")
    print(" 'next' = mean miss over the following 3 games (sigma units;"
          " 0 = shift already absorbed).")
    print(f" If every miss were independent noise, {chance:.1%} of checkpoints"
          " would flag.\n")
    for kind in ("points", "success"):
        for side, column in SIDES.items():
            sigma = sigma_of(frame, kind, column)
            groups = {"up": [], "down": [], "none": []}
            for key, x in sequences(frame, kind, column).items():
                for k in range(RECENT + MIN_SEGMENT, len(x)):
                    early, late = x[:k - RECENT], x[k - RECENT:k]
                    z = (late.mean() - early.mean()) / (
                        sigma * np.sqrt(1 / RECENT + 1 / len(early)))
                    nxt = x[k:k + 3].mean() / sigma
                    bucket = "up" if z >= FLAG_Z else "down" if z <= -FLAG_Z else "none"
                    groups[bucket].append((key, nxt))
            total = sum(len(v) for v in groups.values())
            flagged = len(groups["up"]) + len(groups["down"])
            print(f" {kind + ' ' + side:<14} sigma={sigma * (1 if kind == 'points' else 100):.1f}"
                  f"  flag rate {flagged / total:.1%}")
            for name, rows in groups.items():
                mean, lo, hi = cluster_mean_ci(rows)
                print(f"     {name:<5} n={len(rows):>5}  next {mean:+.2f}  [{lo:+.2f},{hi:+.2f}]")


# --- 4. League-wide ranking ---------------------------------------------

def rank_shifts(frame, team, season) -> None:
    print(f"\n=== 4. How unusual is {team} {season}? (like-for-like game counts) ===")
    for kind in ("points", "success"):
        for side, column in SIDES.items():
            sigma = sigma_of(frame, kind, column)
            seqs = sequences(frame, kind, column)
            mine = seqs.get((season, team))
            if mine is None:
                continue
            n = len(mine)
            stats = {k: scan_split(x[:n], sigma) for k, x in seqs.items()
                     if len(x) >= n}
            z, j = stats[(season, team)]
            beat = sum(abs(v[0]) >= abs(z) for v in stats.values())
            print(f" {kind + ' ' + side:<16} z={z:+.2f} (split after game {j}); "
                  f"{beat} of {len(stats)} FBS team-seasons shifted at least this much "
                  f"in {n} games (top {beat / len(stats):.1%})")


def movers(frame: pd.DataFrame, season: int, top: int = 8) -> None:
    print(f"\n=== Who else is moving in {season}? (success-rate scan, current games) ===")
    for side, column in SIDES.items():
        sigma = sigma_of(frame, "success", column)
        rows = []
        for (s, team), x in sequences(frame, "success", column).items():
            if s != season or len(x) < 2 * MIN_SEGMENT:
                continue
            z, j = scan_split(x, sigma)
            rows.append((abs(z), team, z, j, x[:j].mean() * 100, x[j:].mean() * 100))
        rows.sort(reverse=True)
        print(f"\n {side}ense:" if side == "def" else "\n offense:")
        for _, team, z, j, a, b in rows[:top]:
            print(f"  {team:<20} z={z:+.2f} after game {j}: "
                  f"{a:+.1f} -> {b:+.1f} success-rate pts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default="Oklahoma")
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    frame = load_fbs_misses()
    show_team(frame, args.team, args.season)
    persistence(frame)
    flag_test(frame)
    rank_shifts(frame, args.team, args.season)
    movers(frame, args.season)


if __name__ == "__main__":
    main()
