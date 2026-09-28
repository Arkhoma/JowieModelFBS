"""Cross-check our closing lines (sportsdataverse) against CFBD /lines.

CFBD quotes `spread` from the HOME team's view with homeTeam named, so a
sign flip or wrong-side pick in load_home_lines() shows up as a large
disagreement. Prints agreement stats and the worst mismatches.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_vs_market import load_home_lines  # noqa: E402

PREFERRED = ("consensus", "Bovada", "DraftKings", "ESPN Bet", "William Hill (New Jersey)")


def cfbd_home_lines(season: int) -> dict[str, tuple[float, str, str]]:
    path = ROOT / "data" / "raw" / str(season) / "lines.json.gz"
    if not path.exists():
        return {}
    out = {}
    for game in json.load(gzip.open(path, "rt", encoding="utf-8")):
        spreads = {l.get("provider"): l.get("spread") for l in game.get("lines") or []
                   if l.get("spread") is not None}
        if not spreads:
            continue
        pick = next((spreads[p] for p in PREFERRED if p in spreads),
                    float(np.median(list(spreads.values()))))
        # CFBD spread: negative = home favoured -> home margin = -spread
        out[str(game["id"])] = (-float(pick), game.get("homeTeam"), game.get("awayTeam"))
    return out


def main() -> None:
    ours = load_home_lines()
    rows = []
    for season in range(2021, 2027):
        for gid, (line, home, away) in cfbd_home_lines(season).items():
            if gid in ours:
                rows.append((season, gid, home, away, ours[gid], line))
    df = pd.DataFrame(rows, columns=["season", "game_id", "home", "away", "ours_src", "cfbd"])
    df["diff"] = df.ours_src - df.cfbd
    flipped = (np.sign(df.ours_src) == -np.sign(df.cfbd)) & (df.cfbd.abs() >= 3) \
        & ((df.ours_src + df.cfbd).abs() <= 3)
    print(f"{len(df)} games in both sources")
    print(f"  within 1.5 pts: {(df['diff'].abs() <= 1.5).mean():.1%}")
    print(f"  sign-flipped (|line|>=3): {flipped.sum()}  ({flipped.mean():.1%})")
    print(f"  off by >3 but not a flip: {((df['diff'].abs() > 3) & ~flipped).sum()}")
    print("\nflips by season:", df[flipped].groupby("season").size().to_dict())
    print("\nworst 20:")
    print(df.reindex(df["diff"].abs().sort_values(ascending=False).index)
          .head(20).to_string(index=False))
    df.assign(flipped=flipped).to_csv(ROOT / "data" / "line_crosscheck.csv", index=False)


if __name__ == "__main__":
    main()
