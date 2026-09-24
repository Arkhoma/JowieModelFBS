"""Postseason coverage per season: how many bowl/playoff games load, and where.

    python tools/check_postseason.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import POSTSEASON, available_seasons, load_season  # noqa: E402


def main() -> None:
    print(f"{'season':>6} {'post':>5} {'regMax':>6}  postseason weeks")
    for season in available_seasons():
        games = load_season(season)
        post = [g for g in games if g.season_type == POSTSEASON]
        regular = [g.week for g in games if g.season_type != POSTSEASON]
        weeks = dict(sorted(Counter(g.week for g in post).items()))
        print(f"{season:>6} {len(post):>5} {max(regular, default=0):>6}  {weeks}")


if __name__ == "__main__":
    main()
