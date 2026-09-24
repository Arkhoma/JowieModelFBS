"""Ad-hoc: dump the top of the board plus head-to-head context for teams.

Usage: python tools/inspect_pair.py [Team A] [Team B] ...
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.main import _latest_season, get_season  # noqa: E402
from cfbrank.games import load_season  # noqa: E402


def main() -> None:
    season = _latest_season()
    data = get_season(season)
    games = load_season(season)
    print(f"season={season} week={data.max_week} games={data.n_games} "
          f"epa={data.uses_epa}")
    print("-" * 72)
    for r in data.rankings[:30]:
        print(f"{r['rank']:>3} {r['team']:<24} {r['rating']:>7.2f} "
              f"{r['wins']}-{r['losses']}  sos={r['sos']:>6.2f}")

    by_rank = {r["team"]: r for r in data.rankings}
    for team in sys.argv[1:]:
        row = by_rank.get(team)
        print("-" * 72)
        if row is None:
            close = [t for t in by_rank if team.lower() in t.lower()]
            print(f"{team!r} not found. Close: {close[:8]}")
            continue
        print(f"{row['rank']:>3} {team} {row['rating']:.2f} "
              f"{row['wins']}-{row['losses']} sos={row['sos']:.2f}")
        for g in games:
            if team not in (g.home_team, g.away_team):
                continue
            opp = g.away_team if g.home_team == team else g.home_team
            orank = by_rank.get(opp, {}).get("rank", "-")
            res = "W" if g.winner == team else "L"
            print(f"    wk{g.week:>2} {res} vs {opp:<24} "
                  f"(#{orank})  {g.home_points}-{g.away_points} "
                  f"{'@' if g.away_team == team else 'H'}")


if __name__ == "__main__":
    main()
