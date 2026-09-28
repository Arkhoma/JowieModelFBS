"""How much venue context is filled in, per season (FBS home-site games)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.venue import season_context  # noqa: E402

print(f"{'season':>6} {'games':>5} {'att':>5} {'cap':>5} {'elev':>5} "
      f"{'travel':>6} {'elo':>5} {'medfill':>7}")
for season in range(2014, 2027):
    ctx = season_context(season)
    games = [g for g in load_season(season)
             if g.home_division == "fbs" and not g.neutral_site]
    rows = [ctx.get(str(g.game_id)) for g in games]
    rows = [r for r in rows if r]
    fills = sorted(r.fill for r in rows if r.fill)
    print(f"{season:>6} {len(games):>5} "
          f"{sum(1 for r in rows if r.attendance):>5} "
          f"{sum(1 for r in rows if r.capacity):>5} "
          f"{sum(1 for r in rows if r.elevation_m):>5} "
          f"{sum(1 for r in rows if r.travel_miles):>6} "
          f"{sum(1 for r in rows if r.home_elo and r.away_elo):>5} "
          f"{fills[len(fills) // 2] if fills else float('nan'):>7.2f}")
