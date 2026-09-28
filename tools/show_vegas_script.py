"""Vegas Script vs our number vs the real line, for one week of a season.

    python tools/show_vegas_script.py 2026 4
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import get_season  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.lines import home_lines  # noqa: E402

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
week = int(sys.argv[2]) if len(sys.argv) > 2 else None
data = get_season(season)
lines = home_lines()
print(f"Vegas Script loaded: {data.predictor.vegas is not None} "
      f"(priced games this season: "
      f"{sum(str(g.game_id) in lines for g in load_season(season))})")

import csv  # noqa: E402
sched = ROOT / "data" / "raw" / "schedules" / f"schedules_{season}.csv"
rows = [r for r in csv.DictReader(open(sched, encoding="utf-8"))
        if r["home_division"] == "fbs" and r["away_division"] == "fbs"]
week = week or min(int(r["week"]) for r in rows if r["home_points"] in ("", "NA"))
fbs = data.model.fbs_ratings()
print(f"\nWeek {week}: {'matchup':42s} {'ours':>6} {'script':>7} {'line':>6}")
for r in rows:
    if int(r["week"]) != week or r["home_team"] not in fbs or r["away_team"] not in fbs:
        continue
    p = data.predictor.predict(r["home_team"], r["away_team"], r["neutral_site"] == "TRUE")
    line = lines.get(r["game_id"])
    script = f"{p.vegas_margin:+7.1f}" if p.vegas_margin is not None else "    n/a"
    print(f"  {r['away_team'] + ' @ ' + r['home_team']:48s} {p.predicted_margin:+6.1f} "
          f"{script} {'' if line is None else f'{line:+6.1f}'}")
