"""Sanity-check game_features on one season."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.game_features import season_features  # noqa: E402
from cfbrank.games import load_season  # noqa: E402

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
games = load_season(season)
feats = season_features(season, games)
by_id = {str(g.game_id): g for g in games}
keys = [k for k in feats if k in by_id]
print(f"{len(games)} games, {len(keys)} with features")

luck = np.array([feats[k].home_fumble_luck for k in keys])
margin = np.array([by_id[k].margin for k in keys], dtype=float)
print(f"fumble luck mean {luck.mean():+.3f} sd {luck.std():.3f}")
print(f"margin ~ luck slope {np.polyfit(luck, margin, 1)[0]:.2f} pts per "
      f"extra recovery, corr {np.corrcoef(margin, luck)[0, 1]:.3f}")

decided = [k for k in keys if feats[k].garbage_margin is not None]
print(f"{len(decided)} games reached garbage time")
for k in decided[:8]:
    print(f"  final {by_id[k].margin:+4d}  when decided "
          f"{feats[k].garbage_margin:+5.0f}  {by_id[k].away_team} @ "
          f"{by_id[k].home_team}")
