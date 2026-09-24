"""Where does the +4 bias in the prediction errors come from?

Win probabilities (predict page AND resume) are read off the empirical
error distribution, so a biased distribution shifts every probability.
Splits in-sample residuals by game type to locate the bias.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import get_season  # noqa: E402
from cfbrank.games import load_season  # noqa: E402

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
model = get_season(season).model
groups: dict[str, list[float]] = {}
for g in load_season(season):
    if g.home_team not in model.ratings or g.away_team not in model.ratings:
        continue
    err = g.margin - model.predict_margin(g.home_team, g.away_team,
                                          g.neutral_site)
    kind = ("FBS v FCS" if g.involves_fcs else "FBS v FBS") + (
        " neutral" if g.neutral_site else " home")
    groups.setdefault(kind, []).append(err)
    groups.setdefault("ALL", []).append(err)

for kind, errs in sorted(groups.items()):
    e = np.array(errs)
    print(f"{kind:<22} n={len(e):>4} mean {e.mean():+6.2f} "
          f"median {np.median(e):+6.2f}")
