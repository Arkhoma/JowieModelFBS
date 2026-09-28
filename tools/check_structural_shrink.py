"""Are the structural terms (FCS offset, home field) being shrunk by the blend?

EnsembleModel weights (0.731 + 0.187 = 0.918) deliberately shrink TEAM
quality, but they multiply the whole prediction -- so the FCS offset and
home field shrink too. Residual check: regress (actual - ours) on the FCS
indicator and the home indicator, leave-one-season-out.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from gap_anatomy import BENCH  # noqa: E402
from cfbrank.games import load_season  # noqa: E402

df = pd.read_csv(BENCH, dtype={"game_id": str})
info = {}
for s in df.season.unique():
    for g in load_season(int(s)):
        fcs = 0.0
        if g.home_division != g.away_division:
            fcs = 1.0 if g.away_division == "fcs" else -1.0   # +1 = FBS home
        info[str(g.game_id)] = (fcs, 0.0 if g.neutral_site else 1.0)
df["fcs"] = [info[g][0] for g in df.game_id]
df["home"] = [info[g][1] for g in df.game_id]
df["early"] = (df.week <= 6).astype(float)
df["fcs_early"] = df.fcs * df.early
df["resid"] = df.actual - df.ours

for feats in (["home"], ["fcs"], ["home", "fcs"], ["home", "fcs", "fcs_early"]):
    gain, per_game = 0.0, []
    for season in df.season.unique():
        tr, te = df[df.season != season], df[df.season == season]
        coef, *_ = np.linalg.lstsq(tr[feats].to_numpy(), tr.resid, rcond=None)
        d = np.abs(te.resid) - np.abs(te.resid - te[feats].to_numpy() @ coef)
        per_game.append(d)
    d = np.concatenate(per_game)
    coef, *_ = np.linalg.lstsq(df[feats].to_numpy(), df.resid, rcond=None)
    print(f"{'+'.join(feats):22s} coef {np.round(coef, 2)}  "
          f"LOSO gain {d.mean():+.3f} +/-{1.96 * d.std() / np.sqrt(len(d)):.3f}")
