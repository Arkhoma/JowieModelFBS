"""Is EPA still worth a slot once success rate is in? Uses the cached
walk-forward components from improvement_experiment.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from improvement_experiment import CACHE, loso_stack  # noqa: E402

frame = pd.read_csv(CACHE)
base = loso_stack(frame, ["m_flat", "e_flat"])
for cols in (["m_roster", "e_roster", "s_roster"], ["m_roster", "s_roster"],
             ["m_roster", "e_roster"]):
    pred = loso_stack(frame, cols)
    d = np.abs(frame.actual - pred) - np.abs(frame.actual - base)
    gap = np.abs(frame.actual - pred) - np.abs(frame.actual - frame.line)
    print(f"{' + '.join(cols):<32} vs base {d.mean():+.4f} "
          f"(se {d.std() / np.sqrt(len(d)):.4f})  gap to line {gap.mean():+.3f}")
