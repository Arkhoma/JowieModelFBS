"""Does the roster prior beat the flat carryover prior at predicting
each team's final rating? Out of sample: weights for season S are fit
only on offseasons before S."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.prior import DEFAULT_UNSHRINK, build_prior  # noqa: E402
from cfbrank.prior_model import (  # noqa: E402
    FEATURES, _final, build_roster_prior, fit_weights,
)

print(f"{'season':>7} {'n':>4} {'flat RMSE':>10} {'roster RMSE':>12}")
for season in range(2019, 2026):
    final, div = _final(season)
    flat, roster = build_prior(season), build_roster_prior(season)
    teams = [t for t in final if div.get(t) == "fbs" and t in flat
             and t in roster]
    truth = np.array([final[t] for t in teams]) * DEFAULT_UNSHRINK
    f = np.array([flat[t] for t in teams])
    r = np.array([roster[t] for t in teams])
    print(f"{season:>7} {len(teams):>4} "
          f"{np.sqrt(np.mean((truth - f) ** 2)):>10.2f} "
          f"{np.sqrt(np.mean((truth - r) ** 2)):>12.2f}")

print("\nWeights fit through 2025:")
for name, w in zip(FEATURES, fit_weights(2026)):
    print(f"  {name:<12} {w:+.3f}")
