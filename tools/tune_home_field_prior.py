"""Choose the home-field prior strength by walk-forward MAE.

Centre = mean full-season home field of the PREVIOUS seasons only (no
peeking). Strength is in "equivalent games": 0 = today's unpenalised fit.
Also reports the fitted home field at week 4 for each strength.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
STRENGTHS = (0, 50, 100, 200, 400, 1000)
full_hfa = {s: fit(load_season(s), **PARAMS).home_field
            for s in range(2016, 2026)}

results = {k: [] for k in STRENGTHS}
early_hfa = {k: [] for k in STRENGTHS}
for season in range(2021, 2026):
    centre = float(np.mean([full_hfa[s] for s in range(season - 5, season)]))
    games = sorted(load_season(season), key=lambda g: g.week)
    prior = build_roster_prior_or_empty(season) or None
    for week in sorted({g.week for g in games}):
        history = [g for g in games if g.week < week]
        upcoming = [g for g in games if g.week == week]
        if week < 4 or len(history) < 50:
            continue
        for k in STRENGTHS:
            m = fit(history, prior=prior, **PARAMS,
                    home_field_prior=(centre, k) if k else None)
            if week == 4:
                early_hfa[k].append(m.home_field)
            for g in upcoming:
                if g.home_team in m.ratings and g.away_team in m.ratings:
                    results[k].append(abs(g.margin - m.predict_margin(
                        g.home_team, g.away_team, g.neutral_site)))

base = np.array(results[0])
print(f"{'strength':>9} {'MAE':>7} {'vs none':>8} {'wk4 HFA':>8}")
for k in STRENGTHS:
    r = np.array(results[k])
    print(f"{k:>9} {r.mean():>7.3f} {(r - base).mean():>+8.3f} "
          f"{np.mean(early_hfa[k]):>8.2f}")
print("full-season HFA 2021-25:",
      ", ".join(f"{full_hfa[s]:.2f}" for s in range(2021, 2026)))
