"""Would centring the error distribution improve win-probability calibration?

Predictor.from_model builds win probabilities from in-sample residuals.
If those residuals are biased (mean far from zero), every probability is
shifted. Compares walk-forward Brier score, raw vs centred.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.predict import Predictor, _empirical_win_probability  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
raw_b, cen_b, bias = [], [], []
for season in range(2021, 2026):
    games = sorted(load_season(season), key=lambda g: g.week)
    prior = build_roster_prior_or_empty(season) or None
    for week in sorted({g.week for g in games}):
        history = [g for g in games if g.week < week]
        upcoming = [g for g in games if g.week == week]
        if week < 4 or len(history) < 50:
            continue
        model = fit(history, prior=prior, **PARAMS)
        errors = Predictor.from_model(model, history).errors
        centred = errors - errors.mean()
        for g in upcoming:
            if g.home_team not in model.ratings or g.away_team not in model.ratings:
                continue
            m = model.predict_margin(g.home_team, g.away_team, g.neutral_site)
            won = 1.0 if g.margin > 0 else 0.0
            raw_b.append((_empirical_win_probability(m, errors) - won) ** 2)
            cen_b.append((_empirical_win_probability(m, centred) - won) ** 2)
            bias.append(errors.mean())

print(f"games {len(raw_b)}, mean in-sample residual bias {np.mean(bias):+.2f}")
print(f"Brier raw     {np.mean(raw_b):.4f}")
print(f"Brier centred {np.mean(cen_b):.4f}")
d = np.array(cen_b) - np.array(raw_b)
print(f"diff {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
