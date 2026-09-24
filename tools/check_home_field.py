"""Is the week-3 home-field estimate trustworthy?

Fits the production model on weeks 1-3 only and on the full season, for
past years, and compares the fitted home-field advantage. Early-season
schedules are dominated by home games against FCS and low-major teams,
so an unstable early estimate would inflate "home" and distort every
site-adjusted number downstream (predictions and resume alike).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

PARAMS = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}

print(f"{'season':>7} {'wk1-3 HFA':>10} {'full HFA':>9}")
for season in range(2021, 2027):
    games = load_season(season)
    prior = build_roster_prior_or_empty(season) or None
    early = fit([g for g in games if g.week <= 3], prior=prior, **PARAMS)
    full = fit(games, prior=prior, **PARAMS)
    print(f"{season:>7} {early.home_field:>+10.2f} {full.home_field:>+9.2f}")
