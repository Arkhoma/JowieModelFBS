"""How much would a MARKET-INFORMED rating close the gap?

For each week W, fit the same ridge engine on the closing LINES (not the
results) of every game before W. Lines are public before kickoff, so
this is no leak: it's "what did Vegas think of each team, as of last
week". Then stack it with our production prediction leave-one-season-out.

If this closes most of the gap, the remaining difference between us and
the line is information the market has (injuries, depth charts, sharp
money) that is recoverable from its past prices.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from cfbrank.lines import home_lines as load_home_lines  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402
from cfbrank.vegas_script import (  # noqa: E402
    MARKET_PARAMS, MIN_PRICED_GAMES, priced_games,
)

BENCH = ROOT / "data" / "benchmark_vs_market.csv"
LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else MARKET_PARAMS["lambda_"]
USE_PRIOR = len(sys.argv) > 2 and sys.argv[2] == "prior"
PARAMS = {**MARKET_PARAMS, "lambda_": LAMBDA}


def market_predictions(season: int, lines: dict[str, float]) -> dict[str, float]:
    games = load_season(season)
    priced = priced_games(games, lines)
    out = {}
    prior = (build_roster_prior_or_empty(season) or None) if USE_PRIOR else None
    for week in sorted({g.week for g in games}):
        history = [g for g in priced if g.week < week]
        if len(history) < MIN_PRICED_GAMES:
            continue
        model = fit(history, prior=prior, **PARAMS)
        for g in games:
            if g.week != week:
                continue
            if g.home_team in model.ratings and g.away_team in model.ratings:
                out[str(g.game_id)] = model.predict_margin(
                    g.home_team, g.away_team, g.neutral_site)
    return out


def main() -> None:
    lines = load_home_lines()
    df = pd.read_csv(BENCH, dtype={"game_id": str})
    mk = {}
    for season in sorted(df.season.unique()):
        mk.update(market_predictions(int(season), lines))
    df["mkt_rating"] = df.game_id.map(mk)
    df = df.dropna(subset=["mkt_rating"])
    print(f"lambda={LAMBDA} prior={USE_PRIOR}: {len(df)} games with a market rating")

    err = lambda p: np.abs(df.actual - p)
    print(f"  ours alone        MAE {err(df.ours).mean():.3f}")
    print(f"  market-rating     MAE {err(df.mkt_rating).mean():.3f}")
    print(f"  closing line      MAE {err(df.line).mean():.3f}")

    pred = np.zeros(len(df))
    for season in df.season.unique():
        tr, te = (df.season != season).to_numpy(), (df.season == season).to_numpy()
        X = df[["ours", "mkt_rating"]].to_numpy()
        w, *_ = np.linalg.lstsq(X[tr], df.actual[tr], rcond=None)
        pred[te] = X[te] @ w
    w, *_ = np.linalg.lstsq(df[["ours", "mkt_rating"]], df.actual, rcond=None)
    gap_old = (err(df.ours) - err(df.line))
    gap_new = (err(pred) - err(df.line))
    print(f"  stacked (LOSO)    MAE {err(pred).mean():.3f}   weights {np.round(w, 3)}")
    print(f"\n  gap to line: {gap_old.mean():+.3f} -> {gap_new.mean():+.3f}")
    for name, m in (("wk<=8", df.week <= 8), ("wk>8", df.week > 8)):
        print(f"    {name:6s} {gap_old[m].mean():+.3f} -> {gap_new[m].mean():+.3f}")


if __name__ == "__main__":
    main()
