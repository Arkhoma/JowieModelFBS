"""Ad-hoc: is the published ORDER closer to the betting market's order?

Fits market ratings from this season's closing lines (the Vegas Script's
own fit) and compares them with two versions of our rating: the old RAW
sum (what the table used before 61df08c) and the calibrated one it uses
now.  Rank similarity only -- game predictions never changed.

    python tools/check_rank_vs_market.py [Team ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.main import _latest_season, get_season  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.lines import home_lines  # noqa: E402
from cfbrank.vegas_script import fit_market_ratings  # noqa: E402


def ranks(values: np.ndarray) -> np.ndarray:
    """1 = best."""
    return np.argsort(np.argsort(-values)) + 1


def compare(label: str, ours: np.ndarray, market: np.ndarray) -> None:
    ours_rank, market_rank = ranks(ours), ranks(market)
    spearman = float(np.corrcoef(ours_rank, market_rank)[0, 1])
    pearson = float(np.corrcoef(ours, market)[0, 1])
    mean_gap = float(np.mean(np.abs(ours_rank - market_rank)))
    top25 = len(set(np.where(ours_rank <= 25)[0])
                & set(np.where(market_rank <= 25)[0]))
    print(f"{label:<22} spearman {spearman:.4f}  pearson {pearson:.4f}  "
          f"mean |rank gap| {mean_gap:5.2f}  top-25 overlap {top25}/25")


def main() -> None:
    season = _latest_season()
    data = get_season(season)
    model = data.model
    games = load_season(season)
    market = fit_market_ratings(games, home_lines())
    if market is None:
        raise SystemExit("Too few priced games for market ratings.")

    teams = [t for t in model.fbs_ratings() if t in market.ratings]
    new = np.array([model.rating(t) for t in teams])
    old = np.array([model.margin_weight * model.margin_model.rating(t)
                    + model.epa_weight * model.epa_model.rating_in_points(t)
                    for t in teams])
    mkt = np.array([market.calibrated_rating(t) for t in teams])

    priced = sum(str(g.game_id) in home_lines() for g in games)
    print(f"season {season}: {len(teams)} FBS teams, {priced} priced games\n")
    compare("old (raw sum)", old, mkt)
    compare("new (calibrated)", new, mkt)

    mkt_rank = dict(zip(teams, ranks(mkt)))
    old_rank = dict(zip(teams, ranks(old)))
    new_rank = dict(zip(teams, ranks(new)))
    print(f"\n{'team':<20} {'market':>7} {'old':>5} {'new':>5}")
    for team in sys.argv[1:]:
        print(f"{team:<20} {mkt_rank[team]:>7} {old_rank[team]:>5} "
              f"{new_rank[team]:>5}")


if __name__ == "__main__":
    main()
