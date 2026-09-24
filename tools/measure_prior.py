"""Measure offseason carryover, then show the prior's effect on today.

Answers two questions with data instead of opinion:
  1. How much of last season's rating survives? (rho)
  2. Does using it actually fix the week-3 table?
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import available_seasons, load_season  # noqa: E402
from cfbrank.prior import build_prior, measure_carryover  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402

FITTED = {"lambda_": 5.0, "margin_scale": 28.0, "halflife": 1e6}
TOP_N = 15


def table(model, games, label: str) -> None:
    records: dict[str, list[int]] = {}
    for game in games:
        for team in (game.home_team, game.away_team):
            records.setdefault(team, [0, 0])
        records[game.winner][0] += 1
        records[game.loser][1] += 1

    ranked = sorted(model.fbs_ratings().items(), key=lambda kv: -kv[1])
    print(f"\n{label}")
    print("-" * 46)
    for rank, (team, rating) in enumerate(ranked[:TOP_N], 1):
        wins, losses = records.get(team, (0, 0))
        print(f"{rank:>3}  {team:<26} {rating:>+6.2f}  {wins}-{losses}")


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else max(available_seasons())

    seasons = [s for s in available_seasons() if s - 1 in available_seasons()]
    measured = measure_carryover(seasons)
    print(f"Carryover measured on {measured['n_games']} early-season games "
          f"across {len(seasons)} seasons")
    print(f"  rho          {measured['rho']:.3f}   "
          "(1.0 = teams unchanged year to year, 0.0 = no signal)")
    print(f"  home field   {measured['home_field']:+.2f}")
    print(f"  R^2          {measured['r_squared']:.3f}")

    games = load_season(season)
    table(fit(games, **FITTED), games, f"{season} WITHOUT prior (shipped)")

    prior = build_prior(season, measured["rho"])
    table(fit(games, prior=prior, **FITTED), games,
          f"{season} WITH prior (rho={measured['rho']:.2f})")


if __name__ == "__main__":
    main()
