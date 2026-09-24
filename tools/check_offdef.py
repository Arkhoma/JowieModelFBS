"""Eyeball test for the offense/defense engine on one full season.

    python tools/check_offdef.py 2024
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank import offdef  # noqa: E402
from cfbrank.epa_games import season_epa  # noqa: E402
from cfbrank.epa_ridge import get_ep_model  # noqa: E402
from cfbrank.games import load_season  # noqa: E402


def show(label: str, model: offdef.OffDefResult, n: int = 8) -> None:
    fbs = [t for t in model.teams if model.divisions.get(t) == "fbs"]
    best_off = sorted(fbs, key=model.offense_rating, reverse=True)[:n]
    best_def = sorted(fbs, key=model.defense_rating, reverse=True)[:n]
    worst_off = sorted(fbs, key=model.offense_rating)[:3]
    print(f"\n== {label}: mean {model.mean:.3f}, HFA {model.home_field:.3f} "
          f"({model.home_field_points:.2f} pts), FCS {model.fcs_offset:.3f}, "
          f"edge_scale {model.edge_scale:.2f}")
    print(f"{'offense':<22}{'':>8}   {'defense':<22}")
    for o, d in zip(best_off, best_def):
        print(f"{o:<22}{model.offense_rating(o):>+8.3f}   "
              f"{d:<22}{model.defense_rating(d):>+8.3f}")
    print("worst offenses:", ", ".join(worst_off))


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    games = load_season(season)
    show("points", offdef.fit(offdef.sides_from_games(games), lambda_=3.0))
    records = season_epa(season, get_ep_model())
    sides = offdef.sides_from_efficiency(games, records, "success")
    show("success rate", offdef.fit(sides, lambda_=3.0, is_points=False))


if __name__ == "__main__":
    main()
