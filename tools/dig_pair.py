"""Ad-hoc: why is team A ranked above team B?  Splits each rating into parts.

Usage: python tools/dig_pair.py "Texas A&M" Oklahoma
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.main import _latest_season, get_season  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402


def main() -> None:
    team_a, team_b = sys.argv[1:3]
    season = _latest_season()
    data = get_season(season)
    model = data.model
    prior = build_roster_prior_or_empty(season)
    rows = {r["team"]: r for r in data.rankings}

    print(f"season={season} margin_w={model.margin_weight} "
          f"epa_w={model.epa_weight} lambda={model.lambda_} "
          f"scale={model.margin_scale} halflife={model.halflife}")
    for team in (team_a, team_b):
        row = rows[team]
        print("-" * 60)
        print(f"{team}: rank {row['rank']} rating {row['rating']:.2f} "
              f"record {row['wins']}-{row['losses']}")
        print(f"  margin model rating   {model.margin_model.rating(team):7.2f}"
              f"  -> x{model.margin_weight} = "
              f"{model.margin_weight * model.margin_model.rating(team):.2f}")
        print(f"  success model (pts)   "
              f"{model.epa_model.rating_in_points(team):7.2f}"
              f"  -> x{model.epa_weight} = "
              f"{model.epa_weight * model.epa_model.rating_in_points(team):.2f}")
        print(f"  preseason prior       {prior.get(team, float('nan')):7.2f}")
        print(f"  resume rank {row.get('resume_rank')}  "
              f"off {row.get('off')} def {row.get('def')}")
        for line in row.get("why", []):
            print(f"  why: {line}")

    print("-" * 60)
    for label, home, away, neutral in [
        ("neutral", team_a, team_b, True),
        (f"{team_a} home", team_a, team_b, False),
        (f"{team_b} home", team_b, team_a, False),
    ]:
        p = data.predictor.predict(home, away, neutral)
        print(f"{label:<22} home margin {p.home_score - p.away_score:+.1f}  "
              f"home win {p.home_win_probability:.0%}")

    print_success_rates(season, team_a, team_b)


def print_success_rates(season: int, *teams: str) -> None:
    """Per-game offense/defense success rate behind the efficiency half."""
    from cfbrank.epa_games import season_epa
    from cfbrank.epa_ridge import get_ep_model

    records = season_epa(season, get_ep_model(), drop_garbage_time=True)
    by_game: dict[str, dict] = {}
    for rec in records:
        by_game.setdefault(rec.game_id, {})[rec.team] = rec
    for team in teams:
        print("-" * 60)
        print(f"{team}: success rate per game (off / def / net)")
        offs, defs = [], []
        for sides in by_game.values():
            if team not in sides or len(sides) != 2:
                continue
            opp = next(t for t in sides if t != team)
            me = sides[team]
            offs.append(me.offense_success_rate)
            defs.append(me.defense_success_rate)
            net = me.offense_success_rate - me.defense_success_rate
            print(f"  vs {opp:<22} {me.offense_success_rate:.3f} / "
                  f"{me.defense_success_rate:.3f} / {net:+.3f}")
        print(f"  AVG off {sum(offs)/len(offs):.3f}  def {sum(defs)/len(defs):.3f}")


if __name__ == "__main__":
    main()
