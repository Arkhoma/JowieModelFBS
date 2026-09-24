"""Scratch script: dump rating components + raw EPA edges for named teams.

Not part of the model; just a debugging aid for chasing a specific
"why is this team ranked where it is" question. Safe to delete/rewrite
whenever the next investigation needs something different.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank.games import load_season  # noqa: E402
from cfbrank.prior import build_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit  # noqa: E402
from cfbrank.epa_ridge import (  # noqa: E402
    build_epa_games, fit_epa_ratings, fit_epa_ratings_primed, get_ep_model,
)
from cfbrank.ensemble import EnsembleModel  # noqa: E402

SEASON = 2026
TEAMS = ["New Mexico", "Ole Miss", "LSU", "Mississippi State", "Notre Dame"]


def main() -> None:
    games = load_season(SEASON)
    prior = build_prior_or_empty(SEASON)
    margin_model = fit(games, prior=prior or None, lambda_=5.0,
                        margin_scale=28.0, halflife=1e6)
    epa_games = build_epa_games(SEASON, games, get_ep_model())
    epa_model_raw = fit_epa_ratings(epa_games)
    epa_model = fit_epa_ratings_primed(epa_games, prior or None)
    ens = EnsembleModel(margin_model, epa_model)

    for team in TEAMS:
        print(f"=== {team}")
        print(f"  prior:            {prior.get(team)}")
        print(f"  margin rating:    {margin_model.rating(team)}")
        epa_rating = (epa_model.rating_in_points(team)
                      if team in epa_model.ratings else None)
        epa_rating_raw = (epa_model_raw.rating_in_points(team)
                          if team in epa_model_raw.ratings else None)
        print(f"  epa rating, unprimed (pts): {epa_rating_raw}")
        print(f"  epa rating, primed (pts):   {epa_rating}")
        print(f"  ensemble rating:  {ens.rating(team)}")
        for game in games:
            if team not in (game.home_team, game.away_team):
                continue
            is_home = game.home_team == team
            opp = game.away_team if is_home else game.home_team
            opp_division = game.away_division if is_home else game.home_division
            pts_for = game.home_points if is_home else game.away_points
            pts_against = game.away_points if is_home else game.home_points
            print(f"    wk{game.week}: vs {opp} ({opp_division}) "
                  f"{pts_for}-{pts_against}")
        for eg in epa_games:
            if team not in (eg.home_team, eg.away_team):
                continue
            sign = 1.0 if eg.home_team == team else -1.0
            opp = eg.away_team if sign == 1.0 else eg.home_team
            print(f"    wk{eg.week} raw epa_edge vs {opp}: "
                  f"{sign * eg.epa_edge:+.4f}")


if __name__ == "__main__":
    main()
