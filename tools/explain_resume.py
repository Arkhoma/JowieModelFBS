"""Break down one team's resume game by game: opponent rating, site, and
the benchmark team's win probability that drives best-win / worst-loss.

Usage: python tools/explain_resume.py "Ole Miss" [season]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import _latest_season, get_season  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.resume import benchmark_rating, build_resumes  # noqa: E402


def main() -> None:
    team = sys.argv[1] if len(sys.argv) > 1 else "Ole Miss"
    season = int(sys.argv[2]) if len(sys.argv) > 2 else _latest_season()
    built = get_season(season)
    model = built.model
    games = load_season(season)
    resume = build_resumes(games, model, built.predictor.errors)[team]
    rank = {r["team"]: r["rank"] for r in built.rankings}
    sites = {}
    for g in games:
        if team in (g.home_team, g.away_team):
            opp = g.away_team if g.home_team == team else g.home_team
            sites[(opp, g.week)] = ("neutral" if g.neutral_site else
                                    "home" if g.home_team == team else "away")

    print(f"{team} {season}: benchmark rating "
          f"{benchmark_rating(model):+.1f}, home field "
          f"{model.home_field:+.1f}\n")
    print(f"{'wk':>3} {'res':<3} {'opponent':<22} {'rank':>5} {'rating':>7} "
          f"{'site':<8} {'bench win%':>10}")
    for g in sorted(resume.games, key=lambda g: g.week):
        print(f"{g.week:>3} {'W' if g.won else 'L':<3} {g.opponent:<22} "
              f"{rank.get(g.opponent, '-'):>5} "
              f"{model.rating(g.opponent):>+7.1f} "
              f"{sites[(g.opponent, g.week)]:<8} {g.win_probability:>9.0%}")
    best = resume.best_win
    print(f"\nBest win shown: {best.opponent if best else None}")


if __name__ == "__main__":
    main()
