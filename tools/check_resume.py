"""Ad-hoc: does the resume rating fix the Ole Miss / LSU inversion?"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.main import _latest_season, get_season  # noqa: E402
from cfbrank.games import load_season  # noqa: E402
from cfbrank.resume import build_resumes, rank_resumes  # noqa: E402


def main() -> None:
    season = _latest_season()
    data = get_season(season)
    games = load_season(season)
    resumes = build_resumes(games, data.model, data.predictor.errors)
    ranked = rank_resumes(resumes, data.model)
    pred_rank = {r["team"]: r["rank"] for r in data.rankings}

    print(f"RESUME (strength of record) -- {season} week {data.max_week}")
    print(f"{'#':>3} {'team':<24} {'score':>6} {'rec':>5} "
          f"{'expW':>5} {'pred':>5}  best win")
    print("-" * 82)
    for rank, r in ranked[:25]:
        bw = r.best_win
        note = (f"{bw.opponent} ({bw.win_probability:.0%})" if bw else "-")
        print(f"{rank:>3} {r.team:<24} {r.score:>+6.2f} "
              f"{r.wins}-{r.losses} {r.expected_wins:>5.2f} "
              f"{pred_rank.get(r.team, '-'):>5}  {note}")

    print("\n" + "=" * 82)
    by_team = {r.team: (rank, r) for rank, r in ranked}
    for a, b in (("Ole Miss", "LSU"), ("Oklahoma State", "Oregon")):
        ra, oa = by_team[a]
        rb, ob = by_team[b]
        verdict = "FIXED" if ra < rb else "still inverted"
        print(f"\n{a} #{ra} ({oa.score:+.2f}) vs {b} #{rb} "
              f"({ob.score:+.2f})  resume -> {verdict}")
        print(f"   predictive: {a} #{pred_rank[a]} vs {b} #{pred_rank[b]}")
        for label, r in ((a, oa), (b, ob)):
            print(f"   {label}:")
            for g in sorted(r.games, key=lambda g: g.week):
                print(f"      wk{g.week} {'W' if g.won else 'L'} "
                      f"{g.opponent:<22} benchmark winprob "
                      f"{g.win_probability:>5.0%}  credit {g.credit:+.2f}")


if __name__ == "__main__":
    main()
