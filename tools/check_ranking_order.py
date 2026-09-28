"""Is the ranking ORDER right? Three independent checks.

1. Forward (the honest test). Each week, rank teams using only earlier
   games, then grade the order on that week's games: did the higher-ranked
   team win? Compared on the SAME games with the AP poll released for that
   week, and with the closing-line favourite (which also knows home field,
   so it's a ceiling, not a peer). 2021-25.

2. Hindsight agreement. Final ratings vs SP+ and SRS final ratings: rank
   correlation, top-25 overlap, and how often each final order "explains"
   the season's games (higher-rated team won).

3. Right now. 2026 top 25 vs SP+ and the latest AP poll, with the biggest
   disagreements called out.

    python tools/check_ranking_order.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_vs_market import (  # noqa: E402
    MARGIN_PARAMS, MIN_TRAINING, RELATIVE_LAMBDA, START_WEEK,
)
from cfbrank.ensemble import EFFICIENCY_METRIC, EnsembleModel  # noqa: E402
from cfbrank.epa_ridge import (  # noqa: E402
    build_epa_games, fit_epa_ratings_primed, get_ep_model,
)
from cfbrank.games import load_season  # noqa: E402
from cfbrank.lines import home_lines  # noqa: E402
from cfbrank.prior import home_field_prior  # noqa: E402
from cfbrank.prior_model import build_roster_prior_or_empty  # noqa: E402
from cfbrank.ridge import fit as fit_margin  # noqa: E402

RAW = ROOT / "data" / "raw"
SEASONS = (2021, 2022, 2023, 2024, 2025)
CURRENT = 2026


def _load(season: int, name: str):
    path = RAW / str(season) / f"{name}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def ap_polls(season: int) -> dict[int, dict[str, int]]:
    """week -> {team: AP rank}. CFBD's week-N poll is the one released
    before week N's games (week 1 = preseason)."""
    out = {}
    for week in _load(season, "polls") or []:
        if week["seasonType"] != "regular":
            continue
        for poll in week["polls"]:
            if poll["poll"] == "AP Top 25":
                out[week["week"]] = {r["school"]: r["rank"] for r in poll["ranks"]}
    return out


def final_external(season: int, name: str) -> dict[str, float]:
    return {r["team"]: r["rating"] for r in _load(season, name) or []
            if r.get("rating") is not None}


def build_model(season, history, epa_by_id, prior, hfa_prior):
    epa_history = [epa_by_id[str(g.game_id)] for g in history
                   if str(g.game_id) in epa_by_id]
    if len(history) < MIN_TRAINING or len(epa_history) < MIN_TRAINING:
        return None
    return EnsembleModel(
        margin_model=fit_margin(history, prior=prior,
                                home_field_prior=hfa_prior, **MARGIN_PARAMS),
        epa_model=fit_epa_ratings_primed(epa_history, prior,
                                         relative_lambda=RELATIVE_LAMBDA))


def spearman(a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
    shared = sorted(set(a) & set(b))
    ra = np.argsort(np.argsort([-a[t] for t in shared]))
    rb = np.argsort(np.argsort([-b[t] for t in shared]))
    return float(np.corrcoef(ra, rb)[0, 1]), len(shared)


def top(ratings: dict[str, float], n: int = 25) -> list[str]:
    return [t for t, _ in sorted(ratings.items(), key=lambda kv: -kv[1])[:n]]


def pct(hits: list[bool]) -> str:
    if not hits:
        return "   n/a"
    p = np.mean(hits)
    se = np.sqrt(p * (1 - p) / len(hits))
    return f"{p:6.1%} +/-{1.96 * se:4.1%} (n={len(hits)})"


def forward_check() -> None:
    print("=" * 78)
    print("1. FORWARD: rank with earlier games only, grade on this week's games")
    print("=" * 78)
    lines = home_lines()
    ep = get_ep_model()
    all_games, ap_both, ap_one = [], [], []
    by_gap = {b: [] for b in ("0-3", "3-7", "7-14", "14+")}
    finals = {}

    for season in SEASONS:
        games = sorted(load_season(season), key=lambda g: g.week)
        epa_by_id = {g.game_id: g for g in build_epa_games(
            season, games, ep, metric=EFFICIENCY_METRIC)}
        prior = build_roster_prior_or_empty(season) or None
        hfa = home_field_prior(season)
        polls = ap_polls(season)
        for week in sorted({g.week for g in games}):
            if week < START_WEEK:
                continue
            upcoming = [g for g in games if g.week == week and g.is_fbs_only]
            model = build_model(season, [g for g in games if g.week < week],
                                epa_by_id, prior, hfa)
            if model is None or not upcoming:
                continue
            r = model.fbs_ratings()
            ap = polls.get(week, {})
            for g in upcoming:
                h, a = g.home_team, g.away_team
                if h not in r or a not in r or g.margin == 0:
                    continue
                home_won = g.margin > 0
                ours = (r[h] > r[a]) == home_won
                line = lines.get(str(g.game_id))
                all_games.append((ours, None if line is None
                                  else (line > 0) == home_won))
                gap = abs(r[h] - r[a])
                by_gap["0-3" if gap < 3 else "3-7" if gap < 7
                       else "7-14" if gap < 14 else "14+"].append(ours)
                if h in ap and a in ap:
                    ap_both.append((ours, (ap[h] < ap[a]) == home_won))
                elif (h in ap) != (a in ap):
                    ap_one.append((ours, (h in ap) == home_won))
        finals[season] = model.fbs_ratings()

    priced = [(o, l) for o, l in all_games if l is not None]
    print(f"\nAll FBS-vs-FBS games, weeks {START_WEEK}+  (higher-ranked team won)")
    print(f"  our ranking            {pct([o for o, _ in all_games])}")
    print(f"  our ranking (priced)   {pct([o for o, _ in priced])}")
    print(f"  line favourite (same)  {pct([l for _, l in priced])}"
          "   <- knows home field too")
    print("\nBoth teams in that week's AP Top 25")
    print(f"  our ranking            {pct([o for o, _ in ap_both])}")
    print(f"  AP poll                {pct([p for _, p in ap_both])}")
    print("\nOne team AP-ranked, the other not")
    print(f"  our ranking            {pct([o for o, _ in ap_one])}")
    print(f"  AP poll (ranked wins)  {pct([p for _, p in ap_one])}")
    print("\nBy our rating gap (should climb steadily)")
    for band, hits in by_gap.items():
        print(f"  {band:>5} pts             {pct(hits)}")
    return finals


def hindsight_check(finals: dict[int, dict[str, float]]) -> None:
    print("\n" + "=" * 78)
    print("2. HINDSIGHT: final order vs SP+ and SRS (all three see every game)")
    print("=" * 78)
    print(f"\n{'season':>6} {'vs SP+ rho':>11} {'top25 shared':>13} "
          f"{'vs SRS rho':>11}   games explained: ours / SP+ / SRS")
    for season in SEASONS:
        games = [g for g in load_season(season) if g.is_fbs_only and g.margin]
        ours = finals[season]
        sp = final_external(season, "ratings_sp")
        srs = final_external(season, "ratings_srs")

        def explained(r):
            ok = [(r[g.home_team] > r[g.away_team]) == (g.margin > 0)
                  for g in games if g.home_team in r and g.away_team in r]
            return f"{np.mean(ok):.1%}"

        rho_sp, _ = spearman(ours, sp)
        rho_srs, _ = spearman(ours, srs)
        shared = len(set(top(ours)) & set(top(sp)))
        print(f"{season:>6} {rho_sp:>11.3f} {shared:>10}/25 {rho_srs:>11.3f}"
              f"   {explained(ours)} / {explained(sp)} / {explained(srs)}")


def current_check() -> None:
    print("\n" + "=" * 78)
    print(f"3. RIGHT NOW: {CURRENT}")
    print("=" * 78)
    from app.main import get_season   # heavy import, only needed here
    data = get_season(CURRENT)
    ours = data.model.fbs_ratings()
    sp = final_external(CURRENT, "ratings_sp")
    polls = ap_polls(CURRENT)
    latest = max(polls) if polls else None
    ap = polls.get(latest, {})
    rank_ours = {t: i for i, t in enumerate(top(ours, 999), 1)}
    rank_sp = {t: i for i, t in enumerate(top(sp, 999), 1)}
    rho, n = spearman(ours, sp)
    print(f"\nThrough week {data.max_week}. vs SP+: rho {rho:.3f} over {n} teams, "
          f"top 25 shared {len(set(top(ours)) & set(top(sp)))}/25. "
          f"vs AP (week {latest}): top 25 shared "
          f"{len(set(top(ours)) & set(ap))}/25")
    print(f"\n{'#':>3} {'team':<20} {'SP+':>4} {'AP':>4}   note")
    for team in top(ours):
        s, a = rank_sp.get(team), ap.get(team)
        note = ""
        if s and abs(s - rank_ours[team]) >= 12:
            note = f"SP+ {'much lower' if s > rank_ours[team] else 'much higher'}"
        print(f"{rank_ours[team]:>3} {team:<20} {s or '-':>4} {a or '-':>4}   {note}")
    missing_sp = [t for t in top(sp) if rank_ours.get(t, 999) > 25]
    missing_ap = [t for t in ap if rank_ours.get(t, 999) > 25]
    print("\nSP+ top 25 we leave out: " + ", ".join(
        f"{t} (ours #{rank_ours.get(t, '-')})" for t in missing_sp))
    print("AP top 25 we leave out:  " + ", ".join(
        f"{t} (ours #{rank_ours.get(t, '-')})" for t in missing_ap))


def main() -> None:
    finals = forward_check()
    hindsight_check(finals)
    current_check()


if __name__ == "__main__":
    main()
