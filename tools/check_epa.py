"""Sanity-check the EPA aggregation against known 2024 outcomes.

Before wiring EPA into the ratings, confirm it recovers reality: Ohio
State won the title, Kent State went 0-12. If the leaderboard is
nonsense, the ratings built on it would be worse nonsense.

Also reports what garbage-time filtering actually removes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfbrank import expected_points as ep  # noqa: E402
from cfbrank.epa_games import is_garbage_time, season_epa, to_frame  # noqa: E402
from cfbrank.plays import load_plays  # noqa: E402

TRAIN_SEASONS = (2022, 2023)
TEST_SEASON = 2024


def main() -> None:
    print("Fitting expected-points model on "
          f"{TRAIN_SEASONS} ...")
    import pandas as pd
    training = pd.concat(
        [load_plays(s) for s in TRAIN_SEASONS], ignore_index=True)
    model = ep.fit(training)
    print(f"  {model.n_plays:,} plays, {len(model.table)} state cells")

    plays = load_plays(TEST_SEASON)
    garbage = is_garbage_time(plays)
    print(f"\nGarbage time in {TEST_SEASON}: {garbage.sum():,} of "
          f"{len(plays):,} plays ({garbage.mean():.1%}) removed")

    records = season_epa(TEST_SEASON, model)
    frame = to_frame(records)
    print(f"Aggregated {len(frame):,} team-games")

    season_level = frame.groupby("team").agg(
        off_epa=("off_epa", "mean"),
        def_epa=("def_epa", "mean"),
        net_epa=("net_epa", "mean"),
        off_sr=("off_sr", "mean"),
        games=("game_id", "count"),
    )
    season_level = season_level[season_level["games"] >= 8]

    print(f"\nTOP 15 by net EPA/play ({TEST_SEASON})")
    print(f"{'team':<24} {'net':>7} {'off':>7} {'def':>7} {'off SR':>8}")
    print("-" * 58)
    for team, row in season_level.nlargest(15, "net_epa").iterrows():
        print(f"{team:<24} {row.net_epa:>+7.3f} {row.off_epa:>+7.3f} "
              f"{row.def_epa:>+7.3f} {row.off_sr:>7.1%}")

    print(f"\nBOTTOM 5 by net EPA/play")
    print("-" * 58)
    for team, row in season_level.nsmallest(5, "net_epa").iterrows():
        print(f"{team:<24} {row.net_epa:>+7.3f} {row.off_epa:>+7.3f} "
              f"{row.def_epa:>+7.3f} {row.off_sr:>7.1%}")

    print("\nReality check:")
    for team in ("Ohio State", "Notre Dame", "Oregon", "Kent State"):
        if team in season_level.index:
            rank = int((season_level["net_epa"]
                        > season_level.loc[team, "net_epa"]).sum()) + 1
            print(f"  {team:<14} net EPA rank {rank} of "
                  f"{len(season_level)}")


if __name__ == "__main__":
    main()
