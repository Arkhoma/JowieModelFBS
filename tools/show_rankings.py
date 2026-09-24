"""Print the current top-25 to the terminal -- a quick sanity check
without parsing HTML.

Reuses app.main.get_season directly. An earlier version re-implemented
the model build, which meant any change to the app (a new prior, a new
efficiency metric) silently left this tool reporting the OLD model.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import get_season  # noqa: E402
from cfbrank.games import available_seasons  # noqa: E402

TOP_N = 25


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 else max(available_seasons())
    built = get_season(season)
    model = built.model

    print(f"\n{season} -- {built.n_games} games, through week "
          f"{built.max_week}")
    print(f"Home field {model.home_field:+.1f} | "
          f"FCS offset {model.fcs_offset:+.1f} | "
          f"efficiency component {'yes' if built.uses_epa else 'no'}")
    print()
    print(f"{'#':>3}  {'Team':<26} {'Rating':>7}  {'W-L':>6}  {'Resume':>6}")
    print("-" * 56)
    for row in built.rankings[:TOP_N]:
        print(f"{row['rank']:>3}  {row['team']:<26} {row['rating']:>+6.1f}  "
              f"{row['wins']:>2}-{row['losses']:<3}  "
              f"{row['resume_rank'] or '':>6}")


if __name__ == "__main__":
    main()
