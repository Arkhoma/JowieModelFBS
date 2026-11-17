"""Presentation for the team page's Form section.

Turns cfbrank.form's numbers into chart geometry and plain-English
lines, so the Jinja template only loops over ready-made shapes. Shared by
the live app and the static builder (both go through app.pages).

Honesty rules baked in here, not left to the template:
 * the grey band is one typical game-to-game miss (+/- 1 sigma); a bar
   inside it is ordinary variation and is never described as news;
 * a headline of "moving" appears only when cfbrank.form flagged a shift;
 * each side is charted in the statistic that TRIGGERED its flag, with a
   dashed line at the break, so the picture shows the evidence the
   sentence cites (points otherwise: every game has them);
 * the sentence under the charts always says how big a normal swing is.
"""

from __future__ import annotations

from cfbrank.form import FLAG_Z, FormGame, SeasonForm, Shift

MIN_GAMES = 4                 # fewer scored games and a chart says nothing
WIDTH, HEIGHT = 560, 150
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 44, 8, 8, 24
BAR_FILL = 0.62               # bar width as a share of its slot
MIN_BAR_PX = 1.0              # so a near-zero miss is still visible
Y_STEP = 5                    # axis limit rounds up to a multiple of this
BAND_HEADROOM = 1.6           # axis always shows at least this many sigmas
LABEL_FLIP_PX = 90            # break label flips left this close to the edge

# kind -> (what a reader calls it, unit, factor from stored to displayed)
_STAT = {"points": ("points", "pts", 1.0),
         "success": ("success rate", "success-rate pts", 100.0)}
_SIDES = (("offense", "Offense"), ("defense", "Defense"))
_ATTR = {("offense", "points"): "off_miss", ("defense", "points"): "def_miss",
         ("offense", "success"): "off_success", ("defense", "success"): "def_success"}
_GOOD = {("offense", "points"): "scored more than expected",
         ("defense", "points"): "allowed less than expected",
         ("offense", "success"): "the offense was more efficient than expected",
         ("defense", "success"): "the defense was more efficient than expected"}


def _signed(value: float, kind: str) -> str:
    """A stored miss as the reader sees it, e.g. '+14.2 success-rate pts'."""
    _, unit, factor = _STAT[kind]
    return f"{value * factor:+.1f} {unit}"


def describe_shift(shift: Shift) -> str:
    """One plain sentence for a flagged break."""
    return (f"{shift.side.capitalize()} {_STAT[shift.kind][0]}: "
            f"{_signed(shift.before, shift.kind)} vs. expectation before week "
            f"{shift.from_week}, {_signed(shift.after, shift.kind)} since "
            f"(z = {shift.z:+.1f}).")


def _miss(game: FormGame, side: str, kind: str) -> float | None:
    """One game's miss in DISPLAY units, or None when it has no value."""
    value = getattr(game, _ATTR[(side, kind)])
    return None if value is None else value * _STAT[kind][2]


def _lead_shift(shifts: tuple[Shift, ...], side: str) -> Shift | None:
    """The strongest flagged break on one side, if any."""
    mine = [s for s in shifts if s.side == side]
    return max(mine, key=lambda s: abs(s.z), default=None)


def _axis_limit(values: list[float], band: float) -> float:
    """Symmetric y-range, rounded up to a multiple of Y_STEP."""
    biggest = max([abs(v) for v in values] + [BAND_HEADROOM * band])
    return float(-(-biggest // Y_STEP) * Y_STEP)


def _break_marker(games: tuple[FormGame, ...], shift: Shift, slot: float) -> dict | None:
    """Dashed line between the last 'before' game and the first 'after' one."""
    first_after = next((i for i, g in enumerate(games) if g.week >= shift.from_week), None)
    if not first_after:                      # None, or 0: nothing before it
        return None
    x = PAD_LEFT + first_after * slot
    flip = x > WIDTH - LABEL_FLIP_PX
    return {"x": round(x, 1), "y1": PAD_TOP, "y2": HEIGHT - PAD_BOTTOM,
            "label": f"from wk {shift.from_week}",
            "label_x": round(x + (-4 if flip else 4), 1),
            "label_y": PAD_TOP + 9, "anchor": "end" if flip else "start"}


def _chart(side: str, title: str, kind: str, games: tuple[FormGame, ...],
           sigma: float, limit: float, shift: Shift | None) -> dict:
    stat, unit, factor = _STAT[kind]
    band = sigma * factor
    plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM

    def y_of(value: float) -> float:
        return PAD_TOP + (limit - value) / (2 * limit) * plot_h

    slot = plot_w / len(games)
    bar_w = slot * BAR_FILL
    zero = y_of(0)
    bars, played = [], []
    for i, game in enumerate(games):
        miss = _miss(game, side, kind)
        if miss is None:                 # no fitted row: leave a gap
            continue
        played.append(miss)
        height = max(abs(y_of(miss) - zero), MIN_BAR_PX)
        bars.append({
            "x": round(PAD_LEFT + i * slot + (slot - bar_w) / 2, 1),
            # Grow away from the zero line, so even a hairline bar sits on
            # the correct side of it.
            "y": round(zero - height if miss >= 0 else zero, 1),
            "w": round(bar_w, 1),
            "h": round(height, 1),
            "up": miss >= 0,
            "label_x": round(PAD_LEFT + i * slot + slot / 2, 1),
            "week": game.week,
            "tip": (f"Week {game.week} vs {game.opponent}: "
                    f"{miss:+.1f} {unit} vs. expectation"),
        })
    total = sum(played)
    return {
        "title": title, "stat": stat,
        "width": WIDTH, "height": HEIGHT,
        "zero_y": round(zero, 1),
        "band_y": round(y_of(band), 1),
        "band_h": round(y_of(-band) - y_of(band), 1),
        "plot_left": PAD_LEFT, "plot_right": WIDTH - PAD_RIGHT,
        "ticks": [{"y": round(y_of(v), 1), "label": f"{v:+g}" if v else "0"}
                  for v in (limit, 0.0, -limit)],
        "label_y": HEIGHT - 7,
        "bars": bars,
        "marker": _break_marker(games, shift, slot) if shift else None,
        "summary": (f"{total:+.1f} {unit} vs. expectation over {len(played)} games "
                    f"({total / len(played):+.1f} per game); positive means "
                    f"{_GOOD[(side, kind)]}."),
    }


def _rows(games: tuple[FormGame, ...]) -> list[dict]:
    """The numbers behind the charts, in display units, for the table."""
    def scaled(value, kind):
        return None if value is None else value * _STAT[kind][2]
    return [{"week": g.week, "opponent": g.opponent, "scored": g.scored,
             "expected": g.expected, "off_miss": g.off_miss, "def_miss": g.def_miss,
             "off_success": scaled(g.off_success, "success"),
             "def_success": scaled(g.def_success, "success")} for g in games]


def team_form_context(form: SeasonForm | None, team: str) -> dict | None:
    """Everything the Form section renders, or None to hide it."""
    team_form = form.teams.get(team) if form else None
    if team_form is None or len(team_form.games) < MIN_GAMES:
        return None
    games, shifts = team_form.games, team_form.shifts

    # (side, title, kind, lead shift): chart the stat that tripped the flag.
    specs = []
    for side, title in _SIDES:
        lead = _lead_shift(shifts, side)
        specs.append((side, title, lead.kind if lead else "points", lead))
    # Charts that show the same statistic share one y-range, so they
    # compare by eye; a different statistic gets its own.
    limits = {}
    for kind in {kind for _, _, kind, _ in specs}:
        values, bands = [], []
        for side, _, k, _ in specs:
            if k == kind:
                values += [m for g in games if (m := _miss(g, side, kind)) is not None]
                bands.append(form.sigma[(kind, side)] * _STAT[kind][2])
        limits[kind] = _axis_limit(values, max(bands))

    noise = max(form.sigma[("points", "offense")], form.sigma[("points", "defense")])
    success_noise = max((form.sigma[("success", side)] for side, _ in _SIDES
                         if ("success", side) in form.sigma), default=None)
    return {
        "moving": bool(shifts),
        "headline": ("Worth a look: this team has moved well beyond ordinary "
                     "game-to-game swings." if shifts else
                     "Nothing here we can tell apart from ordinary "
                     "game-to-game swings."),
        "shifts": [describe_shift(s) for s in shifts],
        "charts": [_chart(side, title, kind, games, form.sigma[(kind, side)],
                          limits[kind], lead)
                   for side, title, kind, lead in specs],
        "noise": f"{noise:.0f}",
        # None when the season has no play-by-play (points only).
        "noise_success": (None if success_noise is None
                          else f"{success_noise * _STAT['success'][2]:.0f}"),
        "flag_z": f"{FLAG_Z:g}",
        "rows": _rows(games),
        "has_success": any(g.off_success is not None for g in games),
    }
