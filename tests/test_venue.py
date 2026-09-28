from cfbrank.venue import GameContext, _miles


def _ctx(**kw):
    base = dict(game_id="1", attendance=None, capacity=None, elevation_m=None,
                dome=False, travel_miles=None, home_elo=None, away_elo=None)
    return GameContext(**{**base, **kw})


def test_fill_is_capped_and_needs_both_numbers():
    assert _ctx(attendance=45_000, capacity=90_000).fill == 0.5
    assert _ctx(attendance=120_000, capacity=100_000).fill == 1.1
    assert _ctx(attendance=45_000).fill is None


def test_gainesville_to_oxford_is_about_450_miles():
    gainesville, oxford = (29.65, -82.35), (34.36, -89.53)
    assert 450 < _miles(oxford, gainesville) < 560
