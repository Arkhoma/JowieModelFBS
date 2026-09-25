"""CFBD preseason signals: parsing, coaching-change leakage, graceful gaps."""

from __future__ import annotations

import gzip
import json

import pytest

from cfbrank import cfbd_priors


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """Point cfbd_priors at a scratch data/raw and clear its caches."""
    monkeypatch.setattr(cfbd_priors, "RAW", tmp_path)
    cfbd_priors.load_records.cache_clear()
    cfbd_priors._coach_seasons.cache_clear()
    yield tmp_path
    cfbd_priors.load_records.cache_clear()
    cfbd_priors._coach_seasons.cache_clear()


def write(root, name, records, season=None):
    folder = root / str(season) if season is not None else root
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{name}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(records, f)


def coach(first, school, seasons):
    return {"firstName": first, "lastName": "X",
            "seasons": [{"school": school, "year": y, "games": g}
                        for y, g in seasons]}


def test_missing_files_mean_no_signal(raw):
    assert cfbd_priors.talent(2030) == {}
    assert cfbd_priors.new_coach(2030) == {}


def test_talent_parses_and_skips_blanks(raw):
    write(raw, "talent", [{"team": "A", "talent": 900.5},
                          {"team": "B", "talent": None}], season=2024)
    assert cfbd_priors.talent(2024) == {"A": 900.5}


def test_new_coach_flags_a_real_change(raw):
    write(raw, "coaches", [coach("Old", "A", [(2023, 12)]),
                           coach("New", "A", [(2024, 12)]),
                           coach("Stay", "B", [(2023, 12), (2024, 12)])])
    assert cfbd_priors.new_coach(2024) == {"A": 1.0, "B": 0.0}


def test_midseason_firing_is_not_a_preseason_change(raw):
    """Last year's coach started this season, then got fired in week 6.

    A preseason prior cannot know that yet, so it must not flag it --
    otherwise the prior leaks the result of the season it predicts.
    """
    write(raw, "coaches", [coach("Fired", "A", [(2023, 12), (2024, 5)]),
                           coach("Interim", "A", [(2024, 7)])])
    assert cfbd_priors.new_coach(2024) == {"A": 0.0}


def test_team_without_a_coach_record_yet_is_unknown(raw):
    write(raw, "coaches", [coach("Old", "A", [(2023, 12)])])
    assert cfbd_priors.new_coach(2024) == {}
