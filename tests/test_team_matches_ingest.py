"""Exploding an FBref schedule into per-team calendar rows (2026-09-06)."""

from datetime import datetime

import pandas as pd
import pytest

from data.ingestors.team_matches import UnmappedClubError, build_team_rows


def _schedule(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team"])


def test_a_domestic_match_produces_one_row_per_club():
    out = build_team_rows(
        _schedule([(datetime(2025, 10, 4, 15, 0), "Arsenal", "Liverpool")]),
        season="2025-26", competition="PL",
    )
    assert len(out) == 2
    assert {r["team_code"] for r in out} == {3, 14}


def test_home_and_away_are_recorded_from_each_perspective():
    out = build_team_rows(
        _schedule([(datetime(2025, 10, 4, 15, 0), "Arsenal", "Liverpool")]),
        season="2025-26", competition="PL",
    )
    by_team = {r["team_code"]: r for r in out}
    assert by_team[3]["is_home"] is True
    assert by_team[14]["is_home"] is False
    assert by_team[3]["opponent_name"] == "Liverpool"
    assert by_team[14]["opponent_name"] == "Arsenal"


def test_a_european_tie_yields_only_the_premier_league_side():
    out = build_team_rows(
        _schedule([(datetime(2025, 10, 1, 20, 0), "Manchester City", "Real Madrid")]),
        season="2025-26", competition="UCL",
    )
    assert len(out) == 1
    assert out[0]["team_code"] == 43
    assert out[0]["competition"] == "UCL"
    assert out[0]["opponent_name"] == "Real Madrid"


def test_an_all_foreign_tie_yields_nothing():
    out = build_team_rows(
        _schedule([(datetime(2025, 10, 1, 20, 0), "Real Madrid", "Bayern Munich")]),
        season="2025-26", competition="UCL",
    )
    assert out == []


def test_an_unmapped_club_in_a_pl_schedule_raises():
    """A silently dropped PL club looks permanently rested, which poisons the
    feature instead of merely thinning it."""
    with pytest.raises(UnmappedClubError, match="Real Madrid"):
        build_team_rows(
            _schedule([(datetime(2025, 10, 4, 15, 0), "Arsenal", "Real Madrid")]),
            season="2025-26", competition="PL",
        )


def test_club_name_matching_ignores_case_and_punctuation():
    out = build_team_rows(
        _schedule([(datetime(2025, 10, 4, 15, 0), "ARSENAL", "  liverpool  ")]),
        season="2025-26", competition="PL",
    )
    assert len(out) == 2


def test_rows_without_a_date_are_dropped():
    """FBref lists postponed and not-yet-scheduled matches with a null date."""
    out = build_team_rows(
        _schedule([(pd.NaT, "Arsenal", "Liverpool")]),
        season="2025-26", competition="PL",
    )
    assert out == []


def test_every_row_carries_the_season_it_was_ingested_for():
    out = build_team_rows(
        _schedule([(datetime(2025, 10, 4, 15, 0), "Arsenal", "Liverpool")]),
        season="2021-22", competition="PL",
    )
    assert all(r["season"] == "2021-22" for r in out)
