"""Exploding an FBref schedule into per-team calendar rows (2026-09-06)."""

from datetime import datetime

import pandas as pd
import pytest

from data.ingestors.team_matches import UnmappedClubError, build_team_rows


def _schedule(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team"])


def _schedule_with_time(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "time", "home_team", "away_team"])


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


def test_the_kickoff_time_comes_from_the_date_and_time_columns():
    """FBref's date column is date-only. Reading it alone put every kickoff at
    midnight, which silently turned rest-days into calendar-date subtraction --
    the exact thing projection/congestion.py's tests exist to prevent."""
    sched = _schedule_with_time([(datetime(2025, 10, 4), "20:00", "Arsenal", "Liverpool")])
    out = build_team_rows(sched, season="2025-26", competition="PL")
    assert out[0]["kickoff_time"] == datetime(2025, 10, 4, 20, 0)


def test_a_parenthesised_european_time_uses_the_leading_value():
    """European schedules carry '21:00 (20:00)'; the leading value is the
    venue-local kickoff."""
    sched = _schedule_with_time(
        [(datetime(2025, 10, 1), "21:00 (20:00)", "Manchester City", "Real Madrid")]
    )
    out = build_team_rows(sched, season="2025-26", competition="UCL")
    assert out[0]["kickoff_time"] == datetime(2025, 10, 1, 21, 0)


def test_a_missing_time_falls_back_to_the_date_alone():
    """A postponed or unscheduled fixture has no time. Keep the row -- the date
    still carries real congestion signal -- rather than dropping the match."""
    sched = _schedule_with_time([(datetime(2025, 10, 4), None, "Arsenal", "Liverpool")])
    out = build_team_rows(sched, season="2025-26", competition="PL")
    assert out[0]["kickoff_time"] == datetime(2025, 10, 4, 0, 0)


def test_a_schedule_without_a_time_column_still_works():
    """Defensive: the column's presence is FBref's choice, not ours."""
    out = build_team_rows(
        _schedule([(datetime(2025, 10, 4), "Arsenal", "Liverpool")]),
        season="2025-26", competition="PL",
    )
    assert out[0]["kickoff_time"] == datetime(2025, 10, 4, 0, 0)


def test_leagues_are_registered_before_soccerdata_is_imported():
    """soccerdata caches league_dict.json at import, so registering afterwards
    writes a file the process will never re-read -- which is why every European
    job failed with 'Invalid league' on the first live backfill."""
    import inspect

    from data.ingestors import team_matches

    src = inspect.getsource(team_matches.ingest_competition_season)
    assert src.index("register_leagues()") < src.index("import soccerdata")
