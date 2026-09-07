"""Fixture-congestion features (2026-09-06).

Pure computation over a match calendar, tested without a database so the
semantics are pinned independently of whether the FBref backfill has run."""

import logging
from datetime import datetime

import pandas as pd
import pytest

from projection.congestion import (
    CONGESTION_FEATURE_COLS,
    compute_congestion,
    dateless_kickoff_share,
    warn_on_dateless_kickoffs,
)


def _matches(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["season", "team_id", "kickoff_time", "competition"]
    )


def _anchor(kickoff: datetime, team_id: int = 1, gameweek: int = 5) -> pd.DataFrame:
    return pd.DataFrame([{
        "season": "2025-26", "team_id": team_id,
        "gameweek": gameweek, "anchor_time": kickoff,
    }])


SAT = datetime(2025, 10, 4, 15, 0)


def test_returns_one_row_per_anchor_with_every_feature_column():
    out = compute_congestion(
        _matches([("2025-26", 1, SAT, "PL")]), _anchor(SAT)
    )
    assert len(out) == 1
    for col in CONGESTION_FEATURE_COLS:
        assert col in out.columns


def test_midweek_ucl_before_a_saturday_is_under_three_days_rest():
    """Wed 20:00 -> Sat 15:00 is 2.79 days, not 3.0. Exact timestamps, not
    calendar-date subtraction — that precision is the entire reason this
    design chose true rest-days over a gameweek-window boolean."""
    wed = datetime(2025, 10, 1, 20, 0)
    out = compute_congestion(
        _matches([("2025-26", 1, wed, "UCL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["days_since_last_match"].iloc[0] == pytest.approx(2.7917, abs=0.01)
    assert out["euro_match_in_prev_7d"].iloc[0] == 1.0
    assert out["euro_competition_tier"].iloc[0] == 3.0


def test_a_tuesday_ucl_gives_more_rest_than_a_wednesday_one():
    """The whole reason for true rest-days over a GW-window boolean."""
    tue = datetime(2025, 9, 30, 20, 0)
    wed = datetime(2025, 10, 1, 20, 0)
    tue_rest = compute_congestion(
        _matches([("2025-26", 1, tue, "UCL"), ("2025-26", 1, SAT, "PL")]), _anchor(SAT)
    )["days_since_last_match"].iloc[0]
    wed_rest = compute_congestion(
        _matches([("2025-26", 1, wed, "UCL"), ("2025-26", 1, SAT, "PL")]), _anchor(SAT)
    )["days_since_last_match"].iloc[0]
    assert tue_rest > wed_rest


def test_no_european_football_leaves_the_euro_features_at_zero():
    prev_sat = datetime(2025, 9, 27, 15, 0)
    out = compute_congestion(
        _matches([("2025-26", 1, prev_sat, "PL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["euro_match_in_prev_7d"].iloc[0] == 0.0
    assert out["euro_competition_tier"].iloc[0] == 0.0


def test_tier_ranks_ucl_above_uel_above_uecl():
    wed = datetime(2025, 10, 1, 20, 0)
    tiers = {}
    for comp in ("UCL", "UEL", "UECL"):
        out = compute_congestion(
            _matches([("2025-26", 1, wed, comp), ("2025-26", 1, SAT, "PL")]),
            _anchor(SAT),
        )
        tiers[comp] = out["euro_competition_tier"].iloc[0]
    assert tiers["UCL"] > tiers["UEL"] > tiers["UECL"]


def test_tier_takes_the_highest_when_two_euro_matches_fall_in_the_window():
    out = compute_congestion(
        _matches([
            ("2025-26", 1, datetime(2025, 9, 30, 20, 0), "UEL"),
            ("2025-26", 1, datetime(2025, 10, 2, 20, 0), "UCL"),
            ("2025-26", 1, SAT, "PL"),
        ]),
        _anchor(SAT),
    )
    assert out["euro_competition_tier"].iloc[0] == 3.0


def test_matches_in_prev_14d_counts_every_competition_and_excludes_the_anchor():
    rows = [
        ("2025-26", 1, datetime(2025, 9, 24, 20, 0), "UCL"),
        ("2025-26", 1, datetime(2025, 9, 27, 15, 0), "PL"),
        ("2025-26", 1, datetime(2025, 10, 1, 20, 0), "UCL"),
        ("2025-26", 1, SAT, "PL"),
    ]
    out = compute_congestion(_matches(rows), _anchor(SAT))
    assert out["matches_in_prev_14d"].iloc[0] == 3.0


def test_a_long_gap_is_flagged_as_a_post_international_break():
    long_ago = datetime(2025, 9, 13, 15, 0)
    out = compute_congestion(
        _matches([("2025-26", 1, long_ago, "PL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["is_post_international_break"].iloc[0] == 1.0


def test_an_ordinary_week_is_not_a_break():
    prev_sat = datetime(2025, 9, 27, 15, 0)
    out = compute_congestion(
        _matches([("2025-26", 1, prev_sat, "PL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["is_post_international_break"].iloc[0] == 0.0


def test_days_to_next_match_sees_the_upcoming_midweek_tie():
    tue = datetime(2025, 10, 7, 20, 0)
    out = compute_congestion(
        _matches([("2025-26", 1, SAT, "PL"), ("2025-26", 1, tue, "UCL")]),
        _anchor(SAT),
    )
    assert out["days_to_next_match"].iloc[0] == pytest.approx(3.2, abs=0.1)


def test_a_teams_first_match_of_a_season_gets_the_cap_not_a_nan():
    out = compute_congestion(_matches([("2025-26", 1, SAT, "PL")]), _anchor(SAT))
    assert out["days_since_last_match"].iloc[0] == 14.0
    assert out["days_to_next_match"].iloc[0] == 14.0
    assert not out[CONGESTION_FEATURE_COLS].isna().any().any()


def test_one_teams_calendar_never_leaks_into_another():
    wed = datetime(2025, 10, 1, 20, 0)
    rows = [
        ("2025-26", 1, wed, "UCL"),
        ("2025-26", 1, SAT, "PL"),
        ("2025-26", 2, SAT, "PL"),
    ]
    anchors = pd.DataFrame([
        {"season": "2025-26", "team_id": 1, "gameweek": 5, "anchor_time": SAT},
        {"season": "2025-26", "team_id": 2, "gameweek": 5, "anchor_time": SAT},
    ])
    out = compute_congestion(_matches(rows), anchors).set_index("team_id")
    assert out.loc[1, "euro_match_in_prev_7d"] == 1.0
    assert out.loc[2, "euro_match_in_prev_7d"] == 0.0


def test_seasons_do_not_bleed_across_the_summer():
    may = datetime(2025, 5, 25, 15, 0)
    aug = datetime(2025, 8, 16, 15, 0)
    rows = [("2024-25", 1, may, "PL"), ("2025-26", 1, aug, "PL")]
    anchors = pd.DataFrame([
        {"season": "2025-26", "team_id": 1, "gameweek": 1, "anchor_time": aug}
    ])
    out = compute_congestion(_matches(rows), anchors)
    assert out["days_since_last_match"].iloc[0] == 14.0


def test_empty_matches_yields_defaults_for_every_anchor():
    out = compute_congestion(_matches([]), _anchor(SAT))
    assert len(out) == 1
    assert out["matches_in_prev_14d"].iloc[0] == 0.0
    assert not out[CONGESTION_FEATURE_COLS].isna().any().any()


# --- constant-pinning tests (2026-09-06) -----------------------------------
# Task 3's original suite pinned the shape of this computation but not its
# numbers: mutation testing found the 14-day cap could be deleted, the
# next-match lookup reversed, and every window boundary moved, with all 14
# tests still green. Each test below kills one specific mutant.


def test_a_long_gap_is_capped_at_fourteen_days():
    """Kills M12: removing min(gap, REST_CAP_DAYS) entirely."""
    long_ago = datetime(2025, 8, 20, 15, 0)  # 45 days before SAT
    out = compute_congestion(
        _matches([("2025-26", 1, long_ago, "PL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["days_since_last_match"].iloc[0] == 14.0


def test_a_long_forward_gap_is_also_capped():
    """Kills M12 on the forward branch."""
    far_ahead = datetime(2025, 11, 20, 15, 0)
    out = compute_congestion(
        _matches([("2025-26", 1, SAT, "PL"), ("2025-26", 1, far_ahead, "PL")]),
        _anchor(SAT),
    )
    assert out["days_to_next_match"].iloc[0] == 14.0


def test_days_to_next_match_takes_the_nearest_of_several_future_matches():
    """Kills M10: .iloc[0] -> .iloc[-1] on the future frame."""
    tue = datetime(2025, 10, 7, 20, 0)
    sun = datetime(2025, 10, 12, 15, 0)
    out = compute_congestion(
        _matches([
            ("2025-26", 1, SAT, "PL"),
            ("2025-26", 1, tue, "UCL"),
            ("2025-26", 1, sun, "PL"),
        ]),
        _anchor(SAT),
    )
    assert out["days_to_next_match"].iloc[0] == pytest.approx(3.2083, abs=0.01)


def test_days_since_last_match_takes_the_most_recent_of_several():
    """The backward twin of the above."""
    ten_days = datetime(2025, 9, 24, 15, 0)
    wed = datetime(2025, 10, 1, 20, 0)
    out = compute_congestion(
        _matches([
            ("2025-26", 1, ten_days, "PL"),
            ("2025-26", 1, wed, "UCL"),
            ("2025-26", 1, SAT, "PL"),
        ]),
        _anchor(SAT),
    )
    assert out["days_since_last_match"].iloc[0] == pytest.approx(2.7917, abs=0.01)


def test_a_match_exactly_fourteen_days_before_is_counted():
    """Kills M4: the 14-day cutoff comparison >= becoming >."""
    exactly_14 = SAT - pd.Timedelta(days=14)
    out = compute_congestion(
        _matches([("2025-26", 1, exactly_14, "PL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["matches_in_prev_14d"].iloc[0] == 1.0


def test_a_match_just_outside_fourteen_days_is_not_counted():
    """Kills M2: widening the congestion window from 14 to 21 days."""
    just_outside = SAT - pd.Timedelta(days=14, seconds=1)
    out = compute_congestion(
        _matches([("2025-26", 1, just_outside, "PL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["matches_in_prev_14d"].iloc[0] == 0.0


def test_a_euro_match_exactly_seven_days_before_still_counts():
    """Kills M5: the 7-day European cutoff >= becoming >."""
    exactly_7 = SAT - pd.Timedelta(days=7)
    out = compute_congestion(
        _matches([("2025-26", 1, exactly_7, "UCL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["euro_match_in_prev_7d"].iloc[0] == 1.0
    assert out["euro_competition_tier"].iloc[0] == 3.0


def test_a_euro_match_ten_days_before_is_congestion_but_not_recent_europe():
    """Kills M3: widening the European window from 7 to 14 days. The match must
    still count toward the 14-day total — it happened — but must not register
    as recent European football."""
    ten_days = SAT - pd.Timedelta(days=10)
    out = compute_congestion(
        _matches([("2025-26", 1, ten_days, "UCL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["matches_in_prev_14d"].iloc[0] == 1.0
    assert out["euro_match_in_prev_7d"].iloc[0] == 0.0
    assert out["euro_competition_tier"].iloc[0] == 0.0


def test_a_twelve_day_gap_is_not_an_international_break():
    """Kills M8: lowering the break threshold from 14 to 10 days."""
    twelve_days = SAT - pd.Timedelta(days=12)
    out = compute_congestion(
        _matches([("2025-26", 1, twelve_days, "PL"), ("2025-26", 1, SAT, "PL")]),
        _anchor(SAT),
    )
    assert out["days_since_last_match"].iloc[0] == pytest.approx(12.0, abs=0.01)
    assert out["is_post_international_break"].iloc[0] == 0.0


def test_the_break_flag_agrees_on_every_no_previous_match_path():
    """Finding C1: the flag used to be assigned only inside the loop body, so a
    season opener got 1.0 while a team absent from the calendar and an empty
    calendar both got 0.0 — three situations that are all "no previous match,
    days_since = 14.0"."""
    opener = compute_congestion(
        _matches([("2025-26", 1, SAT, "PL")]), _anchor(SAT)
    )
    absent = compute_congestion(
        _matches([("2025-26", 99, SAT, "PL")]), _anchor(SAT)
    )
    empty = compute_congestion(_matches([]), _anchor(SAT))

    for frame in (opener, absent, empty):
        assert frame["days_since_last_match"].iloc[0] == 14.0
        assert frame["is_post_international_break"].iloc[0] == 1.0


# --- date-only kickoff detection -------------------------------------------
#
# 12a6387 taught the ingest to read FBref's separate `time` column, but the
# backfill is resumable and journals completed jobs, so a table written before
# that fix keeps its midnight timestamps until someone runs --reset. On
# 2026-09-07 the live table was in exactly that state: all 4560 PL rows at
# midnight, all 493 European rows with real kickoffs. Nothing noticed.


def test_a_calendar_of_real_kickoffs_is_not_flagged():
    assert dateless_kickoff_share(_matches([
        ("2025-26", 1, datetime(2025, 10, 1, 20, 0), "UCL"),
        ("2025-26", 1, datetime(2025, 10, 4, 15, 0), "PL"),
    ])) == 0.0


def test_a_wholly_midnight_calendar_is_flagged():
    assert dateless_kickoff_share(_matches([
        ("2025-26", 1, datetime(2025, 10, 1), "UCL"),
        ("2025-26", 1, datetime(2025, 10, 4), "PL"),
    ])) == 1.0


def test_the_mixed_state_that_actually_shipped_is_flagged():
    """The live failure: European rows timed, PL rows not. The bias falls on
    exactly the fixture pattern the feature exists to measure -- a 20:00
    Tuesday UCL followed by a midnight Saturday PL reads 3.17 rest-days
    instead of 3.79, and always UNDERSTATES rest. Uniform midnight was at
    least unbiased, so a half-applied fix is worse than none."""
    assert dateless_kickoff_share(_matches([
        ("2025-26", 1, datetime(2025, 10, 1, 20, 0), "UCL"),
        ("2025-26", 1, datetime(2025, 10, 4), "PL"),
    ])) == pytest.approx(0.5)


def test_an_empty_calendar_does_not_divide_by_zero():
    assert dateless_kickoff_share(_matches([])) == 0.0


def test_string_kickoffs_are_read_as_timestamps_not_compared_as_text():
    """read_sql over SQLite hands back strings, not datetimes."""
    assert dateless_kickoff_share(_matches([
        ("2025-26", 1, "2025-10-01 20:00:00.000000", "UCL"),
        ("2025-26", 1, "2025-10-04 00:00:00.000000", "PL"),
    ])) == pytest.approx(0.5)


def test_the_warning_names_the_command_that_fixes_it(caplog):
    """A warning that does not say what to run gets ignored for weeks -- which
    is how long a stale-schema database ran in production here before."""
    with caplog.at_level(logging.WARNING, logger="projection.congestion"):
        warn_on_dateless_kickoffs(_matches([
            ("2025-26", 1, datetime(2025, 10, 4), "PL"),
        ]))
    assert "backfill_team_matches.py --reset" in caplog.text


def test_no_warning_when_the_calendar_is_sound(caplog):
    with caplog.at_level(logging.WARNING, logger="projection.congestion"):
        warn_on_dateless_kickoffs(_matches([
            ("2025-26", 1, datetime(2025, 10, 4, 15, 0), "PL"),
        ]))
    assert caplog.text == ""
