"""Fixture-congestion features (2026-09-06).

Pure computation over a match calendar, tested without a database so the
semantics are pinned independently of whether the FBref backfill has run."""

from datetime import datetime

import pandas as pd
import pytest

from projection.congestion import CONGESTION_FEATURE_COLS, compute_congestion


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
