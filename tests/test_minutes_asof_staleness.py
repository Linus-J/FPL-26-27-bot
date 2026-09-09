"""A10 — the served minutes features were a gameweek out of date (2026-09-09).

``_build_features`` shifts every rolling column by 1, correctly: a row for
gameweek g must not see g's own outcome. At serve time both public paths then
collapsed to the last row that ACTUALLY EXISTS — the last completed gameweek —
so the features they served summarised only up to the gameweek before that.

Measured on a copy of the live database at 2026-27 GW3, before the fix:

  * player 716 played 0, 0, 90 minutes and was served ``avg_minutes_3gw =
    0.000``. The true as-of-GW3 value is 30.000. A player who had just played
    a full 90 was projected as though he had never played.
  * player 31 played 90, 78, 0 and was served 84.000 against a true 56.000. A
    player who had just been dropped was projected as a nailed starter.
  * all 615 players were stale, in every gameweek of the horizon, and nothing
    logged it. On a held-out GW3 (model trained strictly before it, identical
    512-player set) the stale arm scored log loss 0.7258 / Brier 0.3692 /
    accuracy 0.7207 against 0.6293 / 0.2993 / 0.7930 for the fixed one.

There was a second, quieter half. ``_build_features`` drops rows whose shifted
features are NaN (``dropna(subset=["avg_minutes_5gw", "season_avg_minutes"])``),
which is exactly a player's FIRST row of a season. A player whose entire
history was one gameweek therefore had his only row dropped and got no minutes
prediction at all — 38 of 587 players on the live copy.

The fix appends one outcome-free row per player for the next gameweek, so the
shift's window lands on the last completed gameweek. These tests pin the
behaviour through the real paths; none of them read module source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projection import minutes_model as mm

SEASON = "2026-27"
HISTORY_GWS = [1, 2, 3]
TARGET_GW = 4

# minutes by gameweek. The first two are the live cases above, reproduced: a
# player who has just broken into the side, and one who has just dropped out.
MINUTES = {
    1: [0, 0, 90],
    2: [90, 78, 0],
    3: [90, 90, 90],
}
TEAM_OF = {1: 10, 2: 20, 3: 30}


def _stats(minutes_by_player: dict[int, list[int]] | None = None) -> pd.DataFrame:
    """A ``player_gw_stats``-shaped history frame, strictly before TARGET_GW."""
    minutes_by_player = minutes_by_player or MINUTES
    return pd.DataFrame([
        {
            "player_id": pid, "season": SEASON, "gameweek": gw,
            "minutes": mins, "total_points": 5, "goals_scored": 0, "assists": 0,
            "clean_sheets": 0, "goals_conceded": 0, "saves": 0,
            "yellow_cards": 0, "red_cards": 0, "bonus": 0, "bps": 10,
            "xg": 0.3, "xa": 0.1, "key_passes": 1, "dribbles": 0,
            "position": "MID", "was_home": True, "opponent_team_id": 99,
            "team_id_season": TEAM_OF[pid], "team_id": TEAM_OF[pid],
            "now_cost": 5.0, "selected_by_percent": 1.0,
            "status": "a", "chance_of_playing_next_round": 100,
        }
        for pid, mins in minutes_by_player.items()
        for gw, mins in zip(HISTORY_GWS[:len(mins)], mins, strict=False)
    ])


def _empty_keyed(*extra: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["player_id", "gameweek", "season", "opponent_team_id", *extra]
    )


@pytest.fixture
def hermetic(monkeypatch):
    """No database. Every loader ``_build_features`` reaches for is stubbed."""
    gws = [*HISTORY_GWS, TARGET_GW, TARGET_GW + 1]
    monkeypatch.setattr(mm, "load_congestion", lambda: pd.DataFrame([
        {
            "season": SEASON, "team_id": team, "gameweek": gw,
            "days_since_last_match": 7.0, "days_to_next_match": 7.0,
            "matches_in_prev_14d": 1.0, "euro_match_in_prev_7d": 0.0,
            "euro_competition_tier": 0.0, "is_post_international_break": 0.0,
        }
        for team in TEAM_OF.values() for gw in gws
    ]))
    monkeypatch.setattr(mm, "load_fixture_difficulty", lambda: _empty_keyed())
    monkeypatch.setattr(mm, "load_fixture_odds", lambda: _empty_keyed(
        "my_cs_prob", "opp_cs_prob", "over25_prob"))
    monkeypatch.setattr(mm, "load_player_enrichment", lambda: pd.DataFrame([
        {
            "player_id": pid, "season": SEASON, "gameweek": gw,
            "is_penalty_taker": 0.0, "penalty_xg_per_game": 0.0,
            "is_set_piece_taker": 0.0, "key_passes_per_game": 0.0,
            "injury_severity": 0.0, "price_momentum": 0.0,
            "transfer_velocity": 0.0,
        }
        for pid in TEAM_OF for gw in gws
    ]))


class _FormModel:
    """P(60+) tracks ``avg_minutes_3gw``. A probe, not a plausible minutes
    model: it exists so that a served band is a readable function of the one
    feature this defect corrupted."""

    classes_ = [0, 1, 2]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p2 = np.clip(X["avg_minutes_3gw"].to_numpy() / 90.0, 0.0, 1.0)
        return np.column_stack([1.0 - p2, np.zeros_like(p2), p2])


def _asof(stats: pd.DataFrame) -> pd.DataFrame:
    """The row each serve path collapses to, via the real feature build."""
    built = mm._build_features(mm._append_asof_rows(stats))
    return (
        built.sort_values(["player_id", "season", "gameweek"])
        .drop_duplicates(subset="player_id", keep="last")
        .set_index("player_id")
    )


# --- the defect, stated as the numbers it produced ------------------------

def test_the_asof_row_summarises_the_last_completed_gameweek(hermetic):
    """The live cases. Pre-fix these were 0.000 and 84.000 — the as-of-GW2
    values — because the served row was GW3 and GW3's own outcome is shifted
    out of its own features."""
    asof = _asof(_stats())
    assert asof.loc[1, "avg_minutes_3gw"] == pytest.approx(30.0)   # 0, 0, 90
    assert asof.loc[2, "avg_minutes_3gw"] == pytest.approx(56.0)   # 90, 78, 0
    assert asof.loc[3, "avg_minutes_3gw"] == pytest.approx(90.0)


def test_the_asof_row_is_the_gameweek_about_to_be_played(hermetic):
    """Not the one just finished. The gameweek-keyed features — the congestion
    block — resolve against this, so it has to be the real next round."""
    assert list(_asof(_stats())["gameweek"].unique()) == [TARGET_GW]


def test_the_dnp_streak_counts_the_last_completed_gameweek(hermetic):
    """Player 2 blanked GW3. Pre-fix his served streak was 0 — the as-of-GW2
    value — so the absence override could not fire on the blank that had just
    happened."""
    asof = _asof(_stats())
    assert asof.loc[2, "dnp_streak"] == 1
    assert asof.loc[1, "dnp_streak"] == 0   # played GW3, streak broken


# --- the same thing through the two public paths --------------------------

@pytest.mark.parametrize("serve", ["scalar", "by_gameweek"])
def test_the_last_completed_gameweek_reaches_the_served_bands(hermetic, serve):
    """The acceptance gate, and the one that needs no knowledge of the
    overrides: two frames differing ONLY in the last completed gameweek's
    minutes must not serve identical bands. Pre-fix they did, for every
    player, every week — that gameweek was discarded before the model saw it.
    """
    played = _stats({1: [0, 0, 90]})
    blanked = _stats({1: [0, 0, 0]})

    def bands(stats):
        if serve == "scalar":
            return mm.predict_minutes_bands(stats, _FormModel())[1]
        return mm.predict_minutes_bands_by_gameweek(
            stats, _FormModel(), [TARGET_GW], season=SEASON
        )[(1, TARGET_GW)]

    p_played, p_blanked = bands(played), bands(blanked)
    assert p_played != p_blanked, (
        "the last completed gameweek does not reach the served bands"
    )
    # and in the right direction: having just played 90 must not lower P(60+)
    assert p_played[2] > p_blanked[2]


def test_both_serve_paths_agree_on_the_asof_row(hermetic):
    """They collapse through one helper precisely so they cannot drift. If one
    were fixed and the other left, this is what would catch it."""
    stats = _stats()
    scalar = mm.predict_minutes_bands(stats, _FormModel())
    horizon = mm.predict_minutes_bands_by_gameweek(
        stats, _FormModel(), [TARGET_GW], season=SEASON
    )
    assert set(scalar) == {pid for pid, _ in horizon}
    for pid, band in scalar.items():
        assert band == pytest.approx(horizon[(pid, TARGET_GW)])


def test_a_player_with_one_gameweek_of_history_is_still_projected(hermetic):
    """``_build_features`` drops rows whose shifted features are NaN, which is
    a player's first row of a season. Pre-fix that player's ONLY row was
    dropped and he vanished from the projection entirely — 38 of 587 players
    on the live copy."""
    stats = _stats({1: [90], 2: [90, 78, 0]})
    assert 1 in mm.predict_minutes_bands(stats, _FormModel())


# --- leakage: the appended row must not reach forward ---------------------

def test_the_appended_row_carries_no_outcome(hermetic):
    appended = mm._append_asof_rows(_stats())
    added = appended[appended["gameweek"] == TARGET_GW]
    assert len(added) == len(MINUTES)
    for col in ("minutes", "total_points", "red_cards"):
        assert added[col].isna().all(), f"{col} must not be invented"


def test_features_for_a_gameweek_are_the_same_whether_or_not_it_has_happened(
    hermetic,
):
    """The property the backtest depends on. Walking a season forward, the
    frame at gameweek g holds history up to g-1 and the appended row IS g. If
    that row's features differed from the ones the real g row gets in a frame
    that contains g's outcomes, the fix would be a leak. They must be
    identical, and on the live copy they were: max |diff| 0 over 527 players
    and 11 feature columns."""
    full = _stats()                                     # gameweeks 1..3
    history = full[full["gameweek"] < 3]                # 1..2, g=3 not played

    served = _asof(history)                             # the appended g=3 row
    real = (
        mm._build_features(full)
        .query("gameweek == 3")
        .set_index("player_id")
    )

    cols = [c for c in mm.FEATURE_COLS if c in real.columns]
    assert cols, "no feature columns to compare"
    common = sorted(set(served.index) & set(real.index))
    assert common, "nothing to compare"
    pd.testing.assert_frame_equal(
        served.loc[common, cols].astype(float),
        real.loc[common, cols].astype(float),
        check_like=True,
    )


def test_a_player_whose_history_stops_in_an_earlier_season_keeps_it(hermetic):
    """The rolling features group by (player_id, season). Carrying a departed
    player forward into the current season would blank every one of them, so
    his appended row stays in his own season."""
    old = _stats({3: [90, 90, 90]}).assign(season="2025-26")
    appended = mm._append_asof_rows(pd.concat([_stats({1: [0, 0, 90]}), old]))
    added = appended[appended["minutes"].isna()]
    assert added.loc[added["player_id"] == 3, "season"].tolist() == ["2025-26"]
    assert added.loc[added["player_id"] == 1, "season"].tolist() == [SEASON]


def test_an_empty_frame_is_returned_unchanged():
    empty = pd.DataFrame(columns=["player_id", "season", "gameweek", "minutes"])
    assert mm._append_asof_rows(empty).empty
