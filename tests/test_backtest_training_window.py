"""The backtest trained the minutes model on the wrong frame (A9, 2026-09-08).

All three harnesses in scripts/backtest.py passed ``df_override=history``, where
``history`` is the simulated season's own rows only. Two separate problems:

  * That is the arm that measurably predicts minutes WORSE -- +0.0113 log loss
    and +0.0025 Brier over 103 held-out gameweeks, worse in 79 of them -- and it
    is not the arm the live bot runs. Every strategy number on record was
    validated against a model the bot does not use.
  * Fixing it by dropping the override entirely would have been far worse than
    the bug. ``minutes_model._load_training_data`` has NO season filter -- its
    only WHERE clause is ``s.minutes IS NOT NULL`` -- so a bare
    ``train_minutes()`` inside a walk over 2024-25 trains on 2025-26 and
    2026-27. That is leakage, and leakage that inflates a backtest is the kind
    nobody goes looking for.

Hence ``before=(season, gameweek)``: all history strictly before this point, and
nothing after it. These tests pin both halves -- the cut itself, and the fact
that each of the three call sites applies it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import projection.minutes_model as mm
import scripts.backtest as bt
from projection.minutes_model import rows_strictly_before


def _multi_season_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "season": (
            ["2023-24"] * 3 + ["2024-25"] * 3 + ["2025-26"] * 3 + ["2026-27"] * 3
        ),
        "gameweek": [1, 5, 38] * 4,
        "player_id": list(range(12)),
    })


# --------------------------------------------------------------------------
# the cut itself
# --------------------------------------------------------------------------

def test_earlier_seasons_are_kept_in_full():
    """The whole point of the fix: a walk over 2025-26 gets 2023-24 and
    2024-25 entire, including their GW38s, which happened before it started."""
    kept = rows_strictly_before(_multi_season_frame(), "2025-26", 5)
    assert sorted(kept["season"].unique()) == ["2023-24", "2024-25", "2025-26"]
    assert sorted(kept.loc[kept["season"] == "2024-25", "gameweek"]) == [1, 5, 38]


def test_the_simulated_gameweek_is_excluded_from_its_own_training_set():
    kept = rows_strictly_before(_multi_season_frame(), "2025-26", 5)
    assert sorted(kept.loc[kept["season"] == "2025-26", "gameweek"]) == [1]


def test_later_seasons_are_dropped_entirely():
    """This is the leak. A frame loaded with no season filter carries seasons
    that had not happened when the simulated gameweek was played."""
    kept = rows_strictly_before(_multi_season_frame(), "2025-26", 5)
    assert "2026-27" not in set(kept["season"])


def test_gameweek_1_of_the_earliest_season_leaves_nothing_to_train_on():
    """The boundary case has to be empty, not everything."""
    assert rows_strictly_before(_multi_season_frame(), "2023-24", 1).empty


def test_a_season_label_that_cannot_be_ordered_is_refused():
    """Season strings compare chronologically only because they are YYYY-YY.
    A label that breaks that would compare wrongly and silently, so it raises."""
    with pytest.raises(ValueError, match="YYYY-YY"):
        rows_strictly_before(_multi_season_frame(), "2025/26", 5)


def test_an_unorderable_season_in_the_data_is_refused_too():
    df = _multi_season_frame()
    df.loc[0, "season"] = "2023"
    with pytest.raises(ValueError, match="chronologically"):
        rows_strictly_before(df, "2025-26", 5)


# --------------------------------------------------------------------------
# train() applies it
# --------------------------------------------------------------------------

class _Captured(Exception):
    """Stops the run once we have what we came for, so no model is fitted."""

    def __init__(self, payload):
        self.payload = payload


def test_train_before_cuts_the_loaded_frame_and_not_only_an_override(monkeypatch):
    """Through the real train(): the frame that reaches feature building must
    already be cut, because that frame is what the model sees."""
    monkeypatch.setattr(mm, "_load_training_data", _multi_season_frame)
    monkeypatch.setattr(mm, "_build_features", lambda df: (_ for _ in ()).throw(
        _Captured(df)
    ))

    with pytest.raises(_Captured) as exc:
        mm.train(save=False, fast=True, before=("2025-26", 5))

    got = exc.value.payload
    assert "2026-27" not in set(got["season"]), "trained on a season that had not happened"
    assert sorted(got.loc[got["season"] == "2025-26", "gameweek"]) == [1]
    assert len(got) == 7


def test_train_without_before_is_unchanged(monkeypatch):
    """Live serving passes no cut and must keep getting the whole frame."""
    monkeypatch.setattr(mm, "_load_training_data", _multi_season_frame)
    monkeypatch.setattr(mm, "_build_features", lambda df: (_ for _ in ()).throw(
        _Captured(df)
    ))

    with pytest.raises(_Captured) as exc:
        mm.train(save=False, fast=True)

    assert len(exc.value.payload) == 12


# --------------------------------------------------------------------------
# every backtest harness applies it, through the real code path
# --------------------------------------------------------------------------

def _stub_up_to_the_training_call(monkeypatch, seen: list, gws=(6, 7, 8)):
    """Stub only the I/O the loop touches BEFORE it trains, and make
    train_minutes abort the run. Everything downstream of the model is
    irrelevant to what this file is about and stays unstubbed."""
    all_gws = list(range(1, max(gws) + 1))
    n = 60 * len(all_gws)
    stats = pd.DataFrame({
        "gameweek": [g for g in all_gws for _ in range(60)],
        "player_id": [1] * n,
        "minutes": [90, 0] * (n // 2),
        # run_backtest derives its DGW/BGW counts from the fixture columns
        # before it reaches the model.
        "was_home": [True] * n,
        "team_id_season": [10] * n,
        "opponent_team_id": [11] * n,
    })
    monkeypatch.setattr(bt, "_load_all_stats", lambda season: stats)
    monkeypatch.setattr(bt.assemble, "load_match_odds", lambda season: pd.DataFrame())
    monkeypatch.setattr(bt.assemble, "load_defcon_events", lambda season: pd.DataFrame())
    monkeypatch.setattr(
        bt.assemble, "compute_defcon_field_shares",
        lambda season: {"DEF": {}, "MID_FWD": {}},
    )
    monkeypatch.setattr(
        bt, "_load_players_snapshot",
        lambda season, gw: pd.DataFrame({
            "id": [1, 2, 3], "position": ["MID"] * 3, "team_id": [10] * 3,
            "web_name": ["p1", "p2", "p3"], "now_cost": [5.0] * 3,
        }),
    )

    def fake_train(**kwargs):
        seen.append(kwargs)
        raise _Captured(kwargs)

    monkeypatch.setattr(bt, "train_minutes", fake_train)


@pytest.mark.parametrize("harness", ["run_backtest", "run_naive_xi_backtest",
                                     "run_rebuild_backtest"])
def test_every_harness_trains_on_all_history_before_the_simulated_gameweek(
    monkeypatch, harness
):
    seen: list = []
    _stub_up_to_the_training_call(monkeypatch, seen)

    with pytest.raises(_Captured):
        getattr(bt, harness)(season="2025-26", start_gw=6, end_gw=8, score_2627=False)

    assert len(seen) == 1
    kwargs = seen[0]
    assert kwargs.get("before") == ("2025-26", 6), (
        f"{harness} must train on all history strictly before the simulated "
        f"gameweek; got {kwargs}"
    )
    assert "df_override" not in kwargs, (
        f"{harness} passed df_override, which restricts training to the "
        f"simulated season alone -- the arm the live bot does not use"
    )
