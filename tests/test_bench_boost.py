"""Bench-boost target selection (2026-09-06).

Tier 1 only — the tier that feeds a decision. It searches the persisted 5-GW
projection window for the week whose bench is worth most."""

import dataclasses

import pandas as pd
import pytest

from config.strategy import OPTIMISER
from optimiser.bench_boost import (
    bb_ready_config,
    bench_xpts_by_gameweek,
    select_bb_target_gw,
)

SQUAD = list(range(1, 16))
XI = list(range(1, 12))
BENCH = [12, 13, 14, 15]


def _projections(bench_pts_by_gw: dict[int, float]) -> pd.DataFrame:
    rows = []
    for gw, bench_pts in bench_pts_by_gw.items():
        for pid in XI:
            rows.append({"player_id": pid, "gameweek": gw, "xpts": 6.0})
        for pid in BENCH:
            rows.append({"player_id": pid, "gameweek": gw, "xpts": bench_pts / 4.0})
    return pd.DataFrame(rows)


def test_bb_ready_config_forces_full_bench_weight():
    """Under a bench boost the bench genuinely plays. Full weight is a fact
    about the rules, not a risk preference."""
    base = dataclasses.replace(OPTIMISER, bench_value_weight=0.5)
    out = bb_ready_config(base, target_gw=7)
    assert out.bench_value_weight == 1.0


def test_bb_ready_config_leaves_the_base_untouched():
    base = dataclasses.replace(OPTIMISER, bench_value_weight=0.5)
    bb_ready_config(base, target_gw=7)
    assert base.bench_value_weight == 0.5


def test_bench_totals_are_computed_per_gameweek():
    proj = _projections({4: 8.0, 5: 20.0})
    out = bench_xpts_by_gameweek(SQUAD, proj, current_gw=4)
    assert out[4] == pytest.approx(8.0)
    assert out[5] == pytest.approx(20.0)


def test_the_best_week_in_the_window_is_selected():
    proj = _projections({4: 8.0, 5: 24.0, 6: 9.0})
    target = select_bb_target_gw(SQUAD, proj, current_gw=4, threshold=20.0)
    assert target is not None
    assert target[0] == 5
    assert target[1] == pytest.approx(24.0)


def test_no_target_when_nothing_clears_the_threshold():
    """The live squad's bench is worth 5.35 against a threshold of 20."""
    proj = _projections({4: 5.35, 5: 6.0})
    assert select_bb_target_gw(SQUAD, proj, current_gw=4, threshold=20.0) is None


def test_weeks_before_the_current_one_are_ignored():
    proj = _projections({3: 40.0, 4: 8.0, 5: 22.0})
    target = select_bb_target_gw(SQUAD, proj, current_gw=4, threshold=20.0)
    assert target[0] == 5


def test_the_current_week_can_itself_be_the_target():
    proj = _projections({4: 30.0, 5: 21.0})
    target = select_bb_target_gw(SQUAD, proj, current_gw=4, threshold=20.0)
    assert target[0] == 4


def test_an_empty_projection_frame_yields_no_target():
    assert select_bb_target_gw(SQUAD, pd.DataFrame(), 4, 20.0) is None
