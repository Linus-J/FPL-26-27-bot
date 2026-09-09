"""The pivot price: what reaching a BB-ready squad actually costs
(2026-09-06).

The trap this reports on is arriving at a gameweek with excellent BB fixtures
holding a fodder bench, no bank and few free transfers."""

import types

import pandas as pd
import pytest

from config.strategy import CHIP_TIMING, OPTIMISER
from optimiser.bench_boost import pivot_price


def test_an_identical_squad_costs_nothing():
    squad = list(range(1, 16))
    assert pivot_price(squad, squad)["transfers_required"] == 0


def test_one_swap_costs_one_transfer():
    a = list(range(1, 16))
    b = [*range(1, 15), 99]
    assert pivot_price(a, b)["transfers_required"] == 1


def test_the_players_moving_each_way_are_named():
    a = list(range(1, 16))
    b = [*range(1, 15), 99]
    out = pivot_price(a, b)
    assert out["out"] == [15]
    assert out["in"] == [99]


def test_three_swaps_cost_three_transfers():
    a = list(range(1, 16))
    b = [*range(1, 13), 97, 98, 99]
    assert pivot_price(a, b)["transfers_required"] == 3


def test_order_does_not_affect_the_count():
    a = list(range(1, 16))
    b = list(reversed(range(1, 16)))
    assert pivot_price(a, b)["transfers_required"] == 0


# --- B8 (2026-09-07): the squad the pivot price is measured against ---------

_SQUAD_IDS = list(range(1, 16))


def _frame(bench_pts_by_gw: dict[int, float]) -> pd.DataFrame:
    """Fifteen players a week: eleven at 6.0 xPts and a bench sharing
    ``bench_pts`` four ways, so each week's bench total is exactly what the
    caller asked for."""
    rows = []
    for gw, bench_pts in bench_pts_by_gw.items():
        for pid in _SQUAD_IDS[:11]:
            rows.append({"player_id": pid, "gameweek": gw, "xpts": 6.0})
        for pid in _SQUAD_IDS[11:]:
            rows.append({"player_id": pid, "gameweek": gw, "xpts": bench_pts / 4.0})
    return pd.DataFrame(rows)


def test_the_bb_ready_squad_is_built_with_a_boosted_bench(monkeypatch):
    """The defect this task fixes: the "BB-ready" squad was built with ordinary
    bench weights for the WRONG gameweek, so pivot_price reported the distance
    to the plain single-week optimum. Assert the solver is actually told which
    week to boost, and over a horizon that reaches it."""
    import agent.decision_engine as de

    seen = {}

    def _spy(projections, players, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing the call")

    monkeypatch.setattr(de, "optimise_squad_joint", _spy)

    # next_gw = 4, and the only week clearing the 20 xPts threshold is GW6 --
    # two gameweeks out, so horizon 3 and target index 2.
    projections = _frame({4: 8.0, 5: 9.0, 6: 30.0})
    squad = types.SimpleNamespace(squad=pd.DataFrame({"id": _SQUAD_IDS}))

    with pytest.raises(RuntimeError, match="stop after capturing the call"):
        de._bench_boost_readiness(
            unconstrained_squad=squad,
            projections=projections,
            players=pd.DataFrame(),
            next_gw=4,
            available_budget=100.0,
            ownership=None,
            config=OPTIMISER,
            season=None,
            chip_timing=CHIP_TIMING,
            chips_used=[],
        )

    assert seen["bb_target_week"] == 2
    assert seen["horizon"] == 3
    # ``gameweek=`` feeds the joint-risk covariance matrices; it never selected
    # the horizon's weeks. Passing target_gw there while leaving horizon at 1
    # is what measured the bench at one week on a squad built for another.
    assert "gameweek" not in seen


def test_a_spent_bench_boost_is_not_planned_for(monkeypatch):
    """``optimiser/chips.py::_try_bb`` gates on uses remaining as its very
    first line; this report did not (until 2026-09-09). Once the chip was
    spent for the half it still nominated a target week, solved a second
    squad and quoted the transfer cost of moving to it -- a plan for a chip
    that cannot be played until the half turns over.

    The solver is stubbed to explode, so reaching it at all fails the test:
    the gate has to come before the expensive work, not merely change the
    text afterwards."""
    import agent.decision_engine as de
    from optimiser.chips import Chip

    def _explode(*args, **kwargs):
        raise AssertionError("solved a BB-ready squad for a chip already played")

    monkeypatch.setattr(de, "optimise_squad_joint", _explode)
    monkeypatch.setattr(de, "select_bb_target_gw", _explode)

    out = de._bench_boost_readiness(
        unconstrained_squad=types.SimpleNamespace(
            squad=pd.DataFrame({"id": _SQUAD_IDS})
        ),
        projections=_frame({4: 8.0, 5: 9.0, 6: 30.0}),
        players=pd.DataFrame(),
        next_gw=4,
        available_budget=100.0,
        ownership=None,
        config=OPTIMISER,
        season=None,
        chip_timing=CHIP_TIMING,
        chips_used=[(Chip.BENCH_BOOST, 2)],
    )

    assert out["target_gameweek"] is None
    assert "no use left in this half" in out["reason"]


def test_another_chip_being_spent_does_not_block_the_report(monkeypatch):
    """The gate is per chip. A free hit already played must not stop the bench
    boost being planned for."""
    import agent.decision_engine as de
    from optimiser.chips import Chip

    seen = {}

    def _spy(projections, players, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing the call")

    monkeypatch.setattr(de, "optimise_squad_joint", _spy)

    with pytest.raises(RuntimeError, match="stop after capturing the call"):
        de._bench_boost_readiness(
            unconstrained_squad=types.SimpleNamespace(
                squad=pd.DataFrame({"id": _SQUAD_IDS})
            ),
            projections=_frame({4: 8.0, 5: 9.0, 6: 30.0}),
            players=pd.DataFrame(),
            next_gw=4,
            available_budget=100.0,
            ownership=None,
            config=OPTIMISER,
            season=None,
            chip_timing=CHIP_TIMING,
            chips_used=[(Chip.FREE_HIT, 3)],
        )

    assert seen["bb_target_week"] == 2
