"""A bench-boost week values the bench at 1.0; every other week does not
(2026-09-06).

The superseded design needed a two-aggregate decomposition here because
optimise_squad collapsed the horizon. With per-week variables it is a direct
weight swap, exact by construction — and the same swap in both ILPs, which is
the sync requirement transfers.py:417-431 documents."""

import pytest

from config.strategy import OPTIMISER
from optimiser.bench_weights import (
    bench_gk_weight_for_week,
    bench_slot_weight_for_week,
)


def test_an_ordinary_week_uses_the_configured_slot_weights():
    for k, expected in enumerate(OPTIMISER.bench_slot_weights):
        assert bench_slot_weight_for_week(k, w=0, bb_target_week=None, cfg=OPTIMISER) == (
            pytest.approx(expected * OPTIMISER.bench_value_weight)
        )


def test_the_target_week_values_every_slot_at_one():
    for k in range(len(OPTIMISER.bench_slot_weights)):
        assert bench_slot_weight_for_week(k, w=2, bb_target_week=2, cfg=OPTIMISER) == 1.0


def test_a_non_target_week_is_unaffected_by_a_target_existing():
    assert bench_slot_weight_for_week(0, w=1, bb_target_week=2, cfg=OPTIMISER) == (
        pytest.approx(OPTIMISER.bench_slot_weights[0] * OPTIMISER.bench_value_weight)
    )


def test_the_reserve_keeper_is_full_weight_in_the_target_week_too():
    assert bench_gk_weight_for_week(w=2, bb_target_week=2, cfg=OPTIMISER) == 1.0


def test_both_ilps_price_the_bench_through_one_definition():
    """The sync requirement, enforced structurally rather than by comparison:
    squad.py buys the substitute and transfers.py must not sell him back the
    following week, so both must reach the same function object."""
    import optimiser.squad as squad_mod
    import optimiser.transfers as transfers_mod

    assert (
        squad_mod.bench_slot_weight_for_week
        is transfers_mod.bench_slot_weight_for_week
        is bench_slot_weight_for_week
    )
