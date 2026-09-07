"""Bench slot weights derived from the XI actually chosen (2026-09-06).

The shipped constants (0.53, 0.15, 0.03) were computed once from a GW1 XI
whose mean start probability was 0.93 and never recomputed. An XI carrying
several 0.78s needs a materially fatter slot 1; the live squad bought three
bench players at P(start) <= 0.03 partly because slot 3 is worth 0.03."""

import pytest

from optimiser.bench_weights import (
    at_least_k_probabilities,
    derive_gk_weight,
    derive_slot_weights,
)


def test_a_certain_blank_gives_probability_one():
    assert at_least_k_probabilities([1.0], 1) == pytest.approx([1.0])


def test_nobody_can_blank_gives_probability_zero():
    assert at_least_k_probabilities([0.0, 0.0], 2) == pytest.approx([0.0, 0.0])


def test_two_independent_coin_flips():
    """P(>=1) = 1 - 0.25 = 0.75; P(>=2) = 0.25."""
    assert at_least_k_probabilities([0.5, 0.5], 2) == pytest.approx([0.75, 0.25])


def test_probabilities_are_non_increasing_in_k():
    out = at_least_k_probabilities([0.2, 0.3, 0.4, 0.1, 0.25], 3)
    assert out[0] >= out[1] >= out[2]


def test_asking_for_more_slots_than_players_is_not_an_error():
    out = at_least_k_probabilities([0.5], 3)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)


def test_reproduces_the_shipped_constants_on_the_xi_they_came_from():
    """Ten outfield starters at the documented mean P(start) of 0.93 should
    land near (0.53, 0.15, 0.03) — the numbers this replaces."""
    weights = derive_slot_weights([0.93] * 10)
    assert weights[0] == pytest.approx(0.53, abs=0.03)
    assert weights[1] == pytest.approx(0.15, abs=0.03)
    assert weights[2] == pytest.approx(0.03, abs=0.02)


def test_a_fragile_xi_buys_more_insurance_than_a_nailed_on_one():
    fragile = derive_slot_weights([0.78] * 10)
    nailed = derive_slot_weights([0.97] * 10)
    assert fragile[0] > nailed[0]
    assert fragile[1] > nailed[1]


def test_weights_are_strictly_decreasing_so_the_solver_orders_the_bench():
    """_bench_objective relies on this: any other slot assignment is dominated,
    so no explicit ordering constraint is needed."""
    weights = derive_slot_weights([0.85] * 10)
    assert weights[0] > weights[1] > weights[2]


def test_the_reserve_keeper_is_worth_the_first_choice_keepers_blank_chance():
    assert derive_gk_weight(0.97) == pytest.approx(0.03)
    assert derive_gk_weight(0.80) == pytest.approx(0.20)


def test_an_empty_xi_falls_back_to_zero_rather_than_raising():
    assert derive_slot_weights([]) == pytest.approx((0.0, 0.0, 0.0))


def test_flag_off_reproduces_the_static_weights_exactly(monkeypatch):
    """The whole point of the flag: the change must be measurable, not assumed."""
    import dataclasses

    from helpers_squad import flat_projections, sixteen_players

    from config.strategy import OPTIMISER
    from optimiser.squad import optimise_squad

    players = sixteen_players()
    proj = flat_projections(players, gws=(1,))
    off = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=False)
    a = optimise_squad(proj, players, budget=100.0, horizon=1, config=off)
    b = optimise_squad(proj, players, budget=100.0, horizon=1, config=off)
    assert set(a.squad["id"]) == set(b.squad["id"])


def test_flag_on_lifts_the_bench_of_a_fragile_squad():
    """A squad of rotation risks should buy a better first substitute than the
    static 0.53 would justify."""
    import dataclasses

    from helpers_squad import flat_projections, sixteen_players

    from config.strategy import OPTIMISER
    from optimiser.squad import optimise_squad

    players = sixteen_players()
    players["start_probability"] = 0.70
    proj = flat_projections(players, gws=(1,))

    on = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    out = optimise_squad(proj, players, budget=100.0, horizon=1, config=on)
    assert len(out.squad) == 15
