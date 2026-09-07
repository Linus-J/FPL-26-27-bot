"""The pivot price: what reaching a BB-ready squad actually costs
(2026-09-06).

The trap this reports on is arriving at a gameweek with excellent BB fixtures
holding a fodder bench, no bank and few free transfers."""

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
