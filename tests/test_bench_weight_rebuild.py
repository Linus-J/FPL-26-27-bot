"""Rebuild paths derive their bench weights from the squad they BUILD (B3c).

`optimise_squad` is only ever reached on a path that rebuilds — free hit,
wildcard, cold start. It is never a one- or two-transfer nudge; the ordinary
weekly path calls `evaluate_transfers`. So on the live GW4 frame the solved
squad kept 3 of 15 incumbent players in the wildcard shape and 1 of 15 in the
free-hit shape, and an incumbent-derived weight there describes a squad that
is about to be sold.

Hence one derive-and-re-solve pass, not the fixed-point LOOP B3 shipped. The
loop retained only the immediately preceding XI, so an A->B->A->B cycle never
tripped its equality check, burned the iteration cap, and returned a solution
whose XI was B priced with A's weights. A single pass has no cycle to fall
into and costs exactly two solves.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest
from helpers_squad import flat_projections, sixteen_players

from config.strategy import OPTIMISER
from optimiser.squad import optimise_squad

DERIVED = "derived from the solved XI"


def test_the_rebuild_derives_from_its_own_solved_xi_exactly_once(caplog):
    """Exactly once is the point: the loop is gone, so there is one derivation
    and one re-solve, never a third."""
    players = sixteen_players()
    players["start_probability"] = 0.70
    proj = flat_projections(players, gws=(1,))
    on = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)

    with caplog.at_level(logging.INFO, logger="optimiser.squad"):
        optimise_squad(proj, players, budget=100.0, horizon=1, config=on)

    assert sum(DERIVED in r.message for r in caplog.records) == 1


def test_the_flag_off_derives_nothing(caplog):
    players = sixteen_players()
    players["start_probability"] = 0.70
    proj = flat_projections(players, gws=(1,))
    off = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=False)

    with caplog.at_level(logging.INFO, logger="optimiser.squad"):
        optimise_squad(proj, players, budget=100.0, horizon=1, config=off)

    assert not any(DERIVED in r.message for r in caplog.records)


def test_the_fixed_point_iteration_cap_is_gone():
    """A knob nothing reads is a knob that lies about how the code behaves."""
    assert not hasattr(OPTIMISER, "bench_weight_fixed_point_iterations")


def test_the_cold_start_derives_bench_weights_from_the_squad_it_builds(
    caplog, monkeypatch,
):
    """A cold start has no incumbent squad, so the hoist correctly declines it
    (`derive_bench_config` returns `cfg` unchanged on an empty squad). The
    rebuild shape is what gives it real weights, and it must run there."""
    from projection import cold_start

    players = sixteen_players()
    proj = flat_projections(players, gws=(1,))
    proj["start_probability"] = 0.70

    # `build_initial_squad` merges start_probability onto `players` itself, so
    # the frame it is handed must not already carry the column or the merge
    # suffixes both copies and the derivation silently skips.
    monkeypatch.setattr(
        cold_start, "cold_start_projections",
        lambda *a, **k: (proj, players.drop(columns=["start_probability"]), {}),
    )
    on = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)

    with caplog.at_level(logging.INFO, logger="optimiser.squad"):
        solution, _ = cold_start.build_initial_squad(
            "2026-27", budget=100.0, players=players, config=on,
        )

    assert len(solution.squad) == 15
    assert sum(DERIVED in r.message for r in caplog.records) == 1


# --- the yardstick rule ----------------------------------------------------


def test_chip_options_are_ranked_on_a_bench_weight_free_yardstick():
    """A rebuild path may use re-derived weights to CHOOSE its squad, but the
    xPts it reports for comparison must be on one shared scale, or the engine
    reads a change of yardstick as evidence about the chip. This is the trap
    already recorded for Bench Boost, where a boosted build counts its bench
    at 1.0 and 179.69 vs 178.95 says nothing about which squad is better.

    Established by reading and pinned here: `ChipOption.horizon_xpts` is built
    only from `squad_xpts` (best eleven plus a captain, straight off
    `projections["xpts"]`) and `TransferPlan.net_xpts_gain`, which is itself
    `squad_xpts(after) - squad_xpts(before)` plus the hit cost. No bench weight
    of any kind enters the reported number — they shape the search only. This
    test fails the moment someone reports a solver objective instead.

    A regression guard on an invariant that already held, not a fail-before
    test for new behaviour.
    """
    from optimiser.chip_comparison import build_no_chip_option
    from optimiser.transfers import squad_xpts

    players = sixteen_players()
    proj = flat_projections(players, gws=(1, 2))
    squad_ids = players["id"].tolist()[:15]

    option = build_no_chip_option(
        squad_ids, proj, players, free_transfers=1, horizon=2,
        available_budget=100.0, config=OPTIMISER,
    )
    assert option is not None

    after = [
        t["player_id"] for t in option.plan.transfers_in
    ] + [
        pid for pid in squad_ids
        if pid not in {t["player_id"] for t in option.plan.transfers_out}
    ]
    from config.strategy import TRANSFERS

    expected = squad_xpts(after, proj, horizon=2) + (
        option.plan.hits_taken * TRANSFERS.hit_cost_points
    )
    assert option.horizon_xpts == pytest.approx(expected)
