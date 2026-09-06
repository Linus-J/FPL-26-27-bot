"""The XI objective is built in one place (2026-09-06).

optimiser/squad.py built its objective twice — once for `prob`, once for the
`prob2` max_transfers fallback — in near-identical hand-maintained copies.
`_bench_objective` was already factored out; this does the same for the XI half
so per-week variables cannot be added to one copy and forgotten in the other."""

import pulp

from config.strategy import OPTIMISER
from optimiser.squad import _xi_objective


def _vars(n: int):
    return (
        [pulp.LpVariable(f"sel_{i}", cat="Binary") for i in range(n)],
        [pulp.LpVariable(f"sta_{i}", cat="Binary") for i in range(n)],
        [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in range(n)],
        [pulp.LpVariable(f"vic_{i}", cat="Binary") for i in range(n)],
    )


def test_a_starter_contributes_his_score_once():
    selected, starting, captain, vice = _vars(2)
    expr = _xi_objective(selected, starting, captain, vice, [5.0, 3.0], OPTIMISER)
    assert expr[starting[0]] == 5.0


def test_the_captain_is_counted_a_second_time():
    selected, starting, captain, vice = _vars(2)
    expr = _xi_objective(selected, starting, captain, vice, [5.0, 3.0], OPTIMISER)
    assert expr[captain[0]] == 5.0


def test_the_vice_is_weighted_by_the_configured_share():
    selected, starting, captain, vice = _vars(2)
    expr = _xi_objective(selected, starting, captain, vice, [5.0, 3.0], OPTIMISER)
    assert expr[vice[0]] == 5.0 * OPTIMISER.vice_captain_weight


def test_a_benched_player_contributes_nothing_here():
    """Bench value is _bench_objective's job, not this one's."""
    selected, starting, captain, vice = _vars(2)
    expr = _xi_objective(selected, starting, captain, vice, [5.0, 3.0], OPTIMISER)
    assert selected[0] not in expr
