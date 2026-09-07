"""A solver timeout must be distinguishable from infeasibility (2026-09-07).

optimise_squad raises on any non-Optimal status, generate_squad_pool catches
that and breaks, and its docstring says it breaks "once the cuts exhaust the
legal squads" — so a timeout looked exactly like normal termination and
silently produced a short pool. chip_comparison swallowed the same exception
and silently dropped the Free Hit option."""

import dataclasses

import pytest
from helpers_squad import flat_projections, sixteen_players

from config.strategy import OPTIMISER
from optimiser.squad import SolverTimeout, generate_squad_pool, optimise_squad


def test_solver_timeout_is_a_runtime_error():
    """Existing `except RuntimeError` handlers must keep working."""
    assert issubclass(SolverTimeout, RuntimeError)


def test_an_impossible_time_limit_raises_solver_timeout_not_bare_runtime_error():
    players = sixteen_players()
    proj = flat_projections(players)
    cfg = dataclasses.replace(OPTIMISER, solver_time_limit_seconds=0.001)
    with pytest.raises(SolverTimeout):
        optimise_squad(proj, players, budget=100.0, horizon=3, config=cfg)


def test_an_infeasible_problem_does_not_raise_solver_timeout():
    """A budget no legal squad can meet is infeasible, not slow."""
    players = sixteen_players()
    proj = flat_projections(players)
    with pytest.raises(RuntimeError) as excinfo:
        optimise_squad(proj, players, budget=1.0, horizon=1)
    assert not isinstance(excinfo.value, SolverTimeout)


def test_generate_squad_pool_propagates_a_timeout_instead_of_returning_short():
    """The whole point: a timeout must not masquerade as an exhausted pool."""
    players = sixteen_players()
    proj = flat_projections(players)
    cfg = dataclasses.replace(OPTIMISER, solver_time_limit_seconds=0.001)
    with pytest.raises(SolverTimeout):
        generate_squad_pool(proj, players, n=5, budget=100.0, horizon=3, config=cfg)


def test_an_integer_feasible_solution_is_treated_as_a_timeout():
    """pulp REWRITES a timed-out-with-incumbent CBC result to status
    "Optimal" with sol_status LpSolutionIntegerFeasible, so checking the
    status string alone misses every timeout that actually found a squad --
    the case the guard exists for."""
    import pulp

    from config.strategy import OPTIMISER
    from optimiser.squad import SolverTimeout, _raise_if_not_optimal

    with pytest.raises(SolverTimeout):
        _raise_if_not_optimal(
            "Optimal", OPTIMISER, "test solve",
            sol_status=pulp.constants.LpSolutionIntegerFeasible,
        )


def test_a_genuinely_optimal_solution_does_not_raise():
    import pulp

    from config.strategy import OPTIMISER
    from optimiser.squad import _raise_if_not_optimal

    _raise_if_not_optimal(
        "Optimal", OPTIMISER, "test solve",
        sol_status=pulp.constants.LpSolutionOptimal,
    )
