"""The XI objective is built in one place (2026-09-06).

optimiser/squad.py built its objective twice — once for `prob`, once for the
`prob2` max_transfers fallback — in near-identical hand-maintained copies.
`_bench_objective` was already factored out; this does the same for the XI half
so per-week variables cannot be added to one copy and forgotten in the other.

Those per-week variables landed the same day, so the builder is now keyed
(index, week) and scored by (player_id, week). The assertions below are all at
`weeks == 1`, where the expression it returns is the one the single-week form
returned — which is the property the free-hit path depends on."""

import pulp

from config.strategy import OPTIMISER
from optimiser.squad import _xi_objective

# Player ids deliberately unequal to the variable indices: scores are looked up
# by player id, variables by index, and nothing should conflate the two.
PLAYER_IDS = [101, 102]
SCORES_PW = {(101, 0): 5.0, (102, 0): 3.0}


def _vars(n: int, weeks: int = 1):
    return (
        [pulp.LpVariable(f"sel_{i}", cat="Binary") for i in range(n)],
        {(i, w): pulp.LpVariable(f"sta_{i}_{w}", cat="Binary")
         for i in range(n) for w in range(weeks)},
        {(i, w): pulp.LpVariable(f"cap_{i}_{w}", cat="Binary")
         for i in range(n) for w in range(weeks)},
        {(i, w): pulp.LpVariable(f"vic_{i}_{w}", cat="Binary")
         for i in range(n) for w in range(weeks)},
    )


def _expr(weeks: int = 1, scores_pw=None):
    selected, starting, captain, vice = _vars(2, weeks)
    expr = _xi_objective(
        selected, starting, captain, vice,
        scores_pw or SCORES_PW, PLAYER_IDS, weeks, OPTIMISER,
    )
    return expr, selected, starting, captain, vice


def test_a_starter_contributes_his_score_once():
    expr, _, starting, _, _ = _expr()
    assert expr[starting[(0, 0)]] == 5.0


def test_the_captain_is_counted_a_second_time():
    expr, _, _, captain, _ = _expr()
    assert expr[captain[(0, 0)]] == 5.0


def test_the_vice_is_weighted_by_the_configured_share():
    expr, _, _, _, vice = _expr()
    assert expr[vice[(0, 0)]] == 5.0 * OPTIMISER.vice_captain_weight


def test_a_benched_player_contributes_nothing_here():
    """Bench value is _bench_objective's job, not this one's."""
    expr, selected, _, _, _ = _expr()
    assert selected[0] not in expr


def test_each_week_is_scored_from_its_own_projection():
    """The whole point of the per-week form: a player good in week 0 and
    useless in week 1 must not have the two averaged away before the solver
    sees them."""
    scores_pw = {(101, 0): 5.0, (101, 1): 0.0, (102, 0): 3.0, (102, 1): 9.0}
    expr, _, starting, _, _ = _expr(weeks=2, scores_pw=scores_pw)
    assert expr[starting[(0, 0)]] == 5.0
    assert starting[(0, 1)] not in expr
    assert expr[starting[(1, 1)]] == 9.0
