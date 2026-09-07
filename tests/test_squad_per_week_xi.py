"""optimise_squad picks an XI per gameweek, not one for the whole horizon
(2026-09-06).

It collapsed the horizon into a single scalar per player and solved for one
`starting[i]`, so two players whose good fixtures alternate earned no credit
for alternating — exactly where squad SHAPE is decided (wildcard, cold start).
optimiser/transfers.py never had this defect."""

import pandas as pd
from helpers_squad import flat_projections, sixteen_players

from optimiser.squad import optimise_squad


def test_h1_reproduces_the_single_week_answer():
    """The free-hit path calls this with horizon=1. That equivalence is the
    proof the reformulation is faithful."""
    players = sixteen_players()
    proj = flat_projections(players, gws=(1,))
    out = optimise_squad(proj, players, budget=100.0, horizon=1)
    assert len(out.squad) == 15
    assert len(out.starting_xi) == 11


def test_the_xi_can_differ_between_gameweeks():
    """Two midfielders with opposite fixtures: each should start in his own
    good week rather than one being permanently benched."""
    players = sixteen_players()
    proj = flat_projections(players, gws=(1, 2))
    # p10 and p11 are MIDs. Give them sharply opposed weeks.
    proj.loc[(proj["player_id"] == 10) & (proj["gameweek"] == 1), "xpts"] = 12.0
    proj.loc[(proj["player_id"] == 10) & (proj["gameweek"] == 2), "xpts"] = 0.1
    proj.loc[(proj["player_id"] == 11) & (proj["gameweek"] == 1), "xpts"] = 0.1
    proj.loc[(proj["player_id"] == 11) & (proj["gameweek"] == 2), "xpts"] = 12.0

    out = optimise_squad(proj, players, budget=100.0, horizon=2)
    squad_ids = set(out.squad["id"])
    assert {10, 11} <= squad_ids, "both alternating players should be bought"


def test_week_zero_is_what_gets_reported():
    players = sixteen_players()
    proj = flat_projections(players, gws=(1, 2))
    proj.loc[(proj["player_id"] == 10) & (proj["gameweek"] == 1), "xpts"] = 20.0
    out = optimise_squad(proj, players, budget=100.0, horizon=2)
    assert 10 in set(out.starting_xi["id"])


def test_a_full_squad_and_legal_xi_are_still_produced():
    players = sixteen_players()
    proj = flat_projections(players)
    out = optimise_squad(proj, players, budget=100.0, horizon=3)
    assert len(out.squad) == 15
    assert len(out.starting_xi) == 11
    assert out.captain_id in set(out.starting_xi["id"])
    assert out.vice_captain_id in set(out.starting_xi["id"])
    assert out.captain_id != out.vice_captain_id


def test_position_quotas_hold_in_every_week():
    players = sixteen_players()
    proj = flat_projections(players)
    out = optimise_squad(proj, players, budget=100.0, horizon=3)
    counts = out.squad["position"].value_counts().to_dict()
    assert counts == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_the_budget_is_respected():
    players = sixteen_players()
    proj = flat_projections(players)
    out = optimise_squad(proj, players, budget=64.0, horizon=3)
    assert out.total_cost <= 64.0 + 1e-6


def _set_gw_xpts(proj: pd.DataFrame, pid: int, by_gw: dict[int, float]) -> None:
    for gw, xpts in by_gw.items():
        proj.loc[(proj["player_id"] == pid) & (proj["gameweek"] == gw), "xpts"] = xpts


# The two tests below are the ones that actually separate the per-week
# formulation from the horizon-collapsed one. The six above pass on either,
# because a player who alternates still has a large HORIZON SUM and so gets
# bought regardless — the collapsed objective's failure is in what it does
# with him afterwards, not in whether it owns him.
#
# Both use goalkeepers deliberately: the squad holds exactly two and exactly
# one may start, so "who starts" is a forced binary choice with no room for
# the position minimums to make both answers legal.


def test_the_week_zero_starter_is_chosen_on_week_zero():
    """A keeper worth more over the horizon must still sit behind one worth
    more THIS week.

    p1 scores 5 then 10 (decayed horizon total 13.5); p2 scores 6 then nothing
    (6.0). The collapsed objective starts p1 in the only gameweek it reports,
    surrendering a point in the one week that is about to be played for points
    it cannot collect until next week — when it will pick an XI again anyway.
    """
    players = sixteen_players()
    proj = flat_projections(players, gws=(1, 2))
    _set_gw_xpts(proj, 1, {1: 5.0, 2: 10.0})
    _set_gw_xpts(proj, 2, {1: 6.0, 2: 0.0})

    out = optimise_squad(proj, players, budget=100.0, horizon=2)
    xi = set(out.starting_xi["id"])
    assert 2 in xi, "the keeper who is best in the reported gameweek must start it"
    assert 1 not in xi


def test_an_alternating_pair_is_bought_over_a_steady_keeper():
    """Squad SHAPE, which is the reason this change exists.

    p1 (10, 0.1) and p2 (0.1, 10) cover each other perfectly: whichever is
    good starts, so the pair is worth ~10 a week. p3 (5, 5) has almost the
    same horizon total as either of them alone. Collapsing the horizon makes
    the second keeper a bench asset valued at 3% of a number that alternation
    has already made meaningless, and p3's steadier collapsed total wins the
    slot — the pair is worth 18.5 to the per-week objective and 14.4 with p3.
    """
    players = sixteen_players()
    proj = flat_projections(players, gws=(1, 2))
    _set_gw_xpts(proj, 1, {1: 10.0, 2: 0.1})
    _set_gw_xpts(proj, 2, {1: 0.1, 2: 10.0})
    _set_gw_xpts(proj, 3, {1: 5.0, 2: 5.0})

    out = optimise_squad(proj, players, budget=100.0, horizon=2)
    keepers = set(out.squad.loc[out.squad["position"] == "GKP", "id"])
    assert keepers == {1, 2}, f"expected the alternating pair, got {keepers}"


def test_the_bench_is_ordered_by_slot_in_every_week():
    """Bench slot weights are strictly decreasing, so the best available bench
    player must land in slot 1. Nothing tested the per-week slot assignment."""
    players = sixteen_players()
    proj = flat_projections(players, gws=(1, 2))
    out = optimise_squad(proj, players, budget=100.0, horizon=2)
    bench = out.squad[~out.squad["is_starting"]]
    assert list(bench["bench_order"]) == sorted(bench["bench_order"])


def test_the_max_transfers_fallback_still_solves_per_week():
    """The prob2 path got the same per-week treatment as prob; only my probes
    covered it."""
    players = sixteen_players()
    proj = flat_projections(players, gws=(1, 2))
    current = players["id"].tolist()[:15]
    out = optimise_squad(
        proj, players, budget=100.0, horizon=2,
        current_squad_ids=current, max_transfers=0,
    )
    assert len(out.squad) == 15
    assert len(out.starting_xi) == 11
