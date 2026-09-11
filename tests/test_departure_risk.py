"""Departure-risk gate (v2-build-plan §6.5) — the squad-construction gate
that hard-excludes/force-sells confirmed departures.

Pure functions (confirmed_p_leave, stay_probability_multiplier,
apply_departure_discount, hard_excluded_ids) plus the real bug fix in
evaluate_transfers: an already-OWNED player who becomes status='u' used to
be silently dropped from the ILP's variable set entirely (no `tout`
variable existed for them at all), so they never appeared in the reported
transfers_out even though the squad-size constraint happened to force a
replacement in. Now they're forced out explicitly (tout==1) and correctly
reported.
"""

from __future__ import annotations

import pandas as pd
import pytest

from optimiser.departure_risk import (
    apply_departure_discount,
    confirmed_p_leave,
    hard_excluded_ids,
    is_hard_excluded,
    stay_probability_multiplier,
)
from optimiser.transfers import evaluate_transfers, pin_confirmed_departures


def test_confirmed_p_leave_only_status_u():
    assert confirmed_p_leave("u") == 1.0
    for status in ("a", "d", "i", "s", "n"):
        assert confirmed_p_leave(status) == 0.0


def test_is_hard_excluded_threshold():
    assert is_hard_excluded(0.7) is True
    assert is_hard_excluded(0.69) is False
    assert is_hard_excluded(1.0) is True


def test_stay_probability_multiplier_three_tiers():
    assert stay_probability_multiplier(0.1) == 1.0        # below rumour floor -> no effect
    assert stay_probability_multiplier(0.5) == pytest.approx(0.5)  # rumour tier -> 1 - p_leave
    assert stay_probability_multiplier(0.9) == 0.0         # hard-exclude tier -> zeroed


def test_apply_departure_discount_scales_only_named_players():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "xpts": 10.0, "xpts_mean": 10.0},
        {"player_id": 2, "gameweek": 5, "xpts": 8.0, "xpts_mean": 8.0},
    ])
    out = apply_departure_discount(projections, {1: 0.5})
    assert out.loc[out["player_id"] == 1, "xpts"].iloc[0] == pytest.approx(5.0)
    assert out.loc[out["player_id"] == 2, "xpts"].iloc[0] == 8.0  # untouched


def test_apply_departure_discount_empty_map_is_noop():
    projections = pd.DataFrame([{"player_id": 1, "gameweek": 5, "xpts": 10.0}])
    out = apply_departure_discount(projections, {})
    pd.testing.assert_frame_equal(out, projections)


def test_hard_excluded_ids_from_status_column():
    players = pd.DataFrame([
        {"id": 1, "status": "a"}, {"id": 2, "status": "u"}, {"id": 3, "status": "d"},
    ])
    assert hard_excluded_ids(players) == {2}


def test_hard_excluded_ids_missing_status_column_is_safe():
    assert hard_excluded_ids(pd.DataFrame([{"id": 1}])) == set()


# --- evaluate_transfers: the real bug fix ----------------------------------

def _owned_squad():
    # 2 GKP, 5 DEF, 5 MID, 3 FWD across 5 teams, round-robin so each team
    # gets exactly 3 players (respects max_players_per_club=3)
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    costs = {"GKP": 4.5, "DEF": 4.5, "MID": 5.0, "FWD": 5.5}
    rows = []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": costs[position],
            "team_id": 1 + (i % 5), "status": "a", "web_name": f"p{pid}",
        })
    return pd.DataFrame(rows)


def _replacement_candidates(start_id: int):
    # extra DEF options on teams NOT already at the 3-per-club cap
    rows = []
    for i, team in enumerate((6, 7, 8)):
        rows.append({"id": start_id + i, "position": "DEF", "now_cost": 4.5,
                     "team_id": team, "status": "a", "web_name": f"repl{i}"})
    return pd.DataFrame(rows)


def test_evaluate_transfers_force_sells_confirmed_departure():
    owned = _owned_squad()
    departed_id = owned[owned["position"] == "DEF"]["id"].iloc[0]
    owned.loc[owned["id"] == departed_id, "status"] = "u"  # confirmed departure

    replacements = _replacement_candidates(start_id=100)
    players = pd.concat([owned, replacements], ignore_index=True)
    squad_ids = owned["id"].tolist()

    gws = [10, 11, 12]
    proj_rows = []
    for pid in players["id"]:
        base = 8.0 if pid in replacements["id"].values else 4.0
        base = 0.0 if pid == departed_id else base
        for gw in gws:
            proj_rows.append({
                "player_id": pid, "gameweek": gw, "xpts": base,
                "start_probability": 0.9,
            })
    projections = pd.DataFrame(proj_rows)

    plan = evaluate_transfers(
        current_squad_ids=squad_ids,
        projections=projections,
        players=players,
        free_transfers=1,
        available_budget=100.0,
    )

    out_ids = {t["player_id"] for t in plan.transfers_out}
    in_ids = {t["player_id"] for t in plan.transfers_in}
    assert departed_id in out_ids, "confirmed departure must be reported as sold"
    assert len(in_ids) >= 1, "a replacement must be bought to refill the DEF slot"
    # exactly one DEF slot needed refilling from this scenario
    new_squad = [pid for pid in squad_ids if pid not in out_ids] + list(in_ids)
    assert len(new_squad) == 15
    assert players[players["id"].isin(new_squad)]["position"].value_counts().to_dict() == {
        "DEF": 5, "MID": 5, "FWD": 3, "GKP": 2,
    }


def test_evaluate_transfers_no_departure_is_unaffected():
    owned = _owned_squad()
    replacements = _replacement_candidates(start_id=100)
    players = pd.concat([owned, replacements], ignore_index=True)
    squad_ids = owned["id"].tolist()

    gws = [10, 11, 12]
    proj_rows = [
        {"player_id": pid, "gameweek": gw, "xpts": 4.0, "start_probability": 0.9}
        for pid in players["id"] for gw in gws
    ]
    projections = pd.DataFrame(proj_rows)

    plan = evaluate_transfers(
        current_squad_ids=squad_ids, projections=projections, players=players,
        free_transfers=1, available_budget=100.0,
    )
    # no forced sale needed -- a sensible plan makes 0 transfers when nothing
    # is meaningfully better (all replacement candidates project identically)
    assert plan.hits_taken == 0


# --- dead-weight departures are a money chip, not a week-0 emergency -------
#
# A confirmed departure whose projections are already zero contributes nothing
# to the squad whether he is sold this week or next. Forcing `tout == 1` at
# week 0 spent the free transfer on a slot that was dead either way, and on
# the live GW4 2026-27 frame that cascaded into a -4: Neave (status 'u', 0
# xPts) had to go, the only replacement affordable on a 0.0 bank was worth 0
# xPts, so buying a useful forward needed money, so a second player was sold
# to raise it. Measured cost of the forced timing: ~1.9 xPts of plan quality.
#
# The sale stays mandatory -- a permanently dead 15th slot hurts Bench Boost
# and auto-subs -- but WHICH week it happens in is now the optimiser's call.


def _pool_with_one_big_upgrade(owned: pd.DataFrame) -> pd.DataFrame:
    """Replacement DEFs that are no better than what is owned, plus one MID
    who is worth far more than anyone in the squad."""
    rows = [
        {"id": 100 + i, "position": "DEF", "now_cost": 4.5, "team_id": team,
         "status": "a", "web_name": f"repl{i}"}
        for i, team in enumerate((6, 7, 8))
    ]
    rows.append({"id": 200, "position": "MID", "now_cost": 5.0, "team_id": 9,
                 "status": "a", "web_name": "star"})
    return pd.concat([owned, pd.DataFrame(rows)], ignore_index=True)


def _projections(players: pd.DataFrame, gws: list[int], xpts_by_pid: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": pid, "gameweek": gw,
         "xpts": xpts_by_pid.get(pid, 4.0), "start_probability": 0.9}
        for pid in players["id"] for gw in gws
    ])


def test_a_zero_xpts_departure_does_not_burn_week_zeros_free_transfer():
    owned = _owned_squad()
    departed_id = int(owned[owned["position"] == "DEF"]["id"].iloc[0])
    owned.loc[owned["id"] == departed_id, "status"] = "u"
    players = _pool_with_one_big_upgrade(owned)
    squad_ids = owned["id"].tolist()

    # The departure is worth nothing; the unowned MID is worth a great deal.
    projections = _projections(players, [10, 11, 12], {departed_id: 0.0, 200: 30.0})

    plan = evaluate_transfers(
        current_squad_ids=squad_ids, projections=projections, players=players,
        free_transfers=1, available_budget=100.0,
    )

    out_ids = {t["player_id"] for t in plan.transfers_out}
    in_ids = {t["player_id"] for t in plan.transfers_in}
    assert departed_id not in out_ids, (
        "the dead slot costs the same next week; week 0's free transfer should "
        "buy the upgrade instead"
    )
    assert 200 in in_ids, "the free transfer should have bought the upgrade"
    assert plan.hits_taken == 0, "deferring the dead sale means no hit is needed"


def test_a_departure_still_projecting_points_is_sold_immediately():
    """The relaxation is keyed on being worth nothing, not on the status. A
    hand-veto lands on the same code path and must still be honoured at once,
    or 'the manager's judgement is not up for negotiation' becomes advisory."""
    owned = _owned_squad()
    departed_id = int(owned[owned["position"] == "DEF"]["id"].iloc[0])
    owned.loc[owned["id"] == departed_id, "status"] = "u"
    players = _pool_with_one_big_upgrade(owned)
    squad_ids = owned["id"].tolist()

    # Worth MORE than anyone available, so nothing but the force would sell him.
    projections = _projections(players, [10, 11, 12], {departed_id: 25.0})

    plan = evaluate_transfers(
        current_squad_ids=squad_ids, projections=projections, players=players,
        free_transfers=1, available_budget=100.0,
    )

    assert departed_id in {t["player_id"] for t in plan.transfers_out}


def _veto_model(horizon_xpts: float, weeks: int = 3, prefer_buyback: bool = True):
    """A one-player model carrying only the pins under test, plus the same
    week-0 "you cannot buy a player you already own" sanity pin the real
    optimiser adds, and an objective that wants the buy-back as much as the
    formulation permits."""
    import pulp

    pid = 7
    prob = pulp.LpProblem("veto", pulp.LpMaximize)
    tin = {(pid, w): pulp.LpVariable(f"in_{w}", cat="Binary") for w in range(weeks)}
    tout = {(pid, w): pulp.LpVariable(f"out_{w}", cat="Binary") for w in range(weeks)}
    owned = {(pid, w): pulp.LpVariable(f"sq_{w}", cat="Binary") for w in range(weeks)}
    for w in range(weeks):
        prior = 1 if w == 0 else owned[(pid, w - 1)]
        prob += owned[(pid, w)] == prior + tin[(pid, w)] - tout[(pid, w)]
        prob += tin[(pid, w)] + tout[(pid, w)] <= 1
    prob += tin[(pid, 0)] == 0  # he is already owned at week 0

    pin_confirmed_departures(prob, tin, tout, [pid], {pid: horizon_xpts}, weeks)

    if prefer_buyback:
        prob += pulp.lpSum(owned[(pid, w)] for w in range(weeks))
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    def val(d):
        return [int(round(pulp.value(d[(pid, w)]) or 0)) for w in range(weeks)]

    return val(tin), val(tout), val(owned)


def test_a_vetoed_player_is_not_bought_back_later_in_the_horizon():
    """The veto has to outlast week 0.

    ``tin`` was pinned to 0 only at week 0, and only by the "you cannot buy a
    player you already own" sanity pin. Nothing stopped the model selling a
    vetoed player at week 0 and buying him straight back at week 1, so the veto
    expired after a single gameweek.

    This is asserted against the model rather than a squad scenario on purpose.
    ``evaluate_transfers`` reports week 0 and nothing else, and the buy-back
    provably cannot perturb week 0: the swap that funds it conserves cash, club
    slot and positional slot, and linearly-priced hits decouple every transfer
    decision from every other. Sweeping a rival upgrade across the -4 boundary
    produced byte-identical plans either way, while the solved model showed
    ``tin`` set at week 1 -- see pin_confirmed_departures' docstring.
    """
    tin, tout, owned = _veto_model(horizon_xpts=30.0)

    assert tout == [1, 0, 0], "a player still worth points is sold at week 0"
    assert tin == [0, 0, 0], f"vetoed player re-acquired mid-horizon: {tin}"
    assert owned == [0, 0, 0], "and is gone for the rest of the horizon"


def test_dead_weight_is_sold_somewhere_in_the_horizon_but_never_re_entered():
    """The money-chip branch relaxes WHEN the sale happens; it must not relax
    whether he can come back."""
    tin, tout, owned = _veto_model(horizon_xpts=0.0)

    assert sum(tout) == 1, "sold exactly once, on the optimiser's chosen week"
    assert tin == [0, 0, 0], f"dead weight re-acquired mid-horizon: {tin}"


def test_the_sale_is_mandatory_even_when_nothing_rewards_it():
    """Without the pin an indifferent objective would simply hold him."""
    _, tout, owned = _veto_model(horizon_xpts=30.0, prefer_buyback=False)

    assert tout[0] == 1
    assert owned == [0, 0, 0]


def test_a_squad_member_missing_from_the_player_frame_is_refused():
    """FPL dropping a player from bootstrap entirely must not conjure a free one.

    Every constraint keys off ``pid_list``, which is built from the players
    frame. A squad member absent from that frame therefore has no ``squad``,
    ``tin`` or ``tout`` variable at all -- so ``in_current`` sums to 14 while
    the squad-size constraint still demands 15, and the model closes the gap by
    buying someone with nothing sold against them. The reported plan then has a
    purchase with no matching sale, ``new_squad_ids`` comes back 16 long, and
    the bank was computed off the 14 prices it could see.

    Nothing can be modelled on the missing player's behalf -- there is no
    price, position or club for him -- so this refuses rather than guesses.
    Status 'u' departures keep their bootstrap row all season and are handled
    by the force-sell path above; this is the case where the row is gone.
    """
    owned = _owned_squad()
    vanished_id = int(owned["id"].iloc[3])
    players = owned[owned["id"] != vanished_id]  # dropped from bootstrap

    gws = [10, 11]
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "xpts": 4.0, "start_probability": 0.9}
        for pid in players["id"] for gw in gws
    ])

    with pytest.raises(ValueError, match=str(vanished_id)):
        evaluate_transfers(
            current_squad_ids=owned["id"].tolist(), projections=projections,
            players=players, free_transfers=1, available_budget=100.0,
        )
