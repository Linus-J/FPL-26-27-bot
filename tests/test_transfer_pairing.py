"""Transfers have to be reported as PAIRS, not as two loose lists (2026-09-11).

Reported live on the GW4 2026-27 frame: Telegram and the dashboard both showed
transfers paired across positions -- a forward arriving "for" a defender. The
plan itself was right; only the presentation was wrong, which is the more
insidious kind of wrong, because the squad it describes is legal and the reader
has no way to tell the pairing is fiction.

``agent/notifier.py`` renders ``zip(transfers_in, transfers_out)`` and the
dashboard prints two adjacent columns the eye reads row-by-row. Both are only
meaningful if the two lists agree position-for-position, and nothing made them:
``gw0_in``/``gw0_out`` are built by walking ``pid_list``, so each list came out
in player-id order, independently of the other.

A correct pairing always exists. ``squad`` is constrained to a fixed count per
position in every week of the horizon, so the number sold at a position equals
the number bought at it. Sorting both lists the same way is therefore enough to
make the zip true, and fixes every consumer at once rather than one renderer at
a time.
"""

from __future__ import annotations

import pandas as pd

from optimiser.transfers import evaluate_transfers


def _squad() -> pd.DataFrame:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    costs = {"GKP": 4.5, "DEF": 4.5, "MID": 5.0, "FWD": 5.5}
    return pd.DataFrame([
        {"id": i + 1, "position": pos, "now_cost": costs[pos],
         "team_id": 1 + (i % 5), "status": "a", "web_name": f"p{i + 1}"}
        for i, pos in enumerate(positions)
    ])


def _plan_across_three_positions():
    """One clear upgrade at each of DEF, MID and FWD, on clubs with room and at
    identical prices, so all three happen and none is forced by money.

    The incoming rows are deliberately ordered FWD, DEF, MID against an owned
    squad ordered DEF, MID, FWD. What ``gw0_in``/``gw0_out`` inherit is the
    players frame's ROW order -- not id order -- so that is what has to
    disagree for the bug to show. Two earlier versions of this scenario let the
    two lists line up by accident and passed with the bug still in place.
    """
    owned = _squad()
    incoming = pd.DataFrame([
        {"id": 202, "position": "FWD", "now_cost": 5.5, "team_id": 8,
         "status": "a", "web_name": "new_fwd"},
        {"id": 201, "position": "DEF", "now_cost": 4.5, "team_id": 6,
         "status": "a", "web_name": "new_def"},
        {"id": 200, "position": "MID", "now_cost": 5.0, "team_id": 7,
         "status": "a", "web_name": "new_mid"},
    ])
    players = pd.concat([owned, incoming], ignore_index=True)

    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw,
         "xpts": 20.0 if pid in (200, 201, 202) else 1.0,
         "start_probability": 0.9}
        for pid in players["id"] for gw in (10, 11)
    ])

    return evaluate_transfers(
        current_squad_ids=owned["id"].tolist(), projections=projections,
        players=players, free_transfers=3, available_budget=100.0,
    )


def test_each_incoming_player_is_paired_with_a_same_position_outgoing_one():
    plan = _plan_across_three_positions()

    assert len(plan.transfers_in) == 3, "scenario should move all three positions"
    pairs = list(zip(plan.transfers_in, plan.transfers_out, strict=True))
    mismatched = [
        (i["web_name"], i["position"], o["web_name"], o["position"])
        for i, o in pairs if i["position"] != o["position"]
    ]
    assert not mismatched, f"in/out pairs cross positions: {mismatched}"


def test_the_two_lists_describe_the_same_multiset_of_positions():
    """The weaker invariant the pairing rests on -- worth its own failure
    message, because if this breaks the sort cannot fix anything."""
    plan = _plan_across_three_positions()

    assert (sorted(t["position"] for t in plan.transfers_in)
            == sorted(t["position"] for t in plan.transfers_out))


def test_the_notifier_renders_same_position_pairs():
    """The actual user-visible surface: ``IN ← OUT`` lines on Telegram."""
    from agent.notifier import format_decision_message

    plan = _plan_across_three_positions()
    by_name = {
        t["web_name"]: t["position"]
        for t in plan.transfers_in + plan.transfers_out
    }
    message = format_decision_message({
        "gameweek": 10,
        "transfers_in": plan.transfers_in,
        "transfers_out": plan.transfers_out,
        "hits_taken": plan.hits_taken,
        "net_xpts_gain": plan.net_xpts_gain,
    })

    rendered = [ln for ln in message.splitlines() if "←" in ln]
    assert len(rendered) == 3
    for line in rendered:
        incoming, outgoing = line.split("←")
        in_name = incoming.split("➡️")[1].split("(")[0].strip()
        out_name = outgoing.strip()
        assert by_name[in_name] == by_name[out_name], f"crossed positions: {line}"
