"""Decision history queries for the dashboard: past logged decisions with
projected vs (once backfilled) actual outcomes, and the latest chip/transfer
plan for display-only surfacing."""

from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from config.strategy import POSITION_ORDER


def get_decision_history(
    db: Session, limit_gws: int = 20, sim_manager_id: int | None = None
) -> pd.DataFrame:
    """One row per decision entry within the last ``limit_gws`` gameweeks,
    most recent first, with ``details`` parsed from JSON. ``sim_manager_id``
    (optional): ``None`` (default, every existing call site) reads the real
    ``decision_log`` unchanged; a value reads that persona's own
    ``sim_decision_log`` rows instead. ``sim_decision_log`` has no
    ``dry_run`` column (every sim decision is inherently simulated) --
    reported as a constant ``True`` so the page's rendering stays
    unchanged either way."""
    table = "sim_decision_log" if sim_manager_id is not None else "decision_log"
    sim_filter = " AND sim_manager_id = :sim_manager_id" if sim_manager_id is not None else ""
    dry_run_col = "1 AS dry_run" if sim_manager_id is not None else "dry_run"
    params: dict = {"sim_manager_id": sim_manager_id} if sim_manager_id is not None else {}

    max_gw_row = db.execute(
        text(f"SELECT MAX(gameweek) FROM {table} WHERE 1=1{sim_filter}"), params
    ).fetchone()
    max_gw = max_gw_row[0] if max_gw_row and max_gw_row[0] is not None else 0

    query = text(f"""
        SELECT id, gameweek, decision_type, details, projected_gain,
               actual_outcome, {dry_run_col}, created_at
        FROM {table}
        WHERE gameweek >= :min_gw{sim_filter}
        ORDER BY gameweek DESC, created_at DESC
    """)
    df = pd.read_sql(query, db.bind, params={**params, "min_gw": max_gw - limit_gws + 1})
    if not df.empty:
        df["details"] = df["details"].apply(json.loads)
        df["dry_run"] = df["dry_run"].astype(bool)
    return df


def get_latest_chip_plan(db: Session) -> dict | None:
    row = db.execute(
        text(
            "SELECT details, projected_gain, gameweek FROM decision_log "
            "WHERE decision_type = 'chip' ORDER BY created_at DESC LIMIT 1"
        )
    ).fetchone()
    if not row:
        return None
    details = json.loads(row[0])
    return {
        "gameweek": row[2],
        "chip": details.get("chip"),
        "reason": details.get("reason"),
        "expected_gain": row[1],
    }


def get_latest_transfer_plan(db: Session) -> dict | None:
    row = db.execute(
        text(
            "SELECT details, projected_gain, gameweek FROM decision_log "
            "WHERE decision_type = 'transfers' ORDER BY created_at DESC LIMIT 1"
        )
    ).fetchone()
    if not row:
        return None
    details = json.loads(row[0])
    incoming = details.get("transfers_in", [])
    outgoing = details.get("transfers_out", [])
    positions = positions_for(db, incoming + outgoing)
    return {
        "gameweek": row[2],
        "transfers_in": paired_order(incoming, positions),
        "transfers_out": paired_order(outgoing, positions),
        "hits_taken": details.get("hits_taken", 0),
        "net_xpts_gain": row[1],
    }


def positions_for(db: Session, transfers: list[dict]) -> dict[int, str]:
    """Map player id -> position for every player named in ``transfers``.
    Shared with the site export, which publishes the same pairs."""
    ids = {t["player_id"] for t in transfers if t.get("player_id") is not None}
    if not ids:
        return {}
    rows = db.execute(
        text("SELECT id, position FROM players WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        ),
        {"ids": sorted(ids)},
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def paired_order(transfers: list[dict], positions: dict[int, str]) -> list[dict]:
    """Order a transfer list by position so that zipping the in-list against
    the out-list pairs each arriving player with the departing one they
    actually replace.

    The optimiser now emits both lists in this order, but rows logged before
    that fix are stored crossed AND carry no ``position`` key, so sorting what
    is stored would not be enough -- the position is looked up and attached
    here. That repairs history as well as anything written since, and keeps the
    invariant a property of the read path rather than of whichever writer
    happened to produce the row.

    A player who has left the game entirely may no longer have a row in
    ``players``; he keeps an empty position and sorts to the end rather than
    disappearing from a plan that really did transfer him.
    """
    enriched = [
        {**t, "position": t.get("position") or positions.get(t.get("player_id"), "")}
        for t in transfers
    ]
    # Position only, and nothing else: the sort is stable, so players who
    # share a position -- or whose position could not be resolved at all --
    # keep the order they were stored in. Any pairing among same-position
    # swaps is equally true, so there is no second key worth imposing, and
    # adding one would reshuffle rows the fix has no business touching.
    return sorted(
        enriched,
        key=lambda t: POSITION_ORDER.get(t["position"], len(POSITION_ORDER)),
    )
