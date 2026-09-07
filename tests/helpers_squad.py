"""Shared squad-solver test fixtures.

Moved out of ``test_squad_per_week_xi.py`` (2026-09-07) so
``test_solver_timeout.py`` does not need a cross-test-module import to reach
them. Built verbatim from the private ``_players``/``_flat_projections``
helpers that lived there -- several existing tests depend on their exact
shape."""

import pandas as pd


def sixteen_players() -> pd.DataFrame:
    """Sixteen cheap players so a legal 15 exists with one real choice left."""
    rows = []
    pid = 1
    for pos, count in (("GKP", 3), ("DEF", 6), ("MID", 6), ("FWD", 4)):
        for _ in range(count):
            rows.append({
                "id": pid, "web_name": f"p{pid}", "position": pos,
                "now_cost": 4.0, "team_id": (pid % 12) + 1, "status": "a",
                "start_probability": 0.9,
            })
            pid += 1
    return pd.DataFrame(rows)


def flat_projections(players: pd.DataFrame, gws=(1, 2, 3)) -> pd.DataFrame:
    rows = []
    for gw in gws:
        for _, p in players.iterrows():
            rows.append({
                "player_id": p["id"], "gameweek": gw, "xpts": 2.0,
                "xpts_var": 1.0, "upside": 0.5, "downside": 0.5,
            })
    return pd.DataFrame(rows)
