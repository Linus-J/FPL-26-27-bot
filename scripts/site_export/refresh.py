"""Rebuild the ``history`` block of already-published run files.

Each weekly export writes one ``gw{N}.json`` and never touches it again, which
is right for most of what is in there: the squad and the projections are what
the engine believed that week and are not re-derivable now. The history block
is different -- it is a rendering of ``decision_log``, so correcting a logged
decision leaves every earlier file showing the old version.

That is exactly what happened on 2026-09-12: restating the GW3 free-hit reason
reached gw4.json, because the export rebuilds it, but not gw3.json, which is
the file the GW3 run itself published.

Each file is capped at its own gameweek, so refreshing can never put an event
into a run that had not happened when the run was made.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scripts.site_export.payload import _build_history_entries, _history_positions

_RUN_FILE = re.compile(r"^gw(\d+)\.json$")


def refresh_history_files(
    data_dir: Path, history_df: pd.DataFrame, positions: dict[int, str] | None = None
) -> list[Path]:
    """Rewrite every run file whose history no longer matches the log.
    Returns the paths that changed, so a re-run reports nothing."""
    changed = []
    for path in sorted(data_dir.glob("gw*.json")):
        match = _RUN_FILE.match(path.name)
        if not match:
            continue
        gw = int(match.group(1))

        payload = json.loads(path.read_text())
        rebuilt = _build_history_entries(history_df, up_to_gw=gw, positions=positions or {})
        if payload.get("history") == rebuilt:
            continue

        payload["history"] = rebuilt
        path.write_text(json.dumps(payload, indent=2) + "\n")
        changed.append(path)
    return changed


def refresh_from_db(data_dir: Path, db) -> list[Path]:
    """``refresh_history_files`` against the live decision log."""
    from dashboard.data.decisions import get_decision_history

    history_df = get_decision_history(db, limit_gws=40)
    return refresh_history_files(data_dir, history_df, _history_positions(db, history_df))
