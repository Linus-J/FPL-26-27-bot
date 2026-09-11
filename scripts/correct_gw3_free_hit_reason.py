"""Restate the published GW3 free-hit reason with recomputed figures.

``roll_forward_free_transfers`` used to zero a chip week's transfers and then
add the weekly +1 anyway, so the GW3 free hit was chosen -- and its reason
logged, and that reason published -- on the belief that the chip would hand
its continuation two free transfers. It hands it one. 20a6097 fixed the rule
but could not fix the string already in ``decision_log``.

An earlier pass appended a note saying the figures were optimistic, on the
grounds that they could not be recomputed. They can. ``player_projections``
appends a batch per run rather than overwriting, and ``player_state_snapshots``
keeps prices and availability by day, so the continuation can be re-solved on
exactly the inputs it saw. The replay, on the 14:31:05 run of 2026-09-03
(``decision_log`` id=73, projection batch ``2026-09-03 14:30:22.213893``):

    control, 2 FTs: net +25.44, 1 hit, 3 transfers -> total 278.469
    corrected, 1 FT: net +21.44, 2 hits, 3 transfers -> total 274.469

The control is the whole basis for trusting the other line. It returns the
logged +25.44 against a recorded total of 278.4643 and a recorded margin of
15.06, so the code that has changed since has not moved this solve. Against a
no-chip total of 263.4059 the corrected margin is +11.06.

The continuation makes the same three transfers either way; with one free
transfer it simply pays an extra -4. So the free hit still beat no chip, by
11.1 rather than 15.1, and the decision stands.

The free-hit week itself is unchanged at 68.45: it is a one-week eleven built
by ``optimise_squad_joint``, which never sees a transfer allowance.

``projected_gain`` is corrected alongside the wording, because that column is
what the site renders as the week's number; leaving it optimistic would have
published a figure the corrected sentence beside it contradicts.

Only the newest GW3 chip row is restated, because only that run was replayed
and only that run is published -- the export keeps one decision per gameweek.
The runs it superseded are returned to what they said at the time.

Usage::

    uv run python scripts/correct_gw3_free_hit_reason.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

logger = logging.getLogger(__name__)

GAMEWEEK = 3

# Margin over no chip and the free-hit week's own projection. No free-transfer
# count: with real figures there is nothing left to explain in the one line a
# reader sees.
CORRECTED = "beats no-chip by 11.1 xPts — free hit: 68.45 xPts"

# The margin the run recorded, less the extra -4 the second arm pays. That
# delta is the entire difference between them -- the continuation makes the
# same three transfers either way -- so taking it off the recorded figure
# keeps this number on the same footing as every other gain in the log.
# Substituting the replay's own total would land 0.005 away and import drift
# the correction is not about.
CORRECTED_GAIN = 15.058381098233099 - 4.0

# What the earlier pass did, so it can be undone: the count was reduced and a
# parenthesised note appended.
_NOTE = re.compile(r" \(logged as (\d+) FTs: [^)]*\)$")
_COUNT = re.compile(r"(?<!\d)(\d+) FT(?:\(s\))? (into (?:a )?continuation)")


def strip_correction_note(reason: str) -> str | None:
    """The reason as originally logged, or ``None`` if it was never noted."""
    match = _NOTE.search(reason)
    if not match:
        return None
    restored = reason[: match.start()]
    return _COUNT.sub(f"{match.group(1)} FT(s) \\2", restored, count=1)


def plan_corrections(db: Session) -> list[tuple[int, str, str, float | None]]:
    """``(row id, details JSON, new reason, new gain)`` for every row that needs
    one, with the gain ``None`` where only the wording changes. Reads only, so
    ``--dry-run`` cannot write by accident."""
    rows = db.execute(
        text(
            "SELECT id, details, projected_gain FROM decision_log "
            "WHERE decision_type = 'chip' AND gameweek = :gw ORDER BY id"
        ),
        {"gw": GAMEWEEK},
    ).fetchall()
    if not rows:
        return []

    published_id = rows[-1][0]
    planned = []
    for row_id, raw, gain in rows:
        details = json.loads(raw)
        reason = details.get("reason", "")
        published = row_id == published_id
        wanted = CORRECTED if published else strip_correction_note(reason)
        # Only the published run was replayed, so only it can carry the
        # recomputed gain; the rest keep what they recorded at the time.
        wanted_gain = CORRECTED_GAIN if published else None
        if wanted is None or (wanted == reason and wanted_gain in (None, gain)):
            continue
        details["reason"] = wanted
        planned.append((row_id, json.dumps(details), wanted, wanted_gain))
    return planned


def correct_gw3_free_hit(db: Session) -> int:
    """Restate the published GW3 chip reason and un-note the rest. Returns how
    many rows changed, which is 0 on a second run."""
    planned = plan_corrections(db)
    for row_id, details, reason, gain in planned:
        if gain is None:
            db.execute(
                text("UPDATE decision_log SET details = :details WHERE id = :id"),
                {"details": details, "id": row_id},
            )
        else:
            db.execute(
                text(
                    "UPDATE decision_log SET details = :details, projected_gain = :gain "
                    "WHERE id = :id"
                ),
                {"details": details, "gain": gain, "id": row_id},
            )
        logger.info("row %s -> %s", row_id, reason)
    db.commit()
    return len(planned)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restate the published GW3 free-hit reason")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change without writing"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from data.db import get_session

    db = get_session()
    try:
        if args.dry_run:
            planned = plan_corrections(db)
            for row_id, _, reason, _gain in planned:
                logger.info("row %s -> %s", row_id, reason)
            logger.info("Would change %d row(s)", len(planned))
            return
        logger.info("Changed %d row(s)", correct_gw3_free_hit(db))
    finally:
        db.close()


if __name__ == "__main__":
    main()
