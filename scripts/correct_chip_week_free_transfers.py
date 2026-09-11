"""Correct the free-transfer count in already-logged chip reasons.

``roll_forward_free_transfers`` used to zero a chip week's transfers and then
add the weekly +1 anyway, so every Wildcard and Free Hit invented a transfer
it had not earned. 20a6097 fixed the rule, but the reason strings written
before it are still in ``decision_log`` -- and the portfolio site publishes
them verbatim, so the GW3 free hit reads as having bought two free transfers
into its continuation when it bought one.

The old bug added exactly +1, which makes the correction deterministic rather
than a guess. What cannot be recovered is the rest of the sentence: the
continuation was SOLVED with the extra transfer in hand, so both its value
and the margin over no-chip are optimistic by however much that transfer was
worth, and re-solving now would use projections the decision never saw (the
minutes model changed materially on 2026-09-09). Those two figures are
therefore kept as logged and labelled as such, rather than quietly restated
to numbers that were never computed.

The decision itself is unaffected and is not touched: one transfer makes the
continuation worth less, not negative.

Usage::

    uv run python scripts/correct_chip_week_free_transfers.py [--dry-run]

Idempotent -- the note it appends is also what tells it the row is done.
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

# "2 FT(s) into continuation worth" and "then 2 FT(s) into a continuation
# worth" are both in the live log -- the wording changed during GW3 -- so the
# article and the "then" are matched loosely and preserved as written.
_CONTINUATION = re.compile(r"(?<!\d)(\d+) FT\(s\) (into (?:a )?continuation)")

_NOTE = "logged as {logged} FTs"


def correct_reason(reason: str) -> str | None:
    """The reason with its free-transfer count reduced by the one the chip
    never granted, or ``None`` if there is nothing to correct."""
    if "logged as" in reason:
        return None
    match = _CONTINUATION.search(reason)
    if not match:
        return None

    logged = int(match.group(1))
    actual = max(1, logged - 1)
    # "N FT(s)" was written to dodge the plural; at one it stops being a dodge
    # and starts being wrong.
    count = "1 FT" if actual == 1 else f"{actual} FT(s)"
    corrected = _CONTINUATION.sub(f"{count} \\2", reason, count=1)
    return (
        f"{corrected} ({_NOTE.format(logged=logged)}: a chip week leaves the "
        "allowance unchanged, so the continuation and the margin were both "
        "optimistic)"
    )


def correct_chip_reasons(db: Session) -> int:
    """Rewrite every affected chip row in ``decision_log``. Returns how many
    rows changed, which is 0 on a second run."""
    rows = db.execute(
        text("SELECT id, gameweek, details FROM decision_log WHERE decision_type = 'chip'")
    ).fetchall()

    changed = 0
    for row_id, gameweek, raw in rows:
        details = json.loads(raw)
        corrected = correct_reason(details.get("reason", ""))
        if corrected is None:
            continue
        details["reason"] = corrected
        db.execute(
            text("UPDATE decision_log SET details = :details WHERE id = :id"),
            {"details": json.dumps(details), "id": row_id},
        )
        logger.info("GW%s row %s corrected", gameweek, row_id)
        changed += 1

    db.commit()
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change without writing"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from data.db import get_session

    db = get_session()
    try:
        if args.dry_run:
            rows = db.execute(
                text("SELECT gameweek, details FROM decision_log WHERE decision_type = 'chip'")
            ).fetchall()
            for gw, raw in rows:
                corrected = correct_reason(json.loads(raw).get("reason", ""))
                if corrected is not None:
                    logger.info("GW%s -> %s", gw, corrected)
            return
        logger.info("Corrected %d row(s)", correct_chip_reasons(db))
    finally:
        db.close()


if __name__ == "__main__":
    main()
