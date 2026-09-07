"""backfill_team_matches.py — populate the all-competition calendar.

Six seasons x four competitions = 24 scrapes, every one of them through
FBref's Cloudflare front. Run it on a workstation with a real browser:

    DB_PATH=fpl_bot_v2.db FBREF_HEADED=1 uv run python scripts/backfill_team_matches.py

Progress is journalled after each competition-season, so a run that dies
part-way resumes where it stopped rather than restarting. Re-running a
completed job is harmless anyway — writes are upserts — but skipping it saves
a Cloudflare round trip, which is the scarce resource here.

    --force   ignore the journal and re-scrape everything
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.ingestors.team_matches import ingest_competition_season  # noqa: E402

logger = logging.getLogger(__name__)

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
LEAGUES = [
    "ENG-Premier League",
    "INT-Champions League",
    "INT-Europa League",
    "INT-Conference League",
]

# Premier League first, across all seasons: PL kickoffs are the anchors every
# congestion row is computed against, so a run that only gets half way is far
# more useful with them in place.
ALL_JOBS: list[tuple[str, str]] = [
    (season, league) for league in LEAGUES for season in SEASONS
]

JOURNAL = Path(".omc/state/team_matches_backfill.json")


def pending_jobs(done: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Outstanding (season, league) pairs, in ALL_JOBS order."""
    return [job for job in ALL_JOBS if job not in done]


def _load_journal() -> set[tuple[str, str]]:
    if not JOURNAL.exists():
        return set()
    try:
        return {tuple(entry) for entry in json.loads(JOURNAL.read_text())}
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Journal unreadable; starting from scratch")
        return set()


def _save_journal(done: set[tuple[str, str]]) -> None:
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.write_text(json.dumps(sorted(done)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore the journal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    done = set() if args.force else _load_journal()
    jobs = pending_jobs(done)
    logger.info("%d of %d competition-seasons outstanding", len(jobs), len(ALL_JOBS))

    failures: list[tuple[str, str]] = []
    for season, league in jobs:
        try:
            written = ingest_competition_season(season, league)
            logger.info("OK   %s %s (%d rows)", league, season, written)
            done.add((season, league))
            _save_journal(done)
        except Exception as exc:  # noqa: BLE001 -- one bad season must not end the run
            logger.error("FAIL %s %s: %s", league, season, exc)
            failures.append((season, league))

    if failures:
        logger.error("%d competition-seasons failed: %s", len(failures), failures)
        logger.error("Re-run the same command to retry only these.")
        return 1
    logger.info("Backfill complete: %d competition-seasons", len(ALL_JOBS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
