"""Resumability of the six-season calendar backfill (2026-09-06).

24 competition-seasons behind Cloudflare; a crash at 19 must not restart
from zero."""

from scripts.backfill_team_matches import ALL_JOBS, pending_jobs


def test_nothing_done_means_everything_pending():
    assert pending_jobs(set()) == ALL_JOBS


def test_completed_jobs_are_skipped():
    done = {ALL_JOBS[0], ALL_JOBS[1]}
    remaining = pending_jobs(done)
    assert ALL_JOBS[0] not in remaining
    assert ALL_JOBS[1] not in remaining
    assert len(remaining) == len(ALL_JOBS) - 2


def test_everything_done_means_nothing_pending():
    assert pending_jobs(set(ALL_JOBS)) == []


def test_order_is_deterministic():
    assert pending_jobs(set()) == pending_jobs(set())


def test_covers_six_seasons_and_four_competitions():
    assert len({season for season, _ in ALL_JOBS}) == 6
    assert len({league for _, league in ALL_JOBS}) == 4
    assert len(ALL_JOBS) == 24


def test_the_premier_league_is_scraped_before_the_european_competitions():
    """PL rows are the anchors every congestion row is computed against, so a
    partial run is far more useful with them in place."""
    first_euro = next(
        i for i, (_, league) in enumerate(ALL_JOBS) if league != "ENG-Premier League"
    )
    assert all(
        league == "ENG-Premier League" for _, league in ALL_JOBS[:first_euro]
    )
