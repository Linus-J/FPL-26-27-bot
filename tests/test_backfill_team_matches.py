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


def test_the_journal_path_is_anchored_to_the_repo_not_the_cwd():
    """Run from anywhere but the repo root and a CWD-relative journal makes
    resume silently restart from zero. settings.py records this project losing
    five weeks to exactly that class of bug."""
    from scripts.backfill_team_matches import JOURNAL

    assert JOURNAL.is_absolute()


def test_a_zero_row_job_is_not_recorded_as_done(tmp_path, monkeypatch):
    """I1: journalling a 0-row scrape makes the re-run skip it forever, and a
    missing club's calendar gap reads as REST."""
    import scripts.backfill_team_matches as bf

    monkeypatch.setattr(bf, "JOURNAL", tmp_path / "j.json")
    monkeypatch.setattr(bf, "ALL_JOBS", [("2021-22", "INT-Champions League")])
    monkeypatch.setattr(bf, "ingest_competition_season", lambda season, league: 0)

    rc = bf.main_for_test()

    assert rc == 1, "a zero-row job must be reported as a failure"
    assert bf._load_journal() == set(), "a zero-row job must not be journalled"


def test_a_job_that_wrote_rows_is_recorded_as_done(tmp_path, monkeypatch):
    import scripts.backfill_team_matches as bf

    monkeypatch.setattr(bf, "JOURNAL", tmp_path / "j.json")
    monkeypatch.setattr(bf, "ALL_JOBS", [("2021-22", "INT-Champions League")])
    monkeypatch.setattr(bf, "ingest_competition_season", lambda season, league: 380)

    rc = bf.main_for_test()

    assert rc == 0
    assert bf._load_journal() == {("2021-22", "INT-Champions League")}


def test_a_partly_written_journal_does_not_discard_completed_jobs(tmp_path, monkeypatch):
    """The atomic-write guarantee: a torn file must never silently reset
    progress on a run whose entire purpose is resumability."""
    import scripts.backfill_team_matches as bf

    journal = tmp_path / "j.json"
    monkeypatch.setattr(bf, "JOURNAL", journal)
    bf._save_journal({("2021-22", "ENG-Premier League")})
    assert bf._load_journal() == {("2021-22", "ENG-Premier League")}
    # No stray temp file left behind.
    assert list(tmp_path.iterdir()) == [journal]
