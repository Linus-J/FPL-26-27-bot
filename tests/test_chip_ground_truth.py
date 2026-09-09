"""Chip availability, read from FPL rather than reconstructed (2026-09-09).

``chips_used_this_season`` reads ``decision_log``, which records what the bot
RECOMMENDED. This engine has no submission path (removed 2026-08-18): a human
reads the team sheet and enters it, and can decline. Four separate rules were
added to that function to close the gap between recommendation and reality --
de-duplication of repeated runs (2026-08-16), supersede-by-newer-lineup
(2026-08-28), ignore-a-row-against-the-gameweek-being-decided (2026-09-02), and
before all of them the ``chip_played``-column bug (2026-07-28). FPL publishes
the answer outright at ``/entry/{id}/history/``.

Checked against the real entry 504618 on 2026-09-09 the inference AGREED with
FPL -- 13 chip rows in the live log collapsing to the real ``3xc`` at GW2 and
``freehit`` at GW3 -- so none of this fixes a wrong answer that was being
served. It removes the class. The case no heuristic over the log can reach is a
recommendation the operator declined: it leaves a row saying "played" and no
counter-evidence anywhere, and the chip stays spent for the rest of the half.
``test_a_declined_recommendation_is_not_a_played_chip`` is that case.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import data.ingestors.fpl_api as fpl_api
import optimiser.chips as chips
from data.models import Base, ChipUsage, ChipUsageSync
from optimiser.chips import Chip, chips_played_this_season

SEASON = "2026-27"
ENTRY = 504618


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'chips.db'}")
    Base.metadata.create_all(bind=engine)
    local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fpl_api, "get_session", lambda: local())
    monkeypatch.setattr(chips, "get_session", lambda: local())
    s = local()
    yield s
    s.close()


def _log(*played: tuple[str, int]) -> pd.DataFrame:
    """A decision_log the way the engine writes one: chip name inside the JSON
    ``details`` column, not a column of its own."""
    return pd.DataFrame([
        {
            "gameweek": gw, "decision_type": "chip",
            "details": json.dumps({"chip": name, "reason": "x"}),
            "created_at": datetime(2026, 8, 1),
        }
        for name, gw in played
    ])


def _history(*played: tuple[str, int]) -> dict:
    return {"chips": [
        {"name": name, "event": gw, "time": "2026-08-28T13:21:34.100551Z"}
        for name, gw in played
    ]}


# --- ground truth beats inference -----------------------------------------

def test_fpl_overrules_the_recommendation_log(session):
    """The log says a wildcard went in at GW2. FPL says a triple captain did.
    FPL is the one that decides what the squad may still play."""
    fpl_api.upsert_entry_chips(_history(("3xc", 2)), SEASON, ENTRY)
    played = chips_played_this_season(SEASON, _log(("wildcard", 2)), ENTRY)
    assert played == [(Chip.TRIPLE_CAPTAIN, 2)]


def test_a_declined_recommendation_is_not_a_played_chip(session):
    """The case the log can never get right, and the reason for the tables.

    The bot recommended a free hit; the operator did not play it. FPL's chip
    list is legitimately EMPTY. Inference reads the recommendation row and
    reports the chip spent -- for the rest of the half."""
    fpl_api.upsert_entry_chips(_history(), SEASON, ENTRY)   # synced, nothing played

    log = _log(("freehit", 3))
    assert chips.chips_used_this_season(log) == [(Chip.FREE_HIT, 3)], (
        "precondition: the inference does report the declined chip as played"
    )
    assert chips_played_this_season(SEASON, log, ENTRY) == []


def test_never_asking_falls_back_to_the_log(session):
    """No sync marker at all -- a fresh database, a machine that has not run
    the ingest. Behave as the bot did before this existed, and say so."""
    log = _log(("bboost", 4))
    assert chips_played_this_season(SEASON, log, ENTRY) == [(Chip.BENCH_BOOST, 4)]


def test_an_empty_ground_truth_is_not_the_same_as_no_ground_truth(session):
    """Both produce zero ``chip_usage`` rows. Only one of them means the
    operator played nothing, so the sync marker is what gets consulted."""
    log = _log(("freehit", 3))
    before = chips_played_this_season(SEASON, log, ENTRY)
    fpl_api.upsert_entry_chips(_history(), SEASON, ENTRY)
    after = chips_played_this_season(SEASON, log, ENTRY)

    assert session.execute(select(ChipUsage)).all() == []
    assert before == [(Chip.FREE_HIT, 3)]
    assert after == []


def test_a_persona_never_reads_the_real_entrys_chips(session):
    """Shadow personas and the backtest walk pass entry_id=None: they have no
    FPL entry, and inheriting the real squad's spent chips would look like a
    plausible result rather than a bug."""
    fpl_api.upsert_entry_chips(_history(("3xc", 2), ("freehit", 3)), SEASON, ENTRY)
    assert chips_played_this_season(SEASON, _log(("wildcard", 5)), None) == [
        (Chip.WILDCARD, 5)
    ]


def test_ground_truth_is_scoped_to_the_season_and_the_entry(session):
    fpl_api.upsert_entry_chips(_history(("3xc", 2)), "2025-26", ENTRY)
    fpl_api.upsert_entry_chips(_history(("bboost", 9)), SEASON, ENTRY + 1)
    assert chips_played_this_season(SEASON, _log(("freehit", 3)), ENTRY) == [
        (Chip.FREE_HIT, 3)
    ], "a different season or entry must not count as having synced this one"


# --- the ingest ------------------------------------------------------------

def test_the_sync_marker_is_written_even_when_nothing_was_played(session):
    assert fpl_api.upsert_entry_chips(_history(), SEASON, ENTRY) == 0
    marker = session.execute(select(ChipUsageSync)).scalar_one()
    assert (marker.season, marker.entry_id, marker.chips_seen) == (SEASON, ENTRY, 0)


def test_re_syncing_does_not_duplicate(session):
    for _ in range(3):
        fpl_api.upsert_entry_chips(_history(("3xc", 2), ("freehit", 3)), SEASON, ENTRY)
    assert len(session.execute(select(ChipUsage)).all()) == 2
    assert len(session.execute(select(ChipUsageSync)).all()) == 1


def test_the_played_at_timestamp_is_parsed(session):
    fpl_api.upsert_entry_chips(_history(("3xc", 2)), SEASON, ENTRY)
    row = session.execute(select(ChipUsage)).scalar_one()
    assert row.played_at == datetime(2026, 8, 28, 13, 21, 34, 100551)


def test_a_malformed_chip_entry_is_skipped_not_fatal(session):
    history = {"chips": [
        {"name": "3xc", "event": 2, "time": "2026-08-28T13:21:34.100551Z"},
        {"name": "freehit"},                       # no event
        {"event": 5, "time": "2026-09-01T00:00:00Z"},   # no name
    ]}
    fpl_api.upsert_entry_chips(history, SEASON, ENTRY)
    assert chips_played_this_season(SEASON, pd.DataFrame(), ENTRY) == [
        (Chip.TRIPLE_CAPTAIN, 2)
    ]


def test_a_chip_this_bot_does_not_model_is_reported_loudly(session, caplog):
    """It is a chip genuinely spent. Dropping it silently would leave the
    optimiser believing it still had one."""
    fpl_api.upsert_entry_chips(
        {"chips": [{"name": "manager", "event": 4, "time": None}]}, SEASON, ENTRY
    )
    with caplog.at_level("ERROR"):
        assert chips_played_this_season(SEASON, pd.DataFrame(), ENTRY) == []
    assert "manager" in caplog.text


def test_an_unparseable_timestamp_still_records_the_chip(session):
    fpl_api.upsert_entry_chips(
        {"chips": [{"name": "3xc", "event": 2, "time": "not a time"}]}, SEASON, ENTRY
    )
    row = session.execute(select(ChipUsage)).scalar_one()
    assert row.played_at is None
    assert chips_played_this_season(SEASON, pd.DataFrame(), ENTRY) == [
        (Chip.TRIPLE_CAPTAIN, 2)
    ]


async def _unreachable(entry_id):
    raise AssertionError("must not call FPL without an entry id")


def test_no_entry_id_configured_does_not_call_fpl(session, monkeypatch, caplog):
    """FPL_TEAM_ID defaults to 0. Asking about entry 0 is a guaranteed 404
    every run, and the warning has to name the consequence."""
    import asyncio
    monkeypatch.setattr(fpl_api, "fetch_entry_history", _unreachable)
    with caplog.at_level("WARNING"):
        asyncio.run(fpl_api.ingest_entry_history(SEASON, 0))
    assert "FPL_TEAM_ID is unset" in caplog.text
    assert "recommendation log" in caplog.text
    assert session.execute(select(ChipUsage)).all() == []
