"""The free-transfer allowance, read from FPL rather than carried in the log
(2026-09-09).

Two separate bugs met here and produced one wrong number for the live GW4
decision.

The first was the rule. ``roll_forward_free_transfers`` zeroed the transfers a
chip made and then added the weekly ``+1`` anyway, so every Wildcard and every
Free Hit invented a free transfer -- while its own comment said "you do not
earn an extra one in the week you play the chip". FPL's rule is that the
allowance is unchanged across a chip week: saved transfers are retained, but
the chip consumes that gameweek's own allotment.

The second was the source. Even with the rule fixed, the engine seeds the
count from the last ``decision_log`` row, which records what the bot ADVISED.
There is no submission path (removed 2026-08-18), so an operator who declines
a transfer or takes an unplanned hit leaves the stored allowance wrong for the
rest of the season with nothing to correct it. So the count is reconstructed
from ``/entry/{id}/history/``, exactly as chip usage now is -- see
``tests/test_chip_ground_truth.py``.

Live case this was caught on, entry 504618: 1 free transfer into GW2, two
transfers for a -4 hit there (so the allowance was demonstrably 1), rolling to
1 for GW3, Free Hit played at GW3 -- and FPL showed **1** for GW4 where the
bot believed 2.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import data.ingestors.fpl_api as fpl_api
import optimiser.transfers as transfers_mod
from config.strategy import TRANSFERS
from data.models import Base, ChipUsage, EntryGameweek
from optimiser.transfers import free_transfers_this_gameweek

SEASON = "2026-27"
ENTRY = 504618


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'ft.db'}")
    Base.metadata.create_all(bind=engine)
    local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fpl_api, "get_session", lambda: local())
    monkeypatch.setattr(transfers_mod, "get_session", lambda: local())
    s = local()
    yield s
    s.close()


def _history(*weeks: tuple[int, int, int]) -> dict:
    """``(gameweek, event_transfers, event_transfers_cost)``, FPL's shape."""
    return {"current": [
        {"event": gw, "event_transfers": made, "event_transfers_cost": cost,
         "points": 50, "total_points": 50 * gw}
        for gw, made, cost in weeks
    ]}


def _chip(session, gw: int, name: str) -> None:
    session.add(ChipUsage(season=SEASON, entry_id=ENTRY, chip=name, gameweek=gw,
                          played_at=datetime(2026, 8, 28), synced_at=datetime(2026, 9, 9)))
    session.commit()


# --- the live case --------------------------------------------------------

def test_the_free_hit_gameweek_does_not_hand_out_a_transfer(session):
    """Entry 504618's actual season to the GW4 deadline. FPL said 1; the bot
    said 2, and the whole GW4 transfer plan was built on the extra one."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 2, 4), (3, 0, 0)), SEASON, ENTRY)
    _chip(session, 3, "freehit")

    assert free_transfers_this_gameweek(SEASON, ENTRY, 4) == 1


def test_without_the_free_hit_the_same_season_banks_one(session):
    """The control for the test above: identical history, no chip. The Free
    Hit is doing the work, not the transfer counts."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 2, 4), (3, 0, 0)), SEASON, ENTRY)

    assert free_transfers_this_gameweek(SEASON, ENTRY, 4) == 2


# --- the walk -------------------------------------------------------------

def test_gameweek_one_is_free_so_the_walk_starts_at_two(session):
    """GW1 transfers are unlimited and free. However many were made, GW2's
    allowance is the weekly one, the same for everybody -- which is what makes
    the reconstruction possible at all."""
    fpl_api.upsert_entry_gameweeks(_history((1, 9, 0)), SEASON, ENTRY)

    assert free_transfers_this_gameweek(SEASON, ENTRY, 2) == TRANSFERS.free_transfers_per_gw


def test_quiet_weeks_bank_up_to_the_cap(session):
    weeks = [(1, 0, 0)] + [(gw, 0, 0) for gw in range(2, 10)]
    fpl_api.upsert_entry_gameweeks(_history(*weeks), SEASON, ENTRY)

    assert free_transfers_this_gameweek(SEASON, ENTRY, 10) == TRANSFERS.max_banked_free_transfers


def test_a_chip_at_the_cap_does_not_push_past_it(session):
    """The bug was additive, so it also broke the cap: five banked plus an
    invented one would have been clamped back to five and hidden here. It is
    the floor case and the middle of the range that expose it."""
    weeks = [(1, 0, 0)] + [(gw, 0, 0) for gw in range(2, 8)]
    fpl_api.upsert_entry_gameweeks(_history(*weeks), SEASON, ENTRY)
    _chip(session, 7, "wildcard")

    assert free_transfers_this_gameweek(SEASON, ENTRY, 8) == TRANSFERS.max_banked_free_transfers


def test_a_chip_mid_range_leaves_the_bank_untouched(session):
    """Three available in the wildcard week, three out of it -- clear of both
    the floor and the cap, which is where the invented transfer used to show
    up undisguised. Without the wildcard this run would give four."""
    fpl_api.upsert_entry_gameweeks(
        _history((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)), SEASON, ENTRY
    )
    _chip(session, 4, "wildcard")

    assert free_transfers_this_gameweek(SEASON, ENTRY, 5) == 3


def test_hits_are_paid_in_points_not_in_next_weeks_allowance(session):
    """Four transfers on a one-transfer allowance is -12, and GW3 still opens
    on the weekly allowance rather than a negative one."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 4, 12)), SEASON, ENTRY)

    assert free_transfers_this_gameweek(SEASON, ENTRY, 3) == TRANSFERS.free_transfers_per_gw


# --- the cross-check ------------------------------------------------------

def test_a_hit_that_contradicts_the_walk_is_logged_loudly(session, caplog):
    """``transfers_cost`` is 4 x the transfers beyond the allowance, so any
    week with a hit pins the allowance exactly. It is deliberately not used to
    correct the walk: a disagreement means either this model of FPL's rules is
    wrong or the sync is stale, and both want a human rather than a silent
    patch."""
    # Two transfers costing 4 means the allowance was 1. But three quiet weeks
    # precede it, so the walk thinks it is 4.
    fpl_api.upsert_entry_gameweeks(
        _history((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 2, 4)), SEASON, ENTRY
    )

    with caplog.at_level("ERROR"):
        free_transfers_this_gameweek(SEASON, ENTRY, 6)

    assert "disagrees with FPL at GW5" in caplog.text
    assert "the allowance was 1" in caplog.text


def test_the_live_hit_agrees_with_the_walk(session, caplog):
    """The same check on the real history stays quiet -- 2 transfers for -4 at
    GW2 against an allowance of 1. This is the evidence the whole fix rests
    on, so a change that breaks it must not pass silently."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 2, 4), (3, 0, 0)), SEASON, ENTRY)
    _chip(session, 3, "freehit")

    with caplog.at_level("ERROR"):
        free_transfers_this_gameweek(SEASON, ENTRY, 4)

    assert "disagrees" not in caplog.text


# --- when it must decline to answer ---------------------------------------

def test_no_entry_id_means_no_answer(session):
    """The backtest walk and the shadow personas have no FPL entry. They must
    not inherit the real squad's transfer count any more than they inherit its
    spent chips."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 2, 4)), SEASON, ENTRY)

    assert free_transfers_this_gameweek(SEASON, None, 3) is None
    assert free_transfers_this_gameweek(SEASON, 0, 3) is None


def test_nothing_synced_means_no_answer(session):
    assert free_transfers_this_gameweek(SEASON, ENTRY, 4) is None


def test_gameweek_one_has_no_allowance_to_report(session):
    """GW1 transfers are unlimited, so there is no allowance to reconstruct
    and the caller keeps whatever it had."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0)), SEASON, ENTRY)

    assert free_transfers_this_gameweek(SEASON, ENTRY, 1) is None


def test_another_entrys_history_is_not_read(session):
    """Scoped by entry and season, like the chip ground truth."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 0, 0)), SEASON, 999)

    assert free_transfers_this_gameweek(SEASON, ENTRY, 3) is None
    assert free_transfers_this_gameweek("2025-26", 999, 3) is None


# --- the ingest -----------------------------------------------------------

def test_re_syncing_updates_in_place(session):
    """A gameweek is re-read every run while it is in progress, and the
    numbers move. Duplicated rows would double-count the walk."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 0, 0)), SEASON, ENTRY)
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 2, 4)), SEASON, ENTRY)

    rows = session.execute(select(EntryGameweek).order_by(EntryGameweek.gameweek)).scalars().all()
    assert [(r.gameweek, r.transfers_made, r.transfers_cost) for r in rows] == [
        (1, 0, 0), (2, 2, 4),
    ]


def test_a_malformed_gameweek_is_skipped_not_fatal(session):
    written = fpl_api.upsert_entry_gameweeks(
        {"current": [{"event": None, "event_transfers": 1}, {"event": 2, "event_transfers": 1}]},
        SEASON, ENTRY,
    )
    assert written == 1
    assert session.execute(select(EntryGameweek)).scalar_one().gameweek == 2


def test_missing_transfer_fields_default_to_zero(session):
    """FPL omits nothing in practice, but a null must not become a None that
    crashes the walk three functions away."""
    fpl_api.upsert_entry_gameweeks({"current": [{"event": 2}]}, SEASON, ENTRY)

    row = session.execute(select(EntryGameweek)).scalar_one()
    assert (row.transfers_made, row.transfers_cost) == (0, 0)


def test_both_facts_come_from_one_fetch(session, monkeypatch):
    """Chips and gameweeks are in the same payload. Asking twice would double
    the request count for no reason, and could see two different states."""
    import asyncio

    calls = []

    async def _fetch(entry_id):
        calls.append(entry_id)
        return {
            "chips": [{"name": "freehit", "event": 3, "time": "2026-08-28T13:21:34.100551Z"}],
            "current": [{"event": gw, "event_transfers": 0, "event_transfers_cost": 0}
                        for gw in (1, 2, 3)],
        }

    monkeypatch.setattr(fpl_api, "fetch_entry_history", _fetch)
    asyncio.run(fpl_api.ingest_entry_history(SEASON, ENTRY))

    assert calls == [ENTRY]
    assert session.execute(select(ChipUsage)).scalar_one().chip == "freehit"
    assert len(session.execute(select(EntryGameweek)).all()) == 3
    # And the two together give the answer, without a second round trip: one
    # banked from the quiet GW2, held through the Free Hit at GW3.
    assert free_transfers_this_gameweek(SEASON, ENTRY, 4) == 2


def test_a_network_failure_is_not_fatal(session, monkeypatch):
    """Best-effort by design: the ingest must finish and the consumers fall
    back to the log, which is what they used exclusively until 2026-09-09."""
    import asyncio

    import aiohttp

    async def _boom(entry_id):
        raise aiohttp.ClientError("no")

    monkeypatch.setattr(fpl_api, "fetch_entry_history", _boom)
    asyncio.run(fpl_api.ingest_entry_history(SEASON, ENTRY))

    assert session.execute(select(EntryGameweek)).all() == []
    assert free_transfers_this_gameweek(SEASON, ENTRY, 4) is None


def test_a_stale_sync_declines_rather_than_answering_from_old_rows(session, caplog):
    """A history that stops short of the gameweek before the decision is
    missing transfers, and the walk would return a confidently wrong number
    instead of an absent one. The live database has been pointed at the wrong
    file before (2026-07-31), which is exactly how this goes unnoticed."""
    fpl_api.upsert_entry_gameweeks(_history((1, 0, 0), (2, 0, 0)), SEASON, ENTRY)

    with caplog.at_level("WARNING"):
        assert free_transfers_this_gameweek(SEASON, ENTRY, 5) is None
    assert "stops at GW2" in caplog.text
    # One gameweek behind is not stale: GW3 is in progress and GW4 is the
    # decision, which is the ordinary case every week.
    assert free_transfers_this_gameweek(SEASON, ENTRY, 3) == 2
