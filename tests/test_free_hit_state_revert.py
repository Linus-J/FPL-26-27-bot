"""A Free Hit squad is handed back at the deadline (2026-09-07).

``_load_squad_state`` took ``squad_ids`` from the most recent lineup row
regardless of chip, so after a Free Hit gameweek the engine believed it owned
the Free Hit squad. Live consequence: the GW4 plan proposed transferring out
two players the bot did not own.

Bank, free transfers and purchase prices are NOT affected -- an earlier fix
already carries those across a Free Hit correctly -- so they must still come
from the most recent row, including a Free Hit one."""

import itertools
import json

import pytest
from sqlalchemy import text

from agent.decision_engine import _load_squad_state
from config.strategy import OPTIMISER
from data.db import get_session

# A gameweek can be decided many times before its deadline -- the engine is
# re-run interactively and every run appends a row -- so two rows can share a
# gameweek. sqlite's datetime('now') only has 1-second resolution, and the
# offset used to be keyed on `gw` alone, so two calls for the SAME gameweek
# collided on an identical `created_at` and the walk-back's ORDER BY could not
# tell which one was actually the newer, superseding row. Keying the offset on
# a monotonically increasing call counter instead guarantees every logged row
# gets a strictly later timestamp than the one before it, regardless of
# whether they share a gameweek.
_call_order = itertools.count()


def _log_lineup(gw: int, squad_ids: list[int], chip: str | None,
                 bank: float = 0.0, free_transfers: int = 1,
                 purchase_prices: dict | None = None) -> None:
    db = get_session()
    try:
        db.execute(
            text("""INSERT INTO decision_log
                    (gameweek, decision_type, details, projected_gain, dry_run, created_at)
                    VALUES (:gw, 'lineup', :details, 0.0, 1,
                            datetime('now', '+' || :off || ' seconds'))"""),
            {"gw": gw, "off": next(_call_order),
             "details": json.dumps({
                 "squad_ids": squad_ids,
                 "chip_considered": chip,
                 "bank": bank,
                 "free_transfers": free_transfers,
                 "purchase_prices": {str(k): v for k, v in (purchase_prices or {}).items()},
             })},
        )
        db.commit()
    finally:
        db.close()


def _state(decided_gw: int):
    return _load_squad_state(None, team_id=0, config=OPTIMISER, decided_gw=decided_gw)


@pytest.fixture(autouse=True)
def _clean_log():
    db = get_session()
    try:
        db.execute(text("DELETE FROM decision_log"))
        db.commit()
    finally:
        db.close()
    yield


def test_a_free_hit_squad_is_not_carried_forward():
    _log_lineup(2, [1, 2, 3], chip=None)
    _log_lineup(3, [90, 91, 92], chip="freehit")
    assert _state(4).squad_ids == [1, 2, 3]


def test_a_wildcard_squad_IS_carried_forward():
    """A wildcard squad genuinely persists; only a Free Hit reverts."""
    _log_lineup(2, [1, 2, 3], chip=None)
    _log_lineup(3, [90, 91, 92], chip="wildcard")
    assert _state(4).squad_ids == [90, 91, 92]


def test_an_ordinary_week_is_unaffected():
    _log_lineup(2, [1, 2, 3], chip=None)
    _log_lineup(3, [4, 5, 6], chip=None)
    assert _state(4).squad_ids == [4, 5, 6]


def test_bank_and_free_transfers_still_come_from_the_free_hit_row():
    """An earlier fix already carries these correctly across a Free Hit, so
    they must NOT be walked back with squad_ids."""
    _log_lineup(2, [1, 2, 3], chip=None, bank=9.9, free_transfers=1)
    _log_lineup(3, [90, 91, 92], chip="freehit", bank=4.4, free_transfers=2)
    state = _state(4)
    assert state.squad_ids == [1, 2, 3]
    assert state.bank == 4.4
    assert state.free_transfers == 2


def test_consecutive_free_hits_walk_back_past_all_of_them():
    _log_lineup(1, [1, 2, 3], chip=None)
    _log_lineup(2, [70, 71, 72], chip="freehit")
    _log_lineup(3, [90, 91, 92], chip="freehit")
    assert _state(4).squad_ids == [1, 2, 3]


def test_a_free_hit_with_no_earlier_squad_yields_empty_not_the_free_hit_squad():
    """Better to report no squad than to confidently report a wrong one."""
    _log_lineup(3, [90, 91, 92], chip="freehit")
    assert _state(4).squad_ids == []


def test_no_rows_at_all_is_still_a_cold_start():
    assert _state(4).squad_ids == []


def test_only_the_last_row_of_a_gameweek_counts():
    """A gameweek is decided many times before its deadline; each re-run
    appends a row. Only the last one is the decision that stands."""
    _log_lineup(2, [1, 2, 3], chip=None)
    _log_lineup(3, [40, 41, 42], chip=None)       # superseded alternative
    _log_lineup(3, [90, 91, 92], chip="freehit")  # the decision that stands
    assert _state(4).squad_ids == [1, 2, 3]


def test_a_superseded_free_hit_row_does_not_block_a_later_normal_decision():
    """The mirror case: if a gameweek's LAST decision was not a free hit, its
    squad stands even though an earlier run that week explored one."""
    _log_lineup(2, [1, 2, 3], chip=None)
    _log_lineup(3, [90, 91, 92], chip="freehit")  # superseded
    _log_lineup(3, [40, 41, 42], chip=None)       # the decision that stands
    assert _state(4).squad_ids == [40, 41, 42]


def test_the_newest_row_still_supplies_bank_and_free_transfers():
    """Unchanged: only squad_ids is subject to the per-gameweek collapse."""
    _log_lineup(2, [1, 2, 3], chip=None, bank=9.9, free_transfers=1)
    _log_lineup(3, [90, 91, 92], chip="freehit", bank=4.4, free_transfers=2)
    state = _state(4)
    assert state.squad_ids == [1, 2, 3]
    assert state.bank == 4.4
    assert state.free_transfers == 2
