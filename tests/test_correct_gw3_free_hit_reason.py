"""scripts/correct_gw3_free_hit_reason.py

The GW3 free hit was chosen, and published, on the belief that playing the
chip would hand the continuation two free transfers. It does not. 20a6097
fixed the rule; the reason string already in ``decision_log`` kept the old
claim, and the portfolio site publishes it verbatim.

An earlier pass appended an explanatory note to every GW3 chip row rather
than restating the figures, on the grounds that they could not be recomputed.
They can: ``player_projections`` keeps every batch by ``created_at`` and
``player_state_snapshots`` keeps prices by day, so the continuation can be
re-solved exactly as it was. The control proves it -- re-solving with the
logged count of 2 returns the logged +25.44 and a total of 278.469 against
the recorded 278.464. With the correct count of 1 the same three transfers
are made and an extra -4 is paid: +21.44, margin +11.06 rather than +15.06.

So these tests cover two things. The notes from that earlier pass are
reversed, leaving the superseded runs as the unedited audit trail they were,
and the one row the site actually publishes is set to the recomputed figures.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from data.models import Base, DecisionLog
from scripts.correct_gw3_free_hit_reason import (
    CORRECTED,
    CORRECTED_GAIN,
    correct_gw3_free_hit,
    strip_correction_note,
)

ORIGINAL = (
    "beats no-chip by 15.1 xPts — free hit: 68.45 in GW3, "
    "then 2 FT(s) into a continuation worth +25.44"
)
NOTED = (
    "beats no-chip by 15.1 xPts — free hit: 68.45 in GW3, "
    "then 1 FT into a continuation worth +25.44 (logged as 2 FTs: a chip week "
    "leaves the allowance unchanged, so the continuation and the margin were "
    "both optimistic)"
)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'log.db'}")
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _chip_row(session, gameweek: int, reason: str, gain: float = 15.058381098233099) -> None:
    session.add(
        DecisionLog(
            gameweek=gameweek,
            decision_type="chip",
            details=json.dumps({"chip": "freehit", "reason": reason}),
            projected_gain=gain,
        )
    )
    session.commit()


def test_the_published_reason_carries_the_recomputed_margin():
    assert CORRECTED == "beats no-chip by 11.1 xPts — free hit: 68.45 xPts"


def test_the_published_reason_never_mentions_free_transfers():
    """The whole point of recomputing was to stop explaining the bug in the
    one line a reader sees."""
    assert "FT" not in CORRECTED
    assert "logged as" not in CORRECTED


def test_a_noted_reason_is_returned_to_what_was_logged():
    assert strip_correction_note(NOTED) == ORIGINAL


def test_an_unnoted_reason_is_left_alone():
    assert strip_correction_note(ORIGINAL) is None
    assert strip_correction_note("BGW free hit gain 14.2 xPts") is None


def test_only_the_newest_gw3_chip_row_gets_the_recomputed_figures(session):
    """The replay reconstructs the 14:31 run, so only that run's row can
    honestly carry its numbers. The runs it superseded are restored to what
    they said at the time."""
    _chip_row(session, 3, NOTED)
    _chip_row(session, 3, NOTED)

    correct_gw3_free_hit(session)

    reasons = [
        json.loads(r[0])["reason"]
        for r in session.execute(text("SELECT details FROM decision_log ORDER BY id"))
    ]
    assert reasons == [ORIGINAL, CORRECTED]


def test_other_gameweeks_are_untouched(session):
    _chip_row(session, 2, "TC captain xPts 7.8")
    _chip_row(session, 3, NOTED)

    correct_gw3_free_hit(session)

    reasons = [
        json.loads(r[0])["reason"]
        for r in session.execute(text("SELECT details FROM decision_log ORDER BY id"))
    ]
    assert reasons == ["TC captain xPts 7.8", CORRECTED]


def test_running_it_twice_changes_nothing(session):
    _chip_row(session, 3, NOTED)
    _chip_row(session, 3, NOTED)

    query = text("SELECT details, projected_gain FROM decision_log ORDER BY id")
    correct_gw3_free_hit(session)
    first = session.execute(query).fetchall()
    correct_gw3_free_hit(session)
    second = session.execute(query).fetchall()

    assert first == second


def test_it_runs_on_a_log_that_was_never_noted(session):
    """A fresh clone of the database, or one restored from before the earlier
    pass, still ends up with the published row corrected."""
    _chip_row(session, 3, ORIGINAL)

    correct_gw3_free_hit(session)

    details = json.loads(
        session.execute(text("SELECT details FROM decision_log")).fetchone()[0]
    )
    assert details["reason"] == CORRECTED
    assert details["chip"] == "freehit"


def test_the_recomputed_gain_is_the_recorded_one_less_the_extra_hit():
    """The replay's two arms make the same three transfers and differ by
    exactly -4, so the honest correction takes that delta off the figure the
    run actually recorded rather than substituting the replay's own total --
    which lands 0.005 away and would import drift the correction is not
    about."""
    assert CORRECTED_GAIN == 15.058381098233099 - 4.0
    assert f"{CORRECTED_GAIN:.1f}" == "11.1", "must agree with the published reason"


def test_the_published_row_carries_the_recomputed_gain(session):
    """The site renders every event in the log as a number, so a corrected
    reason beside the optimistic 15.06 would publish a figure its own
    sentence contradicts."""
    _chip_row(session, 3, ORIGINAL)

    correct_gw3_free_hit(session)

    gain = session.execute(text("SELECT projected_gain FROM decision_log")).fetchone()[0]
    assert gain == CORRECTED_GAIN


def test_a_superseded_row_keeps_the_gain_it_recorded(session):
    """Only the published run was replayed. The runs it superseded are the
    audit trail of what was believed at the time."""
    _chip_row(session, 3, ORIGINAL, gain=15.910439841825735)
    _chip_row(session, 3, ORIGINAL)

    correct_gw3_free_hit(session)

    gains = [
        r[0] for r in session.execute(
            text("SELECT projected_gain FROM decision_log ORDER BY id")
        )
    ]
    assert gains == [15.910439841825735, CORRECTED_GAIN]
