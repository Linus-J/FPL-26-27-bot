"""scripts/correct_chip_week_free_transfers.py

The GW3 free hit was chosen, and published, on the belief that playing the
chip would hand the continuation two free transfers. It does not: FPL keeps
your saved transfers across a chip but the chip consumes that gameweek's own
allotment, so the allowance is unchanged. 20a6097 fixed the rule; what it
could not fix is the reason string already sitting in ``decision_log``, which
the portfolio site publishes verbatim.

The old bug added exactly +1 on a chip week, so the correction is
deterministic rather than a guess: whatever count was logged, the truth is
one lower. The two figures either side of it -- the continuation's value and
the margin over no-chip -- were solved with the extra transfer in hand and
cannot be recomputed faithfully now, so they are kept and labelled rather
than quietly restated.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from data.models import Base, DecisionLog
from scripts.correct_chip_week_free_transfers import correct_chip_reasons, correct_reason

GW3 = (
    "beats no-chip by 15.1 xPts — free hit: 68.45 in GW3, "
    "then 2 FT(s) into a continuation worth +25.44"
)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'log.db'}")
    Base.metadata.create_all(bind=engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _chip_row(session, gameweek: int, reason: str) -> None:
    session.add(
        DecisionLog(
            gameweek=gameweek,
            decision_type="chip",
            details=json.dumps({"chip": "freehit", "reason": reason}),
            projected_gain=15.1,
        )
    )
    session.commit()


def test_the_logged_count_is_reduced_by_the_transfer_the_chip_never_granted():
    corrected = correct_reason(GW3)

    assert "1 FT into a continuation" in corrected
    assert "2 FT(s)" not in corrected


def test_a_single_transfer_is_not_described_in_the_plural():
    """``N FT(s)`` was written to dodge the plural; at one it stops being a
    dodge and starts being wrong."""
    assert "1 FT(s)" not in correct_reason(GW3)


def test_the_figures_that_were_solved_with_the_extra_transfer_are_labelled():
    corrected = correct_reason(GW3)

    assert "+25.44" in corrected, "the continuation value is kept, not restated"
    assert "15.1" in corrected, "the margin over no-chip is kept, not restated"
    assert "logged as 2 FTs" in corrected


def test_the_older_phrasing_is_corrected_too():
    """Rows from earlier GW3 runs say "into continuation worth", without the
    "then" and without the article -- the wording changed mid-week."""
    corrected = correct_reason(
        "beats no-chip by 15.9 xPts — free hit: 69.26 in GW3, "
        "2 FT(s) into continuation worth +21.48"
    )

    assert corrected is not None
    assert "1 FT into continuation" in corrected


def test_a_reason_with_no_continuation_clause_is_left_alone():
    assert correct_reason("BGW free hit gain 14.2 xPts") is None
    assert correct_reason("TC captain xPts 7.8") is None


def test_an_already_corrected_reason_is_left_alone():
    """Idempotent: the script can be re-run without stacking notes."""
    assert correct_reason(correct_reason(GW3)) is None


def test_correcting_the_log_rewrites_only_the_affected_rows(session):
    _chip_row(session, 3, GW3)
    _chip_row(session, 2, "TC captain xPts 7.8")

    assert correct_chip_reasons(session) == 1

    reasons = [
        json.loads(r[0])["reason"]
        for r in session.execute(
            text("SELECT details FROM decision_log ORDER BY gameweek")
        )
    ]
    assert reasons[0] == "TC captain xPts 7.8"
    assert "1 FT into a continuation" in reasons[1]


def test_correcting_the_log_twice_changes_nothing_the_second_time(session):
    _chip_row(session, 3, GW3)

    assert correct_chip_reasons(session) == 1
    assert correct_chip_reasons(session) == 0


def test_the_chip_itself_is_untouched(session):
    """Only the prose was wrong. The free hit really was played at GW3 and
    the decision still stands -- with one transfer the continuation is worth
    less, not negative."""
    _chip_row(session, 3, GW3)

    correct_chip_reasons(session)

    details = json.loads(
        session.execute(text("SELECT details FROM decision_log")).fetchone()[0]
    )
    assert details["chip"] == "freehit"
