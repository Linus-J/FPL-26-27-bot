"""A backtest that cannot solve a gameweek drops it, silently (2026-09-07).

Two defects, one cause. Every solver failure in scripts/backtest.py logs and
skips the gameweek:

  * ``SolverTimeout`` subclasses ``RuntimeError`` (optimiser/squad.py:44), so it
    lands in the same ``except RuntimeError`` handlers as a genuine
    infeasibility and gets reported as one. They call for opposite responses --
    infeasibility is a modelling bug to fix, a timeout is a wall-clock budget
    to raise.
  * The completion summary reports ``GW%d-%d`` from min/max, which cannot show
    a hole: "GW6-38" prints identically whether 33 weeks scored or 3. And
    run_backtest's headline is a SUM over the surviving rows, so every dropped
    week deflates the number the exit gate reads.
"""

import pytest

from optimiser.squad import SolverTimeout
from scripts.backtest import _drop_note, _unscored_gameweeks


def test_a_timeout_is_not_reported_as_infeasibility():
    """SolverTimeout subclasses RuntimeError purely so existing handlers keep
    catching it. That is what made it indistinguishable in the log."""
    note = _drop_note(SolverTimeout("time limit"), "infeasible")
    assert "infeasible" not in note
    assert "timed out" in note


def test_the_timeout_note_names_the_knob_that_fixes_it():
    """The response to a timeout is to raise the budget, not to debug a squad."""
    assert "solver_time_limit_seconds" in _drop_note(SolverTimeout("x"), "infeasible")


def test_an_ordinary_solver_failure_keeps_its_own_wording():
    """Each call site describes its own stage; only the timeout branch is shared."""
    assert _drop_note(RuntimeError("Infeasible"), "infeasible") == "infeasible"
    assert _drop_note(ValueError("empty frame"), "failed") == "failed"


def test_a_subclass_of_solver_timeout_still_counts_as_one():
    class Derived(SolverTimeout):
        pass

    assert "timed out" in _drop_note(Derived("x"), "infeasible")


def test_unscored_gameweeks_finds_the_hole_a_min_max_range_hides():
    """GW6-10 with GW8 dropped still prints "GW6-10"."""
    assert _unscored_gameweeks([6, 7, 9, 10], [6, 7, 8, 9, 10]) == [8]


def test_a_fully_scored_span_has_no_unscored_gameweeks():
    assert _unscored_gameweeks([6, 7, 8], [6, 7, 8]) == []


def test_every_attempted_gameweek_is_unscored_when_nothing_was_recorded():
    assert _unscored_gameweeks([], [6, 7, 8]) == [6, 7, 8]


def test_repeated_gameweeks_do_not_produce_phantom_holes():
    """run_rebuild_backtest writes one row per mu_baseline, so a gameweek
    appears several times over. Coverage is about presence, not row count."""
    assert _unscored_gameweeks([6, 6, 6, 7, 7], [6, 7]) == []


def test_a_gameweek_the_loop_never_entered_is_not_reported_as_dropped():
    """All three harnesses iterate the gameweeks present in all_stats, not
    range(start_gw, end_gw + 1). A season with no data for GW8 never attempted
    it -- calling that a dropped gameweek would be noise."""
    assert _unscored_gameweeks([6, 7, 9], [6, 7, 9]) == []


def test_a_scored_gameweek_outside_the_attempted_set_is_ignored():
    assert _unscored_gameweeks([4, 5, 6, 7], [6, 7]) == []


def test_numpy_gameweeks_compare_equal_to_python_ints():
    """df["gameweek"] arrives as numpy int64; a set of those would never match
    a set of Python ints built from available_gws if either side skipped int()."""
    import numpy as np

    assert _unscored_gameweeks(np.array([6, 8]), np.array([6, 7, 8])) == [7]


@pytest.mark.parametrize("scored", [[], [7]])
def test_the_result_is_ordered_ascending(scored):
    got = _unscored_gameweeks(scored, [9, 6, 8, 7])
    assert got == sorted(got)
