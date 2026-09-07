"""Bench Boost holds for a better week inside the window (2026-09-06).

_try_bb only ever looked at the current gameweek, so there was no target to
build toward and no reason to wait. _try_tc already has dgw_visible_ahead for
the same question."""

from optimiser.bench_boost import should_hold_bench_boost


def test_holds_when_a_later_week_clears_the_margin():
    assert should_hold_bench_boost(
        current_gw=4, current_bench=20.0,
        target=(6, 25.0), margin=2.0,
    ) is True


def test_fires_when_the_later_week_is_only_marginally_better():
    """Without a margin, ordinary week-to-week noise defers the chip
    indefinitely and it expires — the failure the 2026-08-18 change to _try_bb
    existed to stop."""
    assert should_hold_bench_boost(
        current_gw=4, current_bench=20.0,
        target=(6, 21.0), margin=2.0,
    ) is False


def test_fires_when_the_current_week_is_itself_the_target():
    assert should_hold_bench_boost(
        current_gw=4, current_bench=25.0,
        target=(4, 25.0), margin=2.0,
    ) is False


def test_fires_when_there_is_no_target_at_all():
    assert should_hold_bench_boost(
        current_gw=4, current_bench=20.0, target=None, margin=2.0,
    ) is False


def test_a_zero_margin_disables_holding_for_a_tie():
    assert should_hold_bench_boost(
        current_gw=4, current_bench=20.0, target=(6, 20.0), margin=0.0,
    ) is False


def test_never_holds_for_a_week_already_past():
    assert should_hold_bench_boost(
        current_gw=6, current_bench=20.0, target=(4, 40.0), margin=2.0,
    ) is False
