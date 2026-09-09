"""The bench-boost bar (2026-09-09).

Replaces the absolute ``bench_boost_min_bench_xpts = 20.0``, which no squad
the optimiser can build was able to reach -- measured on post-A10 projections
at GW3 the carried squad benched 11.97-13.61 over GW4-8, and a squad solved
with ``bb_target_week`` to MAXIMISE the bench topped out at 15.56. The gate
never opened, so ``_panic_shrink`` alone timed the chip and the Bench Boost
fired at GW18 and GW37 every season whatever the fixtures did.

The property that failure had no way of holding, and that this file exists to
pin, is scale invariance: the bar must ask "is this week unusually good for MY
bench", not "is this week worth N xPts on whatever scale the model currently
emits". That scale moved twice in the month before this was written (A9's
training arm, A10's serve-time as-of row), which is how 20.0 got there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from optimiser import chips
from optimiser.bench_boost import bench_boost_bar

SQUAD = list(range(1, 16))
XI = list(range(1, 12))
BENCH = [12, 13, 14, 15]


def _projections(bench_pts_by_gw: dict[int, float], scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for gw, bench_pts in bench_pts_by_gw.items():
        for pid in XI:
            rows.append({"player_id": pid, "gameweek": gw, "xpts": 6.0 * scale})
        for pid in BENCH:
            rows.append({"player_id": pid, "gameweek": gw, "xpts": bench_pts * scale / 4.0})
    return pd.DataFrame(rows)


def _bar(proj, current_gw=4, ratio=1.25, floor=8.0, shrink=1.0):
    return bench_boost_bar(
        SQUAD, proj, current_gw, ratio=ratio, floor=floor, shrink=shrink
    )


# --- the bar itself ---------------------------------------------------------

def test_an_ordinary_week_does_not_clear_a_window_of_ordinary_weeks():
    """The question the chip poses. A flat window has no special week in it,
    so nothing in it should be worth a scarce per-half use."""
    bar = _bar(_projections({4: 13.0, 5: 12.0, 6: 13.1, 7: 13.3, 8: 13.6}))
    assert bar.median_bench == pytest.approx(13.1)
    assert bar.threshold == pytest.approx(16.375)
    assert max(13.0, 12.0, 13.1, 13.3, 13.6) < bar.threshold


def test_a_doubled_bench_clears_it_outright():
    """What ``_try_bb``'s comment always claimed the number expressed, and did
    not: a double gameweek roughly doubles the bench, so it arrives at ~2.0x
    the median and the chip fires without needing a DGW gate."""
    bar = _bar(_projections({4: 13.0, 5: 26.0, 6: 13.0, 7: 13.0, 8: 13.0}))
    assert 26.0 > bar.threshold
    assert 13.0 < bar.threshold


def test_the_bar_is_scale_free():
    """The defect the absolute bar had no defence against.

    Every projection tripled is the same football: the same week is the good
    one and the same weeks are ordinary. A bar in xPts silently changes its
    answer; a ratio cannot. Note the tripled flat window sits at 39 xPts,
    comfortably past the old standing bar of 20.0 -- so under that bar this
    squad would have spent the chip on a wholly unremarkable week.
    """
    weeks = {4: 13.0, 5: 12.0, 6: 13.1, 7: 13.3, 8: 13.6}
    plain = _bar(_projections(weeks))
    tripled = _bar(_projections(weeks, scale=3.0))

    assert tripled.threshold == pytest.approx(plain.threshold * 3.0)
    assert tripled.median_bench == pytest.approx(plain.median_bench * 3.0)
    assert 13.6 * 3.0 > 20.0, "precondition: the old absolute bar would have fired"


# --- the floor --------------------------------------------------------------

def test_the_floor_binds_on_a_worthless_window():
    """The one case a pure ratio gets wrong: 1.25x an already-worthless median
    is still worthless, and a blank gameweek produces exactly that."""
    bar = _bar(_projections({4: 2.0, 5: 1.0, 6: 6.0}))
    assert bar.threshold == pytest.approx(8.0)
    assert bar.ratio * bar.median_bench < bar.threshold


def test_the_floor_is_not_discounted_near_a_halfs_expiry():
    """``_panic_shrink`` exists so a real-but-marginal opportunity clears
    before evaporating. A blank gameweek is not a marginal opportunity, and a
    chip genuinely about to expire is rescued by the salvage force in
    ``recommend_chip``, which bypasses this bar entirely -- so shrinking the
    floor too would only buy the right to boost a bench with no fixtures."""
    assert _bar(_projections({4: 2.0, 5: 1.0}), shrink=0.3).threshold == pytest.approx(8.0)


def test_an_empty_window_leaves_the_floor():
    """There is no median to be unusual against, and a bar of 0.0 would fire
    the chip on nothing at all."""
    bar = _bar(pd.DataFrame(columns=["player_id", "gameweek", "xpts"]))
    assert bar.threshold == pytest.approx(8.0)
    assert bar.median_bench is None


# --- panic shrink -----------------------------------------------------------

def test_panic_shrink_pulls_the_ratio_towards_one_not_the_bar_towards_zero():
    """Shrinking 1.25 to 0.375x the median would be meaningless -- it would
    fire on any week at all. Applied to the ratio's EXCESS over 1.0 it decays
    to 1.075x, i.e. "take a slightly-above-typical week", which is the right
    posture as a half runs out."""
    proj = _projections({4: 20.0, 5: 20.0, 6: 20.0})
    assert _bar(proj, shrink=1.0).threshold == pytest.approx(25.0)
    assert _bar(proj, shrink=0.3).threshold == pytest.approx(21.5)


# --- what the reports say ---------------------------------------------------

def test_the_report_names_the_median_the_bar_came_from():
    """A ratio bar is not a constant a reader can look up, so the readiness
    reports have to quote its derivation or they quote a number from nowhere."""
    described = _bar(_projections({4: 13.0, 5: 13.1, 6: 13.3})).describe()
    assert "16.38 xPts" in described
    assert "1.25x" in described
    assert "13.10" in described


def test_the_report_says_when_the_floor_was_what_bound():
    described = _bar(_projections({4: 2.0, 5: 1.0})).describe()
    assert "8.00 xPts" in described
    assert "floor" in described


# --- through the decision ---------------------------------------------------

@pytest.fixture(autouse=True)
def _fixed_half_boundary(monkeypatch):
    chips._get_wc_half_boundary.cache_clear()
    monkeypatch.setattr(chips, "_get_wc_half_boundary", lambda season=None: 19)
    yield


def _recommend(proj, current_gw, bench_xpts):
    other = [chips.Chip.TRIPLE_CAPTAIN, chips.Chip.FREE_HIT, chips.Chip.WILDCARD]
    return chips.recommend_chip(
        current_gw=current_gw, current_squad_ids=SQUAD, projections=proj,
        players=pd.DataFrame(), available_budget=100.0, free_transfers=1,
        season=None, bench_xpts=bench_xpts,
        chips_used=[(c, current_gw - 1) for c in other], squad_age_gws=0,
    )


def test_the_chip_is_not_spent_on_an_ordinary_week_however_large_the_numbers():
    """End to end, the regression the old bar could not have caught. Tripling
    the projections puts every week past 20.0; the decision must not change."""
    weeks = {10: 13.0, 11: 12.0, 12: 13.1, 13: 13.3, 14: 13.6}
    assert _recommend(_projections(weeks), 10, 13.0).chip is None
    assert _recommend(_projections(weeks, scale=3.0), 10, 39.0).chip is None


def test_the_chip_fires_on_a_week_that_is_genuinely_better_than_the_rest():
    proj = _projections({10: 26.0, 11: 13.0, 12: 13.0, 13: 13.0, 14: 13.0})
    assert _recommend(proj, 10, 26.0).chip == chips.Chip.BENCH_BOOST


def test_persona_aggressiveness_scales_the_ratios_excess_not_the_ratio(monkeypatch):
    """``chip_aggressiveness`` multiplies how far above ordinary a week has to
    be. Applied to 1.25 directly it would demand 1.5x the median at
    aggressiveness 1.2 -- a far larger move than the same factor makes to any
    of the absolute thresholds beside it, which would silently make the shadow
    personas a much wider sweep on this axis than on the others."""
    from agent import decision_engine as de

    seen = {}
    monkeypatch.setattr(de, "_run_decision_cycle", lambda **kw: seen.update(kw) or {})
    persona = SimpleNamespace(
        id=1, risk_level=0.0, max_ownership_differential=0.5, chip_aggressiveness=1.2,
        transfer_planning_horizon_gws=5, bench_value_weight=1.0, mu_baseline=0.0,
        transfer_switching_cost=1.5, ft_terminal_value=1.0,
    )
    de.run_for_persona(persona)

    timing = seen["chip_timing"]
    assert timing.bench_boost_min_bench_ratio == pytest.approx(1.3)
    assert timing.bench_boost_min_bench_floor == pytest.approx(
        de.CHIP_TIMING.bench_boost_min_bench_floor * 1.2
    )
