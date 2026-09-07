"""Bench-boost readiness (2026-09-06).

Absorbs §2 of the superseded
`docs/superpowers/specs/2026-08-20-bench-boost-aware-squad-construction-design.md`.

Two things from that design are deliberately NOT here:

- **Cold-start BB building**, measured on 2026-08-20 at net -15.36 (+3.05 in
  the chip week for -18.41 in normal weeks). The binding constraint is the
  budget against fixed position counts, not the bench weighting. That
  measurement establishes "no BB edge at cold start", not "BB-building never
  pays" — re-run it against a genuine double gameweek before revisiting.
- **Automatic selection** between the unconstrained and BB-ready squads. Both
  get built and the pivot price gets reported; a human chooses. An option-value
  term that decided would need calibration this project has refused elsewhere.

Tier 2 (the projection-free long scan to the half boundary) is advisory only
and must never be read by a squad or chip decision.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def bb_horizon_and_index(next_gw: int, target_gw: int) -> tuple[int, int]:
    """The horizon and target index to hand ``optimise_squad`` for a boost week.

    ``optimise_squad`` slices its weeks as ``sorted(unique gameweeks)[:horizon]``
    from a frame starting at ``next_gw``, so the horizon must REACH the target
    week and ``bb_target_week`` is the target's index within it.

    Optimising over the whole span rather than the target week alone is what
    the design always wanted: ordinary auto-substitution weights in every
    ordinary week, full weight only in the week the chip fires. A squad built
    for the target week in isolation would be boost-optimal and mediocre in
    every other week.

    Replaces `bb_ready_config` (2026-09-07), which was a no-op: it set
    `bench_value_weight` to 1.0, which is already the default, and that field
    is a MULTIPLIER on `bench_slot_weights` (0.53, 0.15, 0.03) rather than a
    replacement for them — so it could never produce the full-weight bench a
    boost actually plays. `bb_target_week` is the switch that does, and it is
    threaded into both integer programs.
    """
    if target_gw < next_gw:
        raise ValueError(f"target gameweek {target_gw} is before {next_gw}")
    return target_gw - next_gw + 1, target_gw - next_gw


def bench_xpts_by_gameweek(
    squad_ids: list[int], projections: pd.DataFrame, current_gw: int
) -> dict[int, float]:
    """Total xPts of the four bench players, per gameweek from `current_gw` on.

    The bench is defined per gameweek as the four lowest-scoring squad members
    that week. That is a LOWER BOUND on what auto-substitution would actually
    field, not a reproduction of it: it ignores FPL's formation and
    goalkeeper rules, so it can name a "bench" holding both keepers, which no
    legal XI would leave there. The error is one-directional and therefore
    safe — a legal bench can only swap one of these players for a
    higher-scoring one, so this figure understates bench value and the chip
    fires less often than it should, never more.
    """
    if projections.empty:
        return {}
    frame = projections[
        projections["player_id"].isin(squad_ids)
        & (projections["gameweek"] >= current_gw)
    ]
    totals: dict[int, float] = {}
    for gw, group in frame.groupby("gameweek"):
        bench = group.nsmallest(4, "xpts")
        totals[int(gw)] = float(bench["xpts"].sum())
    return totals


def select_bb_target_gw(
    squad_ids: list[int],
    projections: pd.DataFrame,
    current_gw: int,
    threshold: float,
) -> tuple[int, float] | None:
    """The best bench-boost week inside the persisted projection window.

    Tier 1: the only tier that feeds a decision. Returns (gameweek, bench xPts)
    or None when nothing in the window clears `threshold`. The window is
    whatever the projection frame holds — `projection_horizon_gws` is a hard
    ceiling (`assert_horizons_consistent`), so this cannot see past it, which is
    exactly why the design has a second, advisory tier.
    """
    totals = bench_xpts_by_gameweek(squad_ids, projections, current_gw)
    if not totals:
        return None
    best_gw = max(totals, key=lambda gw: totals[gw])
    if totals[best_gw] < threshold:
        logger.info(
            "No BB target: best bench in window is GW%d at %.2f xPts (threshold %.2f)",
            best_gw, totals[best_gw], threshold,
        )
        return None
    return best_gw, totals[best_gw]


def should_hold_bench_boost(
    current_gw: int,
    current_bench: float,
    target: tuple[int, float] | None,
    margin: float,
) -> bool:
    """Should the chip be held back for a better week inside the window?

    True only when the target is strictly ahead of `current_gw` and beats this
    week by more than `margin`. A margin rather than a bare argmax because
    projections a few weeks out are mostly strength-model output, and without
    one ordinary noise would defer the chip on nothing but sampling error.

    The margin is NOT what stops the chip being deferred all the way to expiry
    (2026-09-07) — `must_play_a_chip_now` is, via the salvage force in
    `optimiser/chips.py`. A sliding window whose far end always looks 2+ points
    better would defer every single week regardless of how large this margin
    were, right up until salvage forced it through. Anyone tuning `margin`
    expecting it to do that job will be tuning the wrong number.
    """
    if target is None:
        return False
    target_gw, target_bench = target
    if target_gw <= current_gw:
        return False
    return (target_bench - current_bench) > margin


def pivot_price(unconstrained_ids: list[int], bb_ready_ids: list[int]) -> dict:
    """What it costs to move from the unconstrained squad to the BB-ready one.

    This is the deliverable of the whole bench-boost half. The trap it reports
    on is arriving at a gameweek with excellent BB fixtures holding a fodder
    bench, no bank and few free transfers -- and the price of escaping it is
    denominated in transfers, so that is the unit it is quoted in.
    """
    out = sorted(set(unconstrained_ids) - set(bb_ready_ids))
    incoming = sorted(set(bb_ready_ids) - set(unconstrained_ids))
    return {"transfers_required": len(out), "out": out, "in": incoming}
