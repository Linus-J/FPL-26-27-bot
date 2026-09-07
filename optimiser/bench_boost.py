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

import dataclasses
import logging

import pandas as pd

from config.strategy import OptimiserConfig

logger = logging.getLogger(__name__)


def bb_ready_config(base: OptimiserConfig, target_gw: int) -> OptimiserConfig:
    """`base` with the bench valued at full weight for the target week.

    Under a bench boost every bench player actually plays, so weight 1.0 is a
    fact about the rules rather than a risk preference — which is why this
    overrides `bench_value_weight` outright. A persona running at 0.5 must not
    silently end up half as BB-ready as one at 1.0; that would be a risk
    setting leaking into a rules question.
    """
    return dataclasses.replace(base, bench_value_weight=1.0)


def bench_xpts_by_gameweek(
    squad_ids: list[int], projections: pd.DataFrame, current_gw: int
) -> dict[int, float]:
    """Total xPts of the four bench players, per gameweek from `current_gw` on.

    The bench is defined per gameweek as the four lowest-scoring squad members
    that week — which is what the auto-substitution order would make it, and
    what a bench boost would actually play.
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
    projections a few weeks out are mostly strength-model output: without one,
    ordinary noise would defer the chip week after week until it expired.
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
