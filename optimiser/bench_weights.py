"""Bench slot weights derived from the chosen XI (2026-09-06).

A bench is an ordered queue: FPL promotes the first eligible bench player when
a starter does not appear, so slot 1 is worth P(at least one starter blanks)
and slot 3 needs three simultaneous absences.

Until now those probabilities were three constants — (0.53, 0.15, 0.03) —
computed once from the live GW1 XI, whose mean start probability was 0.93, and
never recomputed. They are therefore wrong for any other XI, and wrong in the
direction that matters: a squad carrying several 0.78s needs a much fatter slot
1 than a nailed-on one, and gets the same.

`docs/bench-and-correlation-research-2026-08-18.md` §1 names deriving these per
solve as the better answer and notes the circularity — the weights depend on
the XI, which depends on the weights — which is why the caller runs a short
fixed-point pass rather than a single solve.
"""

from __future__ import annotations

import logging

from config.strategy import OptimiserConfig

logger = logging.getLogger(__name__)

DEFAULT_SLOTS = 3


def at_least_k_probabilities(p_blank: list[float], k_max: int) -> list[float]:
    """[P(>=1 blank), P(>=2), ...] from independent per-player blank chances.

    Exact Poisson-binomial by convolution — the count is at most eleven, so
    there is no reason to approximate. Independence is an assumption: a
    postponed fixture blanks a whole club's players together. That correlation
    would raise the tail and so raise slots 2 and 3, making these weights
    mildly conservative, which is the safe direction.
    """
    dist = [1.0]
    for p in p_blank:
        nxt = [0.0] * (len(dist) + 1)
        for count, mass in enumerate(dist):
            nxt[count] += mass * (1.0 - p)
            nxt[count + 1] += mass * p
        dist = nxt

    out: list[float] = []
    for k in range(1, k_max + 1):
        out.append(float(sum(dist[k:])) if k < len(dist) else 0.0)
    return out


def derive_slot_weights(
    start_probabilities: list[float], n_slots: int = DEFAULT_SLOTS
) -> tuple[float, ...]:
    """Outfield bench slot weights from the chosen XI's start probabilities.

    Returns strictly decreasing weights, which is what lets
    ``_bench_objective`` skip an explicit ordering constraint: with decreasing
    weights any assignment other than best-player-in-slot-1 is dominated.
    """
    if not start_probabilities:
        return tuple([0.0] * n_slots)
    blanks = [1.0 - float(p) for p in start_probabilities]
    return tuple(at_least_k_probabilities(blanks, n_slots))


def derive_gk_weight(keeper_start_probability: float) -> float:
    """The reserve keeper plays only if the first choice does not — one keeper,
    one chance, no queue to inherit from."""
    return max(0.0, 1.0 - float(keeper_start_probability))


def bench_slot_weight_for_week(
    slot: int, w: int, bb_target_week: int | None, cfg: OptimiserConfig
) -> float:
    """Bench slot weight for one slot in one week.

    Under a bench boost all four bench players play, so every slot is worth
    1.0 in the target week and the ordinary auto-substitution weights apply
    everywhere else. A flat 1.0 across the whole horizon would build a squad
    that is BB-optimal and mediocre in the other 37 weeks.

    Lives here, once, and is imported by BOTH optimiser/squad.py and
    optimiser/transfers.py. They must not drift: the squad build pays real
    money for a substitute who plays, and if the weekly transfer ILP valued him
    lower it would sell him the following week and charge a transfer for the
    privilege (see transfers.py:417-431). A single definition makes that
    structural rather than something a test has to notice afterwards.
    """
    if bb_target_week is not None and w == bb_target_week:
        return 1.0
    return cfg.bench_slot_weights[slot] * cfg.bench_value_weight


def bench_gk_weight_for_week(
    w: int, bb_target_week: int | None, cfg: OptimiserConfig
) -> float:
    """As above, for the reserve keeper."""
    if bb_target_week is not None and w == bb_target_week:
        return 1.0
    return cfg.bench_gk_weight * cfg.bench_value_weight
