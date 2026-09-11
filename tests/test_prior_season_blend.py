"""How hard each rate shrinks toward the prior season (2026-09-11).

``_blend_toward_prior`` shrinks a current-season per-match rate toward the same
player's prior-season rate with weight ``n / (n + k)``. ``k`` shipped as a
single untuned 3.0 for every rate. Walk-forward over 2023-24, 2024-25 and
2025-26 -- predicting each played match's realised value from the rate as-of
the gameweek before -- says one constant is the wrong shape: the attacking and
card rates want far MORE shrinkage than 3.0, and the defensive-action volumes
want none at all.

Pooled MSE, rows that have a prior, against the shipped k=3.0:

    xg            k=20  0.05647  (3.0 -> 0.05952, +5.1%)
    xa            k=40  0.03376  (3.0 -> 0.03566, +5.3%)   k=20 within 0.2%
    npxg          k=20  0.04962  (3.0 -> 0.05239, +5.3%)
    key_passes    k=20  1.03574  (3.0 -> 1.08193, +4.3%)
    yellow_cards  k=40  0.11875  (3.0 -> 0.12819, +7.4%)
    red_cards     k=40  0.00467  (3.0 -> 0.00491, +4.9%)
    cbit          k=0   3.43792  (3.0 -> 3.75869, +8.5%)
    cbirt         k=0   7.41425  (3.0 -> 8.41006, +11.8%)
    dribbles      k=2   0.26527  (3.0 -> 0.26597, +0.3%)   flat; left at 3.0

Per season the attacking optima are stable -- xg picks 20/40/20 and
key_passes 20/20/20, and k=20 is within 0.6% of each season's own best. The
cbit/cbirt result rests on 2025-26 ALONE: the defcon event feed is not
backfilled before it, so those columns are identically zero in the two earlier
seasons and their MSE is degenerately 0. Within 2025-26 the direction is not
marginal (k=0 9.17 against k=20 14.63), and it is what you would expect of a
role-and-tactics volume stat that does not survive a transfer or a new
manager, but it is one season of evidence and should be revisited when the
feed is backfilled.

The concrete case that prompted this: De Cuyper carried a 3-game rolling
``goal_weight`` of 0.474 built almost entirely on one 1.38-xG match, against a
2025-26 rate of 0.058. At k=3.0 and n=3 the blend returned 0.266 -- still 4.5x
his prior-season rate -- and he was captained on it. At k=20.0 it returns
0.112.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projection.assemble import (
    _PRIOR_SEASON_BLEND_GWS,
    _PRIOR_SEASON_BLEND_GWS_BY_RATE,
    _blend_toward_prior,
)

RATE_COLS = {"xg": "goal_weight", "cbit": "cbit_rate", "dribbles": "dribble_rate"}


def _frames(current: dict[str, float], prior: dict[str, float], played: int):
    last = pd.DataFrame([{RATE_COLS[k]: v for k, v in current.items()}], index=[1])
    prior_rates = pd.DataFrame([prior], index=[1])
    return last, pd.Series({1: float(played)}), prior_rates


def test_attacking_rate_shrinks_at_the_calibrated_strength():
    """De Cuyper's actual numbers: n=3, rolling 0.474, prior 0.058."""
    last, played, prior = _frames(
        {"xg": 0.474, "cbit": 0.0, "dribbles": 0.0},
        {"xg": 0.058, "cbit": 0.0, "dribbles": 0.0},
        played=3,
    )
    out = _blend_toward_prior(last, played, prior, RATE_COLS)

    w = 3.0 / (3.0 + 20.0)
    assert out["goal_weight"].iloc[0] == pytest.approx(w * 0.474 + (1 - w) * 0.058)
    # and emphatically not what the old single constant gave
    assert out["goal_weight"].iloc[0] < 0.15


def test_defensive_volume_is_not_shrunk_toward_the_prior_season():
    last, played, prior = _frames(
        {"xg": 0.0, "cbit": 12.0, "dribbles": 0.0},
        {"xg": 0.0, "cbit": 2.0, "dribbles": 0.0},
        played=3,
    )
    out = _blend_toward_prior(last, played, prior, RATE_COLS)

    assert out["cbit_rate"].iloc[0] == pytest.approx(12.0)


def test_an_uncalibrated_rate_falls_back_to_the_shipped_constant():
    last, played, prior = _frames(
        {"xg": 0.0, "cbit": 0.0, "dribbles": 4.0},
        {"xg": 0.0, "cbit": 0.0, "dribbles": 1.0},
        played=3,
    )
    out = _blend_toward_prior(last, played, prior, {"dribbles": "dribble_rate"})

    w = 3.0 / (3.0 + _PRIOR_SEASON_BLEND_GWS)
    assert out["dribble_rate"].iloc[0] == pytest.approx(w * 4.0 + (1 - w) * 1.0)


def test_an_explicit_scalar_still_overrides_every_rate():
    last, played, prior = _frames(
        {"xg": 0.474, "cbit": 12.0, "dribbles": 0.0},
        {"xg": 0.058, "cbit": 2.0, "dribbles": 0.0},
        played=3,
    )
    out = _blend_toward_prior(last, played, prior, RATE_COLS, blend_gws=1.0)

    w = 3.0 / 4.0
    assert out["goal_weight"].iloc[0] == pytest.approx(w * 0.474 + (1 - w) * 0.058)
    assert out["cbit_rate"].iloc[0] == pytest.approx(w * 12.0 + (1 - w) * 2.0)


def test_a_player_with_no_prior_row_keeps_their_current_rate():
    last = pd.DataFrame([{"goal_weight": 0.474}], index=[1])
    prior = pd.DataFrame([{"xg": 0.058}], index=[99])
    out = _blend_toward_prior(last, pd.Series({1: 3.0}), prior, {"xg": "goal_weight"})

    assert out["goal_weight"].iloc[0] == pytest.approx(0.474)


def test_every_calibrated_rate_is_a_real_rate_source():
    from projection.assemble import _PRIOR_RATE_SOURCES

    assert set(_PRIOR_SEASON_BLEND_GWS_BY_RATE).issubset(_PRIOR_RATE_SOURCES)
