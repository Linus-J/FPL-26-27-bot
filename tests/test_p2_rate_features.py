"""Phase-2 P2 — rate-feature contract (defect D4).

Model features must be rates, not season-cumulative volume, and identical on the
train/serve paths. These lock that the banned cumulative/proxy columns are gone
from every component's FEATURE_COLS and can't creep back.

points_model.py's own guard was removed 2026-08-01 along with the file itself
(confirmed dead in the live and backtest paths -- superseded by
projection/assemble.py's P10 MC assembly). minutes_model.py is still live
(assemble.py's 3-way minutes-band predictor), so its guard stays.
"""

from __future__ import annotations

import pytest

from projection import minutes_model
from projection.features import CUMULATIVE_BANNED_FEATURES, assert_rate_only


def test_minutes_model_feature_cols_are_rate_only():
    assert CUMULATIVE_BANNED_FEATURES.isdisjoint(minutes_model.FEATURE_COLS)


def test_rolling_rate_features_still_present():
    # the rate signal that replaced cumulative ICT/form must still be there
    assert "avg_minutes_5gw" in minutes_model.FEATURE_COLS


def test_assert_rate_only_guard():
    assert_rate_only(["avg_xg_5gw", "now_cost", "pos_MID"])   # clean → no raise
    for banned in ("ict_index", "influence", "creativity", "threat", "form"):
        with pytest.raises(ValueError, match="banned"):
            assert_rate_only(["avg_xg_5gw", banned])


# --- rolling window must not silently discard the newest gameweek -------------
# 2026-08-18, engine review §20.

def _history(n_gws: int, cbit_by_gw: dict[int, int],
             xg_by_gw: dict[int, float] | None = None):
    import pandas as pd

    xg_by_gw = xg_by_gw or {}
    hist = pd.DataFrame([
        {"player_id": 1, "gameweek": gw, "season": "2026-27", "position": "DEF",
         "team_id_season": 1, "opponent_team_id": 2, "was_home": True,
         "minutes": 90, "total_points": 6,
         "xg": float(xg_by_gw.get(gw, 0.0)), "npxg": 0.0, "xa": 0.0,
         "key_passes": 0.0, "yellow_cards": 0, "red_cards": 0}
        for gw in range(1, n_gws + 1)
    ])
    defcon = pd.DataFrame([
        {"player_id": 1, "gameweek": gw, "season": "2026-27",
         "clearances": cbit_by_gw[gw], "blocks": 0, "interceptions": 0,
         "tackles": 0, "recoveries": 0, "dribbles": 0}
        for gw in range(1, n_gws + 1)
    ])
    return hist, defcon


def test_rolling_rates_use_every_played_gameweek():
    """The rolling build carried a ``shift(1)`` inherited from
    ``points_model._build_features``, where the frame legitimately contains the
    row being predicted so shifting is the only thing preventing a leak. Here
    the frame is already strictly prior to the target gameweek, so the shift
    was guarding against a leak truncation had already prevented — and it threw
    away the most recent, most informative gameweek every single week.

    Distinct CBIT per gameweek, so the resulting rate says unambiguously which
    gameweeks it was built from.
    """
    from projection.assemble import _build_rolling_features

    cbit = {1: 10, 2: 20, 3: 30, 4: 40}
    for n in (2, 3, 4):
        hist, defcon = _history(n, cbit)
        rate = _build_rolling_features(hist, defcon).loc[1, "defcon_rate"]
        expected_all = sum(cbit[g] for g in range(1, n + 1)) / n
        expected_dropping_last = sum(cbit[g] for g in range(1, n)) / (n - 1)
        assert rate == pytest.approx(expected_all), (
            f"{n} gameweeks: got {rate}, all-gameweeks mean is {expected_all}, "
            f"dropping the newest would give {expected_dropping_last}"
        )


def test_rolling_rates_are_non_zero_with_a_single_played_gameweek():
    """The GW2 case, and the reason this mattered most.

    ``shift(1)`` on a one-row group is NaN, and the ``fillna(0.0)`` turned that
    into a confident zero — so at the FIRST in-season decision of the season
    every rate was 0: goal_weight, assist_weight, defcon_rate, key_pass_rate,
    dribble_rate, cards. Attacking returns went unattributed, DefCon could not
    reach its threshold, and projections collapsed to appearance points plus
    clean sheets and saves.
    """
    from projection.assemble import _build_rolling_features

    hist, defcon = _history(1, {1: 14})
    row = _build_rolling_features(hist, defcon).loc[1]
    assert row["defcon_rate"] == pytest.approx(14.0)


def test_rolling_rates_never_see_the_target_gameweek():
    """The leak guarantee the shift used to provide is now owned by the
    function itself, via ``target_gw`` — a caller that passes an untruncated
    frame must still not have the target gameweek folded into its own rate."""
    from projection.assemble import _build_rolling_features

    # GW3 is a huge outlier; projecting GW3 must not be able to see it.
    hist, defcon = _history(3, {1: 10, 2: 10, 3: 1000})
    rate = _build_rolling_features(hist, defcon, target_gw=3).loc[1, "defcon_rate"]
    assert rate == pytest.approx(10.0)


# --- prior-season blending (engine review §20 follow-up) ---------------------

def _prior(cbit_per_match: float, xg_per_match: float = 0.0):
    import pandas as pd

    return pd.DataFrame(
        {"cbit": [cbit_per_match], "cbirt": [cbit_per_match],
         "xg": [xg_per_match]}, index=[1]
    ).rename_axis("player_id")


# These exercise the blend through ``xg``/``goal_weight`` rather than the
# defcon rates they originally used. The mechanism is unchanged; what changed
# (2026-09-11) is that the strength is now calibrated per rate, and the
# defensive-action volumes calibrated to k=0 — so ``defcon_rate`` is
# deliberately no longer blended and can no longer demonstrate blending. See
# tests/test_prior_season_blend.py, and the test directly below for the
# pass-through that replaced it.
_XG_BLEND_GWS = 20.0


def test_early_season_rate_leans_on_last_season():
    """At GW2 a rate rests on ONE match. The engine's implicit answer to "what
    do I know about this player" was whatever that single match said — and,
    before §20, exactly zero — while a whole prior season of real per-match
    rates sat unused.

    Weight is by sample size: with one played gameweek and a prior worth
    ``_XG_BLEND_GWS``, the current season gets 1/(1+k).
    """
    from projection.assemble import _build_rolling_features

    hist, defcon = _history(1, {1: 0}, xg_by_gw={1: 20.0})
    blended = _build_rolling_features(
        hist, defcon, prior_rates=_prior(0.0, xg_per_match=4.0)
    ).loc[1, "goal_weight"]
    w = 1.0 / (1.0 + _XG_BLEND_GWS)
    assert blended == pytest.approx(w * 20.0 + (1 - w) * 4.0)


def test_defensive_volume_is_left_on_its_current_season_rate():
    """The k=0 half of the calibration, at the level the engine actually runs:
    a prior-season CBIT of 4 must not pull a current-season 20 down at all."""
    from projection.assemble import _build_rolling_features

    hist, defcon = _history(1, {1: 20})
    rate = _build_rolling_features(
        hist, defcon, prior_rates=_prior(4.0)
    ).loc[1, "defcon_rate"]
    assert rate == pytest.approx(20.0)


def test_the_current_season_takes_over_as_it_accumulates():
    """The prior must fade, not linger — by the time a real sample exists it
    should be doing almost nothing."""
    from projection.assemble import _build_rolling_features

    weights = []
    for n in (1, 2, 3, 4):
        hist, defcon = _history(n, dict.fromkeys(range(1, n + 1), 0),
                                xg_by_gw=dict.fromkeys(range(1, n + 1), 20.0))
        rate = _build_rolling_features(
            hist, defcon, prior_rates=_prior(0.0, xg_per_match=0.0)
        ).loc[1, "goal_weight"]
        weights.append(rate / 20.0)          # share of the blend the season holds
    assert weights == sorted(weights), "current-season weight must be monotonic"
    assert weights[0] == pytest.approx(1 / (1 + _XG_BLEND_GWS))
    assert weights[-1] == pytest.approx(4 / (4 + _XG_BLEND_GWS))


def test_a_player_with_no_prior_season_keeps_their_own_rate():
    """New signings and promoted-club players have no prior row. They must be
    left alone rather than blended toward a NaN or a league default."""
    import pandas as pd

    from projection.assemble import _build_rolling_features

    hist, defcon = _history(1, {1: 20})
    empty_prior = pd.DataFrame(columns=["cbit", "cbirt"]).rename_axis("player_id")
    rate = _build_rolling_features(hist, defcon, prior_rates=empty_prior).loc[1, "defcon_rate"]
    assert rate == pytest.approx(20.0)


# --- enrichment must not be blanked for an unplayed gameweek ------------------
# 2026-09-11. ``load_player_enrichment`` selects ``FROM player_gw_stats``, so it
# structurally cannot contain a row for a gameweek that has not been played.
# ``predict_minutes_bands``'s as-of row lives at exactly that gameweek, so its
# enrichment merge missed and every column fell back to its 0.0 default.
#
# Measured on the live 2026-27 database at GW4: the set-piece block and the
# transfer-momentum block were zero for all 615 served players, against
# training rows where 21% were set-piece takers and momentum was signed and
# non-zero. Summed served P(60+) was 155.2 against a realised 210.3 starters
# per gameweek, and the league-wide maximum was 0.811 -- no player, however
# nailed, could be predicted to start. Carrying the two blocks forward moved
# the sum to 218.4 and the maximum to 0.972.

def _enrichment(rows):
    import pandas as pd

    return pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "season": "2026-27",
         "is_penalty_taker": ipt, "penalty_xg_per_game": 0.0,
         "is_set_piece_taker": ispt, "key_passes_per_game": kpg,
         "injury_severity": 0.0, "price_momentum": pm, "transfer_velocity": 0.0}
        for pid, gw, ipt, ispt, kpg, pm in rows
    ])

