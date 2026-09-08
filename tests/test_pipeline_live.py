"""P3-0: live-serving projections via the P10 MC assembly.

Covers the two new live-only loaders (fixture context + match odds — the
backtest path gets these from played player_gw_stats rows, which don't
exist yet for a live, unplayed fixture) and run_projections's cold-start
fallback. The full assemble.py MC path itself is already covered by
tests/test_assemble.py and the live backtest harness; these tests are
about the live-specific plumbing around it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Fixture, FixtureOdds, Gameweek, Player, Team
from projection import assemble, pipeline


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'pipeline.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(pipeline, "get_session", lambda: Local())
    monkeypatch.setattr(assemble, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _team(id_, name):
    return Team(id=id_, name=name, short_name=name[:3].upper())


def _player(id_, team_id, position="MID"):
    return Player(
        id=id_, fpl_id=id_, code=id_, first_name="P", second_name=str(id_),
        web_name=f"P{id_}", team_id=team_id, position=position, now_cost=5.0,
    )


def test_build_live_fixture_context_resolves_opponent_and_home_away(session):
    session.add_all([_team(1, "Home"), _team(2, "Away")])
    session.add_all([_player(10, team_id=1), _player(20, team_id=2)])
    session.add(Fixture(id=1, fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
    session.commit()

    out = pipeline._build_live_fixture_context("2026-27", [1])
    home_row = out[out["player_id"] == 10].iloc[0]
    away_row = out[out["player_id"] == 20].iloc[0]
    assert bool(home_row["was_home"]) is True
    assert home_row["opponent_team_id"] == 2
    assert bool(away_row["was_home"]) is False
    assert away_row["opponent_team_id"] == 1


def test_build_live_fixture_context_empty_gws_returns_empty_shape(session):
    out = pipeline._build_live_fixture_context("2026-27", [])
    assert out.empty
    assert list(out.columns) == [
        "player_id", "gameweek", "team_id_season", "opponent_team_id", "was_home",
    ]


def test_build_live_fixture_context_no_fixture_for_gw_is_empty(session):
    session.add_all([_team(1, "Home"), _team(2, "Away")])
    session.add(_player(10, team_id=1))
    session.commit()
    out = pipeline._build_live_fixture_context("2026-27", [5])
    assert out.empty


def test_load_live_match_odds_uses_latest_fetch_before_deadline(session):
    session.add_all([_team(1, "Home"), _team(2, "Away")])
    session.add(Gameweek(id=1, season="2026-27", name="GW1", deadline_time=datetime(2026, 8, 15)))
    session.add(Fixture(id=1, fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
    session.commit()
    # a stale early fetch, a good pre-deadline fetch, and a leaky post-deadline fetch
    session.add(FixtureOdds(
        fixture_id=1, home_win_prob=0.40, draw_prob=0.30, away_win_prob=0.30,
        over25_prob=0.50, fetched_at=datetime(2026, 8, 10),
    ))
    session.add(FixtureOdds(
        fixture_id=1, home_win_prob=0.55, draw_prob=0.25, away_win_prob=0.20,
        over25_prob=0.60, fetched_at=datetime(2026, 8, 14),
    ))
    session.add(FixtureOdds(
        fixture_id=1, home_win_prob=0.99, draw_prob=0.005, away_win_prob=0.005,
        over25_prob=0.99, fetched_at=datetime(2026, 8, 15) + timedelta(minutes=1),
    ))
    session.commit()

    out = pipeline._load_live_match_odds("2026-27", [1])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["home_win_prob"] == pytest.approx(0.55)  # the latest BEFORE the deadline


def test_load_live_match_odds_empty_gws_returns_empty_shape(session):
    out = pipeline._load_live_match_odds("2026-27", [])
    assert out.empty
    assert "home_win_prob" in out.columns


# --- _apply_injury_severity_discount (2026-07-30) --------------------------
# Real bug found: injury_parser.py has parsed FPL's free-text news into
# players.injury_severity since it was written, but nothing downstream ever
# read the column -- a fully-wired, fully-dead signal. LIVE-ONLY (never
# wired into minutes_model.py's shared training/backtest pipeline, which
# would leak today's news onto every historical row).

def test_injury_severity_discount_leaves_healthy_players_unchanged():
    players = pd.DataFrame({"id": [1], "injury_severity": [0]})
    projections = pd.DataFrame({
        "player_id": [1], "gameweek": [10],
        "xpts": [8.0], "xpts_mean": [8.0], "xpts_var": [2.0], "start_probability": [0.9],
    })
    out = pipeline._apply_injury_severity_discount(projections.copy(), players)
    assert out.loc[0, "xpts"] == pytest.approx(8.0)
    assert out.loc[0, "start_probability"] == pytest.approx(0.9)


def test_injury_severity_discount_scales_by_severity_tier():
    players = pd.DataFrame({"id": [1, 2, 3], "injury_severity": [1, 2, 3]})
    projections = pd.DataFrame({
        "player_id": [1, 2, 3], "gameweek": [10, 10, 10],
        "xpts": [8.0, 8.0, 8.0], "xpts_mean": [8.0, 8.0, 8.0],
        "xpts_var": [2.0, 2.0, 2.0], "start_probability": [0.9, 0.9, 0.9],
    })
    out = pipeline._apply_injury_severity_discount(projections.copy(), players)
    assert out.loc[0, "xpts"] == pytest.approx(8.0 * 0.7)
    assert out.loc[1, "xpts"] == pytest.approx(8.0 * 0.35)
    assert out.loc[2, "xpts"] == pytest.approx(8.0 * 0.05)
    # monotonically decreasing in severity
    assert out.loc[0, "xpts"] > out.loc[1, "xpts"] > out.loc[2, "xpts"]


def test_injury_severity_discount_missing_column_is_a_noop():
    players = pd.DataFrame({"id": [1]})  # no injury_severity column at all
    projections = pd.DataFrame({"player_id": [1], "gameweek": [10], "xpts": [8.0]})
    out = pipeline._apply_injury_severity_discount(projections.copy(), players)
    assert out.loc[0, "xpts"] == pytest.approx(8.0)


def test_injury_severity_discount_unknown_player_defaults_to_healthy():
    players = pd.DataFrame({"id": [1], "injury_severity": [3]})
    projections = pd.DataFrame({"player_id": [999], "gameweek": [10], "xpts": [8.0]})
    out = pipeline._apply_injury_severity_discount(projections.copy(), players)
    assert out.loc[0, "xpts"] == pytest.approx(8.0)  # unknown player, no discount


def test_run_projections_cold_start_returns_empty_not_crash(session):
    # no player_gw_stats rows for the season at all (GW1, season hasn't
    # started) -- assemble.load_all_stats returns empty, run_projections
    # must return gracefully, not crash
    session.add(Gameweek(id=1, season="2026-27", name="GW1", is_next=True,
                         deadline_time=datetime(2026, 8, 15)))
    session.commit()
    out = pipeline.run_projections(season="2026-27", horizon=1, persist=False)
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert set(out.columns) == {
        "player_id", "gameweek", "xpts", "xpts_mean", "xpts_var", "start_probability",
    }


# ---------------------------------------------------------------------------
# Training-set choice (A9, 2026-09-08)
# ---------------------------------------------------------------------------
#
# A6 made this choice conditional on the current-season frame's row count, and
# then on whether the European congestion features varied in it. Both gates are
# gone: the minutes model trains on all available history, always. See the note
# above ``run_projections``'s call to ``train_minutes`` for the measurement.
#
# The tests below pin the OUTCOME through ``run_projections`` itself, not the
# source text of the module -- ``tests/test_transfer_banking.py:420`` greps
# module source, and that is exactly how an earlier defect on this branch
# survived review.


def _built_frame(rows: int, euro: list[float]) -> pd.DataFrame:
    """A frame shaped like ``_build_features``'s output, only as far as
    ``run_projections`` reads it: a row count and the congestion columns.

    ``euro`` is tiled across the rows. The flag and the tier co-vary in real
    data -- the tier is only non-zero on a week the flag is set -- so they are
    set together here rather than independently.
    """
    from projection.congestion import CONGESTION_FEATURE_COLS

    df = pd.DataFrame({"player_id": range(rows), "gameweek": [3] * rows})
    for col in CONGESTION_FEATURE_COLS:
        df[col] = 0.0
    flags = (euro * rows)[:rows]
    df["euro_match_in_prev_7d"] = flags
    df["euro_competition_tier"] = [f * 3.0 for f in flags]
    return df


def _stub_everything_after_the_gate(monkeypatch, built: pd.DataFrame) -> dict:
    """Record what ``run_projections`` hands ``train_minutes`` and cut the run
    short."""
    from projection import assemble

    seen: dict = {}

    def fake_train(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(pipeline, "train_minutes", fake_train)
    monkeypatch.setattr(pipeline, "_get_current_and_next_gw", lambda: (3, 4))
    monkeypatch.setattr(pipeline, "_minutes_features", lambda df: built)
    monkeypatch.setattr(
        assemble, "load_all_stats",
        lambda season: pd.DataFrame({"player_id": [1], "gameweek": [3], "season": [season]}),
    )
    monkeypatch.setattr(pipeline, "_build_live_fixture_context", lambda s, g: pd.DataFrame())
    monkeypatch.setattr(pipeline, "_load_live_match_odds", lambda s, g: pd.DataFrame())
    monkeypatch.setattr(assemble, "load_defcon_events", lambda s: pd.DataFrame())
    monkeypatch.setattr(assemble, "compute_defcon_field_shares", lambda s: {})
    monkeypatch.setattr(assemble, "assemble_gw_projections", lambda **kw: pd.DataFrame())
    return seen


def test_a_healthy_current_season_frame_is_still_not_the_training_set(monkeypatch):
    """1200 rows and every European feature varying -- the frame that used to
    win the old gate outright and become ``df_override``.

    Measured on 103 held-out gameweeks across 2023-24, 2024-25 and 2025-26
    (59,246 player-gameweeks): training on the current season alone costs
    +0.0113 log loss and +0.0025 Brier against training on all history, and is
    worse in 79 of the 103. A frame being large and non-degenerate does not make
    it the better training set.
    """
    built = _built_frame(1200, euro=[0.0, 1.0])
    seen = _stub_everything_after_the_gate(monkeypatch, built)

    pipeline.run_projections(season="2026-27", horizon=1, persist=False)

    assert "df_override" not in seen, (
        "the minutes model must train on all available history, never on the "
        f"current season's frame alone; got df_override={seen.get('df_override')}"
    )
    assert seen == {"save": False, "fast": True}


def test_a_european_tie_being_played_cannot_change_the_training_set(monkeypatch):
    """The defect this task exists to remove.

    Under the old gate the training set flipped the first time a European tie
    landed inside a gameweek's ``euro_match_in_prev_7d`` window -- 2026-27's is
    2026-09-08 18:45, inside GW4's -- with no code change and nothing in the
    cron log saying why 152 of 615 players' ``start_probability`` had moved by
    more than 0.10. The two frames below differ in exactly that way and must
    produce the same training set.
    """
    calls = []
    for euro in ([0.0], [0.0, 1.0]):
        seen = _stub_everything_after_the_gate(monkeypatch, _built_frame(1200, euro=euro))
        pipeline.run_projections(season="2026-27", horizon=1, persist=False)
        calls.append(dict(seen))

    assert calls[0] == calls[1], (
        "a European tie having been played changed the training set: "
        f"{calls[0]} vs {calls[1]}"
    )


def test_a_thin_current_season_frame_also_trains_on_full_history(monkeypatch):
    """Early-season behaviour is preserved, not changed: the sub-1000-row branch
    already trained on all history, and still does."""
    seen = _stub_everything_after_the_gate(monkeypatch, _built_frame(40, euro=[0.0, 1.0]))

    pipeline.run_projections(season="2026-27", horizon=1, persist=False)

    assert "df_override" not in seen
    assert seen == {"save": False, "fast": True}


def test_no_usable_rows_still_routes_to_the_cold_start(monkeypatch):
    """NOT part of the gate removed here. A season with exactly one played
    gameweek has played history but no usable rows (every row is a first
    appearance, so the shifted rolling features are null and all rows drop),
    and there is nothing to fit OR to serve from. It routes to the cold start's
    prior-season evidence. Killed run_agent live at GW2 of 2026-27 on
    2026-08-25; must survive the gate removal.
    """
    seen = _stub_everything_after_the_gate(monkeypatch, _built_frame(0, euro=[0.0]))
    cold = pd.DataFrame({
        "player_id": [1], "gameweek": [4], "xpts": [3.0],
        "xpts_mean": [3.0], "xpts_var": [1.0], "start_probability": [0.8],
    })
    monkeypatch.setattr(
        pipeline, "cold_start_projections", lambda *a, **k: (cold, None, None)
    )

    out = pipeline.run_projections(season="2026-27", horizon=1, persist=False)

    assert seen == {}, "no model may be trained when there is nothing to fit"
    assert out.equals(cold)


def test_the_chosen_training_set_is_named_in_the_log(monkeypatch, caplog):
    """A change of training set must be visible to an operator reading a cron
    log. It cannot vary between runs any more, but the line still has to say
    which arm ran and how much of it the current season accounts for --
    silent 0.2 swings in ``start_probability`` are how the old flip went
    unnoticed."""
    _stub_everything_after_the_gate(monkeypatch, _built_frame(1200, euro=[0.0, 1.0]))

    with caplog.at_level(logging.INFO, logger="projection.pipeline"):
        pipeline.run_projections(season="2026-27", horizon=1, persist=False)

    messages = [r.getMessage() for r in caplog.records]
    assert any("all available history" in m and "1200" in m for m in messages), (
        f"no log line names the training set and the current season's share of "
        f"it; got {messages}"
    )
