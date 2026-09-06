"""The press-signals table is dropped on startup for databases that predate
its removal (2026-09-06). `create_all` never drops anything, so without this
a database that ran the agent while the Guardian layer existed keeps a dead
table forever."""

from sqlalchemy import Engine, create_engine, inspect, text

from data.migrations import drop_player_press_signals


def _engine_with_press_table(tmp_path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'press.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE player_press_signals ("
            "id INTEGER PRIMARY KEY, player_id INTEGER, scraped_date VARCHAR(10),"
            " sentiment FLOAT, raw_quote TEXT, source_url VARCHAR)"
        ))
    return engine


def test_drops_the_table_when_present(tmp_path):
    engine = _engine_with_press_table(tmp_path)
    assert drop_player_press_signals(engine) is True
    assert "player_press_signals" not in inspect(engine).get_table_names()


def test_is_idempotent(tmp_path):
    engine = _engine_with_press_table(tmp_path)
    drop_player_press_signals(engine)
    assert drop_player_press_signals(engine) is False


def test_absent_table_is_not_an_error(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    assert drop_player_press_signals(engine) is False
