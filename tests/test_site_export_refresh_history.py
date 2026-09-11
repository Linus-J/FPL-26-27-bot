"""scripts/site_export/refresh.py

Each weekly export writes one ``gw{N}.json`` and never touches it again, so a
run file is a point-in-time record: the squad and the projections in it are
what the engine believed that week, and they should stay that way. Its
``history`` block is not -- that is a rendering of ``decision_log``, and when
a logged decision is corrected every file whose history covers it goes stale.

Found after restating the GW3 free-hit reason (2026-09-12): gw4.json picked
up the correction because the export rebuilds it, while gw3.json -- the file
the GW3 run published -- still showed the old wording.

Refreshing rebuilds only the history, from the same ``_build_history_entries``
the export uses and capped at each file's own gameweek, so a file never gains
an event that had not happened when it was written.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.site_export.refresh import refresh_history_files


def _write(tmp_path, gw: int, history: list[dict], squad=None) -> None:
    (tmp_path / f"gw{gw}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gameweek": gw,
                "squad": squad if squad is not None else [{"web_name": "Haaland"}],
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )


def _history_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def log():
    return _history_df([
        {"gameweek": 3, "decision_type": "chip", "projected_gain": 11.06, "details": {
            "chip": "freehit", "reason": "beats no-chip by 11.1 xPts",
        }},
        {"gameweek": 2, "decision_type": "chip", "projected_gain": 7.8, "details": {
            "chip": "3xc", "reason": "TC captain xPts 7.8",
        }},
    ])


def test_a_stale_reason_is_rewritten_from_the_log(tmp_path, log):
    _write(tmp_path, 3, [{"gameweek": 3, "type": "chip", "chip": "freehit",
                          "reason": "beats no-chip by 15.1 xPts — 2 FT(s)"}])

    assert refresh_history_files(tmp_path, log) == [tmp_path / "gw3.json"]

    history = json.loads((tmp_path / "gw3.json").read_text())["history"]
    assert history[0]["reason"] == "beats no-chip by 11.1 xPts"


def test_a_file_never_gains_an_event_from_after_its_own_gameweek(tmp_path, log):
    """gw2.json is what GW2 published. GW3's free hit had not been decided."""
    _write(tmp_path, 2, [{"gameweek": 2, "type": "chip", "chip": "3xc", "reason": "stale"}])

    refresh_history_files(tmp_path, log)

    history = json.loads((tmp_path / "gw2.json").read_text())["history"]
    assert [e["gameweek"] for e in history] == [2]


def test_everything_but_the_history_is_left_exactly_as_it_was(tmp_path, log):
    """The squad and projections in a run file are that week's record and are
    not re-derivable now; only the history is a view of something else."""
    _write(tmp_path, 3, [{"gameweek": 3, "type": "chip", "chip": "freehit", "reason": "stale"}],
           squad=[{"web_name": "Haaland", "xpts": 7.3}])

    refresh_history_files(tmp_path, log)

    payload = json.loads((tmp_path / "gw3.json").read_text())
    assert payload["squad"] == [{"web_name": "Haaland", "xpts": 7.3}]
    assert payload["schema_version"] == 1
    assert payload["gameweek"] == 3


def test_a_file_already_matching_the_log_is_not_rewritten(tmp_path, log):
    """Reported as changed only when it changed, so re-running does not churn
    the git history of the data directory."""
    _write(tmp_path, 3, [{"gameweek": 3, "type": "chip", "chip": "freehit", "reason": "stale"}])

    assert refresh_history_files(tmp_path, log) == [tmp_path / "gw3.json"]
    assert refresh_history_files(tmp_path, log) == []


def test_the_index_is_not_treated_as_a_run_file(tmp_path, log):
    (tmp_path / "index.json").write_text(json.dumps({"runs": []}) + "\n")
    _write(tmp_path, 3, [{"gameweek": 3, "type": "chip", "chip": "freehit", "reason": "stale"}])

    refresh_history_files(tmp_path, log)

    assert json.loads((tmp_path / "index.json").read_text()) == {"runs": []}


def test_the_trailing_newline_is_preserved(tmp_path, log):
    """write_run_file appends one; without it every refresh would show as a
    whole-file diff."""
    _write(tmp_path, 3, [{"gameweek": 3, "type": "chip", "chip": "freehit", "reason": "stale"}])

    refresh_history_files(tmp_path, log)

    assert (tmp_path / "gw3.json").read_text().endswith("}\n")
