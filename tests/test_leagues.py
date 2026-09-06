"""Custom FBref leagues must be registered from the repo, not from a
machine-local file (2026-09-06).

soccerdata reads custom leagues from ~/soccerdata/config/league_dict.json,
which is outside version control. `fbref_prior.PRIOR_LEAGUES` already depends
on two entries that live only there, so a fresh clone cannot run the
prior-league scrape. This module makes the entries the repo's property and
merges them in on demand."""

import json

from data.ingestors.leagues import (
    COMPETITIONS,
    EUROPEAN_LEAGUES,
    INHERITED_LEAGUES,
    register_leagues,
)


def test_every_registered_league_maps_to_a_competition_code():
    for league in EUROPEAN_LEAGUES:
        assert league in COMPETITIONS


def test_competition_codes_are_the_short_forms_we_store():
    assert set(COMPETITIONS.values()) >= {"UCL", "UEL", "UECL"}


def test_register_writes_entries_into_a_missing_file(tmp_path, monkeypatch):
    target = tmp_path / "league_dict.json"
    monkeypatch.setattr("data.ingestors.leagues._league_dict_path", lambda: target)

    register_leagues()

    written = json.loads(target.read_text())
    for name in EUROPEAN_LEAGUES:
        assert name in written


def test_register_preserves_unrelated_existing_entries(tmp_path, monkeypatch):
    target = tmp_path / "league_dict.json"
    target.write_text(json.dumps({"ESP-Segunda": {"FBref": "Segunda Division"}}))
    monkeypatch.setattr("data.ingestors.leagues._league_dict_path", lambda: target)

    register_leagues()

    written = json.loads(target.read_text())
    assert written["ESP-Segunda"] == {"FBref": "Segunda Division"}
    assert "INT-Champions League" in written


def test_register_repairs_the_inherited_prior_league_entries(tmp_path, monkeypatch):
    """fbref_prior.PRIOR_LEAGUES depends on these; without them a fresh clone
    cannot run the prior-league scrape at all."""
    target = tmp_path / "league_dict.json"
    monkeypatch.setattr("data.ingestors.leagues._league_dict_path", lambda: target)

    register_leagues()

    written = json.loads(target.read_text())
    for name in INHERITED_LEAGUES:
        assert name in written


def test_register_is_idempotent(tmp_path, monkeypatch):
    target = tmp_path / "league_dict.json"
    monkeypatch.setattr("data.ingestors.leagues._league_dict_path", lambda: target)

    register_leagues()
    first = target.read_text()
    register_leagues()

    assert target.read_text() == first
