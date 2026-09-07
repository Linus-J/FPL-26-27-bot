"""Club name to stable FPL code (2026-09-06).

team_id is reassigned every season (19 of 29 clubs change id across the six
backfilled seasons) and the `teams` table holds only the current 20, so the
match calendar is keyed on `code` instead. team_season_strength is the
season-aware registry that turns a code back into that season's team_id."""

import sqlite3
from pathlib import Path

import pytest

from data.ingestors.club_codes import CLUB_CODES, EXPECTED_CODES, resolve_club

_LIVE_DB = Path(__file__).resolve().parents[1] / "fpl_bot_v2.db"


def test_every_expected_club_has_at_least_one_name():
    covered = set(CLUB_CODES.values())
    missing = EXPECTED_CODES - covered
    assert not missing, f"codes with no name variant: {sorted(missing)}"


def test_no_name_maps_to_an_unknown_code():
    assert set(CLUB_CODES.values()) <= EXPECTED_CODES


@pytest.mark.parametrize(
    "name,code",
    [
        ("Manchester Utd", 1), ("Manchester United", 1), ("Man Utd", 1),
        ("Manchester City", 43), ("Man City", 43),
        ("Nott'ham Forest", 17), ("Nottingham Forest", 17), ("Nott'm Forest", 17),
        ("Tottenham", 6), ("Spurs", 6), ("Tottenham Hotspur", 6),
        ("Brighton & Hove Albion", 36), ("Brighton", 36),
        ("Wolverhampton Wanderers", 39), ("Wolves", 39),
        ("Newcastle Utd", 4), ("Newcastle United", 4),
        ("Sheffield Utd", 49), ("Sheffield United", 49),
        ("Leicester City", 13), ("Norwich City", 45), ("Watford", 57),
        ("Burnley", 90), ("Luton Town", 102), ("Southampton", 20),
        ("West Ham", 21), ("West Ham United", 21),
    ],
)
def test_known_spellings_resolve(name, code):
    assert resolve_club(name) == code


def test_resolution_is_case_and_punctuation_insensitive():
    assert resolve_club("  MANCHESTER   UTD  ") == 1
    assert resolve_club("brighton and hove albion") == 36


def test_an_unknown_club_resolves_to_none():
    assert resolve_club("Real Madrid") is None
    assert resolve_club("Bayern Munich") is None


@pytest.mark.skipif(not _LIVE_DB.exists(), reason="requires the live fpl_bot_v2.db")
def test_every_code_in_the_live_registry_has_a_name():
    """The table must cover every club team_season_strength knows about, or the
    backfill dies on a season it cannot name."""
    conn = sqlite3.connect(f"file:{_LIVE_DB}?mode=ro", uri=True)
    try:
        codes = {r[0] for r in conn.execute("SELECT DISTINCT code FROM team_season_strength")}
    finally:
        conn.close()
    assert codes == EXPECTED_CODES, (
        f"registry/table mismatch — missing {sorted(codes - EXPECTED_CODES)}, "
        f"stale {sorted(EXPECTED_CODES - codes)}"
    )
