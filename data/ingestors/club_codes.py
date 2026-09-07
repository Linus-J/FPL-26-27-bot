"""Club name -> stable FPL team code (2026-09-06).

Why this exists: `teams.id` is reassigned every season (19 of 29 clubs change
id across the six backfilled seasons) and `teams` holds only the current 20
clubs, while the match calendar spans 29. `code` is stable, and
`team_season_strength(season, team_id, code)` maps it back to a season's id.

Name variants are generous on purpose. FBref writes "Manchester Utd" and
"Nott'ham Forest"; FPL writes "Man Utd" and "Nott'm Forest"; commentary writes
"Manchester United". A miss here is not a thinned feature but an inverted one —
a club with a calendar gap looks RESTED — so the ingest treats an unresolved
Premier League club as fatal rather than skipping it.

data/ingestors/odds_api.py:140-160 records a measured production incident from
exactly this class of name mismatch, and is why the table is this defensive.
"""

from __future__ import annotations

import re

# The 29 codes present in team_season_strength across 2021-22..2026-27.
EXPECTED_CODES: frozenset[int] = frozenset({
    1, 2, 3, 4, 6, 7, 8, 9, 11, 13, 14, 17, 20, 21, 31, 36, 39, 40,
    43, 45, 49, 54, 56, 57, 88, 90, 91, 94, 102,
})

_RAW: dict[int, tuple[str, ...]] = {
    1: ("Manchester Utd", "Manchester United", "Man Utd", "Man United"),
    2: ("Leeds United", "Leeds"),
    3: ("Arsenal",),
    4: ("Newcastle Utd", "Newcastle United", "Newcastle"),
    6: ("Tottenham", "Tottenham Hotspur", "Spurs"),
    7: ("Aston Villa", "Villa"),
    8: ("Chelsea",),
    9: ("Coventry City", "Coventry"),
    11: ("Everton",),
    13: ("Leicester City", "Leicester"),
    14: ("Liverpool",),
    17: ("Nott'ham Forest", "Nottingham Forest", "Nott'm Forest", "Notts Forest"),
    20: ("Southampton",),
    21: ("West Ham United", "West Ham"),
    31: ("Crystal Palace",),
    36: ("Brighton & Hove Albion", "Brighton and Hove Albion", "Brighton"),
    39: ("Wolverhampton Wanderers", "Wolves", "Wolverhampton"),
    40: ("Ipswich Town", "Ipswich"),
    43: ("Manchester City", "Man City"),
    45: ("Norwich City", "Norwich"),
    49: ("Sheffield Utd", "Sheffield United"),
    54: ("Fulham",),
    56: ("Sunderland",),
    57: ("Watford",),
    88: ("Hull City", "Hull"),
    90: ("Burnley",),
    91: ("Bournemouth", "AFC Bournemouth"),
    94: ("Brentford",),
    102: ("Luton Town", "Luton"),
}


def normalize_club(name: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed.

    Punctuation becomes a SPACE rather than being deleted, so "Brighton & Hove
    Albion" and "Brighton and Hove Albion" do not diverge into
    "brighton  hove albion" and "brighton and hove albion".
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(name).lower())).strip()


CLUB_CODES: dict[str, int] = {
    normalize_club(variant): code
    for code, variants in _RAW.items()
    for variant in variants
}


def resolve_club(name: str) -> int | None:
    """FPL team code for a club name, or None if it is not a PL club."""
    return CLUB_CODES.get(normalize_club(name))
