"""Canonical FBref league registrations, owned by the repo (2026-09-06).

``soccerdata`` resolves custom leagues through
``~/soccerdata/config/league_dict.json`` — a file outside version control. That
is a latent defect this module inherits rather than introduces:
``fbref_prior.PRIOR_LEAGUES`` already names ``ENG-Championship`` and
``ITA-Serie A``, whose entries exist only in that local file, so a fresh clone
on a new machine cannot run the prior-league scrape at all.

Keeping the definitions here and merging them in on demand makes both the
inherited leagues and the new European ones reproducible. The merge is additive
and never removes a key it did not write, so a hand-tuned local file survives.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# UEFA club competitions. `season_start`/`season_end` mirror the domestic
# calendar because FBref labels a European season by the domestic year it sits
# inside — the 2024-25 Champions League runs Sep 2024 to May 2025.
EUROPEAN_LEAGUES: dict[str, dict] = {
    "INT-Champions League": {
        "FBref": "Champions League",
        "season_start": "Aug",
        "season_end": "May",
    },
    "INT-Europa League": {
        "FBref": "Europa League",
        "season_start": "Aug",
        "season_end": "May",
    },
    "INT-Conference League": {
        "FBref": "Europa Conference League",
        "season_start": "Aug",
        "season_end": "May",
    },
}

# Already depended on by fbref_prior.PRIOR_LEAGUES but never version-controlled.
INHERITED_LEAGUES: dict[str, dict] = {
    "ENG-Championship": {
        "FBref": "EFL Championship",
        "season_start": "Aug",
        "season_end": "May",
    },
    "ITA-Serie A": {
        "FBref": "Serie A (M)",
        "season_start": "Aug",
        "season_end": "May",
        "Understat": "Serie A",
    },
}

# soccerdata league id -> the short code stored in TeamMatch.competition.
COMPETITIONS: dict[str, str] = {
    "ENG-Premier League": "PL",
    "INT-Champions League": "UCL",
    "INT-Europa League": "UEL",
    "INT-Conference League": "UECL",
}


def _league_dict_path() -> Path:
    """Indirected through a function so tests can redirect it."""
    return Path.home() / "soccerdata" / "config" / "league_dict.json"


def register_leagues() -> None:
    """Merge this repo's league definitions into soccerdata's config.

    Additive and idempotent: existing keys this module does not own are left
    exactly as they are, and re-running writes nothing new.
    """
    path = _league_dict_path()
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("league_dict.json is not valid JSON; rewriting it")
            existing = {}

    merged = {**existing, **INHERITED_LEAGUES, **EUROPEAN_LEAGUES}
    if merged == existing:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    logger.info("Registered %d leagues in %s", len(merged), path)
