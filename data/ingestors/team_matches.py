"""FBref schedules -> the ``team_matches`` calendar (2026-09-06).

One row per team per match, so a domestic fixture yields two rows and a
European tie involving one PL club yields one. That shape is what makes
rest-day computation a per-team sort in ``projection/congestion.py``.

Reuses the SeleniumBase/soccerdata stack that ``data/ingestors/fbref.py``
already depends on: FBref sits behind Cloudflare, which is why
``scripts/run_weekly.py`` forces ``FBREF_HEADED=1``.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.ingestors.leagues import COMPETITIONS, register_leagues
from data.models import Team, TeamMatch

logger = logging.getLogger(__name__)


class UnmappedClubError(RuntimeError):
    """A club in a Premier League schedule has no ``teams`` row.

    Fatal by design. Dropping the row would leave that club with gaps in its
    calendar, and a gap reads as rest — so a silent drop does not thin the
    feature, it inverts it.
    """


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Same shape as ``fbref._normalize_name`` but for clubs rather than players.
    """
    return re.sub(r"[^a-z0-9 ]", "", str(name).lower()).strip()


def build_team_rows(
    schedule: pd.DataFrame,
    season: str,
    competition: str,
    team_ids: dict[str, int],
) -> list[dict]:
    """Explode an FBref schedule into per-team calendar rows.

    ``team_ids`` maps a NORMALISED club name to an FPL ``teams.id``. A club
    absent from it is treated as non-PL: skipped in a European schedule (Real
    Madrid is not supposed to be there), fatal in a domestic one.
    """
    rows: list[dict] = []
    for _, match in schedule.iterrows():
        kickoff = pd.to_datetime(match["date"], errors="coerce")
        if pd.isna(kickoff):
            continue
        home_raw, away_raw = match["home_team"], match["away_team"]
        home, away = _normalize(home_raw), _normalize(away_raw)

        if competition == "PL":
            for raw, norm in ((home_raw, home), (away_raw, away)):
                if norm not in team_ids:
                    raise UnmappedClubError(
                        f"{raw!r} in the {season} Premier League schedule has no "
                        f"teams row. Add it to soccerdata's teamname_replacements "
                        f"or to the teams table before re-running."
                    )

        for team_norm, opponent_raw, is_home in (
            (home, away_raw, True), (away, home_raw, False)
        ):
            if team_norm not in team_ids:
                continue
            rows.append({
                "season": season,
                "team_id": team_ids[team_norm],
                "kickoff_time": kickoff.to_pydatetime(),
                "competition": competition,
                "opponent_name": str(opponent_raw),
                "is_home": is_home,
            })
    return rows


def build_club_name_map() -> dict[str, int]:
    """Normalised club name -> ``teams.id``, from both name and short_name."""
    db = get_session()
    try:
        mapping: dict[str, int] = {}
        for team in db.query(Team).all():
            mapping[_normalize(team.name)] = team.id
            if getattr(team, "short_name", None):
                mapping[_normalize(team.short_name)] = team.id
        return mapping
    finally:
        db.close()


def write_team_matches(rows: list[dict]) -> int:
    """Upsert calendar rows. Idempotent on (season, team_id, kickoff_time), so
    re-running a season after a partial failure is safe and is the whole point.
    """
    if not rows:
        return 0
    db = get_session()
    try:
        stmt = insert(TeamMatch).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["season", "team_id", "kickoff_time"],
            set_={
                "competition": stmt.excluded.competition,
                "opponent_name": stmt.excluded.opponent_name,
                "is_home": stmt.excluded.is_home,
            },
        )
        db.execute(stmt)
        db.commit()
        return len(rows)
    finally:
        db.close()


def ingest_competition_season(season: str, league: str) -> int:  # pragma: no cover
    """Scrape one competition-season's schedule and write its calendar rows.

    ``league`` is a soccerdata league id, e.g. ``"INT-Champions League"``.
    Excluded from coverage: needs live network and a real browser.
    """
    import soccerdata as sd

    register_leagues()
    competition = COMPETITIONS[league]
    fbref = sd.FBref(leagues=league, seasons=season)
    schedule = fbref.read_schedule().reset_index()
    schedule.columns = [str(c).lower().replace(" ", "_") for c in schedule.columns]

    rows = build_team_rows(schedule, season, competition, build_club_name_map())
    written = write_team_matches(rows)
    logger.info("%s %s: %d calendar rows", league, season, written)
    return written
