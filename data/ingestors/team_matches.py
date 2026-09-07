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

import pandas as pd
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.ingestors.club_codes import resolve_club
from data.ingestors.leagues import COMPETITIONS, register_leagues
from data.models import TeamMatch

logger = logging.getLogger(__name__)


class UnmappedClubError(RuntimeError):
    """A club in a Premier League schedule has no ``teams`` row.

    Fatal by design. Dropping the row would leave that club with gaps in its
    calendar, and a gap reads as rest — so a silent drop does not thin the
    feature, it inverts it.
    """


def build_team_rows(
    schedule: pd.DataFrame,
    season: str,
    competition: str,
) -> list[dict]:
    """Explode an FBref schedule into per-team calendar rows.

    Club names are resolved to the stable FPL ``code`` via
    ``data.ingestors.club_codes.resolve_club``, not to ``teams.id`` — see
    ``data/ingestors/club_codes.py`` for why. A club it cannot resolve is
    treated as non-PL: skipped in a European schedule (Real Madrid is not
    supposed to be there), fatal in a domestic one.
    """
    rows: list[dict] = []
    for _, match in schedule.iterrows():
        kickoff = pd.to_datetime(match["date"], errors="coerce")
        if pd.isna(kickoff):
            continue
        home_raw, away_raw = match["home_team"], match["away_team"]
        home_code, away_code = resolve_club(home_raw), resolve_club(away_raw)

        if competition == "PL":
            for raw, code in ((home_raw, home_code), (away_raw, away_code)):
                if code is None:
                    raise UnmappedClubError(
                        f"{raw!r} in the {season} Premier League schedule has no "
                        f"club code. Add it to data.ingestors.club_codes before "
                        f"re-running."
                    )

        for team_code, opponent_raw, is_home in (
            (home_code, away_raw, True), (away_code, home_raw, False)
        ):
            if team_code is None:
                continue
            rows.append({
                "season": season,
                "team_code": team_code,
                "kickoff_time": kickoff.to_pydatetime(),
                "competition": competition,
                "opponent_name": str(opponent_raw),
                "is_home": is_home,
            })
    return rows


def write_team_matches(rows: list[dict]) -> int:
    """Upsert calendar rows. Idempotent on (season, team_code, kickoff_time),
    so re-running a season after a partial failure is safe and is the whole
    point.
    """
    if not rows:
        return 0
    db = get_session()
    try:
        stmt = insert(TeamMatch).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["season", "team_code", "kickoff_time"],
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
    # BEFORE the import: soccerdata reads league_dict.json at import time, so
    # registering afterwards writes a file this process will never re-read --
    # which made every European job fail with "Invalid league" on the first
    # live backfill.
    register_leagues()

    import soccerdata as sd

    competition = COMPETITIONS[league]
    fbref = sd.FBref(leagues=league, seasons=season)
    schedule = fbref.read_schedule().reset_index()
    schedule.columns = [str(c).lower().replace(" ", "_") for c in schedule.columns]

    rows = build_team_rows(schedule, season, competition)
    written = write_team_matches(rows)
    logger.info("%s %s: %d calendar rows", league, season, written)
    return written
