"""Fixture-congestion features from the all-competition match calendar
(2026-09-06).

The engine had no idea a team played on Wednesday. ``data/ingestors/midweek.py``
looked like this feature and was not: it read the PL-only ``fixtures`` table and
detected DOUBLE GAMEWEEKS, and nothing called it. The observed cost was Foden
carrying an identical ``start_probability`` of 0.781 across GW3-GW7, unmoved by
the first Champions League week of the season, because nothing in the pipeline
could see it.

Everything here is a pure function over a calendar. The database wrapper at the
bottom is the only part that touches SQLAlchemy, so the semantics are testable
without fixtures and — more importantly — there is exactly ONE implementation,
called by both the training path and the live path. Two implementations of one
concept is the failure mode this codebase keeps rediscovering.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text

from data.db import get_session

logger = logging.getLogger(__name__)

CONGESTION_FEATURE_COLS = [
    "days_since_last_match",
    "days_to_next_match",
    "matches_in_prev_14d",
    "euro_match_in_prev_7d",
    "euro_competition_tier",
    "is_post_international_break",
]

# Days. Also the value used when there is no previous/next match at all: a
# fortnight is already "fully rested" for every purpose the model cares about,
# so capping and defaulting to the same number avoids a sentinel the booster
# would have to learn to ignore.
REST_CAP_DAYS = 14.0
CONGESTION_WINDOW_DAYS = 14.0
EURO_WINDOW_DAYS = 7.0
INTERNATIONAL_BREAK_DAYS = 14.0

EURO_TIERS = {"UECL": 1.0, "UEL": 2.0, "UCL": 3.0}


def compute_congestion(matches: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Congestion features for each (season, team_id, gameweek) anchor.

    ``matches``: season, team_id, kickoff_time, competition — every competition.
    ``anchors``: season, team_id, gameweek, anchor_time — the team's own PL
    kickoff for that gameweek.

    Seasons never bleed into each other: a team's last match of May is not rest
    for August. That is enforced by grouping on (season, team_id), not team_id.
    """
    out = anchors.copy()
    for col in CONGESTION_FEATURE_COLS:
        out[col] = 0.0
    out["days_since_last_match"] = REST_CAP_DAYS
    out["days_to_next_match"] = REST_CAP_DAYS

    if matches.empty or anchors.empty:
        return out[["season", "team_id", "gameweek", *CONGESTION_FEATURE_COLS]]

    m = matches.copy()
    m["kickoff_time"] = pd.to_datetime(m["kickoff_time"])
    out["anchor_time"] = pd.to_datetime(out["anchor_time"])
    by_team: dict[tuple, pd.DataFrame] = {
        key: group.sort_values("kickoff_time")
        for key, group in m.groupby(["season", "team_id"], sort=False)
    }

    for i, row in out.iterrows():
        group = by_team.get((row["season"], row["team_id"]))
        if group is None:
            continue
        anchor = row["anchor_time"]
        # Strictly before/after: the anchor match is not its own rest.
        before = group[group["kickoff_time"] < anchor]
        after = group[group["kickoff_time"] > anchor]

        if not before.empty:
            gap = (anchor - before["kickoff_time"].iloc[-1]).total_seconds() / 86400.0
            out.at[i, "days_since_last_match"] = min(gap, REST_CAP_DAYS)
        if not after.empty:
            gap = (after["kickoff_time"].iloc[0] - anchor).total_seconds() / 86400.0
            out.at[i, "days_to_next_match"] = min(gap, REST_CAP_DAYS)

        congestion_cutoff = anchor - pd.Timedelta(days=CONGESTION_WINDOW_DAYS)
        out.at[i, "matches_in_prev_14d"] = float(
            (before["kickoff_time"] >= congestion_cutoff).sum()
        )

        euro_cutoff = anchor - pd.Timedelta(days=EURO_WINDOW_DAYS)
        recent_euro = before[
            (before["kickoff_time"] >= euro_cutoff)
            & (before["competition"].isin(EURO_TIERS))
        ]
        if not recent_euro.empty:
            out.at[i, "euro_match_in_prev_7d"] = 1.0
            out.at[i, "euro_competition_tier"] = float(
                recent_euro["competition"].map(EURO_TIERS).max()
            )

        out.at[i, "is_post_international_break"] = float(
            out.at[i, "days_since_last_match"] >= INTERNATIONAL_BREAK_DAYS
        )

    return out[["season", "team_id", "gameweek", *CONGESTION_FEATURE_COLS]]


def load_congestion(season: str | None = None) -> pd.DataFrame:
    """Congestion features per (season, team_id, gameweek), from the database.

    The anchor is the team's own PL kickoff in that gameweek, read from
    ``team_matches`` itself rather than ``fixtures`` — so this works identically
    for the five backfilled seasons (which have no ``fixtures`` rows at all) and
    for the season being played. On a double gameweek the FIRST PL kickoff of
    the week anchors it, which is the match the rest-days question is about.
    """
    db = get_session()
    try:
        where = "WHERE season = :season" if season else ""
        params = {"season": season} if season else {}
        matches = pd.read_sql(
            text(f"SELECT season, team_id, kickoff_time, competition FROM team_matches {where}"),
            db.bind, params=params,
        )
        anchors = pd.read_sql(
            text(f"""
                SELECT tm.season, tm.team_id, g.id AS gameweek,
                       MIN(tm.kickoff_time) AS anchor_time
                FROM team_matches tm
                JOIN gameweeks g
                  ON g.season = tm.season
                 AND tm.kickoff_time >= g.deadline_time
                 AND tm.kickoff_time < COALESCE(
                       (SELECT MIN(g2.deadline_time) FROM gameweeks g2
                         WHERE g2.season = g.season AND g2.id > g.id),
                       '9999-12-31')
                WHERE tm.competition = 'PL' {"AND tm.season = :season" if season else ""}
                GROUP BY tm.season, tm.team_id, g.id
            """),
            db.bind, params=params,
        )
    finally:
        db.close()

    if anchors.empty:
        logger.warning("load_congestion: no PL anchors found; features will be absent")
        return pd.DataFrame(
            columns=["season", "team_id", "gameweek", *CONGESTION_FEATURE_COLS]
        )
    return compute_congestion(matches, anchors)
