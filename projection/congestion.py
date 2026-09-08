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

STANDING OPERATIONAL REQUIREMENT (A6, 2026-09-08). The current season's
European calendar only ever extends as far as the draws that have been made.
2026-27 currently runs 2026-09-08 to 2027-01-28 — the complete league phase,
and genuinely all that exists, because the knockout draws have not happened.
Completed seasons carry their knockout rounds (2025-26 has 45 matches across
February–May), so from February onwards the LIVE feature will under-report
European congestion relative to what the model learned, until each draw is
made and the calendar re-scraped:

    DB_PATH=fpl_bot_v2.db FBREF_HEADED=1 uv run python \\
        scripts/backfill_team_matches.py --reset

This does not affect a 5-gameweek horizon in September. It is a calendar
maintenance task, deliberately not automated here.
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

# What an unmatched player-gameweek gets. A blanket 0.0 -- which is what both
# neighbouring merge helpers do for their flags -- would say "played yesterday,
# playing again tomorrow", the exact opposite of the truth, so the gap columns
# default to the same cap ``compute_congestion`` uses for "no previous match".
CONGESTION_DEFAULTS: dict[str, float] = {
    "days_since_last_match": REST_CAP_DAYS,
    "days_to_next_match": REST_CAP_DAYS,
    "matches_in_prev_14d": 0.0,
    "euro_match_in_prev_7d": 0.0,
    "euro_competition_tier": 0.0,
    "is_post_international_break": 0.0,
}

# The club column on the LEFT frame. ``player_gw_stats.team_id_season`` is the
# club the player turned out for in THAT season; ``players.team_id`` is where
# they play now. Keying on the latter would file a transferred player's 2023-24
# rows against their 2026-27 club's calendar.
CONGESTION_KEYS = ["season", "team_id", "gameweek"]
CLUB_COL = "team_id_season"

# A calendar that is even partly date-only is worse than one that is wholly so.
# The rest-day gap is (this kickoff - the previous one), so mixing true evening
# kickoffs with midnight ones biases the answer in one direction rather than
# merely coarsening it -- and it biases the European midweek case, which is the
# only case these features exist to measure. Warn above a share this small
# because there is no legitimate reason for ANY fixture to sit at 00:00.
DATELESS_KICKOFF_WARN_SHARE = 0.01


def dateless_kickoff_share(matches: pd.DataFrame) -> float:
    """Fraction of the calendar sitting at exactly midnight.

    No football match kicks off at 00:00, so this is a direct measure of how
    much of ``team_matches`` was written before the ingest learned to read
    FBref's separate ``time`` column (12a6387).
    """
    if matches.empty:
        return 0.0
    kickoffs = pd.to_datetime(matches["kickoff_time"], errors="coerce")
    return float((kickoffs.dt.normalize() == kickoffs).mean())


def warn_on_dateless_kickoffs(matches: pd.DataFrame) -> None:
    """Say so, loudly and with the remedy, when the calendar has no clock.

    Silent staleness is this project's recurring failure: the backfill journals
    completed jobs, so re-running it without ``--reset`` leaves pre-fix rows
    untouched and reports success. Naming the command matters -- a warning that
    does not say what to run is a warning that gets scrolled past.
    """
    share = dateless_kickoff_share(matches)
    if share <= DATELESS_KICKOFF_WARN_SHARE:
        return
    logger.warning(
        "%.0f%% of the match calendar sits at exactly midnight: those rows "
        "predate the kickoff-time ingest fix, so rest-days are understated and "
        "matches can anchor to the wrong gameweek. Re-run: "
        "DB_PATH=fpl_bot_v2.db FBREF_HEADED=1 uv run python "
        "scripts/backfill_team_matches.py --reset",
        share * 100,
    )


def compute_congestion(matches: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Congestion features for each (season, team_id, gameweek) anchor.

    ``matches``: season, team_id, kickoff_time, competition — every competition.
    ``anchors``: season, team_id, gameweek, anchor_time — the team's own PL
    kickoff for that gameweek.

    Seasons never bleed into each other: a team's last match of May is not rest
    for August. That is enforced by grouping on (season, team_id), not team_id.
    """
    anchors = anchors.reset_index(drop=True)
    out = anchors.copy()
    for col in CONGESTION_FEATURE_COLS:
        out[col] = 0.0
    out["days_since_last_match"] = REST_CAP_DAYS
    out["days_to_next_match"] = REST_CAP_DAYS

    if matches.empty or anchors.empty:
        # Assigned once, after the loop, so every path agrees (2026-09-06).
        # Previously this sat inside the loop body, below both a `continue`
        # (team absent from the calendar) and an early return (empty matches
        # frame) — so three situations that are all "no previous match,
        # days_since = 14.0" gave two different answers. The brief's
        # definition table defines this flag purely in terms of
        # days_since_last_match, and 14.0 is 14.0 however it arose, so the
        # consistent answer is the one the table already implies.
        out["is_post_international_break"] = (
            out["days_since_last_match"] >= INTERNATIONAL_BREAK_DAYS
        ).astype(float)
        return out[["season", "team_id", "gameweek", *CONGESTION_FEATURE_COLS]]

    m = matches.copy()
    m["kickoff_time"] = pd.to_datetime(m["kickoff_time"])
    out["anchor_time"] = pd.to_datetime(out["anchor_time"])
    by_team: dict[tuple, pd.DataFrame] = {
        key: group.sort_values("kickoff_time")
        for key, group in m.groupby(["season", "team_id"], sort=False)
    }

    unmatched = 0
    for i, row in out.iterrows():
        group = by_team.get((row["season"], row["team_id"]))
        if group is None:
            unmatched += 1
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

    # A silent total miss is indistinguishable from "nobody played midweek"
    # (2026-09-06, review finding C3). A dtype or season-format mismatch would
    # otherwise return every row at its default and read as a fully-rested
    # league, which is plausible enough to reach a retrain unnoticed.
    if unmatched:
        logger.warning(
            "compute_congestion: %d of %d anchors had no calendar rows; "
            "those rows carry default (fully-rested) values",
            unmatched, len(out),
        )

    # Assigned once, after the loop, so every path agrees (2026-09-06).
    # Previously this sat inside the loop body, below both a `continue` (team
    # absent from the calendar) and an early return (empty matches frame) — so
    # three situations that are all "no previous match, days_since = 14.0"
    # gave two different answers. The brief's definition table defines this
    # flag purely in terms of days_since_last_match, and 14.0 is 14.0 however
    # it arose, so the consistent answer is the one the table already
    # implies.
    out["is_post_international_break"] = (
        out["days_since_last_match"] >= INTERNATIONAL_BREAK_DAYS
    ).astype(float)

    return out[["season", "team_id", "gameweek", *CONGESTION_FEATURE_COLS]]


def load_congestion(season: str | None = None) -> pd.DataFrame:
    """Congestion features per (season, team_id, gameweek), from the database.

    ``team_matches`` is keyed on the STABLE FPL club ``code`` (see
    ``data/ingestors/club_codes.py``: ``teams.id`` is reassigned every season
    and ``teams`` holds only the current 20 clubs, while the calendar spans
    29). Both queries below join ``team_season_strength`` to resolve that code
    back to the season's own ``team_id`` before returning — callers still see
    exactly ``season, team_id, gameweek`` plus the six feature columns, the
    season-correct id. A club with no ``team_season_strength`` row for a
    season was not in the Premier League that season and is correctly dropped
    by the inner join.

    The anchor is the team's own PL kickoff in that gameweek, read from
    ``team_matches`` itself rather than ``fixtures`` — so this works identically
    for the six backfilled seasons (which have no ``fixtures`` rows at all) and
    for the season being played. On a double gameweek the FIRST PL kickoff of
    the week anchors it, which is the match the rest-days question is about.
    """
    db = get_session()
    try:
        where = "WHERE tm.season = :season" if season is not None else ""
        season_clause = "AND tm.season = :season" if season is not None else ""
        params = {"season": season} if season is not None else {}
        matches = pd.read_sql(
            text(f"""
                SELECT tm.season, tss.team_id AS team_id, tm.kickoff_time, tm.competition
                FROM team_matches tm
                JOIN team_season_strength tss
                  ON tss.season = tm.season AND tss.code = tm.team_code
                {where}
            """),
            db.bind, params=params,
        )
        anchors = pd.read_sql(
            text(f"""
                SELECT tm.season, tss.team_id AS team_id, g.id AS gameweek,
                       MIN(tm.kickoff_time) AS anchor_time
                FROM team_matches tm
                JOIN team_season_strength tss
                  ON tss.season = tm.season AND tss.code = tm.team_code
                JOIN gameweeks g
                  ON g.season = tm.season
                 AND tm.kickoff_time >= g.deadline_time
                 AND tm.kickoff_time < COALESCE(
                       (SELECT MIN(g2.deadline_time) FROM gameweeks g2
                         WHERE g2.season = g.season AND g2.id > g.id),
                       '9999-12-31')
                WHERE tm.competition = 'PL' {season_clause}
                GROUP BY tm.season, tss.team_id, g.id
            """),
            db.bind, params=params,
        )
    finally:
        db.close()

    warn_on_dateless_kickoffs(matches)

    if anchors.empty:
        logger.warning("load_congestion: no PL anchors found; features will be absent")
        return pd.DataFrame(
            columns=["season", "team_id", "gameweek", *CONGESTION_FEATURE_COLS]
        )
    return compute_congestion(matches, anchors)


def add_congestion_features(
    df: pd.DataFrame, congestion: pd.DataFrame
) -> pd.DataFrame:
    """Left-merge the congestion features onto a player-gameweek frame.

    Same shape as ``features.add_fdr_features`` / ``add_odds_features``, with
    two deliberate differences.

    The key is ``(season, team_id_season, gameweek)``, not the per-fixture key
    those two use: congestion is a property of the CLUB's week, so both of a
    double gameweek's fixtures share it, and the player's season-correct club
    is what decides whose calendar applies.

    The fill is ``CONGESTION_DEFAULTS``, not ``0.0``. That distinction is the
    whole point — see the constant.

    The merge is strictly non-destructive: the right side is de-duplicated on
    the key first, so no row can fan out, and the join is LEFT, so none can be
    dropped. ``_build_features`` has already discarded its unusable rows by the
    time this runs and must not have that decision revisited here.
    """
    merged = df.copy()
    have_keys = (
        {"season", "gameweek", CLUB_COL}.issubset(merged.columns)
        and not congestion.empty
        and set(CONGESTION_KEYS).issubset(congestion.columns)
    )

    if have_keys:
        right = congestion.loc[
            :, [*CONGESTION_KEYS, *CONGESTION_FEATURE_COLS]
        ].drop_duplicates(subset=CONGESTION_KEYS)
        right = right.rename(columns={"team_id": CLUB_COL})
        before = len(merged)
        merged = merged.merge(
            right, on=["season", CLUB_COL, "gameweek"], how="left"
        )
        # A merge that matches nothing is indistinguishable from a league that
        # never played midweek, and that is precisely how ``press_sentiment``
        # and ``btts_prob`` both reached production reading their default on
        # every row. A dtype or season-format mismatch on either side lands
        # here, so say so rather than serving a silent constant.
        matched = int(merged["euro_match_in_prev_7d"].notna().sum())
        if matched == 0:
            logger.warning(
                "add_congestion_features: none of %d player-gameweeks matched "
                "a congestion row (%d available) -- every row will read as "
                "fully rested. Check the season/team_id key on both sides.",
                before, len(right),
            )
    elif not {"season", "gameweek", CLUB_COL}.issubset(merged.columns):
        # Legitimate: single-row predict frames and the very-early-season
        # frames in test_pipeline_cold_start_gap carry no club at all. They get
        # the rested defaults, which is the honest answer for "unknown", but a
        # NaN here would propagate through the scaler and make the whole
        # prediction NaN, so the columns must still exist and be finite.
        logger.debug(
            "add_congestion_features: frame lacks %s; defaulting every "
            "congestion feature", sorted({"season", "gameweek", CLUB_COL} - set(merged.columns)),
        )

    for col, default in CONGESTION_DEFAULTS.items():
        if col not in merged.columns:
            merged[col] = default
        else:
            merged[col] = merged[col].fillna(default).astype(float)
    return merged
