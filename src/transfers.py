"""
Recent transfer tracking.

Added after Mehul asked how to stay current on which players are in the
Premier League and at which club, especially across the summer and
winter transfer windows.

Source: dcaribou/transfermarkt-datasets (github.com/dcaribou/transfermarkt-datasets)
-- a community project that extracts and republishes real Transfermarkt
data as clean CSV files, refreshed weekly, under a CC0 (public domain)
license. Deliberately NOT scraping Transfermarkt's own site directly:
transfermarkt.co.in is on Claude's own fetch blocklist (a strong signal
it actively resists automated access -- they license their data
commercially), and even a scraper that worked today against a site with
active anti-bot defenses (CAPTCHAs, IP bans) would be too fragile to
run repeatedly from a Colab pipeline. This dataset gives the same
underlying Transfermarkt data without that fragility.

Schema confirmed directly against the live files (2026-09), not
assumed: clubs.csv has 'domestic_competition_id' (Transfermarkt's own
league code -- 'GB1' for the Premier League, matching the code in the
transfermarkt.co.in URL Mehul linked); transfers.csv has player_id,
transfer_date, transfer_season, from_club_id, to_club_id,
from_club_name, to_club_name, transfer_fee, market_value_in_eur,
player_name. A bare request to the download host returns 403 -- also
confirmed directly -- so a real browser User-Agent is required.

Known limitation, confirmed against a real, current example (2026-09):
this table tracks TRANSFER events (a player's registration moving from
one club to another), not every subsequent LOAN move. A player
transferred to a club and then loaned onward from there (e.g. Dastan
Satpaev: Kairat Almaty -> Chelsea per this dataset, but actually out on
loan at Burnley in the Championship as of writing) may still show at
their parent club here even though they're playing elsewhere. This is
a real gap in any transfer dataset -- loans, especially lower-profile
ones, lag. It doesn't undermine the rest of the pipeline though:
whether a player is actually eligible/rostered in the Premier League
THIS gameweek is still decided by src/live_ingest.py's live FPL
snapshot (a player out on loan to the Championship simply won't accrue
real FPL minutes/points there, which forecast_points()/recommend_squad()
already handle correctly). This module is a "what moved recently"
context feed for the Report Agent, not a squad-eligibility check --
and it's optional: if the external host is unreachable, callers get a
clear error rather than the whole crew run breaking.

Second known limitation, confirmed directly (2026-09): this dataset can
also simply be missing a real, high-profile transfer outright, not just
lag on loans. Checked after two real 2026 Liverpool signings (Ronald
Araujo, Victor Munoz) didn't appear in recent_transfers() output --
traced it all the way to the fully raw, unfiltered transfers table (no
PL filter, no date window) and confirmed both are absent there too, so
this isn't a bug in the filtering/date-window logic above (a comparable
real transfer, Jeremy Jacquet's, DOES appear correctly at every stage,
which is what confirms the pipeline itself works). It's a genuine gap
in this specific snapshot of the upstream community dataset. Same
conclusion as the loan-lag case: treat this feed as "recent moves we
know about," not an exhaustive list -- cross-check anything load-bearing
against the official source before treating an absence as confirmation
a transfer didn't happen.
"""

import io
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{table}.csv.gz"
# A bare urllib/pandas request gets a 403 from this host -- confirmed
# directly against the real endpoint -- a browser-like User-Agent is
# required to get the actual data back.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
PREMIER_LEAGUE_COMPETITION_ID = "GB1"  # Transfermarkt's own code, confirmed live
RESULT_COLUMNS = [
    "player_name", "from_club_name", "to_club_name",
    "transfer_date", "transfer_fee", "status",
]


def _fetch_table(table: str) -> pd.DataFrame:
    url = BASE_URL.format(table=table)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Couldn't reach the transfer-data source ({url}): {exc}. "
            "This is an optional enrichment, not core data -- the rest of "
            "the pipeline works fine without it."
        ) from exc
    return pd.read_csv(io.BytesIO(resp.content), compression="gzip")


def recent_transfers(
    days: int = 30,
    competition_id: str = PREMIER_LEAGUE_COMPETITION_ID,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Premier League transfers within `days` days either side of now --
    the last `days` days, plus any already-announced deals taking effect
    within the next `days` days.

    Future-dated rows are real, not a data bug: confirmed directly
    against live data that a small number of Premier League transfers
    are pre-agreed and dated ahead of time (e.g. a January-window deal
    signed and announced in the preceding months). Those are included
    (bounded to the same `days` horizon, not unbounded into the distant
    future) and labeled 'announced' rather than silently dropped or
    mislabeled as already having happened.

    Matches a transfer if EITHER side (from_club or to_club) is a
    Premier League club, so both arrivals and departures show up --
    a player leaving the league entirely is exactly the kind of move
    this is meant to surface.

    Returns columns: player_name, from_club_name, to_club_name,
    transfer_date, transfer_fee, status ('completed' or 'announced').
    Empty DataFrame (same columns, zero rows) if nothing falls in the
    window -- a genuinely quiet stretch outside a transfer window is
    the honest, expected result, not a failure.

    `now` is injectable for testing; defaults to the real current time.
    """
    clubs = _fetch_table("clubs")
    transfers = _fetch_table("transfers")

    if "domestic_competition_id" not in clubs.columns:
        raise ValueError(
            "clubs data has no 'domestic_competition_id' column -- the "
            "upstream schema may have changed since this was written "
            f"(expected column not found; got: {list(clubs.columns)})."
        )
    required_transfer_cols = {
        "from_club_id", "to_club_id", "transfer_date",
        "player_name", "from_club_name", "to_club_name", "transfer_fee",
    }
    missing = required_transfer_cols - set(transfers.columns)
    if missing:
        raise ValueError(
            f"transfers data is missing expected column(s) {missing} -- "
            "the upstream schema may have changed since this was written "
            f"(got: {list(transfers.columns)})."
        )

    pl_club_ids = set(clubs.loc[clubs["domestic_competition_id"] == competition_id, "club_id"])
    pl_transfers = transfers[
        transfers["from_club_id"].isin(pl_club_ids) | transfers["to_club_id"].isin(pl_club_ids)
    ].copy()
    pl_transfers["transfer_date"] = pd.to_datetime(pl_transfers["transfer_date"], errors="coerce")

    today = now if now is not None else pd.Timestamp(datetime.now(timezone.utc).date())
    window_start = today - pd.Timedelta(days=days)
    window_end = today + pd.Timedelta(days=days)  # symmetric: "recent" also
    # bounds how far into the future an "announced" deal counts as
    # relevant news right now, rather than surfacing every pre-agreed
    # transfer ever recorded regardless of how far off it is.
    in_window = pl_transfers[
        pl_transfers["transfer_date"].notna()
        & (pl_transfers["transfer_date"] >= window_start)
        & (pl_transfers["transfer_date"] <= window_end)
    ].copy()
    if in_window.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    in_window["status"] = in_window["transfer_date"].apply(
        lambda d: "completed" if d <= today else "announced"
    )
    in_window = in_window.sort_values("transfer_date", ascending=False)
    return in_window[RESULT_COLUMNS].reset_index(drop=True)
