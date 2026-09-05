"""
Live Ingest — this season's snapshot, straight from the source.

Why this is a separate module from ingest.py: the official FPL API only
ever returns the CURRENT state (today's price, ownership, form) — it has
no history endpoint. ingest.py loads an already-finished historical
season in one shot from an archive; this module instead builds up the
*current* season's time series one snapshot at a time, by being run
again every gameweek. That recurring run is the real job of the Ingest
Agent once Phase 3 wires this in — the "Trigger: run before each
gameweek deadline" box in the pipeline diagram exists specifically for
this.

Field names below were confirmed against the live API response directly
(GET https://fantasy.premierleague.com/api/bootstrap-static/), not
guessed. This module hasn't been run end-to-end in the environment that
wrote it (no outbound internet there) — run it from Colab, which does
have internet, and tell me what happens if anything looks off.

Run with:
    python -m src.live_ingest
"""

import requests
import pandas as pd
from datetime import datetime, timezone
from src.db import get_engine

API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# FPL's element_type -> the same position strings used in the historical
# merged_gw.csv files (confirmed: GK, DEF, MID, FWD).
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_snapshot() -> pd.DataFrame:
    """Pull today's live price/ownership/form for every player.

    Only the columns analysis.py actually uses are kept, so this lines
    up with the historical 'prices' table even though the historical
    ingest keeps extra raw columns the live API doesn't expose the same
    way (minutes, goals_scored, etc. — those come from the per-gameweek
    'live' endpoint if you want them later, not bootstrap-static).

    'code' is the same stable cross-season player id ingest.py maps in
    via players_raw.csv — carrying it here too is what lets
    forecast_points() blend a player's historical average into this
    season's cold start without relying on name-string matching (which
    doesn't work: this API's 'web_name' is a short display form, not the
    historical archive's full/legal name).
    """
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    teams = {t["id"]: t["name"] for t in data["teams"]}
    current_gw = next((e["id"] for e in data["events"] if e["is_current"]), None)
    if current_gw is None:
        # Between gameweeks (deadline passed, match not played yet) or
        # pre-season — fall back to the next scheduled gameweek.
        current_gw = next((e["id"] for e in data["events"] if e["is_next"]), None)

    # Season string (e.g. "2026-27"), read from gameweek 1's own deadline
    # rather than guessed from today's date — this is what lets
    # recommend_squad() tell this season's rows apart from the historical
    # archives' rows even though both reuse gameweek numbers 1-38.
    gw1 = next((e for e in data["events"] if e["id"] == 1), None)
    if gw1 is not None:
        start_year = int(gw1["deadline_time"][:4])
        season = f"{start_year}-{str(start_year + 1)[-2:]}"
    else:
        season = "unknown-season"

    rows = []
    for el in data["elements"]:
        rows.append({
            "code": el["code"],
            "name": el["web_name"],
            "team": teams.get(el["team"], str(el["team"])),
            "position": POSITION_MAP.get(el["element_type"], "UNK"),
            "gameweek": current_gw,
            "season": season,
            "price_m": el["now_cost"] / 10,
            "selected": el["selected_by_percent"],
            "transfers_balance": el["transfers_in_event"] - el["transfers_out_event"],
            "transfers_in": el["transfers_in_event"],
            "transfers_out": el["transfers_out_event"],
            # Deliberately el["event_points"] (this gameweek's score), not
            # el["total_points"] (the season's running cumulative total) —
            # the historical archive's 'total_points' column is per-
            # gameweek, and forecast_points() averages/rolls this column
            # assuming every row is one gameweek's worth of points. Using
            # the cumulative field here would mix units with the
            # historical rows, and get worse every gameweek as the
            # cumulative number climbs.
            "total_points": el["event_points"],
            "kickoff_time": datetime.now(timezone.utc),
        })
    return pd.DataFrame(rows)


def append_snapshot() -> pd.DataFrame:
    """Fetch today's snapshot and append it to the 'prices' table.

    Safe to re-run within the same gameweek — it just adds another
    dated row per player rather than overwriting. If you want strictly
    one row per player per gameweek for forecast_points()/recommend_squad(),
    dedupe on (code, gameweek) keeping the most recent kickoff_time
    before running analysis.
    """
    snapshot = fetch_snapshot()
    engine = get_engine()
    snapshot.to_sql("prices", engine, if_exists="append", index=False)
    gw = snapshot["gameweek"].iloc[0]
    season = snapshot["season"].iloc[0]
    print(f"Appended {len(snapshot)} rows for {season}, gameweek {gw}.")
    return snapshot


if __name__ == "__main__":
    append_snapshot()
