"""
Player availability — injury/suspension/doubt status, straight from
FPL's own official data (Phase 6).

Why this exists: Mehul's own example (Dedic starting for Newcastle at
right-back because Tino Livramento is injured, but expected to lose
those minutes back once Livramento returns) is exactly the kind of
signal forecast_points()'s rolling average can't see on its own — a
rolling average of past points has no idea WHY a player's minutes might
change going forward. The fix isn't scraping news sites (same fragility
problem already hit and avoided once, in src/transfers.py's docstring)
— the official bootstrap-static endpoint live_ingest.py already pulls
from carries exactly this signal for every player already, with no new
external dependency at all.

Why a separate table from 'prices' rather than new columns on it: the
'prices' table already has real rows from two different ingest paths
(ingest.py's historical bulk load, live_ingest.py's per-gameweek
append) with a schema that's been stable in the deployed database.
Appending a DataFrame with extra columns onto an existing SQL table via
to_sql(if_exists="append") only works if the columns already match —
adding new ones here would break every existing deployment's db on the
next live_ingest run. A separate 'availability' table, keyed the same
way (code, season, gameweek), sidesteps that entirely and keeps this
squarely optional and additive, same spirit as src/transfers.py.

Schema (field names match live_ingest.py's own confirmed-live fields
where they overlap; the status/news/chance_of_playing fields are FPL's
documented element fields, not yet independently re-confirmed against
a live pull the way live_ingest.py's fields were — verify directly the
first time this runs in Colab, same ask live_ingest.py's own docstring
makes, and tell me if a field name doesn't match): each element has
'status' (single char — 'a' available, 'd' doubtful, 'i' injured, 's'
suspended, 'u' unavailable/left the league, 'n' not considered for
selection), 'news' (free text, empty string when there's nothing to
report), 'chance_of_playing_next_round' (0/25/50/75/100, or null when
there's no doubt registered at all).

Run with:
    python -m src.availability
"""

from datetime import datetime, timezone

import pandas as pd
import requests

from src.db import get_engine

API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

STATUS_LABELS = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not considered for selection",
}

# Statuses where a manager should NOT be picking this player right now,
# regardless of their forecasted points — separate from 'd' (doubtful),
# which is a real but partial risk, not a reason to drop someone
# outright. recommend_squad() uses this set to exclude, not 'd'.
UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}


def fetch_availability_snapshot() -> pd.DataFrame:
    """Pull today's live status/news/doubt-percentage for every player.

    Reuses the exact same request live_ingest.fetch_snapshot() makes
    (same endpoint, same response) — if you're calling both, it's an
    honest duplicate network call for now (keeps the two modules
    independent, same as ensure_data_tool calling live_ingest and this
    separately) rather than a shared fetch, at the cost of one extra
    request per pipeline run.
    """
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    teams = {t["id"]: t["name"] for t in data["teams"]}
    current_gw = next((e["id"] for e in data["events"] if e["is_current"]), None)
    if current_gw is None:
        current_gw = next((e["id"] for e in data["events"] if e["is_next"]), None)

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
            "season": season,
            "gameweek": current_gw,
            "status": el.get("status", "a"),
            "news": el.get("news") or "",
            "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            "snapshot_time": datetime.now(timezone.utc),
        })
    return pd.DataFrame(rows)


def append_availability_snapshot() -> pd.DataFrame:
    """Fetch today's status snapshot and append it to the 'availability'
    table. Append-only, same reasoning as live_ingest.append_snapshot()
    — latest_availability() below does the dedup on read.
    """
    snapshot = fetch_availability_snapshot()
    engine = get_engine()
    snapshot.to_sql("availability", engine, if_exists="append", index=False)
    print(
        f"Appended {len(snapshot)} availability rows for "
        f"{snapshot['season'].iloc[0]}, gameweek {snapshot['gameweek'].iloc[0]}."
    )
    return snapshot


def latest_availability() -> pd.DataFrame:
    """Most recent status per player. Deduped the same way _load_prices()
    is (latest snapshot_time per code) — this table is append-only for
    the same reason 'prices' is, so a straight SELECT would double-count
    anyone snapshotted more than once.

    Raises RuntimeError if the table doesn't exist yet (nobody's called
    append_availability_snapshot() in this database) — callers that
    treat this as optional (recommend_squad(), the crew tool) catch this
    and degrade gracefully rather than erroring out the whole pipeline.
    """
    engine = get_engine()
    import sqlalchemy

    if not sqlalchemy.inspect(engine).has_table("availability"):
        raise RuntimeError(
            "No 'availability' table found — run "
            "src.availability.append_availability_snapshot() at least once "
            "(same idea as live_ingest.py's append_snapshot(), just for "
            "status instead of price)."
        )
    df = pd.read_sql("SELECT * FROM availability", engine)
    if df.empty:
        return df
    df = df.sort_values("snapshot_time").drop_duplicates(subset=["code"], keep="last")
    df["status_label"] = df["status"].map(STATUS_LABELS).fillna(df["status"])
    return df


if __name__ == "__main__":
    append_availability_snapshot()
