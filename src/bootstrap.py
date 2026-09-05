"""
Bootstrap — make sure the 'prices' table exists and has this season's
live snapshot, downloading and ingesting from scratch if it doesn't.

Why this exists as its own module: in Colab, a human runs ingest.py and
live_ingest.py by hand, one cell at a time, watching the output. Streamlit
Community Cloud isn't like that — every deploy (and every wake from
sleep) starts from a clean checkout of the GitHub repo, with no
data/raw/ or data/processed/ carried over, and nobody there to run cells.
The dashboard (dashboard/app.py) calls ensure_data() once on startup so
it's self-sufficient: same download URLs ingest.py's own docstring
documents, same ingest.py/live_ingest.py logic underneath, just driven
automatically instead of by hand.

Run with:
    python -m src.bootstrap
"""

from pathlib import Path

import requests
import sqlalchemy

from src.db import get_engine
from src.ingest import RAW_DIR, RAW_FILES, SEASON_PATTERN, main as run_ingest
from src.live_ingest import append_snapshot

GW_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
PLAYERS_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/players_raw.csv"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _has_prices_table() -> bool:
    return sqlalchemy.inspect(get_engine()).has_table("prices")


def ensure_historical() -> None:
    """Download + ingest historical seasons. Idempotent — does nothing
    if the 'prices' table already exists, so it's cheap to call on
    every dashboard load.
    """
    if _has_prices_table():
        return

    for filename in RAW_FILES:
        match = SEASON_PATTERN.search(filename)
        if not match:
            continue
        season = match.group(1)

        gw_path = RAW_DIR / filename
        if not gw_path.exists():
            _download(GW_URL.format(season=season), gw_path)

        players_path = RAW_DIR / f"players_raw_{season}.csv"
        if not players_path.exists():
            _download(PLAYERS_URL.format(season=season), players_path)

    run_ingest()


def ensure_data() -> None:
    """Historical (once) + a fresh live snapshot (every call — this is
    what keeps current-season prices from going stale). The dashboard
    wraps this call in an st.cache_resource(ttl=...) so it doesn't hit
    the live API on every rerun, just periodically.
    """
    ensure_historical()
    append_snapshot()


if __name__ == "__main__":
    ensure_data()
