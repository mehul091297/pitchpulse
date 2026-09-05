"""
Ingest Agent (Phase 1 -> Phase 3).

Dataset: Track A — Fantasy Premier League price market.

Source: vaastav/Fantasy-Premier-League (github.com/vaastav/Fantasy-Premier-League)
Each season folder (e.g. data/2023-24/) contains gws/merged_gw.csv — one row
per player per gameweek, confirmed columns include:
  name, position, team, element (player id), round / GW (gameweek number),
  value (price, tenths of £m — 125 = £12.5m), selected (ownership count),
  transfers_in, transfers_out, transfers_balance, total_points, minutes,
  goals_scored, assists, kickoff_time

That's the real time series this project needs: `value` moving week to week
is the "price market," and `transfers_balance` is the demand signal behind
every move. Download one or more seasons' merged_gw.csv into data/raw/
before running this.

Also needed, one per season: players_raw.csv (same repo, same season
folder — e.g. data/2023-24/players_raw.csv). merged_gw.csv's 'element'
column is a season-only id FPL reassigns to a different player every
year, so it can't identify a player across seasons or against the live
API. players_raw.csv has both 'id' (== that season's 'element') and
'code' — FPL's actual stable identifier, unchanged across seasons and
also present in the live bootstrap-static API — side by side, which is
what makes it possible to tell "this is the same player" between a
historical row and a live snapshot. Confirmed columns (verified against
github.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv):
id, code, first_name, second_name, web_name, element_type (among ~100
others). load_raw() below uses it purely as an id -> code lookup.

Phase 1 goal: get this working as a plain script. Phase 3 goal: wrap the
same logic as a CrewAI tool so an actual agent calls it.

Run with:
    python -m src.ingest
"""

import re
from pathlib import Path
import pandas as pd
from src.db import get_engine, REPO_ROOT

# Anchored to the repo root (see src/db.py's REPO_ROOT), not to the
# process's cwd — same reasoning as DEFAULT_DB_PATH there.
RAW_DIR = REPO_ROOT / "data" / "raw"

# Every merged_gw.csv you've downloaded, one per season. Matches the
# seasons the Colab notebook's download cell fetches by default — if you
# add another season there, add its filename here too.
RAW_FILES = ["merged_gw_2023-24.csv", "merged_gw_2024-25.csv"]

SEASON_PATTERN = re.compile(r"(\d{4}-\d{2})")


def _load_code_map(season: str) -> dict:
    """That season's element id -> FPL's stable cross-season 'code',
    read from players_raw_<season>.csv. See the module docstring for why.
    """
    path = RAW_DIR / f"players_raw_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path} — download players_raw.csv for each season "
            "from github.com/vaastav/Fantasy-Premier-League (same folder as "
            "merged_gw.csv) into data/raw/, saved as players_raw_<season>.csv."
        )
    players = pd.read_csv(path, usecols=["id", "code"])
    return dict(zip(players["id"], players["code"]))


def load_raw() -> pd.DataFrame:
    """Read and concatenate every configured merged_gw.csv from data/raw/.

    Critical: gameweek numbers (1-38) repeat every season, so each
    season's rows are tagged with a 'season' column here, before
    concatenation — without it, "gameweek 38" is ambiguous across
    multiple seasons' worth of rows, which silently breaks any code
    that assumes df["gameweek"].max() means "the most recent gameweek
    played" (recommend_squad() relies on exactly this — see analysis.py).

    Each row also gets a 'code' column, mapped from that season's
    players_raw.csv (element -> code). A handful of rows can fail to map
    (a player removed from FPL mid-season) — the row is kept either way,
    since it's still valid input to that season's own rolling average;
    it just won't have a code to blend into a later season's cold start.
    """
    frames = []
    for filename in RAW_FILES:
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path} — download merged_gw.csv for each season "
                "from github.com/vaastav/Fantasy-Premier-League into data/raw/ "
                "and list the filename(s) in RAW_FILES above."
            )
        match = SEASON_PATTERN.search(filename)
        if not match:
            raise ValueError(
                f"Can't tell what season {filename!r} is from its name — "
                "rename it to include the season, e.g. 'merged_gw_2023-24.csv'."
            )
        season = match.group(1)
        frame = pd.read_csv(path)
        frame["season"] = season
        frame["code"] = frame["element"].map(_load_code_map(season))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, dtype fixes, dedup — same moves as the
    Python EDA assignment: check dtypes, coerce dates, drop/flag nulls
    on key columns, drop exact duplicates.
    """
    # 'element' (player id) is reused/reassigned every season by FPL, so
    # deduping on (element, round) ALONE across concatenated seasons can
    # collide players from different seasons — 'season' has to be part
    # of the uniqueness key too.
    df = df.drop_duplicates(subset=["element", "round", "season"])
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce")
    df["price_m"] = df["value"] / 10  # tenths of £m -> £m, easier to read on charts
    gw_col = "round" if "round" in df.columns else "GW"
    df = df.rename(columns={gw_col: "gameweek"})

    # Drop "ghost" player-seasons: a player with zero minutes across every
    # gameweek of a season never actually featured in the Premier League
    # that year. Verified against real data (Mehul caught this) — Harry
    # Kane has 36 rows in the 2023-24 archive, one per gameweek, every one
    # with minutes=0 and total_points=0, price frozen at a single value the
    # entire season, and 'selected' (ownership) steadily decaying as
    # managers gradually sold a player who was never going to play — he'd
    # already transferred to Bayern Munich before the season kicked off.
    # This is a leftover entry in the source archive (FPL's game database
    # hadn't fully dropped him yet when the season's squads locked in), not
    # a bug in this pipeline — but leaving it in would make a departed
    # player show up as a "top priced player" in the dashboard's default
    # chart selection, and would (harmlessly, since his 'code' won't match
    # any current player, but still uselessly) pollute the cold-start
    # average pool in forecast_points(). Filtered per (code, season): a
    # player who didn't feature at all in ONE season but did in others
    # only loses that one season's rows, not their whole history.
    # Group by (code, season), falling back to name for the rare row whose
    # code never mapped (see _load_code_map) — plain groupby(["code", ...])
    # would otherwise merge every NaN-code row into one shared bucket
    # (pandas' default dropna=True drops them from transform() entirely,
    # leaving NaN, which `> 0` reads as False and would wrongly drop a real
    # active player just for lacking a code; dropna=False merges *different*
    # unmapped players' minutes together instead, which is equally wrong).
    group_key = df["code"].fillna(df["name"])
    minutes_by_player_season = df.groupby([group_key, "season"])["minutes"].transform("sum")
    df = df[minutes_by_player_season > 0]

    return df


def main() -> None:
    raw = load_raw()
    clean_df = clean(raw)

    engine = get_engine()
    clean_df.to_sql("prices", engine, if_exists="replace", index=False)
    print(f"Wrote {len(clean_df)} rows to the 'prices' table.")


if __name__ == "__main__":
    main()
