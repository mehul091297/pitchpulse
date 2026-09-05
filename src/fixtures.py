"""
Fixture data and difficulty — used to time chip plays (Phase 6).

Source: https://fantasy.premierleague.com/api/fixtures/ — the same
official FPL API live_ingest.py and availability.py already pull from
(bootstrap-static), just a different endpoint on the same API. No new
external dependency, no scraping: every field used here comes straight
from FPL itself, including its own Fixture Difficulty Rating (FDR) —
we're not inventing a difficulty score, just reading the one FPL
already publishes and uses for its own in-app fixture ticker.

Schema (field names match FPL's documented fixtures response and the
team-id convention live_ingest.py already uses; not yet independently
re-confirmed against a live pull the way live_ingest.py's fields were
— verify directly the first time this runs in Colab, and tell me if a
field name doesn't match): each fixture has 'event' (gameweek number,
null for a fixture not yet slotted into one — postponed/rescheduled),
'team_h'/'team_a' (team ids, matching bootstrap-static's teams[].id),
'team_h_difficulty'/'team_a_difficulty' (FPL's own 1-5 FDR — 1 easiest,
5 hardest — from the HOME/AWAY side's perspective respectively),
'finished', 'kickoff_time'.

Run with:
    python -m src.fixtures
"""

import pandas as pd
import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def fetch_fixtures() -> pd.DataFrame:
    """One row per real fixture: gameweek, home team, away team, each
    side's FDR, finished flag. Fixtures with no gameweek assigned yet
    (event is null) are dropped — chip timing can't plan around a
    fixture that hasn't been slotted into a gameweek.
    """
    teams_resp = requests.get(BOOTSTRAP_URL, timeout=30)
    teams_resp.raise_for_status()
    teams = {t["id"]: t["name"] for t in teams_resp.json()["teams"]}

    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    fixtures = resp.json()

    rows = []
    for f in fixtures:
        if f.get("event") is None:
            continue
        rows.append({
            "gameweek": f["event"],
            "team_h": teams.get(f["team_h"], str(f["team_h"])),
            "team_a": teams.get(f["team_a"], str(f["team_a"])),
            "team_h_difficulty": f["team_h_difficulty"],
            "team_a_difficulty": f["team_a_difficulty"],
            "finished": f["finished"],
            "kickoff_time": f.get("kickoff_time"),
        })
    return pd.DataFrame(rows)


def team_gameweek_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Reshape from one-row-per-match to one-row-per-(team, gameweek) —
    the shape chip timing actually needs, since a chip decision is made
    per squad (many teams) per gameweek, not per individual match.

    A team appearing twice for the same gameweek IS a double gameweek
    (fixture_count=2 — both matches' difficulty averaged into
    avg_difficulty for that week's summary); a team with NO row for a
    gameweek that other teams do have is a blank for them, which
    callers detect by absence (a left join against the full gameweek
    range, not a 0 in this table).
    """
    home = fixtures[["gameweek", "team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team", "team_h_difficulty": "difficulty"}
    )
    away = fixtures[["gameweek", "team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team", "team_a_difficulty": "difficulty"}
    )
    combined = pd.concat([home, away], ignore_index=True)
    return (
        combined.groupby(["team", "gameweek"])
        .agg(fixture_count=("difficulty", "size"), avg_difficulty=("difficulty", "mean"))
        .reset_index()
    )


def load_team_fixtures() -> pd.DataFrame:
    """Convenience one-shot: fetch + reshape in one call, for callers
    (the chip-strategy tool, ad-hoc Colab exploration) that just want
    the per-team-per-gameweek table and don't care about the raw
    per-match rows in between.
    """
    return team_gameweek_fixtures(fetch_fixtures())


def next_actionable_gameweek() -> int:
    """The earliest gameweek a chip decision could still actually apply
    to — i.e. the next one whose transfer deadline HASN'T passed yet.

    Deliberately prefers FPL's own 'is_next' event flag over
    'is_current': by the time a gameweek is "current" its deadline has
    already passed and squads are locked, so a chip recommended for it
    would be advice nobody could still act on. Falls back to
    'is_current' only if there's no 'is_next' at all (the very last
    gameweek of the season).
    """
    resp = requests.get(BOOTSTRAP_URL, timeout=30)
    resp.raise_for_status()
    events = resp.json()["events"]
    nxt = next((e["id"] for e in events if e["is_next"]), None)
    if nxt is not None:
        return nxt
    cur = next((e["id"] for e in events if e["is_current"]), None)
    if cur is not None:
        return cur
    raise RuntimeError(
        "Could not determine the next actionable gameweek from FPL's own "
        "events data — the season may have ended, or the events schema "
        "has changed since this was written."
    )


if __name__ == "__main__":
    tf = load_team_fixtures()
    print(tf.sort_values(["gameweek", "team"]).to_string(index=False))
