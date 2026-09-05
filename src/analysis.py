"""
Analyst Agent logic (Phase 2 -> Phase 5).

Dataset: Track A — Fantasy Premier League price market (see src/ingest.py
for column details). After ingest.py's clean() step, the 'prices' table
has: name, team, position, gameweek, season, code, price_m, selected,
transfers_balance, transfers_in, transfers_out, total_points,
kickoff_time (plus the raw 'value' and 'element' columns for
historically-ingested rows).

'code' matters as much as 'season': it's FPL's stable, cross-season
player id (unlike 'name', which isn't spelled the same way in the
historical archive as in the live API, and unlike 'element'/'id', which
FPL reassigns to a different player every season). forecast_points()'s
cold-start fallback and recommend_squad()'s dedup both key off it.

'season' matters more than it looks: gameweek numbers (1-38) repeat
every year, so anything comparing "gameweek" across multiple loaded
seasons — including a live current-season snapshot alongside historical
archives — has to filter by season first or it'll silently mix seasons
together. recommend_squad() does this; price_trend()/demand_signal()
don't yet (fine for now since they're only plotted per-season in
Phase 2, but worth revisiting if the dashboard ever charts multiple
seasons on one axis).

Phase 2: price_trend() / demand_signal() just need to return DataFrames
the dashboard can plot. Phase 4 adds the predictive layer — forecast_points()
and anomaly_scores(). Phase 5 adds the prescriptive layer — recommend_squad(),
which turns forecast_points()'s output into an actual pick under real
constraints. Together these four functions are what the Analyst Agent
wraps as CrewAI tools in Phase 3, and their output is the "findings +
flags" the Report Agent narrates in the pipeline diagram.
"""

import pandas as pd
import pulp
from src.db import get_engine


def _load_prices() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM prices", engine)


def list_seasons() -> list[str]:
    """Every season present in the 'prices' table, oldest first — for
    populating a season picker (the dashboard's season selectbox uses
    this directly).
    """
    df = _load_prices()
    _require_season_column(df)
    return sorted(df["season"].unique().tolist())


def _season_filtered(df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    """Filter to one season, defaulting to the most recent one present.

    Same collision this module's docstring already warns about:
    gameweek numbers (1-38) repeat every season, so any chart plotting
    'gameweek' on the x-axis has to be scoped to a single season first —
    otherwise gameweek 1 of three different seasons lands on the same
    x-axis point. price_trend()/demand_signal() used to skip this (fine
    when only historical data existed; not fine now that a live
    2026-27 snapshot sits in the same table).
    """
    _require_season_column(df)
    season = season or df["season"].max()
    return df[df["season"] == season]


def price_trend(
    entity_col: str = "name",
    date_col: str = "gameweek",
    price_col: str = "price_m",
    season: str | None = None,
) -> pd.DataFrame:
    """Average price over time, grouped by player (or team, or position),
    within one season (defaults to the most recent season in the data —
    pass `season` explicitly, e.g. from list_seasons(), to look at another).
    """
    df = _season_filtered(_load_prices(), season)
    return (
        df.groupby([entity_col, date_col])[price_col]
        .mean()
        .reset_index()
    )


def demand_signal(
    entity_col: str = "name",
    date_col: str = "gameweek",
    season: str | None = None,
) -> pd.DataFrame:
    """Net transfers (in - out) per gameweek — the demand pressure that
    drives price changes — within one season (see price_trend()'s
    `season` argument). Useful on its own, and as an input to
    anomaly_scores().
    """
    df = _season_filtered(_load_prices(), season)
    return (
        df.groupby([entity_col, date_col])["transfers_balance"]
        .sum()
        .reset_index()
    )


def _require_season_column(df: pd.DataFrame) -> None:
    if "season" not in df.columns:
        raise ValueError(
            "The 'prices' table has no 'season' column — it was probably built "
            "before that fix landed. Drop the table and re-run src/ingest.py "
            "(and src/live_ingest.py) to rebuild it with season tags, otherwise "
            "gameweek numbers collide across seasons (1-38 repeats every year) "
            "and forecasts/recommendations end up picking stale historical rows."
        )


def _require_code_column(df: pd.DataFrame) -> None:
    if "code" not in df.columns:
        raise ValueError(
            "The 'prices' table has no 'code' column — it was probably built "
            "before that fix landed. Drop the table and re-run src/ingest.py "
            "(with players_raw_<season>.csv downloaded — see its module "
            "docstring) and src/live_ingest.py to rebuild it with FPL's stable "
            "player code, otherwise forecast_points() can't reliably tell a "
            "historical row and a live row are the same player (name strings "
            "don't match across the two sources) and its cold-start fallback "
            "silently does nothing for most players."
        )


def forecast_points(window: int = 5, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Phase 4: predict each player's next-gameweek points.

    Naive baseline on purpose: a rolling average of each player's own last
    `window` gameweeks, shifted by one so a gameweek never sees its own
    result. This is the "start simple" baseline the plan recommends before
    reaching for anything fancier (recent form, fixture difficulty,
    minutes-played trend are the natural next refinements) — a model you
    can explain in one sentence beats one you can't.

    Grouped by (code, season), not just code — otherwise a player's
    rolling average would quietly blend last gameweek of one season into
    gameweek 1 of the next as if they were consecutive matches. 'code' —
    not 'name' — is the grouping key: it's FPL's stable id, unchanged
    across seasons and across the live API, whereas 'name' isn't safe to
    match on (see _require_code_column's docstring / ingest.py).

    Returns the full price/points table with a new `predicted_points`
    column, ready to feed straight into recommend_squad().
    """
    df = _load_prices() if df is None else df.copy()
    _require_season_column(df)
    _require_code_column(df)
    df = df.sort_values(["code", "season", "gameweek"])
    df["predicted_points"] = (
        df.groupby(["code", "season"])["total_points"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    )

    # Cold start: a player's first gameweek(s) in a season have no
    # in-season history yet, so the roll above leaves them NaN — most
    # visible right now, early in 2026-27, where most players only have
    # one or two live snapshots so far. Fall back to that player's
    # average points-per-gameweek from whatever OTHER rows of theirs
    # exist in the loaded data (prior seasons, matched via 'code' so a
    # transfer or a shared surname can't cause a wrong match). This is
    # the actual reason forecast_points() cares about historical seasons
    # at all — they're the cold-start prior for a new season, not just a
    # backtest. Rows whose 'code' never mapped (see ingest.py) simply
    # get no fallback and stay NaN — excluded from recommend_squad()'s
    # pool rather than given a wrong one.
    player_avg = df.groupby("code")["total_points"].transform("mean")
    df["predicted_points"] = df["predicted_points"].fillna(player_avg)

    return df


def recommend_squad(
    budget: float = 100.0,
    squad_size: dict | None = None,
    max_per_team: int = 3,
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """Phase 5: the prescriptive layer — pick the 15 players that maximize
    total predicted points without breaking real FPL squad rules.

    This is a constrained optimization problem (integer/linear
    programming), solved with PuLP: maximize sum(predicted_points) subject
    to (a) total price <= budget, (b) exact position counts, (c) at most
    `max_per_team` players from any one real club. It's the same recipe
    used by most public FPL-optimizer tools — the differentiator here is
    that the Report Agent (Phase 3) explains *why*, using each pick's
    predicted_points and price against the alternatives it passed over,
    instead of just printing a list.

    Requires forecast_points() to have been run first (or pass its output
    in via `df`) — recommend_squad() only picks from the most recent
    gameweek of the most recent SEASON that has a predicted_points value.

    Picking "most recent season" matters because gameweek numbers reset
    every year (1-38 repeats), so if the table has both historical
    seasons and this season's live snapshots, a plain
    df["gameweek"].max() would grab gameweek 38 from an old, finished
    season instead of this season's actual latest gameweek — which is
    exactly the bug that made an earlier version of this function
    recommend players by their 2023-24/2024-25 club instead of today's.

    Returns (squad_dataframe, solver_status). Check solver_status == "Optimal"
    before trusting the result — "Infeasible" means the constraints can't
    all be satisfied (e.g. budget too low for the position requirements).
    """
    squad_size = squad_size or {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    df = forecast_points(df=df) if df is None else df
    _require_season_column(df)
    _require_code_column(df)

    latest_season = df["season"].max()  # "2026-27" > "2024-25" sorts correctly as strings
    season_df = df[df["season"] == latest_season]
    latest_gw = season_df["gameweek"].max()
    pool = (
        season_df[season_df["gameweek"] == latest_gw]
        .dropna(subset=["predicted_points"])
        .drop_duplicates(subset=["code"])
        .copy()
    )
    if pool.empty:
        raise ValueError(
            "No players with a predicted_points value at the latest gameweek — "
            "run forecast_points() first, or check the 'window' has enough history."
        )

    prob = pulp.LpProblem("PitchPulse_Squad", pulp.LpMaximize)
    picks = pulp.LpVariable.dicts("pick", pool.index, cat="Binary")

    # Objective: maximize total predicted points across the 15 picks.
    prob += pulp.lpSum(picks[i] * pool.loc[i, "predicted_points"] for i in pool.index)

    # Budget constraint.
    prob += pulp.lpSum(picks[i] * pool.loc[i, "price_m"] for i in pool.index) <= budget

    # Exact squad composition (2 GK, 5 DEF, 5 MID, 3 FWD by default).
    for pos, count in squad_size.items():
        idx = pool[pool["position"] == pos].index
        prob += pulp.lpSum(picks[i] for i in idx) == count

    # Max players from any one real club.
    for _team, group in pool.groupby("team"):
        prob += pulp.lpSum(picks[i] for i in group.index) <= max_per_team

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    chosen = [i for i in pool.index if picks[i].value() == 1]
    squad = pool.loc[chosen, ["name", "team", "position", "price_m", "predicted_points"]]
    squad = squad.sort_values(["position", "predicted_points"], ascending=[True, False])
    return squad, pulp.LpStatus[status]


def anomaly_scores(threshold: float = 2.0) -> pd.DataFrame:
    """Phase 4: flag player-gameweeks where transfers_balance or price
    movement deviates sharply from that player's own recent baseline
    (e.g. z-score over a rolling window). Output feeds directly into the
    Report Agent's alerts — this is the signal a "price spike" report
    line comes from.
    """
    raise NotImplementedError("Implement in Phase 4.")
