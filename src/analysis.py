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
    """All rows from the 'prices' table, deduplicated to one row per
    player-gameweek-season.

    live_ingest.append_snapshot() is deliberately append-only (see its
    docstring) — every time ensure_data_tool/ensure_data() runs (once per
    crew run, and the dashboard's cached call periodically), it adds
    ANOTHER row for the current gameweek rather than overwriting the
    earlier one. Confirmed in practice, not just in theory: two separate
    crew runs against the same gameweek (2026-27 GW2) each logged
    "Appended N rows for 2026-27, gameweek 2" — so the real database now
    has duplicate snapshots of that gameweek, one per run.

    Left un-deduped, that's mostly harmless for a single-point read like
    recommend_squad() (its own drop_duplicates just picks one snapshot,
    close enough), but it would be actively wrong for anomaly_scores():
    diffing between two same-gameweek re-ingests — taken minutes apart,
    with transfers_in/out still accumulating within that gameweek's
    window — would look exactly like a real week-over-week price/demand
    swing and get flagged as a fabricated anomaly. Deduplicated here,
    once, so every caller sees a clean series without needing its own
    copy of this logic.

    Keeps the most recently ingested row per (code-or-name, season,
    gameweek) — ordered by kickoff_time, which live_ingest.py stamps
    with the actual ingestion time (for historical rows it's the real
    match kickoff time instead, but those already have zero duplicates
    per (element, round, season) from ingest.py's own dedup, so which one
    "wins" there is moot). Falls back to 'name' for the rare row whose
    'code' never mapped, same reasoning as ingest.py's ghost-row filter.
    """
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM prices", engine)
    if not {"season", "gameweek"}.issubset(df.columns):
        return df  # let callers' own _require_*_column raise a clearer error
    df = df.copy()
    df["_dedup_key"] = df["code"].fillna(df["name"]) if "code" in df.columns else df["name"]
    df = df.sort_values("kickoff_time")
    df = df.drop_duplicates(subset=["_dedup_key", "season", "gameweek"], keep="last")
    return df.drop(columns=["_dedup_key"])


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


def anomaly_scores(
    threshold: float = 2.0,
    window: int = 5,
    min_periods: int = 3,
    season: str | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Phase 4: flag player-gameweeks where a price change or transfer
    swing deviates sharply from that player's OWN recent baseline — not
    from the league-wide average, since a £4.5m benchwarmer and a £15m
    star have completely different normal ranges for both metrics.

    For each metric — gameweek-over-gameweek price change, and
    transfers_balance — this computes a rolling z-score against that
    player's trailing history within the same season (grouped by 'code',
    not 'name', for the same cross-season-identity reason every other
    function here does; scoped to one season at a time so a summer
    transfer/price reset never gets compared against last year's numbers).
    The rolling mean/std are shifted by one gameweek first, so a
    gameweek's own value can never leak into its own baseline — the same
    look-ahead guard forecast_points() uses.

    Two guards keep this from manufacturing false signal out of thin
    data, the same failure mode price_movers_tool had before its fix:

    - A player's first `min_periods` gameweeks of a season have no
      reliable baseline yet (not enough history to say what's "normal"
      for THEM specifically) — excluded rather than scored. This matters
      most right now: early in 2026-27, nobody has enough in-season
      history to clear this bar yet, so don't be surprised if this
      returns empty against the live current season — that's the honest
      answer, not a bug (see the "insufficient history" test case).
    - A rolling std of exactly zero (a player whose price/demand was
      perfectly constant every prior gameweek — common here, since FPL
      prices sit flat for most players most weeks) can't be divided by
      directly. That's NOT a reason to exclude these rows though: a
      player who hasn't moved AT ALL in min_periods+ weeks and then
      suddenly does is arguably the single clearest anomaly this
      function can find, not one to suppress. A tiny epsilon floor on
      the std turns "any deviation from a perfectly flat baseline" into
      a very large (not infinite/NaN) z-score, capped at a sentinel
      magnitude so it still sorts to the top without printing an
      absurd number, while a value that stays exactly at the constant
      still correctly scores z=0 — so a truly unchanged player is
      never falsely flagged.

    Returns one row per flagged player-gameweek-metric — an alerts
    table, not the full history — sorted by severity (|z-score|
    descending). Columns: code, name, team, season, gameweek, metric
    ('price_change' or 'transfers_balance'), value, z_score. Empty
    DataFrame (same columns, zero rows) if nothing clears the threshold.
    Defaults to the most recent season present, like price_trend()/
    demand_signal() — pass `season` explicitly to check another one.
    """
    columns = ["code", "name", "team", "season", "gameweek", "metric", "value", "z_score"]
    df = _load_prices() if df is None else df.copy()
    _require_season_column(df)
    _require_code_column(df)

    season = season or df["season"].max()
    season_df = df[df["season"] == season].sort_values(["code", "gameweek"]).copy()

    # Anomalies live in the CHANGE, not the level: a player sitting at a
    # flat £6.0m every week isn't news, but a sudden jump from £6.0m to
    # £6.2m is. transfers_balance is already a per-gameweek net swing, so
    # it needs no such transformation.
    season_df["price_change"] = season_df.groupby("code")["price_m"].diff()

    alerts = []
    for metric in ("price_change", "transfers_balance"):
        grouped = season_df.groupby("code")[metric]
        baseline_mean = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
        )
        baseline_std = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).std()
        )
        epsilon = 1e-6  # far below any realistic price/transfer std, so a
        # genuinely nonzero std is essentially unaffected by the floor.
        safe_std = baseline_std.where(baseline_std > 0, epsilon)
        z = (season_df[metric] - baseline_mean) / safe_std
        # The epsilon floor makes a true zero-std baseline's z-score
        # technically correct but absurdly large (a real deviation
        # divided by 1e-6) — fine for ranking severity, useless as a
        # number a report would print. Capped at a sentinel magnitude:
        # still sorts above every "normal" std-based z-score, without
        # printing something like "z-score: 49,900,000,000".
        MAX_Z = 50.0
        z = z.clip(lower=-MAX_Z, upper=MAX_Z)
        has_baseline = baseline_std.notna()  # NaN = not enough history yet
        flagged = season_df[has_baseline & z.notna() & (z.abs() >= threshold)].copy()
        if flagged.empty:
            continue
        flagged["metric"] = metric
        flagged["value"] = flagged[metric]
        flagged["z_score"] = z.loc[flagged.index]
        alerts.append(flagged[columns])

    if not alerts:
        return pd.DataFrame(columns=columns)

    result = pd.concat(alerts, ignore_index=True)
    return result.reindex(result["z_score"].abs().sort_values(ascending=False).index).reset_index(drop=True)
