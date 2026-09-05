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
constraints. Phase 6 extends the prescriptive layer two ways:
recommend_squad() now excludes injured/suspended players by default
(src/availability.py), and recommend_chip_strategy() times the season's
four chip types against real fixture data (src/fixtures.py). Together
these functions are what the Analyst Agent wraps as CrewAI tools in
Phase 3, and their output is the "findings + flags" the Report Agent
narrates in the pipeline diagram.
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
    exclude_unavailable: bool = True,
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

    `exclude_unavailable` (Phase 6, default True): drops anyone whose
    latest known status (src/availability.py) is injured, suspended,
    unavailable, or not considered for selection, plus anyone flagged
    at exactly 0% chance of playing next round. Doubtful players ('d'
    status, partial chance-of-playing) are deliberately kept in the
    pool rather than dropped — a real but partial risk isn't the same
    as "don't pick this player at all," and forecast_points()'s own
    rolling average already reflects reduced minutes for someone who's
    been managed carefully. Degrades gracefully to the old behavior
    (no filtering) if no availability data has been ingested yet — this
    is a real improvement when the data exists, not a hard requirement,
    same spirit as src/transfers.py's optional enrichments.

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

    if exclude_unavailable:
        from src.availability import UNAVAILABLE_STATUSES, latest_availability

        try:
            avail = latest_availability()
        except RuntimeError:
            avail = None  # no availability data ingested yet — skip the filter
        if avail is not None and not avail.empty:
            unavailable_codes = set(
                avail[
                    avail["status"].isin(UNAVAILABLE_STATUSES)
                    | (avail["chance_of_playing_next_round"] == 0)
                ]["code"]
            )
            pool = pool[~pool["code"].isin(unavailable_codes)]

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
    squad = pool.loc[chosen, ["code", "name", "team", "position", "price_m", "predicted_points"]]
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


def player_availability(codes: list | None = None) -> pd.DataFrame:
    """Phase 6: current injury/suspension/doubt status per player, from
    src/availability.py's latest snapshot. `codes` optionally filters to
    a specific set of player codes (e.g. a recommended squad) rather
    than every player in the league. Raises RuntimeError if no
    availability data has been ingested yet — same contract as
    latest_availability() itself.
    """
    from src.availability import latest_availability

    df = latest_availability()
    if codes is not None and not df.empty:
        df = df[df["code"].isin(codes)]
    return df


def _fixture_easiness_factor(avg_difficulty: float) -> float:
    """FPL's own Fixture Difficulty Rating runs 1 (easiest) to 5
    (hardest). Converted here to a symmetric multiplier centered on 1.0
    at the middle rating (3), so a squad's baseline predicted_points is
    scaled up for easy fixtures and down for hard ones. This is an
    explainable adjustment, not a fitted model — same "start simple,
    explain in one sentence" philosophy as forecast_points()'s rolling
    average — capped so one extreme rating can't swing a projection to
    an implausible multiple.
    """
    factor = (6 - avg_difficulty) / 3
    return max(0.3, min(1.7, factor))


def recommend_chip_strategy(
    squad_codes: list,
    current_gameweek: int,
    team_fixtures: pd.DataFrame,
    horizon: int = 8,
    bench_size: int = 4,
    df: pd.DataFrame | None = None,
) -> dict:
    """Phase 6: recommend which upcoming gameweek to play each of the
    season's four chips, for a given squad.

    FPL's real chip rules, which this respects: two of each chip
    (Wildcard, Free Hit, Triple Captain, Bench Boost), one set usable in
    the first half of the season (gameweeks 1-19) and a separate fresh
    set from gameweek 20 onward — unused first-half chips expire rather
    than carrying over. This function only ever recommends gameweeks
    inside whichever half `current_gameweek` currently falls in (and
    only from `current_gameweek` onward within it) — it won't suggest a
    gameweek that's already passed, or one in a half you can't reach
    with this half's chips.

    The core signal: each squad player's forecast_points() baseline
    (their own rolling average — this function doesn't build a new
    predictive model) adjusted, per candidate gameweek, by that
    player's real team's fixture(s) that week — FPL's own difficulty
    rating (via _fixture_easiness_factor) and fixture COUNT (0 for a
    blank gameweek, 1 normally, 2 for a double). That adjusted number is
    what actually varies week to week; the four chips are then timed
    off different views of it:

    - Triple Captain: the gameweek where your single best-forecasted
      player's adjusted points peak (rewarded by an easy fixture or,
      especially, a double).
    - Bench Boost: the gameweek where your bench specifically (the
      `bench_size` squad members with the lowest predicted_points, used
      as a proxy for who'd actually be benched — recommend_squad()'s
      output has no literal starting-11 of its own) has its highest
      combined adjusted total.
    - Free Hit: the single worst ISOLATED gameweek — a dip relative to
      the gameweeks around it. Free Hit only fixes one week before your
      real squad reverts, so it's wasted on anything but a one-off
      trough (a lone blank gameweek, a one-week bad patch).
    - Wildcard: the START of the worst SUSTAINED stretch — a multi-
      gameweek trough, since a wildcard rebuilds your squad for good,
      so its payoff compounds over several weeks rather than one.

    Requires real fixture data — pass in
    src.fixtures.load_team_fixtures()'s output as `team_fixtures`, since
    price/points data alone doesn't carry which gameweek each team
    actually plays.

    Returns a dict with one entry per chip (gameweek + a short reason)
    plus which half of the season this is scoped to and the plain-
    English methodology, so a caller (or the Report Agent) can explain
    the reasoning, not just print a gameweek number. Treat this as a
    starting point for your own judgement, not a guarantee — it's an
    explainable heuristic over real data, not a fitted model, and it
    can't see team news that hasn't happened yet (a player picking up
    an injury next week isn't reflected here until src/availability.py's
    next snapshot).
    """
    df = forecast_points(df=df) if df is None else df
    _require_season_column(df)
    _require_code_column(df)

    if team_fixtures is None or team_fixtures.empty:
        raise ValueError(
            "No fixture data given — call src.fixtures.load_team_fixtures() "
            "first and pass its result in as `team_fixtures`."
        )

    latest_season = df["season"].max()
    season_df = df[df["season"] == latest_season]
    latest_gw = season_df["gameweek"].max()
    squad = (
        season_df[season_df["gameweek"] == latest_gw]
        .drop_duplicates(subset=["code"])
        .set_index("code")
        .reindex(squad_codes)[["name", "team", "position", "predicted_points"]]
    )
    missing = squad[squad["name"].isna()].index.tolist()
    squad = squad.dropna(subset=["name"])
    if squad.empty:
        raise ValueError(
            "None of the given squad_codes matched a player with a current "
            "predicted_points value — check the codes came from this same "
            "season's forecast_points() output."
        )

    half_end = 19 if current_gameweek <= 19 else 38
    half_start = 1 if current_gameweek <= 19 else 20
    requested_gws = list(
        range(max(current_gameweek, half_start), min(half_end, current_gameweek + horizon) + 1)
    )
    # A gameweek with literally zero rows anywhere in team_fixtures means
    # the fixture list doesn't cover it yet (postponements not
    # re-slotted, or a horizon reaching past what's been published) —
    # that's "no data," not "every team blank," and must NOT be treated
    # as a real trough/dip (every squad member would score 0 there
    # purely from missing data, which would wrongly look like the worst
    # possible gameweek to both Free Hit and Wildcard). Dropped from the
    # candidates entirely rather than silently scored as a blank.
    known_gws = set(team_fixtures["gameweek"].unique())
    candidate_gws = [gw for gw in requested_gws if gw in known_gws]
    unknown_gws = [gw for gw in requested_gws if gw not in known_gws]

    result = {
        "half": "first (GW1-19)" if half_end == 19 else "second (GW20-38)",
        "horizon_gameweeks": candidate_gws,
    }
    if missing:
        result["skipped_codes"] = missing
    if unknown_gws:
        result["gameweeks_without_fixture_data"] = unknown_gws
    if not candidate_gws:
        result["note"] = (
            "No upcoming gameweeks in the current chip half have fixture data "
            "to plan around yet."
        )
        return result

    bench_codes = set(squad.sort_values("predicted_points").head(bench_size).index)

    gw_rows = []
    for gw in candidate_gws:
        adj_points = {}
        for code, row in squad.iterrows():
            tf_row = team_fixtures[
                (team_fixtures["team"] == row["team"]) & (team_fixtures["gameweek"] == gw)
            ]
            fixture_count = int(tf_row["fixture_count"].iloc[0]) if not tf_row.empty else 0
            avg_diff = float(tf_row["avg_difficulty"].iloc[0]) if not tf_row.empty else None
            factor = _fixture_easiness_factor(avg_diff) if avg_diff is not None else 0.0
            adj_points[code] = row["predicted_points"] * factor * fixture_count
        gw_rows.append({
            "gameweek": gw,
            "squad_total": sum(adj_points.values()),
            "bench_total": sum(v for c, v in adj_points.items() if c in bench_codes),
            "best_code": max(adj_points, key=adj_points.get),
            "best_points": max(adj_points.values()),
        })
    gw_df = pd.DataFrame(gw_rows).set_index("gameweek")

    bench_boost_gw = int(gw_df["bench_total"].idxmax())
    triple_captain_gw = int(gw_df["best_points"].idxmax())
    best_captain_name = squad.loc[gw_df.loc[triple_captain_gw, "best_code"], "name"]

    # Free Hit: worst ISOLATED dip — squad_total below the average of its
    # immediate neighbours in the horizon.
    rolling_avg = gw_df["squad_total"].rolling(3, center=True, min_periods=1).mean()
    free_hit_gw = int((rolling_avg - gw_df["squad_total"]).idxmax())

    # Wildcard: start of the worst SUSTAINED trough — a multi-gameweek
    # rolling average at its lowest point in the horizon.
    trough_window = min(4, len(gw_df))
    trough_avg = gw_df["squad_total"].rolling(trough_window).mean()
    if trough_avg.notna().any():
        window_end = int(trough_avg.idxmin())
        wildcard_gw = max(gw_df.index.min(), window_end - trough_window + 1)
    else:
        wildcard_gw = gw_df.index.min()

    result.update({
        "bench_boost": {
            "gameweek": bench_boost_gw,
            "expected_bench_points": round(gw_df.loc[bench_boost_gw, "bench_total"], 1),
        },
        "triple_captain": {
            "gameweek": triple_captain_gw,
            "player": best_captain_name,
            "expected_points": round(gw_df.loc[triple_captain_gw, "best_points"], 1),
        },
        "free_hit": {
            "gameweek": free_hit_gw,
            "reason": "Biggest isolated one-week dip in the horizon.",
        },
        "wildcard": {
            "gameweek": int(wildcard_gw),
            "reason": f"Start of the worst {trough_window}-gameweek stretch in the horizon.",
        },
        "methodology": (
            "predicted_points (forecast_points()'s rolling per-player baseline) "
            "adjusted per gameweek by FPL's own fixture difficulty rating and "
            "fixture count (0 for a blank, 2 for a double) — an explainable "
            "heuristic, not a fitted model."
        ),
    })
    return result
