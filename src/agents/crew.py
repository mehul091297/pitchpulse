"""
Agent pipeline (Phase 3): Ingest Agent -> Analyst Agent -> Report Agent.

The actual CrewAI wiring — each agent wraps the same functions the
dashboard uses (src/bootstrap.py, src/analysis.py), so there's exactly
one place that knows how to ingest data or build a squad recommendation;
the agents just call it and reason over the result. Sequential process,
matching the pipeline diagram in the project plan.

Requires an LLM API key. Copy .env.example to .env and fill in
ANTHROPIC_API_KEY or OPENAI_API_KEY (whichever you have) before running
this — the tools/data pipeline itself needs no key, only the agents'
reasoning does.

Run with:
    python -m src.agents.crew
"""

import os

from dotenv import load_dotenv

load_dotenv()

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool


# ---------------------------------------------------------------------------
# Tools — thin wrappers around src/bootstrap.py and src/analysis.py that
# turn their DataFrame/tuple return values into short text summaries an
# LLM can actually reason over. None of this duplicates logic: every
# number here comes from the same functions the dashboard calls.
# ---------------------------------------------------------------------------


@tool("Ensure current data")
def ensure_data_tool() -> str:
    """Make sure the price/points database has both historical seasons
    and this season's latest live snapshot (plus a fresh player
    availability snapshot), downloading and ingesting from scratch if
    needed. Call this first, before any analysis tool — the other
    tools assume the database already exists.
    """
    from src.analysis import list_seasons
    from src.bootstrap import ensure_data

    ensure_data()
    seasons = list_seasons()
    return f"Data is current. Seasons available: {', '.join(seasons)}."


@tool("Get biggest price movers")
def price_movers_tool(season: str = "") -> str:
    """Get the players whose price rose or fell the most over one
    season. Leave `season` blank for the most recent season. Returns
    the top 5 risers and top 5 fallers as text, not the full table.
    """
    from src.analysis import price_trend

    df = price_trend(season=season or None)
    if df.empty:
        return "No price data available for that season."
    df = df.sort_values("gameweek")
    first = df.groupby("name")["price_m"].first()
    last = df.groupby("name")["price_m"].last()
    # Round before filtering — real FPL prices move in exact £0.1m steps,
    # but float subtraction can leave e.g. 5.5 - 5.4 = 0.09999999999999964,
    # which `!= 0` would wrongly treat as "moved."
    change = (last - first).round(1)
    # Early in a season (or a quiet week), almost every player's price is
    # genuinely unchanged — verified against a real run where ALL 10
    # "top movers" the naive top-5/bottom-5 picked showed exactly £0.0m,
    # an arbitrary tie-break dressed up as a finding, not a real signal.
    # Filtering to actual nonzero moves and saying so plainly when there
    # aren't any is the honest version of this tool.
    movers = change[change != 0]

    if movers.empty:
        return (
            "No meaningful price movement yet this season — prices are "
            "still flat this early on. Don't present a top-5 as if it "
            "were signal; there isn't one yet."
        )

    lines = ["Biggest risers:"]
    risers = movers[movers > 0].sort_values(ascending=False).head(5)
    if risers.empty:
        lines.append("  (none yet)")
    for name, chg in risers.items():
        lines.append(f"  {name}: {chg:+.1f}m (now £{last[name]:.1f}m)")

    lines.append("Biggest fallers:")
    fallers = movers[movers < 0].sort_values().head(5)
    if fallers.empty:
        lines.append("  (none yet)")
    for name, chg in fallers.items():
        lines.append(f"  {name}: {chg:+.1f}m (now £{last[name]:.1f}m)")
    return "\n".join(lines)


@tool("Get transfer demand swings")
def demand_swings_tool(season: str = "") -> str:
    """Get the players with the biggest net transfer swings (in minus
    out) in the most recent gameweek of one season. Leave `season`
    blank for the most recent season. Returns the top 5 gainers and
    top 5 losers as text.
    """
    from src.analysis import demand_signal

    df = demand_signal(season=season or None)
    if df.empty:
        return "No demand data available for that season."
    latest_gw = df["gameweek"].max()
    latest = df[df["gameweek"] == latest_gw].sort_values("transfers_balance")

    lines = [f"Gameweek {latest_gw} demand swings:", "Most transferred IN (net):"]
    for _, row in latest.tail(5).sort_values("transfers_balance", ascending=False).iterrows():
        lines.append(f"  {row['name']}: {row['transfers_balance']:+.0f}")
    lines.append("Most transferred OUT (net):")
    for _, row in latest.head(5).iterrows():
        lines.append(f"  {row['name']}: {row['transfers_balance']:+.0f}")
    return "\n".join(lines)


@tool("Get price/demand anomaly alerts")
def anomaly_alerts_tool(season: str = "") -> str:
    """Get player-gameweeks where a price change or transfer swing was
    unusual FOR THAT SPECIFIC PLAYER, compared to their own recent
    history — not a top-5/bottom-5 list, an alert list, so it can come
    back with anywhere from zero to many entries depending on what
    actually happened. Leave `season` blank for the most recent season.
    A player needs a few prior gameweeks of history before this can say
    what's normal for them, so don't be surprised if this returns
    nothing at all early in a season — that means there's genuinely not
    enough history yet, not that the tool failed.
    """
    from src.analysis import anomaly_scores

    df = anomaly_scores(season=season or None)
    if df.empty:
        return (
            "No anomalies flagged for that season. Either nothing unusual "
            "has happened, or (very likely early in a season) there isn't "
            "enough per-player history yet to tell what's normal for them "
            "— don't report this as 'no anomalies detected', report it as "
            "'not enough history yet to detect anomalies.'"
        )

    lines = [f"Anomaly alerts ({len(df)} flagged, most severe first):"]
    for _, row in df.iterrows():
        if row["metric"] == "price_change":
            detail = f"price moved {row['value']:+.1f}m in gameweek {row['gameweek']}"
        else:
            detail = f"net transfers of {row['value']:+.0f} in gameweek {row['gameweek']}"
        lines.append(
            f"  {row['name']} ({row['team']}): {detail} "
            f"— {abs(row['z_score']):.1f}x their normal week-to-week variation"
        )
    return "\n".join(lines)


@tool("Get recent Premier League transfers")
def recent_transfers_tool(days: int = 30) -> str:
    """Get real Premier League transfers (arrivals, departures, and
    already-announced future moves) from the last `days` days — sourced
    from real Transfermarkt data, refreshed weekly. Use this to check
    whether a player is still actually in the league or has since moved
    clubs, especially useful during the summer and winter transfer
    windows. This is a market-news feed, NOT the source of truth for
    whether a player is playing minutes right now — a player recorded
    here can still be out on loan elsewhere without that loan showing
    up in this data; trust the squad/points tools for actual current
    playing status, and this tool for transfer context.

    If this comes back saying the data source is unreachable, that's an
    optional enrichment failing, not a core pipeline problem — proceed
    with the rest of your analysis regardless.
    """
    from src.transfers import recent_transfers

    try:
        df = recent_transfers(days=days)
    except (RuntimeError, ValueError) as exc:
        return f"Recent transfer data unavailable ({exc}). Proceed without it."

    if df.empty:
        return f"No Premier League transfers recorded in the last {days} days."

    lines = [f"Premier League transfers, last {days} days ({len(df)} found):"]
    for _, row in df.iterrows():
        fee = f"£{row['transfer_fee']:,.0f}" if row["transfer_fee"] else "free/undisclosed"
        date_str = row["transfer_date"].strftime("%Y-%m-%d")
        lines.append(
            f"  [{row['status']}] {row['player_name']}: "
            f"{row['from_club_name']} -> {row['to_club_name']} "
            f"({fee}, {date_str})"
        )
    return "\n".join(lines)


@tool("Recommend a squad")
def recommend_squad_tool(budget: float = 100.0) -> str:
    """Build the budget-optimal 15-player squad for the current
    gameweek: maximize predicted points subject to the real FPL rules
    (budget, exact position counts, max 3 players per club). Excludes
    injured/suspended/unavailable players by default (whenever
    availability data has been ingested — see the availability tool).
    Returns the squad, total predicted points, total cost, and solver
    status as text.
    """
    from src.analysis import recommend_squad

    squad, status = recommend_squad(budget=budget)
    if status != "Optimal":
        return f"Solver status: {status} — no feasible squad at this budget."

    lines = [
        f"Solver status: {status}",
        f"Total predicted points: {squad['predicted_points'].sum():.1f}",
        f"Total cost: £{squad['price_m'].sum():.1f}m",
        "",
    ]
    ordered = squad.sort_values(["position", "predicted_points"], ascending=[True, False])
    for _, row in ordered.iterrows():
        lines.append(
            f"  {row['position']} {row['name']} ({row['team']}) — "
            f"£{row['price_m']:.1f}m, {row['predicted_points']:.1f} predicted pts"
        )
    return "\n".join(lines)


@tool("Get player availability / injury news")
def player_availability_tool(top_n: int = 10) -> str:
    """Get current injury/doubt/suspension status for the league's most
    notable affected players, straight from FPL's own official data —
    NOT scraped news, so there's no separate 'source' to go stale or get
    blocked. 'Notable' is proxied by price (FPL prices correlate with
    prominence — a doubtful £12m player matters more to report than a
    doubtful £4.0m fringe player), sorted highest price first, capped at
    `top_n`.

    A player showing 'doubtful' with a chance-of-playing percentage is a
    real but partial risk, not a reason to assume they're out — that
    distinction matters when reporting this, and recommend_squad_tool
    already only excludes the clear-cut unavailable cases (injured,
    suspended, unavailable, not considered for selection), not doubtful
    ones.

    If this comes back saying no availability data has been ingested
    yet, that's an optional enrichment not yet run (call
    src.availability.append_availability_snapshot() first) — not a
    pipeline failure.
    """
    from src.analysis import player_availability
    from src.db import get_engine

    try:
        avail = player_availability()
    except RuntimeError as exc:
        return f"Availability data unavailable ({exc}). Proceed without it."

    if avail.empty:
        return "No availability data ingested yet."

    notable = avail[avail["status"] != "a"]
    if notable.empty:
        return "No injuries, doubts, or suspensions currently reported for any player."

    # Price is only in the 'prices' table, not 'availability' — join in
    # for the prominence ranking rather than duplicating price into
    # every availability snapshot.
    import pandas as pd

    prices = pd.read_sql("SELECT code, price_m FROM prices", get_engine())
    latest_price = prices.groupby("code")["price_m"].last()
    notable = notable.copy()
    notable["price_m"] = notable["code"].map(latest_price).fillna(0)
    notable = notable.sort_values("price_m", ascending=False).head(top_n)

    lines = [f"Player availability news ({len(notable)} shown, most notable by price first):"]
    for _, row in notable.iterrows():
        news = f" — {row['news']}" if row["news"] else ""
        chance = (
            f", {row['chance_of_playing_next_round']:.0f}% chance next round"
            if pd.notna(row["chance_of_playing_next_round"])
            else ""
        )
        lines.append(f"  {row['name']} ({row['team']}, £{row['price_m']:.1f}m): {row['status_label']}{chance}{news}")
    return "\n".join(lines)


@tool("Recommend chip timing strategy")
def chip_strategy_tool(budget: float = 100.0, horizon: int = 8) -> str:
    """Recommend which upcoming gameweek to play each of the season's
    four chips (Wildcard, Free Hit, Triple Captain, Bench Boost) for the
    current recommended squad, using FPL's own real fixture difficulty
    and fixture count (blank/double gameweeks) — see
    src.analysis.recommend_chip_strategy's docstring for the full
    reasoning behind each chip's timing rule.

    Only ever recommends gameweeks within whichever chip-half (GW1-19
    or GW20-38) the next actionable gameweek falls in — first-half
    chips expire rather than carrying over, so a second-half suggestion
    would be useless advice right now.

    If fixture data can't be fetched, this is a real limitation (chip
    timing can't work without knowing which gameweek each team plays)
    — say so plainly rather than guessing.
    """
    from src.analysis import recommend_squad, recommend_chip_strategy
    from src.fixtures import load_team_fixtures, next_actionable_gameweek

    squad, status = recommend_squad(budget=budget)
    if status != "Optimal":
        return f"Can't recommend chip timing — squad solver status was {status}, not Optimal."

    try:
        current_gw = next_actionable_gameweek()
        team_fixtures = load_team_fixtures()
    except Exception as exc:
        return f"Chip-timing data unavailable ({exc}). This needs real fixture data to work at all."

    result = recommend_chip_strategy(
        squad_codes=squad["code"].tolist(),
        current_gameweek=current_gw,
        team_fixtures=team_fixtures,
        horizon=horizon,
    )

    if "note" in result and "bench_boost" not in result:
        return f"Chip strategy ({result['half']}): {result['note']}"

    lines = [f"Chip strategy for the {result['half']} half, gameweeks {result['horizon_gameweeks']}:"]
    bb, tc, fh, wc = result["bench_boost"], result["triple_captain"], result["free_hit"], result["wildcard"]
    lines.append(f"  Bench Boost — GW{bb['gameweek']} (bench expected {bb['expected_bench_points']} pts)")
    lines.append(f"  Triple Captain — GW{tc['gameweek']} on {tc['player']} (expected {tc['expected_points']} pts)")
    lines.append(f"  Free Hit — GW{fh['gameweek']} ({fh['reason']})")
    lines.append(f"  Wildcard — GW{wc['gameweek']} ({wc['reason']})")
    if result.get("gameweeks_without_fixture_data"):
        lines.append(
            f"  (No fixture data yet for gameweek(s) {result['gameweeks_without_fixture_data']} "
            "— excluded from consideration rather than guessed.)"
        )
    lines.append(f"  Methodology: {result['methodology']}")
    return "\n".join(lines)


def _default_llm() -> LLM:
    """Pick a model based on whichever API key is actually set. Model
    identifiers drift over time — if the one picked here is no longer
    valid for your provider, set PITCHPULSE_LLM_MODEL in .env to
    override it (any litellm-style "provider/model" string works).
    """
    override = os.getenv("PITCHPULSE_LLM_MODEL")
    if override:
        return LLM(model=override)
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLM(model="anthropic/claude-3-5-haiku-latest")
    if os.getenv("OPENAI_API_KEY"):
        return LLM(model="openai/gpt-4o-mini")
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return LLM(model="gemini/gemini-3.6-flash")
    raise RuntimeError(
        "No LLM API key found. Copy .env.example to .env and fill in "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY (the "
        "ingest/analysis tools themselves need no key — only the agents' "
        "reasoning does)."
    )


def build_crew(budget: float = 100.0) -> Crew:
    llm = _default_llm()

    ingest_agent = Agent(
        role="Data Ingestion Specialist",
        goal=(
            "Make sure the price/points database has current historical "
            "and live data before any analysis runs."
        ),
        backstory=(
            "You maintain PitchPulse's data pipeline. You know the "
            "database silently goes stale if nobody refreshes it, so you "
            "always check and refresh it first, before handing off."
        ),
        tools=[ensure_data_tool],
        llm=llm,
        verbose=True,
    )

    analyst_agent = Agent(
        role="Football Data Analyst",
        goal=(
            "Analyze price trends, transfer demand, player-specific "
            "anomalies, recent real-world transfers, and player "
            "availability, and produce a budget-optimal squad "
            "recommendation plus a chip-timing strategy, with reasoning "
            "grounded in the actual numbers your tools return."
        ),
        backstory=(
            "You're a Fantasy Premier League analyst who never states a "
            "number you haven't pulled from a tool. You explain WHY a "
            "pick makes sense (price versus predicted points), not just "
            "what the optimizer chose. You also know the difference "
            "between 'nothing unusual happened' and 'there isn't enough "
            "history yet to tell' when your anomaly tool comes back "
            "empty, and you say which one it actually is. You know real "
            "transfer news is context, not a squad-eligibility check — a "
            "player can show up there without it reflecting where "
            "they're actually playing right now (e.g. out on loan), so "
            "you never let it override what the squad/points tools say. "
            "You know the same is true of injury/doubt news: a 'doubtful' "
            "player is a real but partial risk, not confirmation they'll "
            "sit out — you never state a player is definitely out unless "
            "their status is genuinely injured/suspended/unavailable, not "
            "just doubtful. And you know your chip-timing recommendations "
            "are an explainable heuristic over real fixture data, not a "
            "guarantee — you present them as a strong starting point, not "
            "as certainty."
        ),
        tools=[
            price_movers_tool,
            demand_swings_tool,
            anomaly_alerts_tool,
            recent_transfers_tool,
            player_availability_tool,
            recommend_squad_tool,
            chip_strategy_tool,
        ],
        llm=llm,
        verbose=True,
    )

    report_agent = Agent(
        role="Football Analytics Reporter",
        goal=(
            "Turn the analyst's structured findings into a short, honest "
            "narrative report a fantasy manager could act on."
        ),
        backstory=(
            "You write for someone who wants the headline first: who to "
            "buy, who to watch, and why, in plain English — grounded "
            "strictly in what the analyst actually found, never inventing "
            "a reason the data doesn't support."
        ),
        llm=llm,
        verbose=True,
    )

    ingest_task = Task(
        description=(
            "Ensure the PitchPulse database is fully up to date "
            "(historical seasons plus this season's latest live "
            "snapshot) before any analysis begins. Use your tool and "
            "report what seasons are now available."
        ),
        expected_output="A one-line confirmation naming the seasons now available.",
        agent=ingest_agent,
    )

    analyze_task = Task(
        description=(
            "Using the now-current data, pull the biggest price movers, "
            "transfer demand swings, any player-specific anomaly alerts, "
            "recent real-world Premier League transfers, and player "
            "availability/injury news for the most recent season, then "
            "build a squad recommendation with a budget of £{budget}m and "
            "a chip-timing strategy (Wildcard, Free Hit, Triple Captain, "
            "Bench Boost) for that squad. Summarize your findings "
            "clearly, including WHY specific picks make sense relative to "
            "their price. If the anomaly tool returns no alerts, say "
            "plainly whether that's because nothing unusual happened or "
            "because there isn't enough history yet — don't blur the two "
            "together. If the transfer-news, availability, or chip-timing "
            "tools are unavailable, note that plainly too and move on — "
            "they're optional enrichments, not blockers, except the squad "
            "recommendation itself which is the core deliverable."
        ),
        expected_output=(
            "A structured summary covering: (1) notable price "
            "risers/fallers, (2) notable transfer demand swings, (3) any "
            "anomaly alerts (player-specific price/demand moves well "
            "outside that player's own normal range) and, if none, which "
            "of the two honest reasons why, (4) any recent real-world "
            "transfers worth knowing about, (5) notable injury/doubt news "
            "affecting well-known players, (6) the recommended 15-player "
            "squad with total predicted points and total cost, with brief "
            "reasoning for a few standout picks, and (7) the recommended "
            "gameweek to play each of the four chips this half-season, "
            "with a one-line reason for each."
        ),
        agent=analyst_agent,
        context=[ingest_task],
    )

    report_task = Task(
        description=(
            "Turn the analyst's findings into a short narrative report "
            "(a headline, then paragraphs) for a fantasy manager deciding "
            "their transfers and chip plays this gameweek. Lead with the "
            "recommended squad's overall shape and budget, then the "
            "standout picks and why, then the notable price/demand "
            "movers, then any anomaly alerts as a short 'watch list' — "
            "and if there are none, say plainly whether that's because "
            "nothing unusual happened or because there isn't enough "
            "history yet to tell, exactly as the analyst reported it — "
            "then any notable injury/doubt news (being careful to say "
            "'doubtful' when the analyst said doubtful, not 'ruled out'), "
            "then the chip-timing recommendation for all four chips with "
            "the analyst's stated reasoning, and close with any recent "
            "real-world transfers worth knowing about. Stay strictly "
            "grounded in the analyst's numbers — never invent a reason "
            "they didn't give."
        ),
        expected_output="A short markdown report, ready to read as-is.",
        agent=report_agent,
        context=[analyze_task],
        output_file="reports/latest_report.md",
    )

    return Crew(
        agents=[ingest_agent, analyst_agent, report_agent],
        tasks=[ingest_task, analyze_task, report_task],
        process=Process.sequential,
        verbose=True,
    )


def main() -> None:
    crew = build_crew()
    result = crew.kickoff(inputs={"budget": 100.0})
    print(result)


if __name__ == "__main__":
    main()
