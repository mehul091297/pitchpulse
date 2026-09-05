# PitchPulse

An agentic analytics pipeline for English Premier League pricing markets — a Business Analytics portfolio project.

**Live dashboard:** https://pitchpulse-bs7uhkkyuchigcpnxupi29.streamlit.app/

Full project plan, architecture diagram, and phased roadmap: see the plan artifact shared alongside this repo (or rebuild it from `plan.html` in this folder).

## What this is

Four stages, one trigger:

1. **Ingest Agent** — extracts and cleans raw match/price data
2. **SQLite store** — clean, versioned tables
3. **Analyst Agent** — trend analysis, points forecasting, squad optimization, anomaly/spike scoring
4. **Report Agent** — turns the analyst's findings, recommended squad, and anomaly flags into a written narrative report

A Streamlit dashboard reads the same metrics for interactive exploration.

Goes one step past most student analytics projects: it doesn't just describe what happened to prices (descriptive) or flag why (diagnostic) — `forecast_points()` predicts what's next (predictive) and `recommend_squad()` turns that into an actual budget-constrained pick (prescriptive), with the Report Agent explaining the reasoning. See `src/analysis.py`.

## Data track

**Chosen: Track A — Fantasy Premier League price market.** Player prices rise and fall through the season purely from manager transfers: fully real, no synthetic layer, no gambling adjacency.

- Historical, gameweek-by-gameweek: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — each season's `data/<season>/gws/merged_gw.csv` has the per-player, per-gameweek `value` (price), `selected` (ownership), and `transfers_balance` (demand) needed for this project. Download the season(s) you want into `data/raw/` and list the filenames in `src/ingest.py`'s `RAW_FILES`.
- Current season, live: [official FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) (no key required).
- Richer current-season option: [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) (FPL data fused with match stats + Elo).

*(Track B — betting-odds market, from the [EPL Match Data 2000-2025](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025) dataset — stays documented in the project plan as a fallback if the FPL data ever turns out too thin for some analysis.)*

**Historical vs. live — two sources, two jobs.** Neither community archive above has the current season yet (checked directly: `vaastav` stops at 2025-26, FPL-Core-Insights documents up to 2025/26). That's fine, because they're doing a different job than live data would: `src/ingest.py` loads finished historical seasons so `forecast_points()` has something to be trained and checked against. `src/live_ingest.py` pulls this season's actual current prices straight from the official FPL API (`bootstrap-static`) — the only source that's ever truly current, since it *is* the source. Run `python -m src.live_ingest` and it appends one snapshot to the same `prices` table; run it again next gameweek and it adds another, gradually building this season's own time series the same way the historical archives were built, one gameweek at a time. This is what the pipeline's recurring trigger ("run before each gameweek deadline") actually does once Phase 3 wires it in — and it's how `recommend_squad()` ends up picking from *this season's* real prices, not last season's.

## Status

- [x] Dataset track chosen — Track A: FPL price market
- [x] Data ingested into `data/processed/pitchpulse.db` (2 historical seasons + live 2026-27 snapshots, joined on FPL's stable player `code`)
- [x] Core dashboard live — season-safe price/demand charts + squad recommendation tab
- [x] Agent pipeline v1 running end to end (`src/agents/crew.py` — Ingest -> Analyst -> Report agents verified via a real Colab run, all 3 agents/tasks completing successfully on real data; caught and fixed a tie-break bug in the price-movers tool along the way)
- [x] Forecasting wired in (`forecast_points()` — verified against real data)
- [x] Anomaly detection wired in (`anomaly_scores()` — per-player rolling z-score on price change and transfer swings, wired into the crew as a new tool; verified with synthetic tests and a real Colab run — correctly reports 'not enough history yet' this early in 2026-27 instead of fabricating a result)
- [x] Squad optimizer wired in (`recommend_squad()` — verified against real data, matches deployed dashboard output)
- [x] Deployed — live at https://pitchpulse-bs7uhkkyuchigcpnxupi29.streamlit.app/
- [ ] Demo recorded

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your LLM API key before running the agent pipeline (not needed for ingestion or the dashboard alone).

## Project layout

```
data/raw/          # untouched source files (gitignored — download per the dataset links above)
data/processed/    # cleaned tables / sqlite db (gitignored)
src/ingest.py      # historical raw files -> clean SQLite tables
src/live_ingest.py # this season's live snapshot, straight from the official FPL API
src/bootstrap.py   # one-call ensure_data(): downloads + ingests + snapshots — what the deployed dashboard runs on a fresh checkout
src/db.py          # SQLite connection helper
src/analysis.py    # trend, points forecast, squad optimizer (PuLP), anomaly-score functions
src/agents/crew.py # CrewAI pipeline: Ingest -> Analyst -> Report agents
dashboard/app.py   # Streamlit + Plotly dashboard (season-safe trends + squad recommendation)
reports/           # generated narrative reports land here
```

## Running

```bash
# 1. Drop your chosen dataset's raw files into data/raw/, then:
python -m src.ingest

# 2. Explore the dashboard:
streamlit run dashboard/app.py

# 3. Once the agent pipeline is wired up (Phase 3 in the plan):
python -m src.agents.crew
```
