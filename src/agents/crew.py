"""
Agent pipeline (Phase 3): Ingest Agent -> Analyst Agent -> Report Agent.

This is the CrewAI wiring for the diagram in the project plan. It's left
as a scaffold with clear TODOs rather than working code, because it
depends on your chosen dataset (Phase 0) and a working analysis layer
(Phase 2) existing first — build in that order.

Docs: https://docs.crewai.com/
"""

import os
from dotenv import load_dotenv

load_dotenv()

# from crewai import Agent, Task, Crew, Process
# from src import ingest, analysis


def build_crew():
    """TODO (Phase 3):

    1. Define three Agents (role, goal, backstory) mirroring the diagram:
       - Ingest Agent: wraps src.ingest.main() as a @tool
       - Analyst Agent: wraps src.analysis.price_trend / forecast_points /
         recommend_squad / anomaly_scores as tools
       - Report Agent: no custom tool needed — just a strong prompt that
         turns the Analyst Agent's structured output (including a
         recommended squad, once Phase 5 is built) into a short narrative
         report (see reports/ for where output should land)

    2. Define three Tasks, one per agent, chained sequentially.

    3. Wire them into a Crew with Process.sequential and return it.

    Keep the Report Agent's prompt honest: it should only claim what the
    numbers support (e.g. flag a threshold breach, don't invent a cause
    the data doesn't show).
    """
    raise NotImplementedError("Build once Phase 2 (analysis.py) works.")


def main() -> None:
    crew = build_crew()
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
