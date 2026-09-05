"""
PitchPulse dashboard (Phase 2).

Run with: streamlit run dashboard/app.py

Keep this file about presentation only — all the real logic lives in
src/analysis.py so the same functions can be reused by the Report Agent
later without duplicating code.
"""

import streamlit as st
import plotly.express as px

from src.analysis import price_trend

st.set_page_config(page_title="PitchPulse", page_icon="⚽", layout="wide")

st.title("PitchPulse")
st.caption("English Premier League price-market analytics")

try:
    trend_df = price_trend()
    fig = px.line(
        trend_df,
        x=trend_df.columns[1],
        y=trend_df.columns[2],
        color=trend_df.columns[0],
        title="Average price over time",
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.info(
        "No data yet — run `python -m src.ingest` first "
        "(see src/ingest.py for the TODOs to fill in)."
    )
    st.exception(e)
