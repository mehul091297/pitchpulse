"""
PitchPulse dashboard (Phase 2).

Local / Colab:
    streamlit run dashboard/app.py

Deployed (Streamlit Community Cloud): the app has no data of its own on
a fresh checkout, so it calls src/bootstrap.py once on startup to
download and ingest historical seasons and pull a live snapshot — see
that module's docstring for why this differs from the Colab workflow.

Keep this file about presentation only — all the real logic lives in
src/analysis.py so the same functions can be reused by the Report Agent
later without duplicating code.
"""

import streamlit as st
import plotly.express as px

from src.analysis import price_trend, demand_signal, recommend_squad, list_seasons

st.set_page_config(page_title="PitchPulse", page_icon="⚽", layout="wide")


@st.cache_resource(ttl=3600)  # re-check for a fresher live snapshot hourly
def _bootstrap():
    from src.bootstrap import ensure_data
    ensure_data()
    return True


st.title("PitchPulse")
st.caption("English Premier League price-market analytics — an agentic pipeline: ingest → analyze → recommend → report")

try:
    _bootstrap()
    seasons = list_seasons()
except Exception as e:
    st.info(
        "No data yet. Locally/in Colab: run `python -m src.ingest` then "
        "`python -m src.live_ingest` first. Deployed: this shouldn't "
        "happen — src/bootstrap.py should have built the data automatically."
    )
    st.exception(e)
    st.stop()

tab_trends, tab_squad = st.tabs(["Price & demand trends", "Squad recommendation"])

with tab_trends:
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)

    trend_df = price_trend(season=season)
    all_players = sorted(trend_df["name"].unique().tolist())
    default_players = (
        trend_df.groupby("name")["price_m"].mean().sort_values(ascending=False).head(5).index.tolist()
    )
    picked = st.multiselect("Players to chart", all_players, default=default_players)

    if picked:
        fig = px.line(
            trend_df[trend_df["name"].isin(picked)],
            x="gameweek",
            y="price_m",
            color="name",
            title=f"Price over the season — {season}",
            labels={"gameweek": "Gameweek", "price_m": "Price (£m)", "name": "Player"},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Pick at least one player above to see their price trend.")

    st.subheader("Biggest demand swings, most recent gameweek")
    demand_df = demand_signal(season=season)
    latest_gw = demand_df["gameweek"].max()
    latest_demand = demand_df[demand_df["gameweek"] == latest_gw].copy()
    latest_demand["abs_balance"] = latest_demand["transfers_balance"].abs()
    latest_demand = latest_demand.sort_values("abs_balance", ascending=False).head(10)
    fig2 = px.bar(
        latest_demand.sort_values("transfers_balance"),
        x="transfers_balance",
        y="name",
        orientation="h",
        title=f"Net transfers (in − out), gameweek {latest_gw}",
        labels={"transfers_balance": "Net transfers", "name": "Player"},
    )
    st.plotly_chart(fig2, width="stretch")

with tab_squad:
    st.caption(
        "Prescriptive layer: maximizes predicted next-gameweek points "
        "under real FPL squad rules (budget, position counts, max 3 per "
        "club), using this season's latest gameweek — regardless of the "
        "season picked in the other tab."
    )
    budget = st.slider("Budget (£m)", min_value=80.0, max_value=100.0, value=100.0, step=0.5)

    try:
        squad, status = recommend_squad(budget=budget)
    except Exception as e:
        st.warning("Couldn't build a squad recommendation yet.")
        st.exception(e)
    else:
        if status != "Optimal":
            st.warning(f"Solver status: {status} — constraints may be too tight at this budget.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Solver status", status)
            col2.metric("Total predicted points", f"{squad['predicted_points'].sum():.1f}")
            col3.metric("Total cost", f"£{squad['price_m'].sum():.1f}m")

            for pos, label in [("GK", "Goalkeepers"), ("DEF", "Defenders"), ("MID", "Midfielders"), ("FWD", "Forwards")]:
                pos_df = squad[squad["position"] == pos].sort_values("predicted_points", ascending=False)
                if not pos_df.empty:
                    st.markdown(f"**{label}**")
                    st.dataframe(
                        pos_df[["name", "team", "price_m", "predicted_points"]].rename(
                            columns={"name": "Player", "team": "Club", "price_m": "Price (£m)", "predicted_points": "Predicted points"}
                        ),
                        hide_index=True,
                        width="stretch",
                    )
