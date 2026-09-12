"""Simple Streamlit dashboard for buyer segmentation."""

from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULT_PATH = BASE_DIR / "outputs" / "cluster_results.csv"
PROFILE_PATH = BASE_DIR / "outputs" / "investment_profiles.csv"

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Machine learning based market intelligence dashboard")

if not RESULT_PATH.exists() or not PROFILE_PATH.exists():
    st.warning("Run `python run_pipeline.py` first to generate the dashboard data.")
    st.stop()

df = pd.read_csv(RESULT_PATH)
profiles = pd.read_csv(PROFILE_PATH)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Buyers", len(df))
col2.metric("Segments", df["cluster"].nunique())
col3.metric(
    "Investment Buyers",
    f"{df['acquisition_purpose'].eq('Investment').mean()*100:.1f}%"
)
col4.metric("Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f}/10")

st.subheader("Buyer Segment Profiles")
st.dataframe(profiles, use_container_width=True)

st.subheader("Segment Distribution")
segment_counts = df["cluster"].value_counts().sort_index()
st.bar_chart(segment_counts)

st.subheader("Filter Buyers")
selected_cluster = st.selectbox(
    "Select cluster",
    ["All"] + sorted(df["cluster"].unique().tolist())
)

filtered = df if selected_cluster == "All" else df[df["cluster"] == selected_cluster]
st.dataframe(filtered, use_container_width=True)
