"""
Clustering_dashboard.py
Simple Streamlit dashboard to explore clustering outputs.

RUN:
    streamlit run clustering_dashboard.py

EXPECTS:
- feature_with_clustering.csv (must contain: employee_id, week_start, cluster_label, cluster_name, plus features)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import json, os

st.set_page_config(page_title="Clustering Dashboard", layout="wide")
st.title(" Clustering Dashboard")

# --- Load Data ---
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

DATA_PATH = "features_with_clusters.csv"
try:
    df = load_csv(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load {DATA_PATH}: {e}")
    st.stop()

# --- Prepare Data ---
if "week_start" in df.columns:
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce", utc=True)

# --- Slicers ---
weeks = sorted(df["week_start"].dropna().unique())
week_sel = st.selectbox("Week", options=weeks, index=len(weeks)-1 if weeks else 0, format_func=lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "N/A")
cluster_opts = ["All"] + sorted(df["cluster_name"].dropna().unique().tolist())
cluster_sel = st.selectbox("Cluster", options=cluster_opts, index=0)

mask = (df["week_start"] == week_sel) if pd.notna(week_sel) else np.full(len(df), True)
if cluster_sel != "All":
    mask &= (df["cluster_name"] == cluster_sel)
cur = df[mask].copy()

# --- KPIs ---
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Employees (filtered)", value=int(cur["employee_id"].nunique()))
with kpi2:
    st.metric("#Clusters (all)", value=int(df["cluster_label"].nunique()))
with kpi3:
    risk_col = "risk" if "risk" in df.columns else "pct_after_hours" if "pct_after_hours" in df.columns else None
    val = float(cur.get(risk_col, pd.Series(dtype=float)).mean()) if risk_col and not cur.empty else 0.0
    st.metric(f"Avg {risk_col or 'N/A'} (filtered)", value=f"{val:.2f}")

st.divider()

# --- Cluster Validation: Silhouette Score ---
MODEL_INFO_PATH = "_model_info.json"
silhouette = None
if "silhouette_score" in df.columns:
    silhouette = float(df["silhouette_score"].iloc[0])
elif os.path.exists(MODEL_INFO_PATH):
    try:
        with open(MODEL_INFO_PATH, "r") as f:
            info = json.load(f)
            silhouette = info.get("silhouette", None)
    except Exception:
        silhouette = None
if silhouette is not None:
    st.info(f"Cluster Quality (Silhouette Score): **{silhouette:.3f}**")
    if silhouette < 0.25:
        st.warning("Low silhouette score: clusters may overlap or be poorly separated. Consider tuning parameters or features.")
    elif silhouette < 0.5:
        st.info("Moderate silhouette score: clusters are somewhat separated.")
    else:
        st.success("Good silhouette score: clusters are well separated.")

st.caption("KPIs show filtered employee count, total clusters, and average risk (or after-hours %) for selected week/cluster.")

# --- Cluster Distribution ---
st.subheader("Cluster Distribution (selected week)")
st.caption("Treemap shows cluster sizes for the selected week. Hover for cluster label and count.")
dist = (df[df["week_start"] == week_sel]
          .groupby(["cluster_label", "cluster_name"], as_index=False)
          .agg(count=("employee_id", "nunique")))
if not dist.empty:
    fig = px.treemap(dist, path=["cluster_name"], values="count",
                     hover_data={"cluster_label": True, "count": True})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data for the selected week.")

# --- Cluster Profiles ---
st.subheader("Cluster Profiles (feature comparison)")
st.caption("Heatmap compares cluster profiles across key features. Z-score normalization highlights differences.")
numeric_cols = [
    c for c in ["msgs_sent", "msgs_recv", "uniq_contacts", "pct_after_hours",
                "weekend_ratio", "burstiness", "reciprocity", "degree_in",
                "degree_out", "betweenness", "eigenvector", "clustering_coeff"]
    if c in df.columns
]
if numeric_cols:
    prof = (df[df["week_start"] == week_sel]
              .groupby("cluster_name")[numeric_cols]
              .mean()
              .reset_index())
    prof_long = prof.melt(id_vars="cluster_name", var_name="metric", value_name="value")
    heat = prof_long.pivot(index="cluster_name", columns="metric", values="value")
    heat_z = (heat - heat.mean()) / (heat.std() + 1e-9)
    heat_z = heat_z.reset_index().melt(id_vars="cluster_name", var_name="metric", value_name="z")
    fig = px.imshow(
        heat_z.pivot(index="cluster_name", columns="metric", values="z"),
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Cluster Profile Summary Table**")
    st.dataframe(prof, use_container_width=True)
else:
    st.info("No numeric feature columns found to profile clusters.")

# --- Business Relevance: Cluster Label Tooltips ---
st.subheader("Cluster Labels & Descriptions")
st.caption("Business labels help interpret clusters. Hover over cluster names for descriptions.")
if "cluster_name" in df.columns:
    unique_labels = df["cluster_name"].dropna().unique()
    for label in unique_labels:
        st.markdown(f"- **{label}**: {'High risk group' if 'risk' in label.lower() else 'See profile table above for details.'}")

# --- Trends Over Time ---
st.subheader("Trends Over Time")
st.caption("Line chart shows risk/activity trends by cluster. Area chart shows cluster size over time.")
trend_left, trend_right = st.columns(2)
with trend_left:
    metric = "risk" if "risk" in df.columns else "pct_after_hours" if "pct_after_hours" in df.columns else "msgs_sent"
    line = (df.groupby(["week_start", "cluster_name"])[metric]
              .mean()
              .reset_index())
    if cluster_sel != "All":
        line = line[line["cluster_name"] == cluster_sel]
    fig_line = px.line(line, x="week_start", y=metric, color="cluster_name", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)
with trend_right:
    size_over_time = (df.groupby(["week_start", "cluster_name"])["employee_id"]
                        .nunique().reset_index(name="count"))
    if cluster_sel != "All":
        size_over_time = size_over_time[size_over_time["cluster_name"] == cluster_sel]
    fig_area = px.area(size_over_time, x="week_start", y="count", color="cluster_name", groupnorm=None)
    st.plotly_chart(fig_area, use_container_width=True)

# --- Individuals Table ---
st.subheader("Individuals (filtered)")
st.caption("Table shows employees in the selected week/cluster and their feature values.")
show_cols = ["employee_id", "cluster_name"] + [c for c in numeric_cols[:6]]
grid = cur.sort_values(["cluster_name", "employee_id"])[show_cols].head(500)
st.dataframe(grid, use_container_width=True, hide_index=True)

st.caption("Tip: Export cluster assignments and profiles for further analysis. Use chart captions and tooltips for guidance.")

print("Open your dashboard at: http://127.0.0.1:8501/")