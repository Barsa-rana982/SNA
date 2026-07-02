"""
Minimal Feature Analysis Script
Reads enron_cleaned_final.csv, computes weekly features, and exports features.csv for clustering.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

# --- CONFIG ---
INPUT = "enron_cleaned_final_2.csv"  # updated input
OUTPUT = "features.csv"
OUTPUT_WITH_CLUSTERS = "features_with_clusters.csv"

# --- LOAD DATA ---
df = pd.read_csv(INPUT)
# --- Automatic column mapping ---
def auto_map(colnames, candidates):
    for cand in candidates:
        for col in colnames:
            if cand == col.lower():
                return col
    for cand in candidates:
        for col in colnames:
            if cand in col.lower():
                return col
    return None


sender_candidates = ["sender_id", "from", "email_from", "from_email", "sender"]
receiver_candidates = ["receiver_id", "to", "email_to", "to_email", "recipient"]
timestamp_candidates = ["ts_utc", "date", "timestamp", "datetime", "sent"]

cols = [c.lower() for c in df.columns]
sender_col = auto_map(df.columns, sender_candidates)
receiver_col = auto_map(df.columns, receiver_candidates)
timestamp_col = auto_map(df.columns, timestamp_candidates)

if not sender_col or not receiver_col or not timestamp_col:
    raise ValueError(f"Could not find required columns. Found: sender={sender_col}, receiver={receiver_col}, timestamp={timestamp_col}. Available columns: {list(df.columns)}")

df = df.rename(columns={sender_col: "sender_id", receiver_col: "receiver_id", timestamp_col: "ts_utc"})
if "ts_utc" not in df.columns:
    raise ValueError("No timestamp column found.")

# --- PREPROCESS ---
df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
df["week_start"] = df["ts_utc"].dt.to_period("W-MON").dt.start_time

# --- FEATURE ENGINEERING ---
def is_after_hours(dt):
    hour = dt.hour
    weekday = dt.weekday()
    return (hour < 9 or hour > 18) or (weekday >= 5)

def burstiness(times):
    if len(times) < 3:
        return 0.0
    t = np.sort(times.view(np.int64) / 1e9 / 60.0)
    diffs = np.diff(t)
    return float(np.std(diffs) / (np.mean(diffs) + 1e-9)) if np.mean(diffs) > 0 else 0.0

features = []
for (emp, week), group in df.groupby(["sender_id", "week_start"]):
    msgs_sent = len(group)
    uniq_contacts = group["receiver_id"].nunique()
    pct_after = np.mean([is_after_hours(t) for t in group["ts_utc"]]) if msgs_sent > 0 else 0.0
    burst = burstiness(group["ts_utc"])
    features.append({
        "employee_id": emp,
        "week_start": week,
        "msgs_sent": msgs_sent,
        "uniq_contacts": uniq_contacts,
        "pct_after_hours": pct_after,
        "burstiness": burst,
    })

feat_df = pd.DataFrame(features)

# --- RECEIVED COUNTS ---
recv = (df.groupby(["receiver_id", "week_start"])["ts_utc"]
          .count().reset_index()
          .rename(columns={"receiver_id": "employee_id", "ts_utc": "msgs_recv"}))
feat_df = feat_df.merge(recv, on=["employee_id", "week_start"], how="left").fillna({"msgs_recv": 0})

# --- NETWORK METRICS ---
graph_rows = []
for week, week_df in df.groupby("week_start"):
    G = nx.DiGraph()
    for _, r in week_df.iterrows():
        G.add_edge(r["sender_id"], r["receiver_id"])
    nodes = list(G.nodes())
    degree_in = dict(G.in_degree())
    degree_out = dict(G.out_degree())
    try:
        betweenness = nx.betweenness_centrality(G)
    except Exception:
        betweenness = {n: 0.0 for n in nodes}
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eigenvector = {n: 0.0 for n in nodes}
    try:
        clustering = nx.clustering(G.to_undirected())
    except Exception:
        clustering = {n: 0.0 for n in nodes}
    for n in nodes:
        graph_rows.append({
            "employee_id": n,
            "week_start": week,
            "degree_in": degree_in.get(n, 0),
            "degree_out": degree_out.get(n, 0),
            "betweenness": betweenness.get(n, 0.0),
            "eigenvector": eigenvector.get(n, 0.0),
            "clustering_coeff": clustering.get(n, 0.0),
        })

graph_df = pd.DataFrame(graph_rows)
feat_df = feat_df.merge(graph_df, on=["employee_id", "week_start"], how="left").fillna(0)

# --- EXPORT ---
feat_df.to_csv(OUTPUT, index=False)
print(f"✅ Features exported to {OUTPUT}")

# --- SIMPLE CLUSTERING for Power BI ---
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except Exception:
    print("sklearn not available; skipping clustering export. To enable clustering, install scikit-learn in your environment.")
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True

if SKLEARN_AVAILABLE:
    # Prepare features for clustering
    cluster_cols = [
        c for c in ["msgs_sent", "msgs_recv", "uniq_contacts", "pct_after_hours",
                    "burstiness", "degree_in", "degree_out", "betweenness", "eigenvector", "clustering_coeff"]
        if c in feat_df.columns
    ]
    if len(cluster_cols) >= 2:
        X = feat_df[cluster_cols].fillna(0).astype(float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Choose k using a simple heuristic: min(6, sqrt(n_unique_employees))
        n_emp = feat_df['employee_id'].nunique()
        k = min(6, max(2, int(np.sqrt(max(1, n_emp)))))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(Xs)
        feat_df['cluster_label'] = labels

        # Create human-friendly cluster names
        feat_df['cluster_name'] = feat_df['cluster_label'].apply(lambda v: f'Cluster {v}')

        # Attach silhouette score (global) if possible
        try:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(Xs, labels)
        except Exception:
            sil = None

        # Add silhouette score as a column for downstream dashboards (same value repeated)
        if sil is not None:
            feat_df['silhouette_score'] = float(sil)

        # Export CSV ready for Power BI (no index, ISO datetimes)
        out_df = feat_df.copy()
        if 'week_start' in out_df.columns:
            out_df['week_start'] = pd.to_datetime(out_df['week_start']).dt.strftime('%Y-%m-%d')
        out_df.to_csv(OUTPUT_WITH_CLUSTERS, index=False)
        print(f"✅ Features with clusters exported to {OUTPUT_WITH_CLUSTERS}")
    else:
        print("Not enough numeric features to run clustering. Export skipped.")