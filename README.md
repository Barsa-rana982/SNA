# Clustering Dashboard Project

## Overview
This project analyzes features from the Enron dataset, performs clustering, and provides a dashboard for visualizing results.

## File Descriptions
- **clustering_dashboard.py**: Script for launching the dashboard to visualize clustering results and explore features interactively.
- **enron_cleaned_final_2.csv**: Cleaned Enron dataset used as input for analysis and clustering. (updated)
- **feature_Analysis.py**: Script for analyzing dataset features, including statistical analysis and feature selection.
- **features_with_clusters.csv**: Output file containing features and their assigned cluster labels after clustering.

## How to Run
1. Ensure Python is installed on your system.
2. Place all project files in the same directory.
3. (Optional) Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, install packages manually as needed.
4. Run feature analysis:
   ```
   python feature_Analysis.py
   ```
5. Run the clustering dashboard:
   ```
   python clustering_dashboard.py
   ```
6. Use the CSV files as input/output for the scripts as needed.

## Notes
- Update or create `requirements.txt` with all necessary Python packages for easy setup.
- For any issues, check script dependencies and ensure all required packages are installed.

## Data Dictionary: features_with_clusters.csv
| Column Name         | Description                                                                                 |
|---------------------|---------------------------------------------------------------------------------------------|
| employee_id         | Unique identifier for the employee (usually an email address)                               |
| week_start          | Start date of the week for the data record                                                  |
| msgs_sent           | Number of messages sent by the employee during the week                                     |
| uniq_contacts       | Number of unique contacts the employee communicated with                                    |
| pct_after_hours     | Percentage of messages sent after regular working hours                                     |
| burstiness          | Measure of variability in messaging activity (higher values indicate more bursty behavior)  |
| msgs_recv           | Number of messages received by the employee during the week                                 |
| degree_in           | Number of incoming connections (people who sent messages to the employee)                   |
| degree_out          | Number of outgoing connections (people the employee sent messages to)                       |
| betweenness         | Betweenness centrality score (how often the employee acts as a bridge in the network)       |
| eigenvector         | Eigenvector centrality score (influence of the employee in the network)                     |
| clustering_coeff    | Clustering coefficient (how connected the employee’s contacts are to each other)            |
| cluster_label       | Numeric label for the cluster assigned to the employee                                      |
| cluster_name        | Name of the cluster assigned to the employee                                                |


features_with_clusters.csv
What it is:
Weekly-level behavioral and network features for each employee, with clustering labels indicating groupings of similar communication/network patterns.

Key columns:

employee_id: Unique employee identifier (email)
week_start: Start date of week
msgs_sent, msgs_recv: Number of messages sent/received
uniq_contacts: Number of unique contacts communicated with
pct_after_hours: % messages sent after regular working hours
burstiness: Variability in messaging activity (higher = more bursty)
degree_in, degree_out: Number of incoming/outgoing connections
betweenness: Betweenness centrality (network bridge role)
eigenvector: Eigenvector centrality (network influence)
clustering_coeff: How interconnected the employee’s contacts are
cluster_label, cluster_name: Cluster assignment (numeric and descriptive)

What to Visualize

Cluster composition:
Bar chart: Number of employees per cluster (cluster_label/cluster_name)
Feature distributions by cluster:
Boxplot/violin plot: Compare msgs_sent, uniq_contacts, pct_after_hours, etc. across clusters
Network roles:
Scatterplot: betweenness vs eigenvector colored by cluster
Temporal trends:
Line chart: Average msgs_sent or burstiness per cluster over time (week_start)
Cluster profiles:
Radar/spider chart: Median feature values for each cluster (shows behavioral archetypes)
Work-life balance:
Heatmap: pct_after_hours by cluster and week


Insights You Can Derive

Cluster archetypes:
Identify clusters representing high-volume communicators, network bridges, isolated users, or overworked employees.
Temporal shifts:
Spot weeks where certain clusters change in size or behavior (possible organizational events).
Network structure:
Find clusters with high centrality (key influencers) or high clustering coefficient (tight-knit teams).
Work-life balance:
Detect clusters with consistently high after-hours activity (potential burnout risk).


Recommended Visualizations & Tools

Goal	                   Chart Type	                   Tool/Library
Cluster composition	    Bar chart	                   Matplotlib, Power BI
Feature distributions	Boxplot/Violin plot	          Seaborn
Network roles	         Scatterplot	                   Plotly, Seaborn
Temporal trends	      Line chart	                   Matplotlib, Tableau
Cluster profiles	      Radar chart	                   Plotly, Power BI
Work-life balance	      Heatmap	                      Seaborn, Plotly
Example narrative:

“Cluster 2 contains employees with high betweenness and eigenvector scores, acting as network bridges and influencers. Cluster 4 shows high burstiness and after-hours messaging, indicating possible workload stress.”

## Insights (generated)

Summary:
- Rows (employee-weeks): 75,481
- Unique employees: 19,780
- Date range: 1998-12-29 through 2002-12-17
- Number of clusters: 6
- Global silhouette score: ~0.513 (included in `silhouette_score` column)

Cluster summaries (high-level):
- Cluster 0 (24,181 rows, ~11,458 employees): low-volume senders with very high after-hours activity (pct_after_hours ≈ 0.94). Possible late responders or external contacts.
- Cluster 1 (1,326 rows, ~126 employees): very high-volume communicators with many unique contacts and high betweenness — core hubs/managers.
- Cluster 2 (31,028 rows, ~12,368 employees): majority group, low activity and low after-hours (daytime communicators).
- Cluster 3 (6,518 rows, ~3,325 employees): moderate activity with high clustering coefficient — tightly connected teams or group threads.
- Cluster 4 (11 rows, 2 employees): extreme outliers with extremely high message volumes — likely automated accounts or mailing lists; inspect immediately.
- Cluster 5 (12,417 rows, ~2,352 employees): mid/high activity with broader contact lists — active contributors but not central hubs.

Key interpretations:
- High after-hours cluster (Cluster 0) may indicate work-life balance concerns or external communication patterns.
- Cluster 1 represents central communicators (high betweenness) who may be organizational hubs or key coordinators.
- Cluster 3’s high clustering coefficient suggests intra-team communication patterns (tight subgroups).
- Very small outlier cluster (Cluster 4) should be investigated and labeled/removed if it represents system or alias traffic.

Recommended Power BI visuals (quick wins):
- Treemap or bar chart for cluster sizes (distinct employees per cluster).
- Line chart: average `msgs_sent` or `pct_after_hours` over `week_start` per cluster to surface temporal shifts.
- Heatmap: cluster profile (z-scored feature means) to compare clusters across features.
- Scatter: `betweenness` vs `msgs_sent` (or `uniq_contacts`) colored by `cluster_name` to spot hubs vs broadcasters.
- Table: top accounts by `msgs_sent` within Cluster 1 and Cluster 4 for manual review.

Recommended immediate actions:
1. Inspect accounts in Cluster 4 (2 accounts) for automation or listservs; exclude or label as needed.
2. Prepare a HR/people-ops report for Cluster 0 to review potential after-hours workload patterns.
3. Consider aggregating to employee-level (one row per employee) for executive dashboards to reduce noise.
4. Optionally re-run clustering with alternate features or k choices (elbow / silhouette analysis) to refine groups.

If you want, I can generate an aggregated employee-level CSV or a sample Power BI template (PBIX) with these visuals pre-built.