"""
Advanced Clustering Dashboard for Team Member Focus.

This specialized dashboard provides comprehensive clustering analysis capabilities
designed specifically for clustering specialists. It focuses on advanced metrics,
detailed cluster analysis, and professional visualization tools.

Features:
- Advanced clustering performance metrics (Silhouette Score, Cluster Balance, etc.)
- Interactive KPI displays with professional styling
- Detailed cluster profiling and analysis
- Quality assessment and recommendations
- Export capabilities for professional reporting

Author: SNA Team - Clustering Specialist Dashboard
Version: 2.0 (Professional Enhancement)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Clustering Analysis Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ClusteringDashboard:
    """
    Advanced Clustering Dashboard for Team Member Focus.
    
    This class provides comprehensive clustering analysis capabilities
    specifically designed for clustering specialists. It includes:
    
    CORE FUNCTIONALITY:
    - Load and validate clustering data
    - Calculate advanced clustering performance metrics
    - Create interactive visualizations and KPI displays
    - Provide quality assessments and recommendations
    
    DATA SOURCES:
    - cluster_analysis_detailed.csv: Complete analysis dataset with features and clusters
    - clustering_matrix_ready.csv: Feature matrix ready for ML algorithms
    - clustering_results_final.csv: Final cluster assignments and metadata
    
    KEY METRICS:
    - Silhouette Score: Cluster separation quality (-1 to 1, higher is better)
    - Cluster Balance: Size distribution fairness (lower is more balanced)
    - Intra-cluster Distance: Compactness within clusters (lower is better)
    - Stability Analysis: Consistency across different random seeds
    
    Attributes:
        data (DataFrame): Main analysis dataset with features and cluster labels
        features (DataFrame): Feature matrix for ML algorithms
        cluster_results (DataFrame): Cluster assignments and metadata
        silhouette_scores (dict): Cached silhouette scores for performance
    """
    
    def __init__(self):
        """Initialize ClusteringDashboard with empty data containers."""
        # Primary data containers
        self.data = None                # Main analysis dataset
        self.features = None            # Feature matrix for ML
        self.cluster_results = None     # Cluster assignments
        self.silhouette_scores = {}     # Cached performance metrics
        
    def load_data(self):
        """Load clustering data and results"""
        try:
            # Load main analysis data
            self.data = pd.read_csv("data/cluster_analysis_detailed.csv")
            
            # Load feature matrix
            self.features = pd.read_csv("data/clustering_matrix_ready.csv")
            
            # Validation checks
            if self.data.empty or self.features.empty:
                st.error("Data files are empty or corrupted")
                return False
                
            if 'kmeans_cluster' not in self.data.columns:
                st.error("Clustering results not found in data")
                return False
                
            st.success(f"Successfully loaded data: {len(self.data)} samples, {len(self.data.columns)} features")
            return True
            
        except FileNotFoundError as e:
            st.error(f"Data file not found: {str(e)}")
            st.info("Please ensure clustering analysis has been completed and data files exist")
            return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def calculate_clustering_kpis(self):
        """Calculate advanced clustering performance metrics"""
        # Basic KPIs
        total_samples = len(self.data)
        num_clusters = self.data['kmeans_cluster'].nunique()
        
        # Silhouette score for current clustering
        feature_cols = [col for col in self.data.columns if col not in ['person', 'kmeans_cluster']]
        X = self.data[feature_cols].values
        labels = self.data['kmeans_cluster'].values
        
        silhouette_avg = silhouette_score(X, labels)
        silhouette_samples_scores = silhouette_samples(X, labels)
        
        # Cluster size distribution
        cluster_sizes = self.data['kmeans_cluster'].value_counts().sort_index()
        cluster_balance = cluster_sizes.std() / cluster_sizes.mean()  # Lower is more balanced
        
        # Intra-cluster distances
        intra_cluster_distances = []
        for cluster in range(num_clusters):
            cluster_data = X[labels == cluster]
            if len(cluster_data) > 1:
                centroid = cluster_data.mean(axis=0)
                distances = np.sqrt(np.sum((cluster_data - centroid) ** 2, axis=1))
                intra_cluster_distances.append(distances.mean())
        
        avg_intra_distance = np.mean(intra_cluster_distances)
        
        return {
            'total_samples': total_samples,
            'num_clusters': num_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_balance': cluster_balance,
            'avg_intra_distance': avg_intra_distance,
            'cluster_sizes': cluster_sizes,
            'silhouette_samples': silhouette_samples_scores
        }

    def create_clustering_kpi_section(self, kpis):
        """Create KPI metrics section"""
        st.subheader("Clustering Performance KPIs")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Samples",
                f"{kpis['total_samples']:,}",
                help="Total number of data points clustered"
            )
        
        with col2:
            st.metric(
                "Number of Clusters",
                f"{kpis['num_clusters']}",
                help="Optimal number of clusters identified"
            )
        
        with col3:
            # Silhouette score with color coding
            score = kpis['silhouette_score']
            if score > 0.7:
                delta_color = "normal"
                help_text = "Excellent clustering quality"
            elif score > 0.5:
                delta_color = "normal" 
                help_text = "Good clustering quality"
            else:
                delta_color = "inverse"
                help_text = "Poor clustering quality"
            
            st.metric(
                "Silhouette Score",
                f"{score:.3f}",
                delta=f"{'Excellent' if score > 0.7 else 'Good' if score > 0.5 else 'Poor'}",
                help=help_text
            )
        
        with col4:
            balance = kpis['cluster_balance']
            st.metric(
                "Cluster Balance",
                f"{balance:.2f}",
                delta=f"{'Balanced' if balance < 1.0 else 'Imbalanced'}",
                help="Lower values indicate more balanced cluster sizes"
            )
        
        with col5:
            st.metric(
                "Avg Intra-Distance",
                f"{kpis['avg_intra_distance']:.3f}",
                help="Average distance within clusters (lower is better)"
            )

def main():
    """Main dashboard application"""
    
    # Title and description
    st.title("Advanced Clustering Analysis Dashboard")
    st.markdown("**Specialized Dashboard for Clustering Team Member** | Deep dive into clustering performance, patterns, and insights")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = ClusteringDashboard()
    
    # Load data
    if not dashboard.load_data():
        st.stop()
    
    # Calculate KPIs
    kpis = dashboard.calculate_clustering_kpis()
    
    # Create KPI section
    dashboard.create_clustering_kpi_section(kpis)
    
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("Dashboard Filters")
    
    # Cluster filter
    available_clusters = sorted(dashboard.data['kmeans_cluster'].unique())
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters to Analyze",
        available_clusters,
        default=available_clusters,
        help="Choose specific clusters for detailed analysis"
    )
    
    # Feature selection for analysis
    feature_cols = [col for col in dashboard.data.columns if col not in ['person', 'kmeans_cluster']]
    selected_features = st.sidebar.multiselect(
        "Select Features for Analysis",
        feature_cols,
        default=feature_cols[:5],
        help="Choose features for detailed visualization"
    )
    
    # Analysis mode
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Overview", "Cluster Profiles", "Feature Analysis", "Quality Assessment", "Export Results"],
        help="Choose the type of analysis to perform"
    )
    
    # Filter data based on selections
    filtered_data = dashboard.data[dashboard.data['kmeans_cluster'].isin(selected_clusters)]
    
    if analysis_mode == "Overview":
        st.header("Clustering Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster size distribution
            fig = px.bar(
                x=kpis['cluster_sizes'].index,
                y=kpis['cluster_sizes'].values,
                title="Cluster Size Distribution",
                labels={'x': 'Cluster', 'y': 'Number of Samples'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Silhouette analysis
            if len(selected_features) > 0:
                X_selected = filtered_data[selected_features].values
                labels_selected = filtered_data['kmeans_cluster'].values
                
                if len(np.unique(labels_selected)) > 1:
                    silhouette_scores = silhouette_samples(X_selected, labels_selected)
                    
                    fig = go.Figure()
                    
                    for cluster in sorted(filtered_data['kmeans_cluster'].unique()):
                        cluster_scores = silhouette_scores[labels_selected == cluster]
                        fig.add_trace(go.Box(
                            y=cluster_scores,
                            name=f'Cluster {cluster}',
                            boxmean=True
                        ))
                    
                    fig.update_layout(
                        title="Silhouette Score Distribution by Cluster",
                        yaxis_title="Silhouette Score",
                        xaxis_title="Cluster"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_mode == "Cluster Profiles":
        st.header("Detailed Cluster Profiles")
        
        if len(selected_features) >= 3:
            # Create radar chart for cluster profiles
            cluster_profiles = filtered_data.groupby('kmeans_cluster')[selected_features].mean()
            
            # Normalize data for radar chart
            scaler = StandardScaler()
            normalized_profiles = pd.DataFrame(
                scaler.fit_transform(cluster_profiles.T).T,
                columns=cluster_profiles.columns,
                index=cluster_profiles.index
            )
            
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            for i, cluster in enumerate(normalized_profiles.index):
                if cluster in selected_clusters:
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_profiles.loc[cluster].values.tolist() + [normalized_profiles.loc[cluster].values[0]],
                        theta=list(normalized_profiles.columns) + [normalized_profiles.columns[0]],
                        fill='toself',
                        name=f'Cluster {cluster}',
                        line_color=colors[i % len(colors)]
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-3, 3]
                    )),
                showlegend=True,
                title="Cluster Feature Profiles (Standardized)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("Cluster Characteristics Summary")
        summary_stats = filtered_data.groupby('kmeans_cluster')[selected_features].agg(['mean', 'std']).round(3)
        st.dataframe(summary_stats, use_container_width=True)
    
    elif analysis_mode == "Quality Assessment":
        st.header("Clustering Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Quality Metrics Interpretation")
            
            # Silhouette score interpretation
            score = kpis['silhouette_score']
            if score > 0.7:
                st.success(f"Excellent clustering quality (Score: {score:.3f})")
                st.info("Clusters are well-separated and internally cohesive")
            elif score > 0.5:
                st.warning(f"Good clustering quality (Score: {score:.3f})")
                st.info("Clusters are reasonably well-defined")
            else:
                st.error(f"Poor clustering quality (Score: {score:.3f})")
                st.info("Consider adjusting clustering parameters or preprocessing")
            
            # Cluster balance assessment
            balance = kpis['cluster_balance']
            if balance < 1.0:
                st.success(f"Well-balanced clusters (Balance: {balance:.2f})")
            else:
                st.warning(f"Imbalanced clusters (Balance: {balance:.2f})")
                st.info("Some clusters may be significantly larger than others")
        
        with col2:
            st.subheader("Recommendations")
            
            recommendations = []
            
            if kpis['silhouette_score'] < 0.5:
                recommendations.append("Consider increasing/decreasing the number of clusters")
                recommendations.append("Review feature selection and preprocessing")
            
            if kpis['cluster_balance'] > 1.5:
                recommendations.append("Investigate outliers that might form small clusters")
                recommendations.append("Consider different clustering algorithms")
            
            if not recommendations:
                recommendations.append("Current clustering configuration appears optimal")
                recommendations.append("Consider validation with business stakeholders")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    elif analysis_mode == "Export Results":
        st.header("Export Clustering Results")
        
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Cluster Assignments"):
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="cluster_assignments.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download KPI Summary"):
                kpi_summary = pd.DataFrame([{
                    'Metric': k,
                    'Value': v if not isinstance(v, (pd.Series, np.ndarray)) else str(v)
                } for k, v in kpis.items() if k not in ['cluster_sizes', 'silhouette_samples']])
                
                csv = kpi_summary.to_csv(index=False)
                st.download_button(
                    label="Download KPI CSV",
                    data=csv,
                    file_name="clustering_kpis.csv",
                    mime="text/csv"
                )
        
        st.subheader("Current Analysis Summary")
        st.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Total Samples:** {kpis['total_samples']:,}")
        st.write(f"**Number of Clusters:** {kpis['num_clusters']}")
        st.write(f"**Silhouette Score:** {kpis['silhouette_score']:.3f}")
        st.write(f"**Selected Clusters:** {selected_clusters}")
        st.write(f"**Selected Features:** {len(selected_features)} features")

if __name__ == "__main__":
    main()