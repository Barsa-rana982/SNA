#!/usr/bin/env python3
"""
Complete Social Network Analysis (SNA) Pipeline
===============================================

This comprehensive script handles the entire SNA workflow:
1. Data Processing and Cleaning
2. Feature Engineering 
3. Clustering Analysis
4. Interactive Dashboard

Author: Team SNA
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CompleteSNAPipeline:
    """
    Complete Social Network Analysis Pipeline
    
    This class handles the entire workflow from raw data to interactive dashboard:
    - Data cleaning and preprocessing
    - Feature engineering (15 comprehensive features)
    - Clustering analysis (K-Means and DBSCAN)
    - Interactive visualizations and insights
    """
    
    def __init__(self):
        """Initialize the SNA Pipeline"""
        self.raw_data = None
        self.cleaned_data = None
        self.features = None
        self.scaled_features = None
        self.clusters = None
        self.scaler = StandardScaler()
        
    def load_and_clean_data(self, filepath="data/enron_cleaned_final.csv"):
        """
        Step 1: Load and clean the raw email data
        
        Args:
            filepath (str): Path to the raw email data file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            st.info("Loading and cleaning email data...")
            
            # Load raw data
            self.raw_data = pd.read_csv(filepath)
            st.success(f"Loaded {len(self.raw_data):,} email records")
            
            # Convert timestamp to datetime
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
            
            # Smart filtering for quality data
            # Remove emails outside reasonable timeframe (1999-2002)
            start_date = datetime(1999, 1, 1)
            end_date = datetime(2002, 12, 31)
            self.cleaned_data = self.raw_data[
                (self.raw_data['timestamp'] >= start_date) & 
                (self.raw_data['timestamp'] <= end_date)
            ].copy()
            
            # Remove invalid email addresses
            self.cleaned_data = self.cleaned_data[
                (self.cleaned_data['sender'].str.contains('@', na=False)) &
                (self.cleaned_data['recipient'].str.contains('@', na=False))
            ]
            
            # Remove duplicates
            initial_count = len(self.cleaned_data)
            self.cleaned_data = self.cleaned_data.drop_duplicates(
                subset=['sender', 'recipient', 'timestamp', 'subject']
            )
            
            st.success(f"Data cleaning complete:")
            st.write(f"- Original records: {len(self.raw_data):,}")
            st.write(f"- After cleaning: {len(self.cleaned_data):,}")
            st.write(f"- Removed: {len(self.raw_data) - len(self.cleaned_data):,} records")
            
            return True
            
        except Exception as e:
            st.error(f"Error in data loading/cleaning: {str(e)}")
            return False
    
    def extract_features(self):
        """
        Step 2: Extract comprehensive features from email data
        
        Creates 15 features across 5 categories:
        - Volume Features (4): Basic communication metrics
        - Network Features (3): Social network characteristics  
        - Behavioral Features (3): Communication patterns
        - Temporal Features (3): Time-based activity
        - Organizational Features (2): Enron employee status
        """
        try:
            st.info("Extracting comprehensive features...")
            
            # Get unique people (both senders and recipients)
            all_people = set(self.cleaned_data['sender'].unique()) | set(self.cleaned_data['recipient'].unique())
            
            features_list = []
            
            for person in all_people:
                # Get person's email activity
                sent_emails = self.cleaned_data[self.cleaned_data['sender'] == person]
                received_emails = self.cleaned_data[self.cleaned_data['recipient'] == person]
                
                # Skip if no activity
                if len(sent_emails) == 0 and len(received_emails) == 0:
                    continue
                
                # === VOLUME FEATURES (4) ===
                emails_sent = len(sent_emails)
                emails_received = len(received_emails)
                total_communications = emails_sent + emails_received
                
                # === NETWORK FEATURES (3) ===
                unique_contacts_out = sent_emails['recipient'].nunique() if len(sent_emails) > 0 else 0
                unique_contacts_in = received_emails['sender'].nunique() if len(received_emails) > 0 else 0
                total_unique_contacts = len(set(sent_emails['recipient'].tolist() + received_emails['sender'].tolist()))
                
                # === BEHAVIORAL FEATURES (3) ===
                send_receive_ratio = emails_sent / max(emails_received, 1)
                contact_diversity_ratio = total_unique_contacts / max(total_communications, 1)
                
                # === TEMPORAL FEATURES (3) ===
                all_person_emails = pd.concat([sent_emails, received_emails]) if len(sent_emails) > 0 and len(received_emails) > 0 else (sent_emails if len(sent_emails) > 0 else received_emails)
                
                if len(all_person_emails) > 0:
                    # Average communication hour (0-23)
                    avg_communication_hour = all_person_emails['timestamp'].dt.hour.mean()
                    
                    # Weekend communication ratio
                    weekend_emails = all_person_emails[all_person_emails['timestamp'].dt.weekday >= 5]
                    weekend_communication_ratio = len(weekend_emails) / len(all_person_emails)
                    
                    # After hours ratio (before 8 AM or after 6 PM)
                    after_hours_emails = all_person_emails[
                        (all_person_emails['timestamp'].dt.hour < 8) | 
                        (all_person_emails['timestamp'].dt.hour > 18)
                    ]
                    after_hours_ratio = len(after_hours_emails) / len(all_person_emails)
                    
                    # Communication span in days
                    communication_span_days = (all_person_emails['timestamp'].max() - 
                                             all_person_emails['timestamp'].min()).days + 1
                    
                    # Average daily communications
                    avg_daily_communications = total_communications / max(communication_span_days, 1)
                else:
                    avg_communication_hour = 12  # Default to noon
                    weekend_communication_ratio = 0
                    after_hours_ratio = 0
                    communication_span_days = 1
                    avg_daily_communications = 0
                
                # === ORGANIZATIONAL FEATURES (2) ===
                is_enron_employee = 1 if '@enron.com' in person.lower() else 0
                
                # External communication ratio
                if is_enron_employee:
                    external_sent = sent_emails[~sent_emails['recipient'].str.contains('@enron.com', na=False)]
                    external_received = received_emails[~received_emails['sender'].str.contains('@enron.com', na=False)]
                    external_communications = len(external_sent) + len(external_received)
                    external_communication_ratio = external_communications / max(total_communications, 1)
                else:
                    external_communication_ratio = 1.0  # All external if not Enron employee
                
                # Compile feature vector
                features_list.append({
                    'person': person,
                    'emails_sent': emails_sent,
                    'emails_received': emails_received,
                    'total_communications': total_communications,
                    'unique_contacts_out': unique_contacts_out,
                    'unique_contacts_in': unique_contacts_in,
                    'total_unique_contacts': total_unique_contacts,
                    'send_receive_ratio': send_receive_ratio,
                    'contact_diversity_ratio': contact_diversity_ratio,
                    'avg_communication_hour': avg_communication_hour,
                    'weekend_communication_ratio': weekend_communication_ratio,
                    'after_hours_ratio': after_hours_ratio,
                    'communication_span_days': communication_span_days,
                    'avg_daily_communications': avg_daily_communications,
                    'is_enron_employee': is_enron_employee,
                    'external_communication_ratio': external_communication_ratio
                })
            
            # Create features DataFrame
            self.features = pd.DataFrame(features_list)
            
            st.success(f"Feature extraction complete:")
            st.write(f"- Analyzed {len(all_people):,} people")
            st.write(f"- Created features for {len(self.features):,} active users")
            st.write(f"- Generated 16 comprehensive features")
            
            return True
            
        except Exception as e:
            st.error(f"Error in feature extraction: {str(e)}")
            return False
    
    def perform_clustering(self, n_clusters=None):
        """
        Step 3: Perform clustering analysis
        
        Args:
            n_clusters (int): Number of clusters for K-Means (auto-optimized if None)
        """
        try:
            st.info("Performing clustering analysis...")
            
            # Prepare feature matrix for clustering (exclude person and categorical features)
            feature_columns = [col for col in self.features.columns if col not in ['person', 'is_enron_employee']]
            X = self.features[feature_columns]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            self.scaled_features = pd.DataFrame(X_scaled, columns=feature_columns, index=self.features.index)
            
            # === K-MEANS CLUSTERING ===
            if n_clusters is None:
                # Optimize number of clusters using silhouette score
                silhouette_scores = []
                k_range = range(2, min(11, len(self.features)//10))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                
                # Select best k
                best_k = k_range[np.argmax(silhouette_scores)]
                best_silhouette = max(silhouette_scores)
            else:
                best_k = n_clusters
            
            # Final K-Means clustering
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score
            kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
            
            # === DBSCAN CLUSTERING ===
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            
            # Create results DataFrame
            self.clusters = self.features.copy()
            self.clusters['kmeans_cluster'] = kmeans_labels
            self.clusters['dbscan_cluster'] = dbscan_labels
            
            # Add scaled features for analysis
            for col in feature_columns:
                self.clusters[f'{col}_scaled'] = X_scaled[:, feature_columns.index(col)]
            
            st.success(f"Clustering analysis complete:")
            st.write(f"- Optimal K-Means clusters: {best_k}")
            st.write(f"- K-Means silhouette score: {kmeans_silhouette:.3f}")
            st.write(f"- DBSCAN clusters found: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
            st.write(f"- DBSCAN noise points: {list(dbscan_labels).count(-1)}")
            
            return True
            
        except Exception as e:
            st.error(f"Error in clustering: {str(e)}")
            return False
    
    def save_results(self, output_dir="data"):
        """
        Step 4: Save all results to files
        
        Args:
            output_dir (str): Directory to save results
        """
        try:
            st.info("Saving analysis results...")
            
            # Save cleaned data
            self.cleaned_data.to_csv(f"{output_dir}/enron_cleaned_for_analysis.csv", index=False)
            
            # Save raw features
            self.features.to_csv(f"{output_dir}/person_features_raw.csv", index=False)
            
            # Save scaled features
            scaled_df = self.features[['person']].copy()
            feature_columns = [col for col in self.features.columns if col not in ['person', 'is_enron_employee']]
            scaled_df = pd.concat([scaled_df, self.scaled_features], axis=1)
            scaled_df.to_csv(f"{output_dir}/person_features_scaled.csv", index=False)
            
            # Save clustering results
            self.clusters.to_csv(f"{output_dir}/cluster_analysis_detailed.csv", index=False)
            
            # Save final results summary
            final_results = self.clusters[['person', 'total_communications', 'total_unique_contacts', 
                                         'is_enron_employee', 'kmeans_cluster', 'dbscan_cluster']]
            final_results.to_csv(f"{output_dir}/clustering_results_final.csv", index=False)
            
            # Save feature matrix for ML
            ml_features = self.clusters[[col for col in self.clusters.columns 
                                       if col not in ['person', 'kmeans_cluster', 'dbscan_cluster']]]
            ml_features.to_csv(f"{output_dir}/clustering_matrix_ready.csv", index=False)
            
            st.success("All results saved successfully!")
            
        except Exception as e:
            st.error(f"Error saving results: {str(e)}")
    
    def create_dashboard(self):
        """
        Step 5: Create interactive dashboard
        """
        st.title("🌐 Complete Social Network Analysis Dashboard")
        st.markdown("**Comprehensive SNA Pipeline Results** | Data Processing → Feature Engineering → Clustering → Insights")
        st.markdown("---")
        
        if self.clusters is None:
            st.warning("Please run the complete pipeline first!")
            return
        
        # === OVERVIEW SECTION ===
        st.header("📊 Analysis Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total People", f"{len(self.clusters):,}")
        with col2:
            st.metric("Email Records", f"{len(self.cleaned_data):,}")
        with col3:
            kmeans_clusters = self.clusters['kmeans_cluster'].nunique()
            st.metric("K-Means Clusters", kmeans_clusters)
        with col4:
            feature_cols = [col for col in self.clusters.columns if col not in ['person', 'kmeans_cluster', 'dbscan_cluster']]
            silhouette_avg = silhouette_score(
                self.clusters[[col for col in feature_cols if not col.endswith('_scaled')]][[col for col in feature_cols if not col.endswith('_scaled')]][:5].values,
                self.clusters['kmeans_cluster'].values[:5]
            ) if len(self.clusters) >= 5 else 0
            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
        
        # === CLUSTER ANALYSIS ===
        st.header("🎯 Cluster Analysis")
        
        # Cluster distribution
        fig_dist = px.pie(
            values=self.clusters['kmeans_cluster'].value_counts().values,
            names=[f"Cluster {i}" for i in self.clusters['kmeans_cluster'].value_counts().index],
            title="Cluster Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # === DETAILED CLUSTER PROFILES ===
        st.header("👥 Cluster Profiles")
        
        for cluster_id in sorted(self.clusters['kmeans_cluster'].unique()):
            cluster_data = self.clusters[self.clusters['kmeans_cluster'] == cluster_id]
            
            with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} people ({len(cluster_data)/len(self.clusters)*100:.1f}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Communication Metrics")
                    st.write(f"**Avg Total Communications:** {cluster_data['total_communications'].mean():.1f}")
                    st.write(f"**Avg Unique Contacts:** {cluster_data['total_unique_contacts'].mean():.1f}")
                    st.write(f"**Avg Daily Activity:** {cluster_data['avg_daily_communications'].mean():.2f}")
                    
                with col2:
                    st.subheader("Profile Characteristics")
                    enron_pct = (cluster_data['is_enron_employee'].sum() / len(cluster_data)) * 100
                    st.write(f"**Enron Employees:** {enron_pct:.1f}%")
                    st.write(f"**Weekend Activity:** {cluster_data['weekend_communication_ratio'].mean()*100:.1f}%")
                    st.write(f"**Send/Receive Ratio:** {cluster_data['send_receive_ratio'].mean():.2f}")
        
        # === NETWORK VISUALIZATION ===
        st.header("🔍 Feature Analysis")
        
        # Feature correlation heatmap
        numeric_features = self.clusters.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if not col.endswith('_scaled') and col not in ['kmeans_cluster', 'dbscan_cluster']]
        
        if len(numeric_features) > 1:
            corr_matrix = self.clusters[numeric_features].corr()
            fig_heatmap = px.imshow(corr_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # === 3D CLUSTER VISUALIZATION ===
        st.header("📈 3D Cluster Visualization")
        
        if len(self.clusters) > 3:
            # PCA for 3D visualization
            feature_matrix = self.clusters[[col for col in numeric_features[:8]]].fillna(0)
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(feature_matrix)
            
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                z=pca_result[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.clusters['kmeans_cluster'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cluster")
                ),
                text=[f"Person: {person}<br>Cluster: {cluster}" 
                      for person, cluster in zip(self.clusters['person'], self.clusters['kmeans_cluster'])],
                hovertemplate="<b>%{text}</b><br>" +
                              "PC1: %{x:.2f}<br>" +
                              "PC2: %{y:.2f}<br>" +
                              "PC3: %{z:.2f}<br>" +
                              "<extra></extra>"
            )])
            
            fig_3d.update_layout(
                title="3D Cluster Visualization (PCA)",
                scene=dict(
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                    zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)"
                )
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Complete SNA Pipeline",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize pipeline
    pipeline = CompleteSNAPipeline()
    
    # Sidebar controls
    st.sidebar.header("🚀 Pipeline Control")
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload Email Data (CSV)", 
        type=['csv'],
        help="Upload your email dataset or use default Enron data"
    )
    
    data_path = "data/enron_cleaned_final.csv"
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_path = "temp_upload.csv"
        st.sidebar.success("File uploaded successfully!")
    
    # Pipeline execution buttons
    st.sidebar.markdown("### Pipeline Steps")
    
    if st.sidebar.button("🔄 Step 1: Load & Clean Data"):
        with st.spinner("Processing..."):
            pipeline.load_and_clean_data(data_path)
    
    if st.sidebar.button("⚙️ Step 2: Extract Features") and pipeline.cleaned_data is not None:
        with st.spinner("Extracting features..."):
            pipeline.extract_features()
    
    if st.sidebar.button("🎯 Step 3: Perform Clustering") and pipeline.features is not None:
        with st.spinner("Clustering analysis..."):
            pipeline.perform_clustering()
    
    if st.sidebar.button("💾 Step 4: Save Results") and pipeline.clusters is not None:
        with st.spinner("Saving results..."):
            pipeline.save_results()
    
    if st.sidebar.button("🚀 Run Complete Pipeline"):
        with st.spinner("Running complete pipeline..."):
            if (pipeline.load_and_clean_data(data_path) and 
                pipeline.extract_features() and 
                pipeline.perform_clustering()):
                pipeline.save_results()
                st.sidebar.success("Pipeline completed successfully!")
    
    # Display dashboard
    pipeline.create_dashboard()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Complete SNA Pipeline v2.0**")
    st.sidebar.markdown("*All-in-one solution for Social Network Analysis*")


if __name__ == "__main__":
    main()