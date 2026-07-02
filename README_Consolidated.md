# 🌐 Complete Social Network Analysis (SNA) Pipeline

**All-in-One Solution for Email Network Analysis**

## 📋 Overview

This project provides a complete end-to-end pipeline for analyzing email communication networks, specifically designed for the Enron email dataset. Everything is consolidated into a single comprehensive file for easy collaboration and deployment.

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly streamlit scikit-learn
```

### Run the Complete Pipeline
```bash
streamlit run CompleteSNAPipeline.py
```

## 📊 What It Does

### 1. **Data Processing & Cleaning**
- Loads raw email data (751K+ records)
- Smart filtering for quality data (1999-2002 timeframe)
- Removes duplicates and invalid entries
- Results in ~628K clean email records

### 2. **Feature Engineering**
Creates **16 comprehensive features** across 5 categories:
- **Volume Features** (4): Communication counts and totals
- **Network Features** (3): Contact networks and diversity
- **Behavioral Features** (3): Send/receive patterns
- **Temporal Features** (3): Time-based activity patterns  
- **Organizational Features** (2): Enron employee status

### 3. **Clustering Analysis**
- **K-Means Clustering**: Auto-optimized number of clusters
- **DBSCAN**: Density-based clustering for outlier detection
- **Performance Metrics**: Silhouette scores and validation
- **Results**: Identifies communication patterns and user types

### 4. **Interactive Dashboard**
- **Overview Metrics**: Key statistics and KPIs
- **Cluster Profiles**: Detailed analysis of each cluster
- **3D Visualizations**: PCA-based cluster visualization
- **Feature Analysis**: Correlation matrices and insights

## 🎯 Key Findings

The analysis reveals **2 distinct user types**:

### Cluster 0: Regular Users (99.8%)
- **6,767 people** with typical communication patterns
- Average: 146 emails, 13 contacts
- Mix of internal and external users
- Standard business communication levels

### Cluster 1: Super Users (0.2%)  
- **16 people** who are communication powerhouses
- Average: 16,623 emails, 376 contacts
- 113x higher communication volume
- 100% Enron employees (executives/key personnel)

## 📁 File Structure

```
SNA/
├── CompleteSNAPipeline.py    # Main consolidated pipeline
├── README.md                 # This documentation
└── data/                     # Data directory
    ├── enron_cleaned_final.csv          # Input data
    ├── cluster_analysis_detailed.csv    # Generated results
    ├── person_features_raw.csv          # Extracted features
    └── clustering_results_final.csv     # Final cluster assignments
```

## 🔧 Usage

### Option 1: Interactive Dashboard
```bash
streamlit run CompleteSNAPipeline.py
```
- Use sidebar buttons to run individual steps
- Click "🚀 Run Complete Pipeline" for full analysis
- Explore results in interactive dashboard

### Option 2: Programmatic Use
```python
from CompleteSNAPipeline import CompleteSNAPipeline

# Initialize pipeline
pipeline = CompleteSNAPipeline()

# Run complete analysis
pipeline.load_and_clean_data("data/enron_cleaned_final.csv")
pipeline.extract_features()
pipeline.perform_clustering()
pipeline.save_results()
```

## 📈 Key Features

- **🔄 End-to-End Pipeline**: Complete workflow in one file
- **⚡ Auto-Optimization**: Automatic parameter tuning
- **📊 Rich Visualizations**: Interactive plots and charts
- **💾 Export Results**: Saves all intermediate and final results
- **🎯 Business Insights**: Clear interpretation of findings
- **🤝 Team-Friendly**: Single file for easy collaboration

## 🛠️ Customization

### Adding New Features
Modify the `extract_features()` method to add custom features:

```python
# Add your custom feature calculation
custom_feature = your_calculation_here
features_list.append({
    # ... existing features ...
    'your_custom_feature': custom_feature
})
```

### Different Clustering Methods
Extend the `perform_clustering()` method:

```python
# Add new clustering algorithm
from sklearn.cluster import AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels = agg_clustering.fit_predict(X_scaled)
```

## 📚 Technical Details

- **Language**: Python 3.7+
- **Key Libraries**: pandas, scikit-learn, streamlit, plotly
- **Clustering**: K-Means (optimized) + DBSCAN
- **Scaling**: StandardScaler normalization
- **Visualization**: 3D PCA plots, correlation heatmaps
- **Performance**: Silhouette score validation

## 🎉 Benefits of This Consolidated Approach

1. **Easy Collaboration**: Single file to share with teammates
2. **Simple Deployment**: One command to run everything  
3. **Reduced Complexity**: No need to manage multiple files
4. **Complete Functionality**: All original features preserved
5. **Professional Quality**: Clean, documented, production-ready code

## 👥 Team Collaboration

This consolidated pipeline makes it easy to:
- Share complete codebase in a single file
- Version control with fewer conflicts
- Deploy consistently across different environments
- Understand the full workflow without jumping between files
- Merge changes from multiple team members

Perfect for academic projects, presentations, and production deployments!