# clustering.py
import pandas as pd
import umap
import hdbscan
import warnings

# Suppress the specific UMAP warnings so your terminal stays clean
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

def run_clustering(ml_ready_df, human_readable_df):
    # n_neighbors=50 forces UMAP to look at the "big picture" and connect the islands
    # min_dist=0.0 packs the clusters tighter together
    reducer = umap.UMAP(n_components=5, n_neighbors=50, min_dist=0.0, random_state=42)
    umap_embeddings = reducer.fit_transform(ml_ready_df)

    # min_cluster_size=50 allows medium-sized groups to be clusters
    # min_samples=10 makes the algorithm much less aggressive about labeling things as noise
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    cluster_probs = clusterer.probabilities_

    human_readable_df['cluster_label'] = cluster_labels
    human_readable_df['cluster_probability'] = cluster_probs
    
    return human_readable_df

def extract_outliers_and_exemplars(human_readable_df):
    exemplars = []
    
    # Get EVERY group, including the -1 Outlier group
    all_clusters = human_readable_df['cluster_label'].unique()

    for cluster_id in all_clusters:
        cluster_rows = human_readable_df[human_readable_df['cluster_label'] == cluster_id]
        
        if cluster_id == -1:
            # It's the massive attack wave! Just grab the first representative packet.
            exemplars.append(cluster_rows.iloc[0])
        else:
            # It's a normal cluster, grab the mathematical center
            best_idx = cluster_rows['cluster_probability'].idxmax()
            exemplars.append(cluster_rows.loc[best_idx])
            
    exemplars_df = pd.DataFrame(exemplars)
    
    # We return an empty DataFrame for outliers because we are now handling 
    # the entire Outlier pool as a single "Exemplar" to save API calls!
    outliers_df = pd.DataFrame()
    
    return outliers_df, exemplars_df