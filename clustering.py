import pandas as pd
import umap
import hdbscan
import warnings

# Suppress the specific UMAP warnings so your terminal stays clean
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

def run_clustering(ml_ready_df, human_readable_df):
    # n_neighbors=100 forces UMAP to look at the "big picture" and connect the islands
    # min_dist=0.0 packs the clusters tighter together
    reducer = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.01, random_state=42)
    umap_embeddings = reducer.fit_transform(ml_ready_df)

    # min cluster size and n neighbors need changed to account for smaller dataset.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=3, metric='euclidean')
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    cluster_probs = clusterer.probabilities_

    human_readable_df['cluster_label'] = cluster_labels
    human_readable_df['cluster_probability'] = cluster_probs
    
    return human_readable_df

def extract_outliers_and_exemplars(human_readable_df):
    exemplars = []
    
    # 1. Grab ALL the noise/outliers (-1). We return the whole dataframe of them.
    # We copy it to avoid Pandas SettingWithCopy warnings later.
    outliers_df = human_readable_df[human_readable_df['cluster_label'] == -1].copy()
    
    # 2. Get only the valid clusters (ignore -1)
    valid_clusters = [c for c in human_readable_df['cluster_label'].unique() if c != -1]

    for cluster_id in valid_clusters:
        cluster_rows = human_readable_df[human_readable_df['cluster_label'] == cluster_id]
        
        # It's a normal, dense cluster. Grab the mathematical center.
        best_idx = cluster_rows['cluster_probability'].idxmax()
        exemplars.append(cluster_rows.loc[best_idx])
            
    # Create the DataFrame for our valid cluster centers
    exemplars_df = pd.DataFrame(exemplars)
    
    return outliers_df, exemplars_df