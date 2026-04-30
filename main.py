import joblib
import config
import gc
import os
import pandas as pd
from dataLoader import load_data
from features import process_features
from clustering import run_clustering, extract_outliers_and_exemplars
from classifier import row_to_llm_context, init_llm, get_labels
from sklearn.model_selection import train_test_split

# Define cache filenames
CACHE_ML_READY = 'cache_ml_ready.pkl'
CACHE_HUMAN = 'cache_human_readable.pkl'
CACHE_OUTLIERS = 'cache_outliers.pkl'
CACHE_EXEMPLARS = 'cache_exemplars.pkl'

def main():
    # --- PHASE 1: CACHE CHECK OR COMPUTE ---
    if os.path.exists(CACHE_OUTLIERS) and os.path.exists(CACHE_EXEMPLARS):
        print("[*] Found cached clustering data! Skipping UMAP & HDBSCAN...")
        ml_ready_df = pd.read_pickle(CACHE_ML_READY)
        human_readable_df = pd.read_pickle(CACHE_HUMAN)
        outliers_df = pd.read_pickle(CACHE_OUTLIERS)
        exemplars_df = pd.read_pickle(CACHE_EXEMPLARS)
    else:
        print("Loading data...")
        ml_ready_df, human_readable_df = load_data(config.FILE_PATH)


        print("Creating a balanced subset of 10,000 alerts...")
        
        # Calculate exactly how many we need per category to hit 10,000
        num_categories = ml_ready_df['alert.category'].nunique()
        per_category_target = int(10000 / num_categories) + 1
        
        balanced_df = ml_ready_df.groupby('alert.category', group_keys=False).apply(
            lambda x: x.sample(min(len(x), per_category_target), random_state=42)
        )
        
        # If it slightly overshoots due to rounding, trim it to exactly 10k
        if len(balanced_df) > 10000:
            balanced_df = balanced_df.sample(n=10000, random_state=42)

        ml_ready_df = balanced_df
        human_readable_df = human_readable_df.loc[ml_ready_df.index]
        
        print(f"Balanced dataset size: {len(ml_ready_df)} alerts.")
        print("Processing features...")
        ml_ready_df, scaler = process_features(ml_ready_df)
        joblib.dump(scaler, 'robust_scaler.pkl') # Save scaler early
        ml_ready_df = ml_ready_df.astype('float32')
        
        print("Running UMAP and HDBSCAN...")
        human_readable_df = run_clustering(ml_ready_df, human_readable_df)

        print("Extracting outliers and exemplars...")
        outliers_df, exemplars_df = extract_outliers_and_exemplars(human_readable_df)

        print("Caching intermediate results to disk...")
        ml_ready_df.to_pickle(CACHE_ML_READY)
        human_readable_df.to_pickle(CACHE_HUMAN)
        outliers_df.to_pickle(CACHE_OUTLIERS)
        exemplars_df.to_pickle(CACHE_EXEMPLARS)

    # Flush RAM before loading the LLM
    gc.collect()

    # --- PHASE 2: LLM ADJUDICATION ---
    print("\nPreparing LLM contexts...")
    outlier_contexts = [row_to_llm_context(row) for _, row in outliers_df.iterrows()]
    exemplar_contexts = [row_to_llm_context(row) for _, row in exemplars_df.iterrows()]

    print("Initializing Gemini LLM...")
    model = init_llm(config.THREAT_LABELS)

    print("Generating labels via API...")
    print("--- Processing Outliers ---")
    outlier_labels = get_labels(outlier_contexts, model, config.THREAT_LABELS, checkpoint_file="outliers_checkpoint.pkl")
    
    print("\n--- Processing Exemplars ---")
    exemplar_labels = get_labels(exemplar_contexts, model, config.THREAT_LABELS, checkpoint_file="exemplars_checkpoint.pkl")

    # --- PHASE 3: LABEL PROPAGATION & DATASET BUILDING ---
    print("\nBuilding training dataset...")
    ml_ready_df['target_label'] = "Unclassified / Background Noise"
    
    # Initialize our new metrics column
    ml_ready_df['llm_latency_sec'] = 0.0
    
    # 1. Apply Outlier Labels (One-to-One)
    for i, (idx, row) in enumerate(outliers_df.iterrows()):
        # Unpack all 3 variables!
        label, score, latency = outlier_labels[i]
        ml_ready_df.at[idx, 'target_label'] = label
        ml_ready_df.at[idx, 'llm_latency_sec'] = latency

    # 2. Apply Exemplar Labels to ENTIRE Clusters (Label Propagation)
    for i, (idx, row) in enumerate(exemplars_df.iterrows()):
        cluster_id = row['cluster_label']
        label, score, latency = exemplar_labels[i]
        
        # Find every row belonging to this cluster
        member_ids = human_readable_df[human_readable_df['cluster_label'] == cluster_id].index
        
        # Broadcast the label to all members
        ml_ready_df.loc[member_ids, 'target_label'] = label
        
        # AMORTIZED LATENCY MATH:
        cluster_size = len(member_ids)
        amortized_latency = round(latency / cluster_size, 5)
        
        ml_ready_df.loc[member_ids, 'llm_latency_sec'] = amortized_latency

    print("Saving outputs...")
    ml_ready_df.to_csv('catboost_training_data.csv', index=False)

    print("\n--- Final Label Distribution ---")
    print(ml_ready_df['target_label'].value_counts())
    
    # --- PHASE 4: DATA SPLITTING ---
    # Handle rare classes before split (Stratify crashes if a class has only 1 sample)
    label_counts = ml_ready_df['target_label'].value_counts()
    valid_labels = label_counts[label_counts > 1].index
    ml_ready_df = ml_ready_df[ml_ready_df['target_label'].isin(valid_labels)]

    print("\nPerforming Stratified 80/20 Split...")
    train_df, test_df = train_test_split(
        ml_ready_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=ml_ready_df['target_label']
    )

    train_df.to_csv('generic_train.csv', index=False)
    test_df.to_csv('generic_test.csv', index=False)

    print(f"\n=== FINAL DATASET METRICS ===")
    print(f"Total Flows Processed: {len(ml_ready_df)}")
    print(f"Training Set: {len(train_df)} rows")
    print(f"Testing Set: {len(test_df)} rows")
    
    print("\nDone. Dataset ready.")

    total_latency = ml_ready_df['llm_latency_sec'].sum()
    total_api_calls = len(outliers_df) + len(exemplars_df)
    avg_latency_per_call = total_latency / total_api_calls if total_api_calls > 0 else 0
    print(f"\n=== LLM PERFORMANCE METRICS ===")
    print(f"Total API Calls Made: {total_api_calls}")
    print(f"Total Time Waiting on API: {total_latency:.2f} seconds ({total_latency/60:.2f} minutes)")
    print(f"Average Latency per Call: {avg_latency_per_call:.2f} seconds")

if __name__ == "__main__":
    main()