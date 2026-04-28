# main.py
import joblib
import config
from dataLoader import load_data
from features import process_features
from clustering import run_clustering, extract_outliers_and_exemplars
from classifier import row_to_llm_context, init_llm, get_labels
from sklearn.model_selection import train_test_split
import sys

def main():
    print("Loading data...")
    ml_ready_df, human_readable_df = load_data(config.FILE_PATH)

    print("Processing features...")
    ml_ready_df, scaler = process_features(ml_ready_df)

    print("Running UMAP and HDBSCAN...")
    human_readable_df = run_clustering(ml_ready_df, human_readable_df)

    print("Extracting outliers and exemplars...")
    outliers_df, exemplars_df = extract_outliers_and_exemplars(human_readable_df)

    print("Preparing LLM contexts...")
    outlier_contexts = [row_to_llm_context(row) for _, row in outliers_df.iterrows()]
    exemplar_contexts = [row_to_llm_context(row) for _, row in exemplars_df.iterrows()]

    print("Initializing Gemini LLM...")
    # Pass the labels list directly to the init function so it gets baked into the prompt
    model = init_llm(config.THREAT_LABELS)

    print("Generating labels via API...")
    outlier_labels = get_labels(outlier_contexts, model, config.THREAT_LABELS)
    exemplar_labels = get_labels(exemplar_contexts, model, config.THREAT_LABELS)

    print("Building training dataset...")
    training_df = ml_ready_df.copy()
    
    # Initialize everything as a safe default
    training_df['target_label'] = "Unclassified / Background Noise"
    
    # 1. Apply Outlier Labels (One-to-One)
    # Outliers don't belong to clusters, so we apply these individually
    for i, (idx, row) in enumerate(outliers_df.iterrows()):
        context = exemplar_contexts[i]
        if "444" in context:
            print("\n--- ATTACK CLUSTER CONTEXT SENT TO GEMINI ---")
            print(context)
        # This will show you exactly what Gemini saw before it chose "HTTP"
        label, score = outlier_labels[i]
        training_df.at[idx, 'target_label'] = label

    # 2. Apply Exemplar Labels to ENTIRE Clusters (One-to-Many)
    # This is the "Label Propagation" step
    for i, (idx, row) in enumerate(exemplars_df.iterrows()):
        cluster_id = row['cluster_label'] # Get the Cluster ID (e.g., 4)
        label, score = exemplar_labels[i]
        
        # Find every row in the ORIGINAL human_readable_df that belongs to this cluster
        member_ids = human_readable_df[human_readable_df['cluster_label'] == cluster_id].index
        
        # Broadcast the LLM's label to every member of that cluster in training_df
        training_df.loc[member_ids, 'target_label'] = label

    print("Saving outputs...")
    training_df.to_csv('catboost_training_data.csv')
    joblib.dump(scaler, 'robust_scaler.pkl')

    print("\n--- Final Label Distribution ---")
    print(training_df['target_label'].value_counts())
    print("\nDone. Dataset ready.")
# 4. Final Stratified 80/20 Split
    print("\nPerforming Stratified 80/20 Split...")
    
    # We use 'stratify' to ensure the minority class (the attack) is perfectly balanced
    train_df, test_df = train_test_split(
        training_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=training_df['target_label']
    )

    # 5. Dynamic File Saving based on what we are processing
    # If the file path contains 'heartbleed', save as heartbleed. If 'dos', save as dos.
    if "heartbleed" in config.FILE_PATH.lower():
        train_filename = 'heartbleed_train.csv'
        test_filename = 'heartbleed_test.csv'
        print("Detected Heartbleed dataset. Saving files...")
    elif "dos" in config.FILE_PATH.lower():
        train_filename = 'dos_train.csv'
        test_filename = 'dos_test.csv'
        print("Detected DoS dataset. Saving files...")
    else:
        train_filename = 'generic_train.csv'
        test_filename = 'generic_test.csv'

    # Save the artifacts
    train_df.to_csv(train_filename)
    test_df.to_csv(test_filename)

    # 6.
    print(f"\n=== FINAL DATASET METRICS ===")
    print(f"Total Flows Processed: {len(training_df)}")
    print(f"Training Set: {len(train_df)} rows")
    print(f"Testing Set: {len(test_df)} rows")
    
    print("\n--- Training Set Label Distribution ---")
    print(train_df['target_label'].value_counts())
    
    print("\n--- Testing Set Label Distribution ---")
    print(test_df['target_label'].value_counts())

if __name__ == "__main__":
    main()