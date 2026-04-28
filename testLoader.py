import dataLoader
import pandas as pd
import features
import clustering
import classifier
import config

# Set pandas to show us all the columns instead of hiding them with "..."
pd.set_option('display.max_columns', None)

# Path to your freshly generated log
file_path = './thesis_output/smart_sample.json'

print(f"Loading data from {file_path}...")

# Run your newly fixed loader!
ml_ready_df, human_readable_df = dataLoader.load_data(file_path)

# --- Sanity Checks ---

print("\n=== THE VERIFICATION ===")
print(f"Total unique connections (flow_ids) processed: {len(ml_ready_df)}")

# 1. Did the DNS fix work?
print("\n--- Checking DNS ---")
if 'dns.rrname' in ml_ready_df.columns:
    print("SUCCESS: 'dns.rrname' column exists!")
    # Print the first 5 non-null DNS names to prove it extracted the text
    print(ml_ready_df['dns.rrname'].dropna().head())
else:
    print("FAIL: 'dns.rrname' is missing. The JSON flattening didn't catch it.")

# 2. Did the Flow metadata attach correctly?
print("\n--- Checking Flow Metrics ---")
if 'flow.pkts_toserver' in ml_ready_df.columns:
    print("SUCCESS: Flow metrics exist!")
    print(ml_ready_df[['flow.pkts_toserver', 'flow.bytes_toclient', 'flow.state']].dropna().head())
else:
    print("FAIL: Flow metrics are missing.")

# 3. Did the Alerts trigger?
print("\n--- Checking Alerts ---")
if 'alert.severity' in ml_ready_df.columns:
    print("SUCCESS: Alerts triggered and severity was logged!")
    print(ml_ready_df['alert.severity'].dropna().value_counts())
else:
    print("FAIL: No alerts found in this dataset.")




# --- NEW TEST CODE ---
print("\n=== TESTING FEATURE EXTRACTION ===")

# 2. Pass the raw data into your features script
# FIX: Use 'process_features' and catch both the DataFrame and the Scaler!
final_feature_vector, fitted_scaler = features.process_features(ml_ready_df)

# 3. Verify the shape and columns
print(f"Final Feature Vector Shape: {final_feature_vector.shape}")
print("\nFirst 3 rows of the fully processed ML data:")
print(final_feature_vector.head(3))

# --- NEW TEST CODE ---
print("\n=== TESTING CLUSTERING (UMAP + HDBSCAN) ===")

# 1. Run the clustering using the math matrix, but attach labels to the human data!
clustered_df = clustering.run_clustering(final_feature_vector, human_readable_df)

# 2. Extract the LLM-ready targets
outliers_df, exemplars_df = clustering.extract_outliers_and_exemplars(clustered_df)

# 3. Print the results
valid_clusters = [lbl for lbl in clustered_df['cluster_label'].unique() if lbl != -1]
print(f"Total Distinct Traffic Behaviors (Clusters) found: {len(valid_clusters)}")
print(f"Total Anomalies (Outliers) found: {len(outliers_df)}")

print("\n--- The Exemplars (Normal Traffic Representatives) ---")
# Let's peek at the human-readable DNS or IPs of the exemplars
if 'dns.rrname' in exemplars_df.columns:
    print(exemplars_df[['cluster_label', 'dns.rrname', 'dest_port']].head())
else:
    print(exemplars_df[['cluster_label', 'dest_ip', 'dest_port']].head())


print("\n=== TESTING LLM CLASSIFIER ===")

try:
    # 1. Initialize the model using your config labels
    llm_model = classifier.init_llm(config.THREAT_LABELS)
    print("Gemini model initialized successfully!")

    # 2. Translate the Pandas rows into LLM context
    print(f"Translating {len(exemplars_df)} exemplars to LLM context...")
    contexts = [classifier.row_to_llm_context(row) for _, row in outliers_df.iterrows()]
    # 3. Get predictions
    print("Querying Gemini API (this will take a few seconds)...")
    predictions = classifier.get_labels(contexts, llm_model, config.THREAT_LABELS)

    # 4. Display the results!
    for i, text in enumerate(contexts):
        print("\n--- Network Flow ---")
        # Print just the first 3 lines of your context so it doesn't flood the screen
        print("\n".join(text.split("\n")[:3]) + " ...")
        print(f"👉 LLM CLASSIFICATION: {predictions[i][0]}")

except ValueError as e:
    print(f"\n🚨 ERROR: {e}")
    print("Did you forget to run: export GEMINI_API_KEY='your_key' ?")


print("\n=== TESTING LABEL PROPAGATION ===")

# Create a mock training_df
test_training_df = ml_ready_df.copy()
test_training_df['target_label'] = "Pending"

# Test the "One-to-Many" broadcast logic
for i, (idx, row) in enumerate(exemplars_df.iterrows()):
    cluster_id = row['cluster_label']
    label = predictions[i][0] # The label Gemini just gave us
    
    # Find all members
    members = human_readable_df[human_readable_df['cluster_label'] == cluster_id].index
    test_training_df.loc[members, 'target_label'] = label
    print(f"Cluster {cluster_id} propagated label '{label}' to {len(members)} flows.")

# Check the final counts
print("\nFinal Resulting Label Distribution:")
print(test_training_df['target_label'].value_counts())