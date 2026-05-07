import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import config
import time
import psutil
import numpy as np
import os
import gc # Added for memory management

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print("1. Loading lightweight CSV dataset...")
    # load feature extracted csv
    dtype_mapping = {col: np.float32 for col in config.NUMERIC_FEATURES}
    
    # passed the dtype_mapping into the read function to save RAM
    df = pd.read_csv('suricata_features_extracted.csv', dtype=dtype_mapping)
    
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # drop rows where the Label is missing to prevent train_test_split crash
    nan_count = df['Label'].isna().sum()
    if nan_count > 0:
        print(f"Warning: Dropping {nan_count} rows with missing Labels.")
        df = df.dropna(subset=['Label'])

    print("Grouping DoS attacks into a single 'Malicious' class...")
    df['Label'] = df['Label'].apply(lambda x: 'Malicious' if x != 'BENIGN' else 'BENIGN')
    # forcefill all NaN with missing and make sure they are strings
    for col in config.CAT_FEATURES:
      df[col] = df[col].fillna('Missing').astype(str)
      # catch any literal nan strings pandas may have created
      df[col] = df[col].replace('nan', 'Missing')

    # Separate X features and target y feature
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Stratified 80/20 split
    print("\n2. Splitting and stratifying data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y 
    )

    # Check class outputs
    print("\n--- Class Distribution Check ---")
    
    # Combine counts and percentage in summary DF
    train_dist = pd.DataFrame({
        'Train Count': y_train.value_counts(),
        'Train %': y_train.value_counts(normalize=True) * 100
    })
    
    test_dist = pd.DataFrame({
        'Test Count': y_test.value_counts(),
        'Test %': y_test.value_counts(normalize=True) * 100
    })
    
    distribution_summary = train_dist.join(test_dist)
    print(distribution_summary.round(2).to_string())
    print("--------------------------------\n")

    # Initialize and train catboost
    print("3. Initializing CatBoost Training...")
    
    # Convert to catboost pool objects for efficiency
    train_pool = Pool(data=X_train, label=y_train, cat_features=config.CAT_FEATURES)
    test_pool = Pool(data=X_test, label=y_test, cat_features=config.CAT_FEATURES)

    # purge memory before training to prevent OOM Killer
    print("Purging unused Pandas DataFrames from memory...")
    del df, X, y, X_train, y_train 
    gc.collect()

    # catboost model creation, changing lossfunction due to binary classes
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='LogLoss', 
        eval_metric='F1',
        task_type="CPU",
        # auto_class_weights='Balanced', 
        random_seed=24,
        thread_count=4, # limits CPU threads to prevent memory spikes
        border_count=32 # reduces memory footprint heavily
    )

    # Setup the process tracker
    process = psutil.Process(os.getpid())
    
    # training profiling
    print("\n--- Starting Training ---")
    start_train_time = time.perf_counter()
    start_train_ram = process.memory_info().rss / (1024 * 1024) # Convert bytes to MB

    model.fit(
        train_pool,
        eval_set=test_pool,
        verbose=100, 
        early_stopping_rounds=50 
    )

    end_train_time = time.perf_counter()
    end_train_ram = process.memory_info().rss / (1024 * 1024)
    

    train_duration = end_train_time - start_train_time
    ram_used_during_train = end_train_ram - start_train_ram

    print(f"\n[METRICS] Training Latency: {train_duration:.4f} seconds")
    print(f"[METRICS] RAM footprint after training: {end_train_ram:.2f} MB (+{ram_used_during_train:.2f} MB during fit)")

    print("\n4. Training Complete!")
    
    # Save the model
    model.save_model("suricata_catboost_model.cbm")
    print("Model saved to 'suricata_catboost_model.cbm'")


    # inference profiling
    print("\n--- Starting Inference (Testing) Profiling ---")
    
    # isolate testing featres
    num_test_samples = len(X_test)
    
    start_infer_time = time.perf_counter()
    
    # model predictions for test set
    predictions = model.predict(X_test)
    
    end_infer_time = time.perf_counter()
    #

    total_infer_duration = end_infer_time - start_infer_time
    
    # calculate milliseconds to classify one flow
    per_flow_latency_ms = (total_infer_duration / num_test_samples) * 1000
    
    # throughput calc
    throughput_fps = num_test_samples / total_infer_duration

    print(f"\n--- THESIS INFERENCE METRICS ---")
    print(f"Total Test Samples:    {num_test_samples:,}")
    print(f"Total Inference Time:  {total_infer_duration:.4f} seconds")
    print(f"Per-Flow Latency:      {per_flow_latency_ms:.6f} milliseconds")
    print(f"Throughput:            {throughput_fps:,.2f} flows/second")
    print(f"--------------------------------")