import pandas as pd
import numpy as np
import time
import config
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==========================================
# THESIS INFERENCE & EVALUATION SCRIPT
# ==========================================
if __name__ == "__main__":
    print("1. Loading dataset and preparing Test Set...")
    
    # load data
    dtype_mapping = {col: np.float32 for col in config.NUMERIC_FEATURES}
    df = pd.read_csv('suricata_features_extracted.csv', dtype=dtype_mapping)

    # clean missing labels
    df = df.dropna(subset=['Label'])

    # handle missing cat vals
    for col in config.CAT_FEATURES:
        df[col] = df[col].fillna('Missing').astype(str)
        df[col] = df[col].replace('nan', 'Missing')

    # 4. Separate X and y
    X = df.drop('Label', axis=1)
    y = df['Label']

    # recreate exact same split with same random state
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=24, 
        stratify=y 
    )

    print(f"Test Set Ready: {len(X_test):,} flows to classify.")

    # loading pretrained model from trainCB.py
    print("\n2. Loading Pre-trained CatBoost Model...")
    model = CatBoostClassifier()
    
    try:
        model.load_model("suricata_catboost_model.cbm")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'suricata_catboost_model.cbm' is in the same folder as this script.")
        exit()

    # profiling performance
    print("\n3. Running Inference...")
    num_test_samples = len(X_test)
    
    start_infer_time = time.perf_counter()
    
    # generating predictions
    predictions = model.predict(X_test)
    
    end_infer_time = time.perf_counter()

    # speed metrics
    total_infer_duration = end_infer_time - start_infer_time
    per_flow_latency_ms = (total_infer_duration / num_test_samples) * 1000
    throughput_fps = num_test_samples / total_infer_duration

    print(f"\n--- THESIS SPEED METRICS ---")
    print(f"Total Test Samples:    {num_test_samples:,}")
    print(f"Total Inference Time:  {total_infer_duration:.4f} seconds")
    print(f"Per-Flow Latency:      {per_flow_latency_ms:.6f} milliseconds")
    print(f"Throughput:            {throughput_fps:,.2f} flows/second")
    print(f"----------------------------")

    # F1 precision recall
    print("\n4. Generating Classification Report...")
    
    # flatten array
    preds_flat = predictions.flatten() if predictions.ndim > 1 else predictions
    
    # generate scikit-learn report
    report = classification_report(y_test, preds_flat, digits=4)
    
    print("\n--- THESIS CLASSIFICATION METRICS ---")
    print(report)
    print("-------------------------------------")