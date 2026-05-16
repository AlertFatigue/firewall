import pandas as pd
import numpy as np
import time
import os
import psutil
import config
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# THESIS INFERENCE & EVALUATION SCRIPT
# ==========================================
if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    print("1. Loading dataset and preparing Test Set...")
    
    # load data
    dtype_mapping = {col: np.float32 for col in config.NUMERIC_FEATURES}
    df = pd.read_csv('suricata_features_extracted.csv', dtype=dtype_mapping)
    
    # === THE FIX: ALIGN COLUMNS BY HIDING ID IN INDEX ===
    if 'community_id' in df.columns:
        df.set_index('community_id', inplace=True)
        print("Success: community_id moved to index.")
    # ====================================================
    
    # clean missing labels
    df = df.dropna(subset=['Label'])

    print("Grouping DoS attacks into a single 'Malicious' class...")
    df['Label'] = df['Label'].apply(lambda x: 'Malicious' if x != 'BENIGN' else 'BENIGN')
    
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

    # record ram before loading model
    ram_before_model = process.memory_info().rss / (1024 * 1024)
    
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

    # recording ram after loading model
    ram_after_model = process.memory_info().rss / (1024 * 1024)
    model_footprint = ram_after_model - ram_before_model
    
    # profiling performance
    print("\n3. Running Inference...")
    num_test_samples = len(X_test)
    
    start_infer_time = time.perf_counter()
    
    # generating predictions
    predictions = model.predict(X_test)
    
    end_infer_time = time.perf_counter()
    
    # calculate ram after inference is complete
    ram_after_inference = process.memory_info().rss / (1024 * 1024)
    inference_overhead = ram_after_inference - ram_after_model
    
    # speed metrics
    total_infer_duration = end_infer_time - start_infer_time
    per_flow_latency_ms = (total_infer_duration / num_test_samples) * 1000
    throughput_fps = num_test_samples / total_infer_duration

    print(f"\n--- THESIS SPEED METRICS ---")
    print(f"Total Test Samples:    {num_test_samples:,}")
    print(f"Total Inference Time:  {total_infer_duration:.4f} seconds")
    print(f"Per-Flow Latency:      {per_flow_latency_ms:.6f} milliseconds")
    print(f"Throughput:            {throughput_fps:,.2f} flows/second")
    print(f"Model RAM Footprint:   {model_footprint:.2f} MB")
    print(f"Inference RAM Spike:   {inference_overhead:.2f} MB")
    print(f"----------------------------")

    # F1 precision recall
    print("\n4. Generating Classification Report...")
    
    # flatten array
    preds_flat = predictions.flatten() if predictions.ndim > 1 else predictions
    
    # generate scikit-learn report
    report = classification_report(y_test, preds_flat, digits=4)
    
    print("\n--- THESIS CLASSIFICATION METRICS ---")
    print(report)
    
    # ==========================================
    # PR-AUC CALCULATION BLOCK
    # ==========================================
    # Convert string labels to binary for PR-AUC math (Malicious = 1, BENIGN = 0)
    y_test_binary = (y_test == 'Malicious').astype(int)
    
    # Find which column index CatBoost assigned to the 'Malicious' class
    malicious_idx = list(model.classes_).index('Malicious')
    
    # Get the raw probability scores instead of hard labels
    y_pred_proba = model.predict_proba(X_test)[:, malicious_idx]
    
    # Calculate PR-AUC score
    pr_auc = average_precision_score(y_test_binary, y_pred_proba)
    print(f"PR-AUC Score:          {pr_auc:.6f}")
    # ==========================================
    print("-------------------------------------")


    # visualization by confusion matrix
    print("\n5. Generating Visual Confusion Matrix...")
    cm = confusion_matrix(y_test, preds_flat)
    labels = sorted(y_test.unique())

    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Firewall Classification: Predicted vs Actual')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")

    print("\n6. Extracting False Negatives...")
    
    fn_mask = (y_test == 'Malicious') & (preds_flat == 'BENIGN')
    false_negatives_df = X_test[fn_mask].copy()
    
    false_negatives_df['Actual_Label'] = y_test[fn_mask]
    false_negatives_df['Predicted_Label'] = preds_flat[fn_mask]
    false_negatives_df['Malicious_Probability'] = y_pred_proba[fn_mask]
    
    # === PULL THE ID OUT OF THE INDEX ===
    false_negatives_df.reset_index(inplace=True) 
    # Now 'community_id' is a normal column again!
    # ====================================
    
    csv_filename = "exact_false_negatives.csv"
    false_negatives_df.to_csv(csv_filename, index=False)

    # ==========================================
    # 7. EXTRACTING FALSE POSITIVES (FP)
    # ==========================================
    print("\n7. Extracting False Positives...")
    
    # Mask: Actual is BENIGN, but Predicted is Malicious
    fp_mask = (y_test == 'BENIGN') & (preds_flat == 'Malicious')
    
    false_positives_df = X_test[fp_mask].copy()
    
    # Add metrics
    false_positives_df['Actual_Label'] = y_test[fp_mask]
    false_positives_df['Predicted_Label'] = preds_flat[fp_mask]
    # Probability of being Malicious (even though it's actually Benign)
    false_positives_df['Malicious_Probability'] = y_pred_proba[fp_mask]
    
    # Pull the ID back out of the index
    false_positives_df.reset_index(inplace=True) 
    
    fp_filename = "exact_false_positives.csv"
    false_positives_df.to_csv(fp_filename, index=False)
    
    print(f"Successfully saved {len(false_positives_df)} False Positives to '{fp_filename}'.")