import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

def main():
    print("Step 1: Scanning Ground Truth for Attacks (Memory Safe)...")
    master_df = pd.read_csv('master_validation_with_ips.csv', index_col=0)

    # 1. We will store the exact Attack IP/Port combinations in a tiny set
    attack_tuples = set()
    
    # Read the massive CSV in chunks of 100,000 rows so RAM never goes up
    for chunk in pd.read_csv('wednesdayGroundTruth.csv', encoding='cp1252', chunksize=100000):
        chunk.columns = chunk.columns.str.strip()
        
        src_ip_col = next((c for c in chunk.columns if 'source ip' in c.lower()), None)
        dest_ip_col = next((c for c in chunk.columns if 'destination ip' in c.lower()), None)
        dest_port_col = next((c for c in chunk.columns if 'destination port' in c.lower()), None)
        label_col = next((c for c in chunk.columns if 'label' in c.lower()), None)
        
        # Keep ONLY the attacks
        attacks_only = chunk[chunk[label_col].str.contains('Heartbleed', case=False, na=False)]
        
        # Add their IP/Port combos to our lookup set
        for _, row in attacks_only.iterrows():
            src = str(row[src_ip_col]).strip()
            dest = str(row[dest_ip_col]).strip()
            # Safely handle the port number format
            try:
                port = str(int(float(row[dest_port_col])))
            except:
                port = str(row[dest_port_col])
                
            attack_tuples.add((src, dest, port))

    print(f"Found {len(attack_tuples)} unique attack streams in Ground Truth.")

    # 2. Clean your master data
    for col in ['src_ip', 'dest_ip']:
        master_df[col] = master_df[col].astype(str).str.strip()
        
    master_df['dest_port_str'] = master_df['dest_port'].astype(float).astype('Int64').astype(str)

    print("Step 2: Training Model...")
    train_df, test_df = train_test_split(master_df, test_size=0.2, random_state=42, stratify=master_df['target_label'])
    
    cols_to_drop = ['target_label', 'src_ip', 'dest_ip', 'src_port', 'dest_port', 'dest_port_str']
    if 'flow_id' in master_df.columns: cols_to_drop.append('flow_id')
    
    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_df['target_label']
    X_test = test_df.drop(columns=cols_to_drop, errors='ignore')

    model = CatBoostClassifier(iterations=500, learning_rate=0.1, auto_class_weights='Balanced', verbose=0)
    model.fit(X_train, y_train)
    
    test_df = test_df.copy()
    test_df['CatBoost_Prediction'] = model.predict(X_test).ravel()

    print("Step 3: Labeling (Without Merging!)...")
    
    # Check if each row in your test_df is in our set of known attacks
    def is_actual_attack(row):
        return 1 if (row['src_ip'], row['dest_ip'], row['dest_port_str']) in attack_tuples else 0

    test_df['Actual_Is_Attack'] = test_df.apply(is_actual_attack, axis=1)
    
    test_df['Predicted_Is_Attack'] = np.where(
        test_df['CatBoost_Prediction'].str.contains('Malformed|Vulnerability|Suspicious', case=False, na=False), 1, 0
    )

    print("\n" + "="*30)
    print("   FINAL THESIS PERFORMANCE")
    # --- DEBUG: What did the AI call the rows that were actually attacks? ---
    actual_attacks = test_df[test_df['Actual_Is_Attack'] == 1]
    if len(actual_attacks) > 0:
        print("\n--- [DEBUG] AI Predictions for known Heartbleed rows ---")
        print(actual_attacks['CatBoost_Prediction'].value_counts())
    else:
        print("\n--- [DEBUG] No Heartbleed rows made it into the Test Split ---")
    print("="*30)
    
    y_true = test_df['Actual_Is_Attack']
    y_pred = test_df['Predicted_Is_Attack']

    if len(y_true.unique()) < 2:
        print("CRITICAL: Still only found one class. The test_df didn't get the attack row.")
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"True Positives (Caught Attacks):  {tp}")
        print(f"False Positives (False Alarms):   {fp}")
        print(f"Recall: {tp/(tp+fn):.4f}" if (tp+fn)>0 else "Recall: 0")

if __name__ == "__main__":
    main()