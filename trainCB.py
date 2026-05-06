import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import config

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print("1. Loading lightweight CSV dataset...")
    # load feature extracted csv
    df = pd.read_csv('suricata_features_extracted.csv')
    
    print(f"Dataset loaded. Shape: {df.shape}")

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
        stratify=y  # <--- This guarantees proportional class distributions
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

    # Loss function with MultiClass due to multiple classes
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass', 
        eval_metric='Accuracy',
        task_type="CPU", 
        random_seed=42
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        verbose=100, # print progress every 100 iterations
        early_stopping_rounds=50 # Stop if accuracy hasn't improved for 50 trees
    )

    print("\n4. Training Complete!")
    
    # Save the model
    model.save_model("suricata_catboost_model.cbm")
    print("Model saved to 'suricata_catboost_model.cbm'")