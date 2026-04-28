# train_model.py
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Load the Stratified Data
    # Change this to 'dos_train.csv' when you want to train the DoS model
    train_file = 'heartbleed_train.csv'
    test_file = 'heartbleed_test.csv'
    
    print(f"Loading data from {train_file} and {test_file}...")
    train_df = pd.read_csv(train_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)

    # 2. Separate Features (X) and Target Labels (y)
    X_train = train_df.drop(columns=['target_label'])
    y_train = train_df['target_label']
    
    X_test = test_df.drop(columns=['target_label'])
    y_test = test_df['target_label']

    # 3. Initialize CatBoost
    # auto_class_weights='Balanced' is the secret weapon here. 
    # It tells CatBoost to pay extra attention to the rare attack rows!
    print("\nInitializing CatBoost Classifier...")
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        auto_class_weights='Balanced', 
        random_seed=42,
        verbose=100 # Print progress every 100 trees
    )

    # 4. Train the Model
    print("Training model...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    # 5. Evaluate the Model
    print("\n=== MODEL EVALUATION ===")
    predictions = model.predict(X_test)
    
    # Flatten the CatBoost predictions array
    predictions = [pred[0] for pred in predictions]

    # Print the Classification Report (Accuracy, Precision, Recall)
    print(classification_report(y_test, predictions))

    # 6. Extract Feature Importance (Crucial for Thesis Defense)
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    feature_importances = model.get_feature_importance()
    feature_names = X_train.columns
    
    # Sort and print the top 10 features CatBoost used to catch the attacks
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
    print(importance_df)

if __name__ == "__main__":
    main()