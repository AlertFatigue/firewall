# trainModel.py
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

import config


NON_FEATURE_COLS = [
    "target_label",
    "flow_id",
    "cluster_label",
    "cluster_probability",
]


def _prepare_xy(df: pd.DataFrame):
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors="ignore")
    # Keep only numeric columns so accidental raw text columns do not break CatBoost.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["target_label"].astype(str)
    return X, y


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Train CatBoost on LLM-propagated alert labels.")
    parser.add_argument("--dataset", default=config.DATASET_NAME, help="Dataset name used by main.py.")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="Directory containing train/test CSV files.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args(argv)

    train_file = os.path.join(args.output_dir, f"{args.dataset}_train.csv")
    test_file = os.path.join(args.output_dir, f"{args.dataset}_test.csv")
    model_file = os.path.join(args.output_dir, f"{args.dataset}_catboost_model.cbm")
    metrics_file = os.path.join(args.output_dir, f"{args.dataset}_metrics.json")
    feature_importance_file = os.path.join(args.output_dir, f"{args.dataset}_feature_importance.csv")

    print(f"Loading data from {train_file} and {test_file}...")
    train_df = pd.read_csv(train_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)

    if "target_label" not in train_df.columns or "target_label" not in test_df.columns:
        raise ValueError("target_label column missing. Run main.py first.")

    X_train, y_train = _prepare_xy(train_df)
    X_test, y_test = _prepare_xy(test_df)

    if y_train.nunique() < 2:
        raise ValueError("Training data has only one class. CatBoost classification needs at least two classes.")

    print("Initializing CatBoost Classifier...")
    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
    )

    print("Training model...")
    start_train = time.perf_counter()
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    train_seconds = time.perf_counter() - start_train

    print("Evaluating model...")
    start_pred = time.perf_counter()
    predictions = model.predict(X_test).ravel().astype(str)
    pred_seconds = time.perf_counter() - start_pred

    latency_per_alert = pred_seconds / max(len(X_test), 1)
    eps = len(X_test) / pred_seconds if pred_seconds > 0 else float("inf")

    report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=sorted(y_test.unique()))

    print("\n=== MODEL EVALUATION ===")
    print(report_text)
    print("\nConfusion matrix labels:")
    print(sorted(y_test.unique()))
    print(cm)
    print("\n=== LATENCY / EFFICIENCY ===")
    print(f"Training time: {train_seconds:.4f} seconds")
    print(f"Prediction time: {pred_seconds:.6f} seconds for {len(X_test)} alerts")
    print(f"Latency per alert: {latency_per_alert:.8f} seconds")
    print(f"Throughput: {eps:.2f} alerts/second")

    print("Saving model and metrics...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_model(model_file)

    feature_importances = model.get_feature_importance()
    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)
    importance_df.to_csv(feature_importance_file, index=False)

    metrics = {
        "dataset": args.dataset,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_seconds": train_seconds,
        "prediction_seconds": pred_seconds,
        "latency_per_alert_seconds": latency_per_alert,
        "events_per_second": eps,
        "classification_report": report_dict,
        "confusion_matrix_labels": sorted(y_test.unique()),
        "confusion_matrix": cm.tolist(),
        "model_file": model_file,
        "feature_importance_file": feature_importance_file,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(importance_df.head(10))
    print(f"\nModel saved to: {model_file}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
