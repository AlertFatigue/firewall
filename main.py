# main.py
from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
from sklearn.model_selection import train_test_split

import config
from classifier import get_labels, init_llm, row_to_llm_context
from clustering import extract_outliers_and_exemplars, run_clustering
from dataLoader import load_data
from features import process_features


def _safe_stratify(labels):
    counts = labels.value_counts()
    if len(counts) < 2:
        print("WARNING: Only one target class found. Train/test split will not be stratified.")
        return None
    if counts.min() < 2:
        print("WARNING: At least one class has fewer than 2 rows. Train/test split will not be stratified.")
        return None
    return labels


def _apply_label_by_position(training_df, target_index, labels_dict, position: int):
    label, _score = labels_dict[position]
    training_df.loc[target_index, "target_label"] = label


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Run UMAP/HDBSCAN + LLM label propagation pipeline.")
    parser.add_argument("--input", default=config.FILE_PATH, help="Path to Suricata EVE JSON-lines file.")
    parser.add_argument("--dataset", default=config.DATASET_NAME, help="Dataset name used for output filenames.")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="Directory for output CSV/model files.")
    parser.add_argument("--no-llm", action="store_true", help="Do not call Gemini; use deterministic fallback labels.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size. Default: 0.2")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    labeled_file = os.path.join(args.output_dir, f"{args.dataset}_labeled.csv")
    train_file = os.path.join(args.output_dir, f"{args.dataset}_train.csv")
    test_file = os.path.join(args.output_dir, f"{args.dataset}_test.csv")
    clustered_human_file = os.path.join(args.output_dir, f"{args.dataset}_clustered_human_readable.csv")
    scaler_file = os.path.join(args.output_dir, f"{args.dataset}_robust_scaler.pkl")

    print(f"Loading data from: {args.input}")
    raw_df, human_readable_df = load_data(args.input)

    print("Processing features...")
    feature_df, scaler = process_features(raw_df)

    print("Running UMAP + HDBSCAN...")
    clustered_human_df = run_clustering(feature_df, human_readable_df)

    print("Extracting individual outliers and normal cluster exemplars...")
    outliers_df, exemplars_df = extract_outliers_and_exemplars(clustered_human_df)
    print(f"Normal cluster exemplars: {len(exemplars_df)}")
    print(f"Outliers selected for individual labeling: {len(outliers_df)}")

    print("Preparing LLM contexts...")
    outlier_contexts = [row_to_llm_context(row) for _, row in outliers_df.iterrows()]
    exemplar_contexts = [row_to_llm_context(row) for _, row in exemplars_df.iterrows()]

    model = None if args.no_llm else init_llm(config.THREAT_LABELS)
    if model is None:
        print("Gemini not available or --no-llm used. Using deterministic fallback labels for debugging.")
    else:
        print("Gemini initialized successfully.")

    print("Generating labels for outliers...")
    outlier_labels = get_labels(outlier_contexts, model, config.THREAT_LABELS)

    print("Generating labels for cluster exemplars...")
    exemplar_labels = get_labels(exemplar_contexts, model, config.THREAT_LABELS)

    print("Building labeled training dataset...")
    training_df = feature_df.copy()
    training_df["target_label"] = "Unclassified / Background Noise"
    training_df["flow_id"] = training_df.index.astype(str)

    # Mark every HDBSCAN outlier as suspicious by default, then overwrite selected
    # outliers with individual LLM/fallback labels.
    all_outlier_ids = clustered_human_df[clustered_human_df["cluster_label"] == -1].index
    if len(all_outlier_ids) > 0:
        training_df.loc[all_outlier_ids, "target_label"] = "Malformed protocol anomaly or unknown suspicious behavior"

    # Individual outlier labels.
    for position, (idx, _row) in enumerate(outliers_df.iterrows()):
        _apply_label_by_position(training_df, idx, outlier_labels, position)

    # Cluster label propagation: exemplar label -> all members of that normal cluster.
    for position, (_idx, row) in enumerate(exemplars_df.iterrows()):
        cluster_id = row["cluster_label"]
        label, _score = exemplar_labels[position]
        member_ids = clustered_human_df[clustered_human_df["cluster_label"] == cluster_id].index
        training_df.loc[member_ids, "target_label"] = label

    # Add cluster metadata for traceability but do not use it as model features later.
    training_df["cluster_label"] = clustered_human_df["cluster_label"]
    training_df["cluster_probability"] = clustered_human_df["cluster_probability"]

    print("Saving labeled artifacts...")
    training_df.to_csv(labeled_file)
    clustered_human_df.to_csv(clustered_human_file)
    joblib.dump(scaler, scaler_file)

    print("Performing train/test split...")
    stratify = _safe_stratify(training_df["target_label"])
    train_df, test_df = train_test_split(
        training_df,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    train_df.to_csv(train_file)
    test_df.to_csv(test_file)

    print("\n=== FINAL DATASET METRICS ===")
    print(f"Total flows processed: {len(training_df)}")
    print(f"Training set: {len(train_df)} rows -> {train_file}")
    print(f"Testing set: {len(test_df)} rows -> {test_file}")
    print(f"Labeled full dataset -> {labeled_file}")
    print(f"Human-readable clustered file -> {clustered_human_file}")
    print("\n--- Label Distribution ---")
    print(training_df["target_label"].value_counts())
    print("\nDone.")


if __name__ == "__main__":
    main()
