# evaluate.py
from __future__ import annotations

import argparse
import os
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

import config
from trainModel import NON_FEATURE_COLS


SUSPICIOUS_REGEX = "|".join(config.SUSPICIOUS_LABEL_KEYWORDS)


def _norm_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _norm_port(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _find_col(columns, *needles) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for col_lower, original in lower_map.items():
        if all(n.lower() in col_lower for n in needles):
            return original
    return None


def load_attack_tuples(groundtruth_file: str, attack_label: str) -> Set[Tuple[str, str, str]]:
    """Loads attack tuples from CIC-style ground-truth CSV without keeping the whole file in RAM."""
    attack_tuples: Set[Tuple[str, str, str]] = set()

    for chunk in pd.read_csv(groundtruth_file, encoding="cp1252", chunksize=100000):
        chunk.columns = chunk.columns.str.strip()
        src_ip_col = _find_col(chunk.columns, "source", "ip")
        dest_ip_col = _find_col(chunk.columns, "destination", "ip")
        dest_port_col = _find_col(chunk.columns, "destination", "port")
        label_col = _find_col(chunk.columns, "label")

        if not all([src_ip_col, dest_ip_col, dest_port_col, label_col]):
            raise ValueError("Could not identify source IP, destination IP, destination port, and label columns.")

        attacks_only = chunk[chunk[label_col].astype(str).str.contains(attack_label, case=False, na=False)]
        for _, row in attacks_only.iterrows():
            attack_tuples.add(
                (_norm_str(row[src_ip_col]), _norm_str(row[dest_ip_col]), _norm_port(row[dest_port_col]))
            )

    return attack_tuples


def _prepare_x(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors="ignore")
    # Also remove common human-readable columns that might exist in validation files.
    X = X.drop(
        columns=[
            c
            for c in [
                "timestamp",
                "src_ip",
                "src_port",
                "dest_ip",
                "dest_port",
                "proto",
                "app_proto",
                "alert.signature",
                "alert.category",
                "alert.severity",
                "flow_id_human",
            ]
            if c in X.columns
        ],
        errors="ignore",
    )
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate predicted attack/non-attack against a ground-truth CSV.")
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--validation-file", default=None, help="CSV with features + IP/port fields.")
    parser.add_argument("--groundtruth-file", default="wednesdayGroundTruth.csv")
    parser.add_argument("--attack-label", default="Heartbleed")
    parser.add_argument("--model-file", default=None)
    args = parser.parse_args(argv)

    validation_file = args.validation_file or os.path.join(args.output_dir, f"{args.dataset}_validation_with_ips.csv")
    model_file = args.model_file or os.path.join(args.output_dir, f"{args.dataset}_catboost_model.cbm")

    if not os.path.exists(validation_file):
        raise FileNotFoundError(f"Missing validation file: {validation_file}. Run validation.py first.")
    if not os.path.exists(args.groundtruth_file):
        raise FileNotFoundError(f"Missing ground-truth CSV: {args.groundtruth_file}")

    print("Step 1: Loading ground truth attack tuples...")
    attack_tuples = load_attack_tuples(args.groundtruth_file, args.attack_label)
    print(f"Found {len(attack_tuples)} unique attack tuples for label '{args.attack_label}'.")

    print("Step 2: Loading validation data...")
    df = pd.read_csv(validation_file, index_col=0)

    required = ["src_ip", "dest_ip", "dest_port"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Validation file is missing required columns: {missing}")

    df["Actual_Is_Attack"] = df.apply(
        lambda row: 1
        if (_norm_str(row["src_ip"]), _norm_str(row["dest_ip"]), _norm_port(row["dest_port"])) in attack_tuples
        else 0,
        axis=1,
    )

    print("Step 3: Getting model predictions...")
    if "CatBoost_Prediction" in df.columns:
        predictions = df["CatBoost_Prediction"].astype(str)
    else:
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Missing CatBoost model: {model_file}. Run trainModel.py first.")
        model = CatBoostClassifier()
        model.load_model(model_file)
        X = _prepare_x(df)
        predictions = pd.Series(model.predict(X).ravel().astype(str), index=df.index)
        df["CatBoost_Prediction"] = predictions

    df["Predicted_Is_Attack"] = np.where(
        predictions.str.contains(SUSPICIOUS_REGEX, case=False, na=False), 1, 0
    )

    print("\n=== FINAL BINARY ATTACK EVALUATION ===")
    y_true = df["Actual_Is_Attack"]
    y_pred = df["Predicted_Is_Attack"]

    if y_true.nunique() < 2:
        print("WARNING: Ground truth contains only one class in this validation file.")
        print("This means recall/false-negative analysis is not reliable for this run.")

    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix [labels 0=non-attack, 1=attack]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    output_file = os.path.join(args.output_dir, f"{args.dataset}_binary_attack_evaluation.csv")
    df.to_csv(output_file)
    print(f"\nSaved evaluated rows to: {output_file}")


if __name__ == "__main__":
    main()
