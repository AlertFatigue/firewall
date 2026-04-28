# testLoader.py
from __future__ import annotations

import argparse
import os

import pandas as pd

import classifier
import clustering
import config
import dataLoader
import features

pd.set_option("display.max_columns", None)


def main():
    parser = argparse.ArgumentParser(description="Sanity-test the data loading, features, clustering, and LLM contexts.")
    parser.add_argument("--input", default=config.FILE_PATH, help="Path to EVE JSON-lines file.")
    parser.add_argument("--sample-contexts", type=int, default=5)
    parser.add_argument("--call-llm", action="store_true", help="Actually call Gemini if GEMINI_API_KEY is configured.")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    raw_df, human_readable_df = dataLoader.load_data(args.input)

    print("\n=== VERIFICATION ===")
    print(f"Total unique flows processed: {len(raw_df)}")

    print("\n--- Checking DNS ---")
    if "dns.rrname" in raw_df.columns:
        print("SUCCESS: dns.rrname column exists.")
        print(raw_df["dns.rrname"].dropna().head())
    else:
        print("WARNING: dns.rrname is missing in this dataset.")

    print("\n--- Checking Flow Metrics ---")
    flow_cols = [c for c in ["flow.pkts_toserver", "flow.bytes_toclient", "flow.state"] if c in raw_df.columns]
    if flow_cols:
        print("SUCCESS: Flow metrics exist.")
        print(raw_df[flow_cols].dropna().head())
    else:
        print("WARNING: Flow metrics are missing.")

    print("\n--- Checking Alerts ---")
    if "alert.severity" in raw_df.columns:
        print("SUCCESS: Alerts exist in this dataset.")
        print(raw_df["alert.severity"].dropna().value_counts())
    else:
        print("WARNING: No alert.severity column found in this dataset.")

    print("\n=== TESTING FEATURE EXTRACTION ===")
    feature_df, fitted_scaler = features.process_features(raw_df)
    print(f"Final feature vector shape: {feature_df.shape}")
    print(feature_df.head(3))

    print("\n=== TESTING CLUSTERING ===")
    clustered_df = clustering.run_clustering(feature_df, human_readable_df)
    outliers_df, exemplars_df = clustering.extract_outliers_and_exemplars(clustered_df)

    valid_clusters = [lbl for lbl in clustered_df["cluster_label"].unique() if lbl != -1]
    print(f"Total normal clusters found: {len(valid_clusters)}")
    print(f"Total selected outliers: {len(outliers_df)}")

    print("\n--- Exemplars ---")
    preview_cols = [c for c in ["cluster_label", "cluster_probability", "dns.rrname", "dest_ip", "dest_port"] if c in exemplars_df.columns]
    print(exemplars_df[preview_cols].head() if not exemplars_df.empty else "No exemplars found.")

    print("\n=== TESTING CONTEXT GENERATION ===")
    # FIX: use exemplars_df for exemplar contexts, not outliers_df.
    contexts = [classifier.row_to_llm_context(row) for _, row in exemplars_df.head(args.sample_contexts).iterrows()]
    print(f"Generated {len(contexts)} exemplar contexts.")
    for i, text in enumerate(contexts[: args.sample_contexts], start=1):
        print(f"\n--- Context {i} ---")
        print("\n".join(text.split("\n")[:8]))

    if args.call_llm:
        print("\n=== TESTING LLM CLASSIFIER ===")
        llm_model = classifier.init_llm(config.THREAT_LABELS)
        if llm_model is None:
            print("Gemini is unavailable. Set GEMINI_API_KEY or install google-generativeai.")
        predictions = classifier.get_labels(contexts, llm_model, config.THREAT_LABELS)
        for i, (label, confidence) in predictions.items():
            print(f"Context {i}: {label} (confidence marker: {confidence})")

    print("\nSanity test complete.")


if __name__ == "__main__":
    main()
