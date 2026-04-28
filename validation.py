# validation.py
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd

import config
from dataLoader import load_data


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Join labeled ML rows back with human-readable IP/port fields.")
    parser.add_argument("--input", default=config.FILE_PATH, help="Original Suricata EVE JSON-lines file.")
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    args = parser.parse_args(argv)

    labeled_file = os.path.join(args.output_dir, f"{args.dataset}_labeled.csv")
    output_file = os.path.join(args.output_dir, f"{args.dataset}_validation_with_ips.csv")

    if not os.path.exists(labeled_file):
        raise FileNotFoundError(f"Missing labeled file: {labeled_file}. Run main.py first.")

    print("Re-linking IPs and ports for validation...")
    labeled_df = pd.read_csv(labeled_file, index_col=0)
    _, original_df = load_data(args.input)

    keep_cols = [
        "flow_id",
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
    ]
    available = [c for c in keep_cols if c in original_df.columns]

    validation_df = labeled_df.join(original_df[available], rsuffix="_human")
    os.makedirs(args.output_dir, exist_ok=True)
    validation_df.to_csv(output_file)
    print(f"Done. Validation file created: {output_file}")


if __name__ == "__main__":
    main()
