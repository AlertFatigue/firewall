# clustering.py
from __future__ import annotations

import warnings
from typing import Tuple

import hdbscan
import pandas as pd
import umap

import config

warnings.filterwarnings("ignore", category=UserWarning, module="umap")


def run_clustering(ml_ready_df: pd.DataFrame, human_readable_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs UMAP + HDBSCAN and attaches cluster_label + cluster_probability to
    the human-readable DataFrame.

    Fix: this function now handles small datasets safely instead of crashing when
    n_neighbors or min_cluster_size are larger than the data.
    """
    if len(ml_ready_df) == 0:
        raise ValueError("Cannot cluster an empty DataFrame.")

    clustered = human_readable_df.copy()

    if len(ml_ready_df) < 3:
        clustered["cluster_label"] = 0
        clustered["cluster_probability"] = 1.0
        return clustered

    n_neighbors = max(2, min(config.UMAP_NEIGHBORS, len(ml_ready_df) - 1))
    n_components = max(2, min(config.UMAP_COMPONENTS, len(ml_ready_df) - 1))

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=config.UMAP_MIN_DIST,
        random_state=42,
    )
    umap_embeddings = reducer.fit_transform(ml_ready_df)

    min_cluster_size = max(2, min(config.HDBSCAN_MIN_CLUSTER_SIZE, len(ml_ready_df)))
    min_samples = max(1, min(config.HDBSCAN_MIN_SAMPLES, min_cluster_size))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    cluster_probs = clusterer.probabilities_

    clustered["cluster_label"] = cluster_labels
    clustered["cluster_probability"] = cluster_probs
    return clustered


def extract_outliers_and_exemplars(human_readable_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - outliers_df: rows with HDBSCAN label -1, sampled for individual LLM review
      - exemplars_df: one representative row per normal cluster

    Important fix:
      The old code treated the whole -1 outlier pool as one exemplar. This version
      keeps outliers separate so anomalies are not all forced into one label.
    """
    if "cluster_label" not in human_readable_df.columns:
        raise ValueError("cluster_label column missing. Run run_clustering() first.")

    exemplars = []

    # Individual outliers.
    outliers_df = human_readable_df[human_readable_df["cluster_label"] == -1].copy()
    if len(outliers_df) > config.MAX_OUTLIERS_TO_LABEL:
        # Use lowest probability first; these are the most outlier-like rows.
        outliers_df = outliers_df.sort_values("cluster_probability", ascending=True).head(
            config.MAX_OUTLIERS_TO_LABEL
        )

    # One exemplar per non-outlier cluster.
    for cluster_id in sorted(human_readable_df["cluster_label"].unique()):
        if cluster_id == -1:
            continue
        cluster_rows = human_readable_df[human_readable_df["cluster_label"] == cluster_id]
        if cluster_rows.empty:
            continue
        best_idx = cluster_rows["cluster_probability"].idxmax()
        exemplars.append(cluster_rows.loc[best_idx])

    exemplars_df = pd.DataFrame(exemplars)
    if not exemplars_df.empty:
        exemplars_df.index = [row.name for row in exemplars]

    return outliers_df, exemplars_df
