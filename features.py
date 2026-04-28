# features.py
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Tuple

import pandas as pd
from sklearn.preprocessing import RobustScaler

import config


def calculate_entropy(text) -> float:
    if pd.isna(text) or text == "":
        return 0.0
    text = str(text)
    if not text:
        return 0.0
    probabilities = [n_x / len(text) for n_x in Counter(text).values()]
    return float(-sum(p * math.log2(p) for p in probabilities if p > 0))


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    return df


def process_features(
    ml_ready_df: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Converts flattened Suricata events into a stable numerical feature matrix.

    Important:
      - The returned DataFrame keeps the original index, so cluster labels can be
        joined back to human-readable rows safely.
      - MASTER_COLUMNS guarantees identical columns across different datasets.
      - Pass fit_scaler=False with an existing scaler when transforming new/unseen data.
    """
    df = ml_ready_df.copy()

    # Feature extraction for high-cardinality strings.
    for col in config.HIGH_CARD_COLS:
        if col in df.columns:
            df[f"{col}_length"] = df[col].fillna("").astype(str).apply(len)
            df[f"{col}_entropy"] = df[col].apply(calculate_entropy)
            df.drop(columns=[col], inplace=True)
        else:
            df[f"{col}_length"] = 0
            df[f"{col}_entropy"] = 0.0

    # DNS failure indicator.
    if "dns.rcode" in df.columns:
        df["is_dns_failed"] = df["dns.rcode"].apply(
            lambda x: 1 if str(x).upper() in {"REFUSED", "NXDOMAIN", "SERVFAIL"} else 0
        )
    else:
        df["is_dns_failed"] = 0

    # Low-cardinality one-hot encoding.
    cols_to_encode = [col for col in config.LOW_CARD_COLS if col in df.columns]
    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, dummy_na=True)

    # Remove raw identifiers/text not used directly by the model.
    drop_cols = [col for col in config.COLS_TO_DROP if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Keep only the stable thesis feature schema.
    df = df.reindex(columns=config.MASTER_COLUMNS, fill_value=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    # Robust scaling for continuous fields.
    _ensure_columns(df, config.CONTINUOUS_COLS)
    if scaler is None:
        scaler = RobustScaler()
        fit_scaler = True

    if fit_scaler:
        df[config.CONTINUOUS_COLS] = scaler.fit_transform(df[config.CONTINUOUS_COLS])
    else:
        df[config.CONTINUOUS_COLS] = scaler.transform(df[config.CONTINUOUS_COLS])

    return df, scaler
