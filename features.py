# features.py
import pandas as pd
import math
from collections import Counter
from sklearn.preprocessing import RobustScaler

import config

def calculate_entropy(text):
    if pd.isna(text) or text == "":
        return 0
    text = str(text)
    probabilities = [n_x / len(text) for x, n_x in Counter(text).items()]
    return -sum(p * math.log2(p) for p in probabilities)

def process_features(ml_ready_df):
    # Feature Extraction
    for col in config.HIGH_CARD_COLS:
        if col in ml_ready_df.columns:
            ml_ready_df[f'{col}_length'] = ml_ready_df[col].fillna('').astype(str).apply(len)
            ml_ready_df[f'{col}_entropy'] = ml_ready_df[col].apply(calculate_entropy)
            ml_ready_df.drop(columns=[col], inplace=True)

    if 'dns.rcode' in ml_ready_df.columns:
        ml_ready_df['is_dns_failed'] = ml_ready_df['dns.rcode'].apply(
            lambda x: 1 if str(x) in ['REFUSED', 'NXDOMAIN', 'SERVFAIL'] else 0
        )

    # Encoding
    cols_to_encode = [col for col in config.LOW_CARD_COLS if col in ml_ready_df.columns]
    ml_ready_df = pd.get_dummies(ml_ready_df, columns=cols_to_encode, dummy_na=True)

    # Clean + Align
    ml_ready_df.drop(columns=[col for col in config.COLS_TO_DROP if col in ml_ready_df.columns], inplace=True)
    ml_ready_df = ml_ready_df.reindex(columns=config.MASTER_COLUMNS, fill_value=0)
    ml_ready_df = ml_ready_df.astype(float).fillna(0)

    # Scaling
    scaler = RobustScaler()
    ml_ready_df[config.CONTINUOUS_COLS] = scaler.fit_transform(ml_ready_df[config.CONTINUOUS_COLS])
    
    return ml_ready_df, scaler