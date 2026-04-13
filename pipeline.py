# CLEANED + IMPROVED VERSION (LESS TERMINAL SPAM + EXEMPLAR LABELING)

import pandas as pd
import numpy as np
import math
import json
import joblib
import os
import umap
import hdbscan
from collections import Counter
from sklearn.preprocessing import RobustScaler
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

# ==========================================
# STEP 0: LOAD DATA
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'eve.json')

parsed_json_list = []
with open(file_path, 'r') as f:
    for line in f:
        if line.strip():
            parsed_json_list.append(json.loads(line))

ml_ready_df = pd.json_normalize(parsed_json_list)
ml_ready_df = ml_ready_df.groupby('flow_id').first()
human_readable_df = ml_ready_df.copy()

# ==========================================
# ENTROPY FUNCTION
# ==========================================
def calculate_entropy(text):
    if pd.isna(text) or text == "":
        return 0
    text = str(text)
    probabilities = [n_x / len(text) for x, n_x in Counter(text).items()]
    return -sum(p * math.log2(p) for p in probabilities)

# ==========================================
# FEATURE EXTRACTION
# ==========================================
high_card_cols = ['tls.sni', 'tls.subject', 'dns.rrname', 'http.hostname', 'http.http_user_agent', 'http.url']

for col in high_card_cols:
    if col in ml_ready_df.columns:
        ml_ready_df[f'{col}_length'] = ml_ready_df[col].fillna('').astype(str).apply(len)
        ml_ready_df[f'{col}_entropy'] = ml_ready_df[col].apply(calculate_entropy)
        ml_ready_df.drop(columns=[col], inplace=True)

if 'dns.rcode' in ml_ready_df.columns:
    ml_ready_df['is_dns_failed'] = ml_ready_df['dns.rcode'].apply(
        lambda x: 1 if str(x) in ['REFUSED', 'NXDOMAIN', 'SERVFAIL'] else 0
    )

# ==========================================
# ENCODING
# ==========================================
low_card_cols = ['proto', 'app_proto', 'flow.state', 'flow.reason', 'http.http_method', 'alert.severity']
cols_to_encode = [col for col in low_card_cols if col in ml_ready_df.columns]
ml_ready_df = pd.get_dummies(ml_ready_df, columns=cols_to_encode, dummy_na=True)

# ==========================================
# CLEAN + ALIGN
# ==========================================

cols_to_drop = ['timestamp', 'src_ip', 'dest_ip', 'in_iface', 'event_type', 'flow.start', 'flow.end', 'flow.alerted', 'icmp_type', 'icmp_code']
ml_ready_df.drop(columns=[col for col in cols_to_drop if col in ml_ready_df.columns], inplace=True)

master_columns = [
    'flow.pkts_toserver', 'flow.pkts_toclient', 'flow.bytes_toserver', 'flow.bytes_toclient', 'flow.age',
    'tls.sni_length', 'tls.sni_entropy', 'tls.subject_length', 'tls.subject_entropy',
    'dns.rrname_length', 'dns.rrname_entropy', 'http.hostname_length', 'http.hostname_entropy',
    'http.http_user_agent_length', 'http.http_user_agent_entropy', 'http.url_length', 'http.url_entropy',
    'is_dns_failed', 'proto_TCP', 'proto_UDP', 'proto_ICMP', 'proto_IPv6-ICMP', 'proto_nan',
    'app_proto_http', 'app_proto_tls', 'app_proto_dns', 'app_proto_smb', 'app_proto_ftp', 'app_proto_smtp', 'app_proto_failed', 'app_proto_nan',
    'flow.state_new', 'flow.state_established', 'flow.state_closed', 'flow.state_nan',
    'flow.reason_timeout', 'flow.reason_shutdown', 'flow.reason_nan',
    'alert.severity_1.0', 'alert.severity_2.0', 'alert.severity_3.0', 'alert.severity_nan'
]

ml_ready_df = ml_ready_df.reindex(columns=master_columns, fill_value=0)
ml_ready_df = ml_ready_df.astype(float).fillna(0)

# ==========================================
# SCALING
# ==========================================
continuous_cols = [
    'flow.pkts_toserver', 'flow.pkts_toclient', 'flow.bytes_toserver', 'flow.bytes_toclient', 'flow.age',
    'tls.sni_length', 'tls.sni_entropy', 'tls.subject_length', 'tls.subject_entropy',
    'dns.rrname_length', 'dns.rrname_entropy', 'http.hostname_length', 'http.hostname_entropy',
    'http.http_user_agent_length', 'http.http_user_agent_entropy', 'http.url_length', 'http.url_entropy'
]

scaler = RobustScaler()
ml_ready_df[continuous_cols] = scaler.fit_transform(ml_ready_df[continuous_cols])

# ==========================================
# UMAP + HDBSCAN
# ==========================================
reducer = umap.UMAP(n_components=5, random_state=42)
umap_embeddings = reducer.fit_transform(ml_ready_df)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(umap_embeddings)
cluster_probs = clusterer.probabilities_

human_readable_df['cluster_label'] = cluster_labels
human_readable_df['cluster_probability'] = cluster_probs

# ==========================================
# OUTLIERS + EXEMPLARS
# ==========================================
outliers_df = human_readable_df[human_readable_df['cluster_label'] == -1]

exemplars = []
exemplar_flow_ids = []

valid_clusters = [lbl for lbl in human_readable_df['cluster_label'].unique() if lbl != -1]

for cluster_id in valid_clusters:
    cluster_rows = human_readable_df[human_readable_df['cluster_label'] == cluster_id]
    
    # best_idx is the index of the row. Since our index is flow_id, best_idx IS the flow_id!
    best_idx = cluster_rows['cluster_probability'].idxmax()
    
    exemplar_row = cluster_rows.loc[best_idx]
    exemplars.append(exemplar_row)
    
    # THE FIX: Just append best_idx directly instead of trying to look up the column!
    exemplar_flow_ids.append(best_idx)
    
exemplars_df = pd.DataFrame(exemplars)

# ==========================================
# TEXT CONVERSION
# ==========================================
def row_to_llm_context(row):
    proto = row.get('proto', 'unknown protocol')
    dest_port = row.get('dest_port', 'unknown port')
    pkts = row.get('flow.pkts_toserver', 0)
    bytes_sent = row.get('flow.bytes_toserver', 0)
    app_proto = row.get('app_proto', 'no application layer') # Grab the app protocol!

    return f"This is a {proto} connection over {app_proto} on port {dest_port} sending {pkts} packets and {bytes_sent} bytes."

outlier_contexts = [row_to_llm_context(row) for _, row in outliers_df.iterrows()]
exemplar_contexts = [row_to_llm_context(row) for _, row in exemplars_df.iterrows()]

# ==========================================
# LOAD LLM (NO SPAM)
# ==========================================
model_path = "knowledgator/gliclass-edge-v3.0"
model = GLiClassModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipeline = ZeroShotClassificationPipeline(model, tokenizer, device='cpu')

threat_labels = [
    "Routine encrypted web browsing and standard HTTPS traffic",
    "Benign background network noise and routine ICMP pings",
    "Suspicious command and control beaconing or malware activity",
    "Aggressive brute force attack or password guessing",
    "Database SQL injection attack",
    "Malformed protocol anomaly or unknown suspicious behavior"
]

# ==========================================
# LABEL FUNCTION (QUIET)
# ==========================================
def get_labels(contexts):
    results_dict = {}
    for i, text in enumerate(contexts):
        res = pipeline(text, threat_labels, threshold=0.3)[0]
        if not res:
            results_dict[i] = ("Protocol Anomaly or Suspicious Behavior", 0)
        else:
            results_dict[i] = (res[0]['label'], res[0]['score'])
    return results_dict

# ==========================================
# BUILD TRAINING DATA (THE NUCLEAR FIX)
# ==========================================
training_df = ml_ready_df.copy()
training_df['target_label'] = "Normal Network Traffic"

# 1. THE ARMOR: Force the master index into a pure text string.
# .replace('.0', '') strips out any decimals Pandas secretly added.
training_df.index = training_df.index.astype(str).str.replace('.0', '', regex=False)

# 2. Map the Outliers
outlier_labels = get_labels(outlier_contexts)
for i, flow_id in enumerate(outliers_df.index):
    # Armor the lookup ID too
    safe_id = str(flow_id).replace('.0', '') 
    label, score = outlier_labels[i]
    
    if safe_id in training_df.index:
        training_df.at[safe_id, 'target_label'] = label

# 3. Map the Exemplars
exemplar_labels = get_labels(exemplar_contexts)
for i, flow_id in enumerate(exemplars_df.index):
    safe_id = str(flow_id).replace('.0', '') 
    label, score = exemplar_labels[i]
    
    if score > 0.3 and "Routine encrypted web browsing" not in label:
        if safe_id in training_df.index:
            training_df.at[safe_id, 'target_label'] = label

# ==========================================
# SAVE
# ==========================================
training_df.to_csv('catboost_training_data.csv')
joblib.dump(scaler, 'robust_scaler.pkl')

print("\n--- Final Label Distribution ---")
print(training_df['target_label'].value_counts())
print("\nDone. Dataset ready.")