import pandas as pd
import json
import joblib
import math
from collections import Counter
from catboost import CatBoostClassifier

print("--- Starting Real-Time Inference Pipeline ---")

# ==========================================
# 1. LOAD THE TRAINED BRAIN AND SCALER
# ==========================================
print("Loading AI Model and Scaler...")
model = CatBoostClassifier()
model.load_model('suricata_threat_model.cbm')

scaler = joblib.load('robust_scaler.pkl')

# ==========================================
# 2. INGEST LIVE LOGS (The new 5-minute batch)
# ==========================================
file_path = 'live_eve.json' # Point this to your fresh logs
parsed_json_list = []
with open(file_path, 'r') as f:
    for line in f:
        if line.strip():
            parsed_json_list.append(json.loads(line))

live_df = pd.json_normalize(parsed_json_list)
live_df = live_df.groupby('flow_id').first().reset_index()

# Keep a human-readable copy to print the alerts later!
alert_context_df = live_df.copy()

# ==========================================
# 3. FEATURE EXTRACTION (Same as Script 1)
# ==========================================
def calculate_entropy(text):
    if pd.isna(text) or text == "": return 0
    text = str(text)
    probabilities = [n_x / len(text) for x, n_x in Counter(text).items()]
    return -sum(p * math.log2(p) for p in probabilities)

high_card_cols = ['tls.sni', 'tls.subject', 'dns.rrname', 'http.hostname', 'http.http_user_agent', 'http.url']
for col in high_card_cols:
    if col in live_df.columns:
        live_df[f'{col}_length'] = live_df[col].fillna('').astype(str).apply(len)
        live_df[f'{col}_entropy'] = live_df[col].apply(calculate_entropy)
        live_df.drop(columns=[col], inplace=True)

if 'dns.rcode' in live_df.columns:
    live_df['is_dns_failed'] = live_df['dns.rcode'].apply(
        lambda x: 1 if str(x) in ['REFUSED', 'NXDOMAIN', 'SERVFAIL'] else 0
    )

low_card_cols = ['proto', 'app_proto', 'flow.state', 'flow.reason', 'http.http_method', 'alert.severity']
cols_to_encode = [col for col in low_card_cols if col in live_df.columns]
live_df = pd.get_dummies(live_df, columns=cols_to_encode, dummy_na=True)

live_df.set_index('flow_id', inplace=True)
alert_context_df.set_index('flow_id', inplace=True) # Ensure indices match

cols_to_drop = ['timestamp', 'src_ip', 'dest_ip', 'in_iface', 'event_type', 'flow.start', 'flow.end', 'flow.alerted', 'icmp_type', 'icmp_code']
live_df.drop(columns=[col for col in cols_to_drop if col in live_df.columns], inplace=True)

# ==========================================
# 4. ALIGN TO MASTER FEATURES AND SCALE
# ==========================================
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

live_df = live_df.reindex(columns=master_columns, fill_value=0)
live_df = live_df.astype(float).fillna(0)

continuous_cols = [
    'flow.pkts_toserver', 'flow.pkts_toclient', 'flow.bytes_toserver', 'flow.bytes_toclient', 'flow.age',
    'tls.sni_length', 'tls.sni_entropy', 'tls.subject_length', 'tls.subject_entropy',
    'dns.rrname_length', 'dns.rrname_entropy', 'http.hostname_length', 'http.hostname_entropy',
    'http.http_user_agent_length', 'http.http_user_agent_entropy', 'http.url_length', 'http.url_entropy'
]

# CRITICAL: We use transform(), NOT fit_transform() here!
live_df[continuous_cols] = scaler.transform(live_df[continuous_cols])

# ==========================================
# 5. EXECUTE REAL-TIME PREDICTIONS
# ==========================================
print("\nScanning network flows...")
# CatBoost classifies the entire matrix in milliseconds
predictions = model.predict(live_df)

# Attach predictions back to the readable data so we know what IP is attacking
alert_context_df['AI_Threat_Diagnosis'] = predictions

# Filter out the boring stuff and only show actual threats
threats = alert_context_df[alert_context_df['AI_Threat_Diagnosis'] != "Normal Network Traffic"]

print(f"\nScan Complete! Evaluated {len(live_df)} sessions.")
print(f"Found {len(threats)} malicious or anomalous events.\n")

# Print the actionable intelligence for the SOC Analyst (You)
for flow_id, row in threats.iterrows():
    print(f"[{row['AI_Threat_Diagnosis']}] detected from {row.get('src_ip', 'Unknown')} to {row.get('dest_ip', 'Unknown')}")