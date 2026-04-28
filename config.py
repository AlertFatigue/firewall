# config.py
"""
Central configuration for the thesis alert-classification pipeline.

You can override most values with environment variables, for example:
    DATASET_NAME=heartbleed FILE_PATH=./filtered_eve.json python main.py
"""

from __future__ import annotations

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Input / output
# -------------------------
DATASET_NAME = os.environ.get("DATASET_NAME", "heartbleed").strip() or "heartbleed"
FILE_PATH = os.environ.get("FILE_PATH", os.path.join(SCRIPT_DIR, "filtered_eve.json"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(SCRIPT_DIR, "outputs"))

# File names produced by the pipeline
LABELED_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_labeled.csv")
TRAIN_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_train.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_test.csv")
CLUSTERED_HUMAN_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_clustered_human_readable.csv")
SCALER_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_robust_scaler.pkl")
MODEL_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_catboost_model.cbm")
METRICS_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_metrics.json")
FEATURE_IMPORTANCE_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_feature_importance.csv")
VALIDATION_WITH_IPS_FILE = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_validation_with_ips.csv")

# -------------------------
# Feature engineering
# -------------------------
HIGH_CARD_COLS = [
    "tls.sni",
    "tls.subject",
    "dns.rrname",
    "http.hostname",
    "http.http_user_agent",
    "http.url",
]

LOW_CARD_COLS = [
    "proto",
    "app_proto",
    "flow.state",
    "flow.reason",
    "http.http_method",
    "alert.severity",
]

# Columns kept for human-readable validation but removed from ML matrix.
COLS_TO_DROP = [
    "timestamp",
    "src_ip",
    "dest_ip",
    "src_port",
    "dest_port",
    "in_iface",
    "event_type",
    "flow.start",
    "flow.end",
    "flow.alerted",
    "icmp_type",
    "icmp_code",
]

MASTER_COLUMNS = [
    "flow.pkts_toserver",
    "flow.pkts_toclient",
    "flow.bytes_toserver",
    "flow.bytes_toclient",
    "flow.age",
    "tls.sni_length",
    "tls.sni_entropy",
    "tls.subject_length",
    "tls.subject_entropy",
    "dns.rrname_length",
    "dns.rrname_entropy",
    "http.hostname_length",
    "http.hostname_entropy",
    "http.http_user_agent_length",
    "http.http_user_agent_entropy",
    "http.url_length",
    "http.url_entropy",
    "is_dns_failed",
    "proto_TCP",
    "proto_UDP",
    "proto_ICMP",
    "proto_IPv6-ICMP",
    "proto_nan",
    "app_proto_http",
    "app_proto_tls",
    "app_proto_dns",
    "app_proto_smb",
    "app_proto_ftp",
    "app_proto_smtp",
    "app_proto_failed",
    "app_proto_nan",
    "flow.state_new",
    "flow.state_established",
    "flow.state_closed",
    "flow.state_nan",
    "flow.reason_timeout",
    "flow.reason_shutdown",
    "flow.reason_nan",
    "alert.severity_1.0",
    "alert.severity_2.0",
    "alert.severity_3.0",
    "alert.severity_nan",
]

CONTINUOUS_COLS = [
    "flow.pkts_toserver",
    "flow.pkts_toclient",
    "flow.bytes_toserver",
    "flow.bytes_toclient",
    "flow.age",
    "tls.sni_length",
    "tls.sni_entropy",
    "tls.subject_length",
    "tls.subject_entropy",
    "dns.rrname_length",
    "dns.rrname_entropy",
    "http.hostname_length",
    "http.hostname_entropy",
    "http.http_user_agent_length",
    "http.http_user_agent_entropy",
    "http.url_length",
    "http.url_entropy",
]

# -------------------------
# Clustering
# -------------------------
UMAP_COMPONENTS = int(os.environ.get("UMAP_COMPONENTS", "5"))
UMAP_NEIGHBORS = int(os.environ.get("UMAP_NEIGHBORS", "50"))
UMAP_MIN_DIST = float(os.environ.get("UMAP_MIN_DIST", "0.0"))
HDBSCAN_MIN_CLUSTER_SIZE = int(os.environ.get("HDBSCAN_MIN_CLUSTER_SIZE", "50"))
HDBSCAN_MIN_SAMPLES = int(os.environ.get("HDBSCAN_MIN_SAMPLES", "10"))
MAX_OUTLIERS_TO_LABEL = int(os.environ.get("MAX_OUTLIERS_TO_LABEL", "100"))

# -------------------------
# LLM labeling
# -------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
LLM_SLEEP_SECONDS = float(os.environ.get("LLM_SLEEP_SECONDS", "1.0"))

THREAT_LABELS = [
    "Standard DNS Resolution and Naming Services",
    "Routine Unencrypted Web Traffic (HTTP)",
    "Routine Encrypted Web Traffic (HTTPS/TLS)",
    "Benign Background Network Noise (ICMP, ARP, DHCP)",
    "Standard Internal IT and Directory Services (LDAP, SMB, Active Directory)",
    "Suspicious Command and Control (C2) Beaconing",
    "Known Vulnerability Exploitation or Active Attack",
    "Malformed protocol anomaly or unknown suspicious behavior",
    "Unclassified / Background Noise",
]

SUSPICIOUS_LABEL_KEYWORDS = [
    "suspicious",
    "vulnerability",
    "active attack",
    "malformed",
    "unknown suspicious",
    "c2",
]
