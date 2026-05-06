import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this points to your newly merged output!
FILE_PATH = os.path.join(SCRIPT_DIR, 'suricata_logs/eve_labeled.json') 

# ==========================================
# 1. COLUMNS TO DROP
# ==========================================
COLS_TO_DROP = [
    # removing identifiers and routing
    'timestamp', 'flow_id', 'tx_id', 'community_id', 'in_iface', 'event_type',
    'src_ip', 'dest_ip', 'src_port', 'dest_port', 
    
    # preventing memorization of timeline by model
    'pcap_cnt', 'alert.signature', 'alert.signature_id',
    
    # zero variance & noise being dropped
    'pkt_src', 'alert.gid', 'alert.rev', 'flow.start', 'flow.end', 'icmp_type', 'icmp_code',
    
    # --- Raw High-Cardinality Strings ---
    # dropping raw text after calculating length
    'tls.sni', 'tls.subject', 'dns.rrname', 'http.hostname', 'http.http_user_agent', 'http.url'
]

# ==========================================
# 2. CATBOOST CATEGORICAL FEATURES
# ==========================================
# passing this list to catboost directly to natively deal with it

CAT_FEATURES = [
    'proto', 
    'app_proto', 
    'http.http_method', 
    'alert.severity',  # Can be treated as categorical or numeric, categorical is safer for discrete severity tiers
    'alert.category'
]

# ==========================================
# 3. CONTINUOUS NUMERIC FEATURES
# ==========================================
# tree algs scale invariant so no need to scale
NUMERIC_FEATURES = [
    'flow.pkts_toserver', 
    'flow.pkts_toclient', 
    'flow.bytes_toserver', 
    'flow.bytes_toclient', 
    'flow.age',
    
    # keeping length of high cardinality strings
    'tls.sni_length', 
    'tls.subject_length', 
    'dns.rrname_length', 
    'http.hostname_length', 
    'http.http_user_agent_length', 
    'http.url_length'
]