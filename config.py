# config.py
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, 'filtered_eve.json')

HIGH_CARD_COLS = ['tls.sni', 'tls.subject', 'dns.rrname', 'http.hostname', 'http.http_user_agent', 'http.url']
LOW_CARD_COLS = ['proto', 'app_proto', 'flow.state', 'flow.reason', 'http.http_method', 'alert.severity']
COLS_TO_DROP = ['timestamp', 'src_ip', 'dest_ip', 'in_iface', 'event_type', 'flow.start', 'flow.end', 'flow.alerted', 'icmp_type', 'icmp_code']

MASTER_COLUMNS = [
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

CONTINUOUS_COLS = [
    'flow.pkts_toserver', 'flow.pkts_toclient', 'flow.bytes_toserver', 'flow.bytes_toclient', 'flow.age',
    'tls.sni_length', 'tls.sni_entropy', 'tls.subject_length', 'tls.subject_entropy',
    'dns.rrname_length', 'dns.rrname_entropy', 'http.hostname_length', 'http.hostname_entropy',
    'http.http_user_agent_length', 'http.http_user_agent_entropy', 'http.url_length', 'http.url_entropy'
]

THREAT_LABELS = [
    "Standard DNS Resolution and Naming Services",
    "Routine Unencrypted Web Traffic (HTTP)",
    "Routine Encrypted Web Traffic (HTTPS/TLS)",
    "Benign Background Network Noise (ICMP, ARP, DHCP)",
    "Standard Internal IT and Directory Services (LDAP, SMB, Active Directory)",
    "Suspicious Command and Control (C2) Beaconing",
    "Known Vulnerability Exploitation or Active Attack",
    "Malformed protocol anomaly or unknown suspicious behavior"
]