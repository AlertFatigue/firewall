import json
import csv
import config

INPUT_FILE = config.FILE_PATH
OUTPUT_FILE = 'suricata_features_extracted.csv'

# combine featuers to get csv columns
FIELDNAMES = config.CAT_FEATURES + config.NUMERIC_FEATURES + ['Label']

print("Starting memory-efficient extraction from JSON to CSV...")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}\n")

processed_count = 0
written_count = 0

# open both files, read one line at a time
with open(INPUT_FILE, 'r') as infile, open(OUTPUT_FILE, 'w', newline='') as outfile:
    
    # csv writer with features as columns
    writer = csv.DictWriter(outfile, fieldnames=FIELDNAMES)
    writer.writeheader()
    
    for line in infile:
        if not line.strip():
            continue
            
        processed_count += 1
        event = json.loads(line)
        label = event.get('label', 'UNLABELED')
        
        # skip unlabeled alerts
        if label == 'UNLABELED':
            continue
            
        # initialize row dict
        row = {'Label': label}
        
        # CATEGORICAL FEATURES
        row['proto'] = event.get('proto', 'Missing')
        row['app_proto'] = event.get('app_proto', 'Missing')
        
        alert = event.get('alert', {})
        row['alert.severity'] = str(alert.get('severity', 'Missing'))
        row['alert.category'] = alert.get('category', 'Missing')
        
        http = event.get('http', {})
        row['http.http_method'] = http.get('http_method', 'Missing')
        
        # NUMERIC FEATURES
        flow = event.get('flow', {})
        row['flow.pkts_toserver'] = flow.get('pkts_toserver', 0)
        row['flow.pkts_toclient'] = flow.get('pkts_toclient', 0)
        row['flow.bytes_toserver'] = flow.get('bytes_toserver', 0)
        row['flow.bytes_toclient'] = flow.get('bytes_toclient', 0)
        row['flow.age'] = flow.get('age', 0)
        
        # feature engineering
        tls = event.get('tls', {})
        dns = event.get('dns', {})
        
        row['tls.sni_length'] = len(tls.get('sni', '')) if tls.get('sni') else 0
        row['tls.subject_length'] = len(tls.get('subject', '')) if tls.get('subject') else 0
        row['dns.rrname_length'] = len(dns.get('rrname', '')) if dns.get('rrname') else 0
        row['http.hostname_length'] = len(http.get('hostname', '')) if http.get('hostname') else 0
        row['http.http_user_agent_length'] = len(http.get('http_user_agent', '')) if http.get('http_user_agent') else 0
        row['http.url_length'] = len(http.get('url', '')) if http.get('url') else 0
        
        # write to row to free up memory
        writer.writerow(row)
        written_count += 1
        
        # print progress every 100000 alerts so we know not frozen
        if processed_count % 100000 == 0:
            print(f"Processed {processed_count:,} lines... (Written: {written_count:,})")

print("\n--- Extraction Complete! ---")
print(f"Total lines processed: {processed_count:,}")
print(f"Total rows written to CSV: {written_count:,}")
print(f"Ready at: {OUTPUT_FILE}")