import json
import pandas as pd

# ==========================================
# 1. LOAD THE FINGERPRINTS
# ==========================================
print("1. Loading the 736 missed attacks...")
fn_df = pd.read_csv('false_negatives_analysis.csv')

# Ensure the features are integers so they match the JSON exactly
fn_df['flow.pkts_toserver'] = fn_df['flow.pkts_toserver'].fillna(0).astype(int)
fn_df['flow.bytes_toserver'] = fn_df['flow.bytes_toserver'].fillna(0).astype(int)
fn_df['flow.pkts_toclient'] = fn_df['flow.pkts_toclient'].fillna(0).astype(int)
fn_df['flow.bytes_toclient'] = fn_df['flow.bytes_toclient'].fillna(0).astype(int)

# Create a fast-lookup dictionary of fingerprints
target_fingerprints = {}
for index, row in fn_df.iterrows():
    # The unique fingerprint tuple
    fingerprint = (
        row['flow.pkts_toserver'],
        row['flow.bytes_toserver'],
        row['flow.pkts_toclient'],
        row['flow.bytes_toclient']
    )
    # Save the probability score to match it up later
    target_fingerprints[fingerprint] = row['Malicious_Probability']

print(f"Created {len(target_fingerprints)} unique flow fingerprints to hunt for.")

# ==========================================
# 2. STREAM THE JSON LINE-BY-LINE (LOW RAM)
# ==========================================
# UPDATE THIS PATH if your json is somewhere else!
JSON_PATH = 'suricata_logs/eve_labeled.json' 
OUTPUT_CSV = 'Recovered_False_Negatives.csv'

recovered_flows = []
lines_scanned = 0

print(f"\n2. Scanning {JSON_PATH} line-by-line...")
print("This uses almost no RAM. Please wait...")

try:
    with open(JSON_PATH, 'r') as f:
        for line in f:
            lines_scanned += 1
            if not line.strip(): continue
            
            try:
                event = json.loads(line)
                
                # Only check events that contain flow statistics
                if 'flow' in event:
                    f_stats = event['flow']
                    
                    # Generate the fingerprint for the current JSON line
                    current_fingerprint = (
                        f_stats.get('pkts_toserver', 0),
                        f_stats.get('bytes_toserver', 0),
                        f_stats.get('pkts_toclient', 0),
                        f_stats.get('bytes_toclient', 0)
                    )
                    
                    # WE FOUND A MATCH!
                    if current_fingerprint in target_fingerprints:
                        recovered = {
                            'timestamp': event.get('timestamp'),
                            'src_ip': event.get('src_ip'),
                            'dest_ip': event.get('dest_ip'),
                            'src_port': event.get('src_port'),
                            'dest_port': event.get('dest_port'),
                            'app_proto': event.get('app_proto', 'unknown'),
                            'actual_label': event.get('label'), # The ground truth label
                            'pkts_to_server': current_fingerprint[0],
                            'bytes_to_server': current_fingerprint[1],
                            'Model_Confidence_Score': target_fingerprints[current_fingerprint]
                        }
                        recovered_flows.append(recovered)
                        
                        # Print an update every 50 finds so you know it isn't frozen
                        if len(recovered_flows) % 50 == 0:
                            print(f" -> Found {len(recovered_flows)} missed attacks so far...")
                            
            except json.JSONDecodeError:
                pass # Skip broken lines if any exist
                
except FileNotFoundError:
    print(f"ERROR: Could not find {JSON_PATH}. Please check the path.")
    exit()

# ==========================================
# 3. SAVE RESULTS
# ==========================================
print(f"\nFinished scanning {lines_scanned:,} lines.")
print(f"Successfully recovered IP data for {len(recovered_flows)} attack profiles!")

if recovered_flows:
    output_df = pd.DataFrame(recovered_flows)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to '{OUTPUT_CSV}'. You can now open this to see exactly who the model missed.")
else:
    print("No matches found. Ensure your JSON file is the exact one used to generate your features.")