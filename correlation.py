import pandas as pd
import json
import communityid

# Initialize the Community ID generator
cid_gen = communityid.CommunityID()

print("1. Loading and cleaning ground truth CSV...")
df = pd.read_csv('wednesdayGroundTruth.csv')

# Strip pesky whitespaces from CIC-IDS-2017 headers
df.columns = df.columns.str.strip()

print("2. Generating Community IDs for CSV flows...")
def calculate_cid(row):
    try:
        src_ip = str(row['Source IP'])
        dst_ip = str(row['Destination IP'])
        src_port = int(row['Source Port'])
        dst_port = int(row['Destination Port'])
        # CIC-IDS-2017 uses integer protocols (6 = TCP, 17 = UDP)
        proto = int(row['Protocol']) 
        
        # Create the tuple and hash it
        tpl = communityid.FlowTuple(proto, src_ip, dst_ip, src_port, dst_port)
        return cid_gen.calc(tpl)
    except Exception as e:
        return None

df['community_id'] = df.apply(calculate_cid, axis=1)

print("3. Building correlation mapping...")
# Drop duplicates in case of flow fragmentation, keeping the first occurrence 
mapping_df = df.dropna(subset=['community_id']).drop_duplicates(subset=['community_id'])
label_map = mapping_df.set_index('community_id')['Label'].to_dict()

print("4. Correlating and writing labeled eve.json...")
labeled_count = 0
unlabeled_count = 0

with open('./suricata_logs/eve.json', 'r') as infile, open('eve_labeled.json', 'w') as outfile:
    for line in infile:
        if not line.strip():
            continue
            
        event = json.loads(line)
        cid = event.get('community_id')
        
        # Correlate via Community ID
        if cid and cid in label_map:
            event['label'] = label_map[cid]
            labeled_count += 1
        else:
            event['label'] = 'UNLABELED'
            unlabeled_count += 1
            
        # Write the enriched JSON object back out
        outfile.write(json.dumps(event) + '\n')

print(f"Done! Labeled: {labeled_count} | Unlabeled: {unlabeled_count}")