import pandas as pd
import communityid

print("1. Loading exact false errors...")
# Load both error types
fn_df = pd.read_csv('exact_false_negatives.csv')
fp_df = pd.read_csv('exact_false_positives.csv')

# Label them so you don't get confused in the final file
fn_df['Error_Type'] = 'False Negative (Missed Attack)'
fp_df['Error_Type'] = 'False Positive (False Alarm)'

# Combine them into one dataframe
all_errors = pd.concat([fn_df, fp_df])
print(f"Total model errors to map: {len(all_errors)}")

print("2. Loading Wednesday Ground Truth...")
gt_df = pd.read_csv('wednesdayGroundTruth.csv')

# Strip whitespaces from the ground truth columns
gt_df.columns = gt_df.columns.str.strip()

print("3. Generating Community IDs for Ground Truth (This may take a minute)...")
# Initialize the Community ID generator
cid_gen = communityid.CommunityID()

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

# Apply the generator to recreate the IDs
gt_df['community_id'] = gt_df.apply(calculate_cid, axis=1)

print("4. Performing 1:1 mapping...")
# Drop duplicates in case of flow fragmentation, keeping the first occurrence 
gt_df_unique = gt_df.dropna(subset=['community_id']).drop_duplicates(subset=['community_id'])

# Inner merge perfectly aligns the two datasets based on the ID
final_analysis = all_errors.merge(gt_df_unique, on='community_id', how='inner')

output_file = 'Full_Model_Error_Analysis.csv'
final_analysis.to_csv(output_file, index=False)

print(f"\nSuccess! Mapped exactly {len(final_analysis)} errors directly back to their original Ground Truth rows.")
print(f"File saved as: {output_file}")