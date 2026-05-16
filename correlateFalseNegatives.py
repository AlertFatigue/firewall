import pandas as pd

# 1. Load the 736 False Negatives we found
print("Loading False Negative Row IDs...")
fn_df = pd.read_csv('false_negatives_analysis.csv')

# Convert the row IDs into a fast-lookup Python Set
fn_indexes = set(fn_df['Original_Row_ID'])
print(f"Looking for {len(fn_indexes)} specific flows...")

# 2. Point this to your MASSIVE raw file that still has IPs/Timestamps
# (Change this to whatever your big file is actually named)
BIG_FILE_PATH = 'wednesdayGroundTruth.csv' 

# 3. Read the massive file in chunks to save RAM
chunk_size = 100_000 # Read 100,000 rows at a time
matched_rows = []

print(f"Scanning massive dataset: {BIG_FILE_PATH} in chunks...")

try:
    # pandas chunking automatically tracks the global row index perfectly!
    for chunk in pd.read_csv(BIG_FILE_PATH, chunksize=chunk_size, low_memory=False):
        
        # Check if any row in this chunk has an index that matches our FN list
        matching = chunk[chunk.index.isin(fn_indexes)]
        
        if not matching.empty:
            matched_rows.append(matching)
            print(f"Found {len(matching)} matches in current chunk...")

    # 4. Combine all the found needles in the haystack
    if matched_rows:
        final_fn_raw_data = pd.concat(matched_rows)
        
        # Merge the model's confidence scores from the FN analysis file
        final_merged = final_fn_raw_data.merge(
            fn_df[['Original_Row_ID', 'Malicious_Probability']], 
            left_index=True, 
            right_on='Original_Row_ID'
        )

        output_file = 'RAW_false_negatives_with_IPs.csv'
        final_merged.to_csv(output_file, index=False)
        print(f"\nSuccess! Saved complete raw data for the missed attacks to '{output_file}'")
    else:
        print("\nCould not find the rows. Ensure the BIG_FILE_PATH points to the exact file used to generate your features.")

except Exception as e:
    print(f"An error occurred: {e}")