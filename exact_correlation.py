import pandas as pd

# Load both error types
fn_df = pd.read_csv('exact_false_negatives.csv')
fp_df = pd.read_csv('exact_false_positives.csv')

# Label them so you don't get confused in the final file
fn_df['Error_Type'] = 'False Negative (Missed Attack)'
fp_df['Error_Type'] = 'False Positive (False Alarm)'

# Combine them
all_errors = pd.concat([fn_df, fp_df])

# Load Ground Truth
gt_df = pd.read_csv('wednesdayGroundTruth.csv')
gt_df.columns = gt_df.columns.str.strip()

# Final Merge
final_analysis = all_errors.merge(gt_df, on='community_id', how='inner')

final_analysis.to_csv('Full_Model_Error_Analysis.csv', index=False)
print("Done! Check 'Full_Model_Error_Analysis.csv' for the complete breakdown.")