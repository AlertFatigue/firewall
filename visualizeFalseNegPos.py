import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("1. Loading error analysis data...")
try:
    df = pd.read_csv('Full_Model_Error_Analysis.csv')
except FileNotFoundError:
    print("Error: Could not find 'Full_Model_Error_Analysis.csv'.")
    exit()

# Set a professional academic style for the charts
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Split datasets for easier plotting
fn_df = df[df['Error_Type'] == 'False Negative']
fp_df = df[df['Error_Type'] == 'False Positive']

print("2. Generating figures...")

# ==========================================
# FIGURE 1: MISSED ATTACK TAXONOMY
# ==========================================
plt.figure(figsize=(10, 6))
# FIX: Added hue='Label' and legend=False to support the palette
sns.countplot(
    data=fn_df, 
    y='Label', 
    hue='Label',
    order=fn_df['Label'].value_counts().index, 
    palette='Reds_r',
    legend=False
)
plt.title('False Negative Attack Breakdown', fontsize=14, fontweight='bold')
plt.xlabel('Flow Count', fontsize=12)
plt.ylabel('Ground Truth Attack Type', fontsize=12)
plt.tight_layout()
plt.savefig('Thesis_Fig1_Missed_Attacks.png', dpi=300)
plt.close()

# ==========================================
# FIGURE 2: MODEL CONFIDENCE DISTRIBUTION
# ==========================================
plt.figure(figsize=(10, 6))
# histplot handles palettes safely via the hue argument naturally
sns.histplot(
    data=df, 
    x='Malicious_Probability', 
    hue='Error_Type', 
    bins=50, 
    kde=True, 
    palette={'False Negative': '#d62728', 'False Positive': '#ff7f0e'},
    alpha=0.6
)
# Draw the 50% threshold line
plt.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
plt.title('Probability Distribution of Model Errors', fontsize=14, fontweight='bold')
plt.xlabel('Model Output Probability', fontsize=12)
plt.ylabel('Flow Count', fontsize=12)
plt.xlim(0, 1)
plt.legend(title='Type of Error')
plt.tight_layout()
plt.savefig('Thesis_Fig2_Confidence_Distribution.png', dpi=300)
plt.close()



print("\n--- Success! ---")
