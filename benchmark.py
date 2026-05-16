import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# THESIS BENCHMARK VISUALS (BINARY MODELS)
# ==========================================

# Set professional style
plt.style.use('seaborn-v0_8-muted')
sns.set_context("paper", font_scale=1.4)

# Data extracted strictly from Binary Classification Related Work
data = {
    'System': ['Your CatBoost', 'D-MAGIC', 'MSF-IDS', 'ZAD-ML', 'Hybrid IDS'],
    
    # Latency: Hybrid = 2.3s (2300ms), ZAD-ML ~20s batch (estimated 20ms for chart scaling)
    'Latency_ms': [0.000887, 0.07, 0.11, 20.0, 2300.0], 
    
    # F1-Scores / Accuracy metrics
    'F1_Score': [0.9891, 0.7966, 0.9055, 0.9420, 0.9520], 
    
    # False Positive Rates
    'FPR_Percent': [1.06, 4.03, 6.71, 2.90, 4.10], 
    
    # EPS strictly from text. NaN drops the others from the Throughput chart.
    'Throughput_FPS': [1127961, 14272, 910, np.nan, np.nan] 
}

df = pd.DataFrame(data)

def plot_latency_comparison():
    print("Generating Latency Chart...")
    plt.figure(figsize=(10, 6))
    
    # Sort for visual hierarchy
    plot_df = df.sort_values('Latency_ms')
    ax = sns.barplot(x='System', y='Latency_ms', data=plot_df, palette='Reds_r')
    
    # CRITICAL: Log scale so CatBoost (0.0008ms) and Hybrid (2300ms) can fit on the same screen
    ax.set_yscale('log') 
    
    plt.title('Inference Latency', fontsize=16, pad=15)
    plt.ylabel('Latency in ms (Log Scale)')
    plt.xticks(rotation=45)
    
    # Add exact numbers above bars for clarity
    for p in ax.patches:
        val = p.get_height()
        label = f"{val:.4f}" if val < 1 else f"{val:,.1f}"
        ax.annotate(label, (p.get_x() + p.get_width() / 2., val),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('comparison_latency.png', dpi=300)

def plot_f1_vs_fpr():
    print("Generating F1 vs FPR Scatter Plot...")
    plt.figure(figsize=(10, 7))
    
    # The "Holy Grail" is Top-Left (High F1, Low FPR)
    sns.scatterplot(data=df, x='FPR_Percent', y='F1_Score', hue='System', 
                    s=300, palette='viridis', edgecolor='black')
    
    # Annotate points
    for i in range(df.shape[0]):
        # Slight offsets so text doesn't overlap the dots
        plt.text(df.FPR_Percent[i] + 0.15, df.F1_Score[i], df.System[i], 
                 fontsize=12, fontweight='bold')

    plt.title('Detection Performance: F1-Score vs. FPR', fontsize=16, pad=15)
    plt.xlabel('False Positive Rate (%)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust axis limits so labels don't get cut off
    plt.xlim(0, max(df.FPR_Percent) * 1.2)
    plt.ylim(0.75, 1.01)
    
    # Remove redundant legend since we labeled the points
    plt.legend([],[], frameon=False) 
    
    plt.tight_layout()
    plt.savefig('comparison_accuracy.png', dpi=300)

def plot_throughput_comparison():
    print("Generating Throughput Chart...")
    plt.figure(figsize=(10, 6))
    
    # Filter out NaN (ZAD-ML and Hybrid) since they didn't report exact EPS
    speed_df = df.dropna(subset=['Throughput_FPS']).sort_values('Throughput_FPS', ascending=False)
    
    ax = sns.barplot(x='System', y='Throughput_FPS', data=speed_df, palette='Blues_r')
    
    # Use Log Scale here too, because 1.1M vs 910 breaks standard charts
    ax.set_yscale('log')
    
    plt.title('Data Processing Speed', fontsize=16, pad=15)
    plt.ylabel('Events/Flows Per Second (Log Scale)', fontsize=12)
    
    # Add exact numbers above bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', 
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparison_throughput.png', dpi=300)

if __name__ == "__main__":
    print("--- GENERATING BINARY BENCHMARKS ---")
    plot_latency_comparison()
    plot_f1_vs_fpr()
    plot_throughput_comparison()
    print("Success! Check your folder for the three comparison graphics.")