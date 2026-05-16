import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# ==========================================
# THESIS VISUALIZATION SCRIPT
# ==========================================

# Set global thesis style for professional looking charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def plot_pipeline_funnel():
    print("Generating Data Pipeline Funnel...")
    
    stages = ['Raw eve.json Events', 'Aggregated CSV Flows', 'Test Set Evaluated', 'Correctly Classified']
    
    # Updated with your exact terminal output and split logic
    counts = [2310060, 2302380, 460476, 455410] 

    plt.figure(figsize=(10, 6))
    colors = ['#2c3e50', '#34495e', '#2980b9', '#27ae60']
    
    # Reverse the order so the largest is at the top of the funnel
    bars = plt.barh(stages[::-1], counts[::-1], color=colors[::-1])
    
    plt.title('Data Reduction and Inference Pipeline', fontsize=16, pad=20)
    plt.xlabel('Number of Records', fontsize=12)
    
    # Add the text labels to the right of the bars (fixed padding error)
    for bar in bars:
        width = bar.get_width()
        # width * 1.01 adds a small 1% visual gap between the bar and the text
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'{int(width):,}', 
                 ha='left', va='center', fontweight='bold', fontsize=12)

    # Disable scientific notation on the X axis
    plt.ticklabel_format(style='plain', axis='x')
    
    # Add some extra room on the right so the text doesn't get cut off
    plt.xlim(0, max(counts) * 1.2) 
    
    plt.tight_layout()
    plt.savefig('thesis_data_funnel.png', dpi=300)
    print("Saved 'thesis_data_funnel.png'")

def plot_feature_importance():
    print("\nGenerating Feature Importance Chart...")
    
    # Load the model
    model = CatBoostClassifier()
    try:
        model.load_model("suricata_catboost_model.cbm")
    except Exception as e:
        print(f"Error loading model for feature importance: {e}")
        return
    
    # Get importance scores and feature names
    importances = model.get_feature_importance()
    feature_names = model.feature_names_
    
    # Put into a dataframe and sort
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15) # Top 15 features

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    
    plt.title('Top 15 Most Important Network Features for Detection', fontsize=16)
    plt.xlabel('CatBoost Importance Score', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('thesis_feature_importance.png', dpi=300)
    print("Saved 'thesis_feature_importance.png'")

def plot_throughput_benchmark():
    print("\nGenerating Throughput Benchmark...")
    
    # Your model's throughput vs. typical enterprise network requirements
    labels = ['CatBoost Inference\n(Your Model)', 'Typical Enterprise Edge\n(10 Gbps Approx)', 'Small Office Router\n(1 Gbps Approx)']
    
    # Flows per second (Using your exact 1.12M result)
    flows_per_sec = [1127961, 500000, 50000] 

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, flows_per_sec, color=['#e74c3c', '#95a5a6', '#bdc3c7'])
    
    plt.title('Model Throughput vs. Real-World Network Demands', fontsize=16, pad=20)
    plt.ylabel('Flows Processed Per Second', fontsize=12)
    
    # Add exact numbers slightly above bars (fixed padding error)
    for bar in bars:
        height = bar.get_height()
        # height * 1.02 pushes the text slightly above the top of the bar
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')

    plt.ticklabel_format(style='plain', axis='y')
    
    # Add extra room at the top so the highest number doesn't touch the edge
    plt.ylim(0, max(flows_per_sec) * 1.15)
    
    plt.tight_layout()
    plt.savefig('thesis_throughput_benchmark.png', dpi=300)
    print("Saved 'thesis_throughput_benchmark.png'")

if __name__ == "__main__":
    print("--- THESIS VISUALIZATION GENERATOR ---")
    plot_pipeline_funnel()
    plot_feature_importance()
    plot_throughput_benchmark()
    print("\nAll visuals generated successfully! Check your project folder for the .png files.")