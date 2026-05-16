import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Set global academic styling
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.autolayout': True 
})

def plot_attack_distribution():
    # Summed data from your prompt
    labels = ['BENIGN', 'DoS Hulk', 'DoS slowloris', 'DoS Slowhttptest']
    counts = [1934377, 160909, 107580, 99511]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use log scale for X-axis because BENIGN is massive compared to the others
    bars = ax.barh(labels, counts, color=['#4C72B0', '#C44E52', '#DD8452', '#D65F5F'])
    ax.set_xscale('log')
    
    ax.set_xlabel('Total Flow Count (Log Scale)')
    ax.set_title('Dataset Distribution Prior to Binary Grouping')
    
    # Annotate exact numbers on the bars
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{int(width):,}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', fontsize=11)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('attack_distribution.pdf', format='pdf', dpi=300)
    print("Saved attack_distribution.pdf")

def plot_training_resources():
    # Data from your trainCB.py output
    metrics = ['Training Latency\n(Seconds)', 'Active RAM Spike\n(MB)']
    values = [93.52, 3622.14]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(metrics, values, width=0.4, color=['#55A868', '#8172B2'])

    ax.set_ylabel('Value')
    ax.set_title('CatBoost Training Resource Overhead')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), f'{yval:,.2f}', ha='center', va='bottom', fontweight='bold')

    # Extend Y-axis slightly so labels fit
    plt.ylim(0, 4200)
    plt.savefig('training_resources.pdf', format='pdf', dpi=300)
    print("Saved training_resources.pdf")

def plot_feature_importance():
    print("Loading model to extract feature importance...")
    try:
        model = CatBoostClassifier()
        model.load_model("suricata_catboost_model.cbm")
        
        # Get feature importances and names directly from the model
        importances = model.get_feature_importance()
        feature_names = model.feature_names_
        
        # Create a DataFrame and sort it
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(20) # Get top 20
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(df_imp['Feature'], df_imp['Importance'], color='#4C72B0')
        
        ax.set_xlabel('Feature Importance Score')
        ax.set_title('Top 20 Most Influential Features (CatBoost)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.savefig('feature_importance.pdf', format='pdf', dpi=300)
        print("Saved feature_importance.pdf")
        
    except Exception as e:
        print(f"Could not load model for feature importance: {e}")

if __name__ == "__main__":
    plot_attack_distribution()
    plot_training_resources()
    plot_feature_importance()