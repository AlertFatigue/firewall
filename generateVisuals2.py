import matplotlib.pyplot as plt
import numpy as np

# Set global academic styling
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.autolayout': True # Prevents cutoff labels
})

def plot_classification_metrics():
    labels = ['Precision', 'Recall', 'F1-Score']
    benign_scores = [0.9981, 0.9889, 0.9934]
    malicious_scores = [0.9441, 0.9900, 0.9665]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, benign_scores, width, label='BENIGN', color='#4C72B0')
    rects2 = ax.bar(x + width/2, malicious_scores, width, label='Malicious', color='#C44E52')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_title('CatBoost Classification Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.85, 1.02]) # Zoom in to highlight the differences
    ax.legend(loc='lower center')

    # Attach a text label above each bar, displaying its height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('classification_metrics.pdf', format='pdf', dpi=300)
    print("Saved classification_metrics.pdf")

def plot_ram_footprint():
    labels = ['Saved Model Size', 'Inference RAM Spike']
    values_mb = [3.12, 73.68]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values_mb, width=0.5, color=['#55A868', '#DD8452'])

    ax.set_ylabel('Memory (MB)')
    ax.set_title('CatBoost Memory Footprint (Inference)')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval} MB', ha='center', va='bottom')

    plt.ylim(0, 90)
    plt.savefig('ram_footprint.pdf', format='pdf', dpi=300)
    print("Saved ram_footprint.pdf")

if __name__ == "__main__":
    plot_classification_metrics()
    plot_ram_footprint()