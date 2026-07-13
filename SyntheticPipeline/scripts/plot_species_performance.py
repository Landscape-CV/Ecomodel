import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    csv_path = r"d:\Projects\PyTLidar\SyntheticPipeline\output\benchmark_results_single.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    # Filter out invalid rows where f1 score is missing or error occurred
    df = df.dropna(subset=['trunk_f1_score'])
    
    # Extract species from tile name (e.g. ca_black_oak_tile_0 -> ca_black_oak)
    # The convention is everything before '_tile_'
    df['species'] = df['tile'].apply(lambda x: x.split('_tile_')[0].replace('_', ' ').title())
    
    metrics = ['trunk_precision', 'trunk_recall', 'trunk_f1_score', 'trunk_iou']
    
    # Set the style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.barplot(
            data=df,
            x='species',
            y=metric,
            hue='algorithm',
            ax=axes[i],
            errorbar=None
        )
        axes[i].set_title(f"{metric.replace('_', ' ').title()} by Species")
        axes[i].set_ylabel("Score")
        axes[i].set_xlabel("Species")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1.05)
        
        # Place legend nicely
        if i == 0:
            axes[i].legend(title='Algorithm', loc='upper right')
        else:
            axes[i].get_legend().remove()
            
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(csv_path), "visualizations", "species_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved species breakdown plot to {out_path}")

if __name__ == '__main__':
    main()
