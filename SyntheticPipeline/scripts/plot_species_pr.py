import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot PR by species")
    parser.add_argument("--csv", type=str, required=True, help="Path to benchmark_results_single.csv")
    parser.add_argument("--out", type=str, required=True, help="Path to output png")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    
    # Extract species from tile name (e.g., 'barley_tile_1' -> 'barley')
    df['species'] = df['tile'].apply(lambda x: x.split('_tile_')[0])
    
    # Group by species and algorithm, and calculate mean metrics
    # Layout: Top row = Precision, Bottom row = Recall
    #         Left col = Trunk, Right col = Canopy
    metrics = ['trunk_precision', 'canopy_precision', 'trunk_recall', 'canopy_recall']
    summary = df.groupby(['species', 'algorithm'])[metrics].mean().reset_index()
    
    # Set up the plot grid (2x2 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    sns.set_theme(style="whitegrid")
    
    titles = {
        'trunk_precision': "Trunk (Wood) Precision",
        'trunk_recall': "Trunk (Wood) Recall",
        'canopy_precision': "Canopy (Leaf) Precision",
        'canopy_recall': "Canopy (Leaf) Recall"
    }
    
    for ax, metric in zip(axes.flatten(), metrics):
        sns.barplot(x="species", y=metric, hue="algorithm", data=summary, palette="magma", ax=ax)
        ax.set_title(titles[metric], fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=45)
        # Only show legend on the first subplot to save space
        if ax != axes[0, 0]:
            ax.get_legend().remove()
        else:
            ax.legend(title='Algorithm', loc='upper right')
            
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive PR species comparison plot to {args.out}")

if __name__ == "__main__":
    main()
