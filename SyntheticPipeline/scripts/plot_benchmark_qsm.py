import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Plot QSM benchmark results")
    parser.add_argument("--csv", type=str, default=os.path.join(root_dir, "SyntheticPipeline", "output", "qsm_benchmark", "benchmark_results_qsm.csv"))
    parser.add_argument("--out", type=str, default=os.path.join(root_dir, "SyntheticPipeline", "output", "visualizations", "benchmark_results_qsm.png"))
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        return
        
    df = pd.read_csv(args.csv)
    if "Status" in df.columns:
        df = df[df["Status"] == "ok"]
    if df.empty:
        print("No successful benchmark rows to plot.")
        return
    
    # Primary whole-tree score plus radius-partition summaries.
    # Group by Condition, Species, Algorithm
    metrics = ['Whole_F1', 'Whole_IoU', 'Trunk_F1', 'Branch_F1']
    summary = df.groupby(['Species', 'Algorithm', 'Condition'])[metrics].mean().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    sns.set_theme(style="whitegrid")
    
    titles = {
        'Whole_F1': "Whole-tree F1 Score",
        'Whole_IoU': "Whole-tree Voxel IoU",
        'Trunk_F1': "Trunk F1 Score",
        'Branch_F1': "Branch F1 Score",
    }
    
    # Create combined label for Species + Condition
    summary['Species_Cond'] = summary['Species'] + " (" + summary['Condition'] + ")"
    
    for ax, metric in zip(axes.flatten(), metrics):
        sns.barplot(x="Species_Cond", y=metric, hue="Algorithm", data=summary, palette="Set2", ax=ax)
        ax.set_title(titles[metric], fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=45)
        
        # Add labels to empty bars to make it clear they are 0.0
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
            
        if ax != axes[0, 0]:
            ax.get_legend().remove()
        else:
            ax.legend(title='Algorithm', loc='upper right')
            
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved QSM benchmark plot to {args.out}")

if __name__ == "__main__":
    main()
