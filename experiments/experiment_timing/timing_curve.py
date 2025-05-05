#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import random
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
from matplotlib import rcParams

# Set up better styling for plots
plt.style.use('seaborn-v0_8-darkgrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 20
rcParams['figure.titleweight'] = 'bold'

def set_figure_style(fig):
    """Apply custom styling to the figure"""
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Bold titles and labels
        ax.title.set_weight('bold')
        ax.xaxis.label.set_weight('bold')
        ax.yaxis.label.set_weight('bold')
        
        # Set tick parameters
        ax.tick_params(direction='out', width=1.5, length=6)
    return fig

def main(args):
    results_dir = f"{sys.path[0]}/results/{args.id}/"
    path = os.path.join(results_dir, "results.pkl")
    
    print(f"Loading timing data from {path}")
    with open(path, 'rb') as f:
        data = pkl.load(f)
    
    data = np.array(data, dtype=object)
    # Data[:, 1] already contains time values in milliseconds from CUDA timing, no need to convert
    
    df = pd.DataFrame(data, columns=["Number of Sampled Configurations", "Time [ms]", "DOF"])
    
    # Group by DOF and Number of Sampled Configurations to get average time
    # (data already contains multiple trial runs for each configuration)
    grouped_df = df.groupby(["DOF", "Number of Sampled Configurations"])["Time [ms]"].mean().reset_index()
    
    # Create figure with enhanced styling
    plt.figure(figsize=(12, 8))
    plt.title("Average Inference Time vs Number of Configurations", fontsize=22, fontweight='bold')
    
    # Create line plot with custom palette
    out = plt.gca()
    sns_plot = plt.gcf()
    
    # Sort DOFs numerically, not lexicographically
    # First ensure DOF is treated as a string consistently
    df["DOF"] = df["DOF"].astype(str)
    # Get unique DOFs and sort them numerically
    unique_dofs = sorted(df["DOF"].unique(), key=lambda x: int(x))
    
    # Use a color palette that makes differentiating lines easier
    palette = plt.cm.viridis(np.linspace(0, 1, len(unique_dofs)))
    
    # Plot each DOF line in correct numerical order
    for i, dof in enumerate(unique_dofs):
        subset = grouped_df[grouped_df["DOF"] == dof]
        plt.plot(subset["Number of Sampled Configurations"],
                subset["Time [ms]"],
                label=f"{dof} DOF",
                linewidth=2.5,
                marker='o',
                markersize=8,
                color=palette[i])
    
    plt.xlabel("Number of Sampled Configurations")
    plt.ylabel("Average Time [ms]")
    plt.legend(title="Robot DOF", title_fontsize=14)
    
    # Apply enhanced styling
    set_figure_style(plt.gcf())
    plt.tight_layout()
    
    # Save plots to the same directory as results.pkl
    print(f"Saving plots to {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as PDF
    plt.savefig(
        os.path.join(results_dir, "timing_plot.pdf"),
        bbox_inches='tight',
        dpi=300
    )
    
    # Save as PNG for easy viewing
    plt.savefig(
        os.path.join(results_dir, "timing_plot.png"),
        bbox_inches='tight',
        dpi=300
    )
    
    print(f"Plot generation complete. Saved as PDF and PNG in {results_dir}")

if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    
    args = parser.parse_args()
    main(args)