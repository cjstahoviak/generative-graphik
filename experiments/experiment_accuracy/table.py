#!/usr/bin/env python3
import importlib.util
import json
import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
import random
import copy
import pandas as pd
import time

import matplotlib.pyplot as plt
import numpy as np
from generative_graphik.args.utils import str2bool
import argparse
import seaborn as sns
from matplotlib import rcParams

# Set up better styling for plots
plt.style.use('seaborn-v0_8-darkgrid')
# Use default system fonts instead of specific fonts that might not be available
rcParams['font.family'] = 'sans-serif'  # Use system sans-serif font
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
    # Create output directory if it doesn't exist
    results_dir = f"{sys.path[0]}/results/{args.id}/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the pickle file
    data = pd.read_pickle(f"{results_dir}/results.pkl")
    stats = data.reset_index()
    
    # Process error metrics
    stats["Err. Position"] = stats["Err. Position"]*1000  # Convert to mm
    stats["Err. Rotation"] = stats["Err. Rotation"]*(180/np.pi)  # Convert to degrees
    
    # Filter outliers
    q_pos = stats["Err. Position"].quantile(0.99)
    q_rot = stats["Err. Rotation"].quantile(0.99)
    stats = stats.drop(stats[stats["Err. Position"] > q_pos].index)
    stats = stats.drop(stats[stats["Err. Rotation"] > q_rot].index)

    # Calculate statistics
    stats_summary = stats.groupby(["Robot", "Id"])[["Err. Position", "Err. Rotation"]].describe().groupby("Robot").mean()
    stats_summary = stats_summary.drop(["count", "std", "50%"], axis=1, level=1)
    
    # Rename columns for better readability
    stats_summary.rename(columns = {'75%': 'Q$_{3}$', '25%': 'Q$_{1}$','Err. Position':'Err. Pos. [mm]', 'Err. Rotation':'Err. Rot. [deg]'}, inplace = True)

    # Swap to follow paper order
    cols = stats_summary.columns.tolist()
    ins = cols.pop(4)
    cols.insert(2, ins)
    ins = cols.pop(9)
    cols.insert(7, ins)
    stats_summary = stats_summary[cols]

    # Create visualizations with enhanced styling
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    fig.suptitle(f"Inverse Kinematics Accuracy Analysis", fontsize=22, fontweight='bold')
    
    # Position error boxplot with custom palette
    sns.boxplot(x="Robot", y="Err. Position", data=stats, ax=axs[0], 
                hue="Robot", palette="Blues", width=0.6, linewidth=2, legend=False)
    axs[0].set_title("Position Error by Robot Type")
    axs[0].set_ylabel("Position Error (mm)")
    axs[0].set_xlabel("Robot Type")
    
    # Rotation error boxplot with custom palette
    sns.boxplot(x="Robot", y="Err. Rotation", data=stats, ax=axs[1], 
                hue="Robot", palette="Oranges", width=0.6, linewidth=2, legend=False)
    axs[1].set_title("Rotation Error by Robot Type")
    axs[1].set_ylabel("Rotation Error (degrees)")
    axs[1].set_xlabel("Robot Type")
    
    # Apply enhanced styling
    fig = set_figure_style(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    # Save the first figure
    fig.savefig(f"{results_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
    
    # Check if additional data columns exist
    additional_cols = [col for col in stats.columns if col not in 
                      ["Robot", "Id", "Err. Position", "Err. Rotation", "index", 
                       "Goal Pose", "Sol. Config", "Sol. Time"]]
    
    # Create per-robot plots for additional metrics if they exist
    if additional_cols:
        # Get unique robot types
        robot_types = stats["Robot"].unique()
        
        # Create a figure for each robot type
        fig2, axs2 = plt.subplots(len(robot_types), len(additional_cols[:3]), 
                                 figsize=(15, 4*len(robot_types)), 
                                 squeeze=False)
        
        fig2.suptitle(f"Distribution of Additional Metrics by Robot Type", 
                      fontsize=22, fontweight='bold')
        
        # Plot each additional metric for each robot
        for i, robot in enumerate(robot_types):
            robot_data = stats[stats["Robot"] == robot]
            
            for j, col in enumerate(additional_cols[:3]):  # Limit to first 3 additional columns
                ax = axs2[i, j]
                
                # Create a KDE plot for this metric and robot
                sns.histplot(robot_data[col], kde=True, ax=ax, 
                            color=plt.cm.tab10(i), linewidth=2)
                
                ax.set_title(f"{robot}: {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
        
        # Apply enhanced styling
        fig2 = set_figure_style(fig2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the second figure
        fig2.savefig(f"{results_dir}/additional_metrics.png", dpi=300, bbox_inches='tight')
    
    # Handle LaTeX table export
    latex = None
    if args.save_latex:
        s = stats_summary.style
        s.format(precision=1)
        s.format_index(axis=1,level=[0,1])
        latex = s.to_latex(hrules=True, multicol_align="c")
        print(latex)

        # Save the LaTeX table
        latex_file = f"{results_dir}/results_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to {latex_file}")
    
    # Create a summary markdown file with all results
    with open(f"{results_dir}/summary.md", 'w') as f:
        f.write(f"# Results Summary for {args.id}\n\n")
        f.write("## Error Statistics\n\n")
        f.write("```\n")
        f.write(stats_summary.to_string())
        f.write("\n```\n\n")
        f.write("## Figures\n\n")
        f.write("![Error Analysis](error_analysis.png)\n\n")
        if additional_cols:
            f.write("![Additional Metrics](additional_metrics.png)\n\n")
        if latex:
            f.write("## LaTeX Table\n\n")
            f.write("```latex\n")
            f.write(latex)
            f.write("\n```\n")
    
    print(f"Results processed and saved to {results_dir}")

if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--save_latex", type=str2bool, default=True, help="Save latex table.")

    args = parser.parse_args()
    main(args)