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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
        if hasattr(ax, 'spines'):  # 3D axes don't have spines in the same way
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        
        # Bold titles and labels
        if hasattr(ax, 'title'):
            ax.title.set_weight('bold')
        if hasattr(ax, 'xaxis') and hasattr(ax.xaxis, 'label'):
            ax.xaxis.label.set_weight('bold')
        if hasattr(ax, 'yaxis') and hasattr(ax.yaxis, 'label'):
            ax.yaxis.label.set_weight('bold')
        
        # Set tick parameters for 2D plots
        if hasattr(ax, 'tick_params'):
            ax.tick_params(direction='out', width=1.5, length=6)
    return fig

def plot_scaling_analysis(df, results_dir):
    """Create plots showing how computation time scales with DOF."""
    plt.figure(figsize=(12, 8))
    plt.title("Average Computation Time vs. DOF", fontsize=22, fontweight='bold')
    
    # Ensure DOF is treated as a string consistently
    df["DOF"] = df["DOF"].astype(str)
    
    # Convert DOF to numeric for proper sorting, but keep original strings for display
    df["DOF_numeric"] = df["DOF"].apply(lambda x: int(x))
    
    # Select a few representative sample sizes for analysis
    sample_sizes = [1, 16, 64, 128, 256]
    markers = ['o', 's', '^', 'D', 'x']
    
    # Get unique DOFs in correct numerical order
    unique_dofs = sorted(df["DOF"].unique(), key=lambda x: int(x))
    x_positions = np.arange(len(unique_dofs))
    
    for i, sample_size in enumerate(sample_sizes):
        # Filter data for this sample size
        subset = df[df["Number of Sampled Configurations"] == sample_size]
        
        # Create a DataFrame with all DOF values to ensure consistent ordering
        ordered_data = []
        for dof in unique_dofs:
            dof_subset = subset[subset["DOF"] == dof]
            if not dof_subset.empty:
                ordered_data.append({
                    "DOF": dof,
                    "Time [ms]": dof_subset["Time [ms]"].mean()
                })
            else:
                # If no data for this DOF, add with NaN
                ordered_data.append({
                    "DOF": dof,
                    "Time [ms]": np.nan
                })
        
        ordered_df = pd.DataFrame(ordered_data)
        
        plt.plot(x_positions, ordered_df["Time [ms]"], 
                label=f"{sample_size} samples",
                linewidth=2.5,
                marker=markers[i],
                markersize=8)
    
    # Set x-ticks to the DOF values in correct order
    plt.xticks(x_positions, unique_dofs)
    
    plt.xlabel("Degrees of Freedom (DOF)")
    plt.ylabel("Average Time [ms]")
    plt.legend(title="Sample Size")
    plt.grid(True, alpha=0.3)
    
    # Apply enhanced styling
    set_figure_style(plt.gcf())
    plt.tight_layout()
    
    # Save plot
    plt.savefig(
        os.path.join(results_dir, "dof_scaling_analysis.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_efficiency(df, results_dir):
    """Create plots showing time per sample for different configurations."""
    plt.figure(figsize=(12, 8))
    plt.title("Average Time per Sample vs. Sample Size", fontsize=22, fontweight='bold')
    
    # Ensure DOF is treated as a string consistently
    df["DOF"] = df["DOF"].astype(str)
    
    # Calculate time per sample
    df["Time per Sample [ms]"] = df["Time [ms]"] / df["Number of Sampled Configurations"]
    
    # Sort DOFs numerically
    unique_dofs = sorted(df["DOF"].unique(), key=lambda x: int(x))
    
    # Use a color palette that makes differentiating lines easier
    palette = plt.cm.viridis(np.linspace(0, 1, len(unique_dofs)))
    
    # Get all unique sample sizes in the data
    all_sample_sizes = sorted(df["Number of Sampled Configurations"].unique())
    
    for i, dof in enumerate(unique_dofs):
        subset = df[df["DOF"] == dof]
        
        # Create a DataFrame to ensure all sample sizes are included
        # This makes the x-axis match between different plots
        efficiency_data = []
        for sample_size in all_sample_sizes:
            sample_subset = subset[subset["Number of Sampled Configurations"] == sample_size]
            if not sample_subset.empty:
                efficiency_data.append({
                    "Number of Sampled Configurations": sample_size,
                    "Time per Sample [ms]": sample_subset["Time per Sample [ms]"].mean()
                })
            else:
                # If no data for this sample size, add with NaN
                efficiency_data.append({
                    "Number of Sampled Configurations": sample_size,
                    "Time per Sample [ms]": np.nan
                })
                
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df = efficiency_df.sort_values("Number of Sampled Configurations")
        
        plt.plot(efficiency_df["Number of Sampled Configurations"], 
                efficiency_df["Time per Sample [ms]"], 
                label=f"{dof} DOF",
                linewidth=2.5,
                marker='o',
                markersize=8,
                color=palette[i])
    
    plt.xlabel("Number of Sampled Configurations")
    plt.ylabel("Average Time per Sample [ms]")
    plt.legend(title="Robot DOF", title_fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Apply log scale to y-axis only, keeping x-axis linear
    plt.yscale('log')
    
    # Set specific x-ticks to match the timing_plot
    plt.xticks(all_sample_sizes, all_sample_sizes)
    
    # Apply enhanced styling
    set_figure_style(plt.gcf())
    plt.tight_layout()
    
    # Save plot
    plt.savefig(
        os.path.join(results_dir, "efficiency_analysis.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_3d_surface(df, results_dir):
    """Create 3D surface plot showing relationship between all three variables."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ensure DOF is treated as a string consistently
    df["DOF"] = df["DOF"].astype(str)
    
    # Group by DOF and Sample Size to get mean times
    grouped = df.groupby(["DOF", "Number of Sampled Configurations"])["Time [ms]"].mean().reset_index()
    
    # Convert DOF to numeric for proper ordering
    grouped["DOF_numeric"] = grouped["DOF"].apply(lambda x: int(x))
    
    # Get unique DOFs in correct numerical order
    unique_dofs = sorted(df["DOF"].unique(), key=lambda x: int(x))
    unique_dofs_numeric = [int(dof) for dof in unique_dofs]
    unique_samples = sorted(grouped["Number of Sampled Configurations"].unique())
    
    # Build the grid
    X, Y = np.meshgrid(unique_dofs_numeric, unique_samples)
    Z = np.zeros(X.shape)
    
    # Fill in Z values
    for i, dof in enumerate(unique_dofs):
        for j, sample_size in enumerate(unique_samples):
            # Find the matching row in the grouped data
            mask = (grouped["DOF"] == dof) & (grouped["Number of Sampled Configurations"] == sample_size)
            if mask.any():
                Z[j, i] = grouped.loc[mask, "Time [ms]"].values[0]
    
    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, 
                          linewidth=0, antialiased=True,
                          alpha=0.7)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Average Time [ms]')
    
    # Label axes
    ax.set_xlabel('Degrees of Freedom (DOF)')
    ax.set_ylabel('Number of Samples')
    ax.set_zlabel('Average Time [ms]')
    ax.set_title('3D Visualization of Average Computation Time', fontsize=20, fontweight='bold')
    
    # Set the exact x-ticks to show the DOF values in correct order
    ax.set_xticks(unique_dofs_numeric)
    ax.set_xticklabels(unique_dofs)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "3d_surface_plot.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_heatmap(df, results_dir):
    """Create a heatmap showing computation time for different combinations."""
    plt.figure(figsize=(12, 8))
    plt.title("Average Computation Time Heatmap", fontsize=22, fontweight='bold')
    
    # Ensure DOF is treated as a string consistently
    df["DOF"] = df["DOF"].astype(str)
    
    # Group by DOF and Sample Size
    grouped = df.groupby(["DOF", "Number of Sampled Configurations"])["Time [ms]"].mean().reset_index()
    
    # Sort DOFs numerically
    unique_dofs = sorted(grouped["DOF"].unique(), key=lambda x: int(x))
    samples = sorted(grouped["Number of Sampled Configurations"].unique())
    
    # Create an ordered grid for the heatmap
    heatmap_data = np.zeros((len(unique_dofs), len(samples)))
    
    # Fill the grid with values
    for i, dof in enumerate(unique_dofs):
        for j, sample in enumerate(samples):
            # Find matching rows
            mask = (grouped["DOF"] == dof) & (grouped["Number of Sampled Configurations"] == sample)
            if mask.any():
                heatmap_data[i, j] = grouped.loc[mask, "Time [ms]"].values[0]
    
    # Create the heatmap using imshow
    plt.figure(figsize=(12, 8))
    plt.title("Average Computation Time Heatmap", fontsize=22, fontweight='bold')
    im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    # Set the ticks
    plt.xticks(np.arange(len(samples)), samples)
    plt.yticks(np.arange(len(unique_dofs)), unique_dofs)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Time [ms]', rotation=270, fontsize=14, labelpad=20, weight='bold')
    
    # Add text annotations with white text for better contrast
    for i in range(len(unique_dofs)):
        for j in range(len(samples)):
            plt.text(j, i, f"{heatmap_data[i, j]:.1f}",
                    ha="center", va="center", 
                    color="white",  # Use white text for all values
                    fontweight='bold',  # Make text bold
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.3))  # Add background highlight
    
    # Label axes
    plt.xlabel("Number of Sampled Configurations")
    plt.ylabel("Degrees of Freedom (DOF)")
    
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(results_dir, "heatmap_analysis.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def main(args):
    results_dir = f"{sys.path[0]}/results/{args.id}/"
    path = os.path.join(results_dir, "results.pkl")
    
    print(f"Loading timing data from {path}")
    with open(path, 'rb') as f:
        data = pkl.load(f)
    
    data = np.array(data, dtype=object)
    # Data is already in milliseconds from CUDA timing events, no need to convert
    
    df = pd.DataFrame(data, columns=["Number of Sampled Configurations", "Time [ms]", "DOF"])
    
    # Create all plots
    print("Generating additional plots...")
    
    print("1. Creating DOF scaling analysis plot")
    plot_scaling_analysis(df, results_dir)
    
    print("2. Creating efficiency analysis plot")
    plot_efficiency(df, results_dir)
    
    print("3. Creating 3D surface plot")
    plot_3d_surface(df, results_dir)
    
    print("4. Creating heatmap analysis")
    plot_heatmap(df, results_dir)
    
    print(f"All plots generated and saved to {results_dir}")

if __name__ == "__main__":
    random.seed(17)
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    
    args = parser.parse_args()
    main(args)