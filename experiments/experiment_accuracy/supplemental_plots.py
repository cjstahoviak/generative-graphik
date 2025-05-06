#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import argparse
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

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

def plot_error_distribution(stats, results_dir):
    """
    Create a violin plot showing the distribution of position and rotation errors.
    """
    plt.figure(figsize=(14, 8))
    plt.title("Error Distribution Across Robot Types", fontsize=22, fontweight='bold')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Create violin plots for position error - using hue instead of palette directly
    sns.violinplot(x="Robot", y="Err. Position", hue="Robot", data=stats, ax=ax1, 
                  palette="Blues", inner="quartile", cut=0, legend=False)
    ax1.set_title("Position Error Distribution")
    ax1.set_ylabel("Position Error (mm)")
    ax1.set_xlabel("Robot Type")
    
    # Create violin plots for rotation error - using hue instead of palette directly
    sns.violinplot(x="Robot", y="Err. Rotation", hue="Robot", data=stats, ax=ax2, 
                  palette="Oranges", inner="quartile", cut=0, legend=False)
    ax2.set_title("Rotation Error Distribution")
    ax2.set_ylabel("Rotation Error (degrees)")
    ax2.set_xlabel("Robot Type")
    
    # Apply enhanced styling
    set_figure_style(fig)
    plt.tight_layout()
    
    # Save plot (PNG only)
    plt.savefig(
        os.path.join(results_dir, "error_distribution.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_error_relationship(stats, results_dir):
    """
    Create a scatter plot showing relationship between position and rotation errors.
    Color points by robot type and add density contours.
    """
    plt.figure(figsize=(12, 10))
    plt.title("Relationship Between Position and Rotation Errors", fontsize=22, fontweight='bold')
    
    # Set up a custom palette for the robots
    robot_types = stats["Robot"].unique()
    palette = sns.color_palette("husl", len(robot_types))
    robot_colors = {robot: color for robot, color in zip(robot_types, palette)}
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each robot's data with different colors
    for robot in robot_types:
        robot_data = stats[stats["Robot"] == robot]
        
        # Create the scatter plot
        ax.scatter(
            robot_data["Err. Position"], 
            robot_data["Err. Rotation"],
            alpha=0.6,
            label=robot,
            color=robot_colors[robot],
            edgecolor='none'
        )
        
        # Add density contours
        if len(robot_data) > 10:  # Need enough points for density estimation
            try:
                # Calculate kernel density estimate
                x = robot_data["Err. Position"]
                y = robot_data["Err. Rotation"]
                xy = np.vstack([x, y])
                
                # Add density contour if there are enough unique points
                if len(np.unique(xy, axis=1)) > 5:
                    kde = gaussian_kde(xy)
                    
                    # Create a grid for density calculation
                    x_grid = np.linspace(min(x), max(x), 100)
                    y_grid = np.linspace(min(y), max(y), 100)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    
                    # Calculate density at each grid point
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = kde(positions).reshape(X.shape)
                    
                    # Plot contours
                    contour = ax.contour(
                        X, Y, Z, 
                        levels=5, 
                        colors=[robot_colors[robot]],
                        alpha=0.5,
                        linewidths=1.5
                    )
            except Exception as e:
                print(f"Could not generate density contour for {robot}: {e}")
    
    # Add labels and legend
    ax.set_xlabel("Position Error (mm)")
    ax.set_ylabel("Rotation Error (degrees)")
    ax.legend(title="Robot Type")
    
    # Add a grid
    ax.grid(True, alpha=0.3)
    
    # Apply enhanced styling
    set_figure_style(fig)
    plt.tight_layout()
    
    # Save plot (PNG only)
    plt.savefig(
        os.path.join(results_dir, "error_relationship.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_solution_time_vs_accuracy(stats, results_dir):
    """
    Create a plot showing relationship between solution time and error.
    """
    plt.figure(figsize=(12, 8))
    plt.title("Solution Time vs. Accuracy", fontsize=22, fontweight='bold')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Convert solution time to milliseconds for better readability
    stats["Solution Time (ms)"] = stats["Sol. Time"] * 1000
    
    # Plot solution time vs position error
    g1 = sns.scatterplot(
        x="Solution Time (ms)", 
        y="Err. Position", 
        hue="Robot", 
        data=stats, 
        ax=ax1,
        alpha=0.7,
        s=50  # Point size
    )
    ax1.set_title("Solution Time vs. Position Error")
    ax1.set_xlabel("Solution Time (ms)")
    ax1.set_ylabel("Position Error (mm)")
    
    # Plot solution time vs rotation error
    g2 = sns.scatterplot(
        x="Solution Time (ms)", 
        y="Err. Rotation", 
        hue="Robot", 
        data=stats, 
        ax=ax2,
        alpha=0.7,
        s=50  # Point size
    )
    ax2.set_title("Solution Time vs. Rotation Error")
    ax2.set_xlabel("Solution Time (ms)")
    ax2.set_ylabel("Rotation Error (degrees)")
    
    # Remove duplicate legends (keep only one)
    ax2.get_legend().remove()
    
    # Adjust legend position on first subplot
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, title="Robot Type", loc='upper left', bbox_to_anchor=(0, 1))
    
    # Apply enhanced styling
    set_figure_style(fig)
    plt.tight_layout()
    
    # Save plot (PNG only)
    plt.savefig(
        os.path.join(results_dir, "time_vs_accuracy.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_joint_configuration_analysis(stats, results_dir):
    """
    Create visualizations analyzing the joint configurations used in solutions.
    """
    # Get a list of all robot types
    robot_types = sorted(stats["Robot"].unique())
    
    # Create a combined violin plot showing joint angle distributions
    # First, gather joint data for all robots
    all_joint_data = []
    
    # Create subscript mapping (for θ subscripts instead of θ_n)
    subscripts = {
        1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉"
    }
    
    for robot in robot_types:
        # Filter data for this robot
        robot_data = stats[stats["Robot"] == robot]
        
        # Extract the joint configurations
        joint_configs = np.vstack(robot_data["Sol. Config"].values)
        
        # Get number of joints for this robot
        num_joints = joint_configs.shape[1]
        
        # Add joint data to collection
        for i in range(num_joints):
            for angle in joint_configs[:, i]:
                # Use subscript notation
                joint_label = f"θ{subscripts[i+1]}"
                all_joint_data.append({
                    "Robot": robot,
                    "Joint": joint_label,
                    "Angle (rad)": angle
                })
    
    # Convert to DataFrame
    joint_df = pd.DataFrame(all_joint_data)
    
    # Create a figure with robot types as columns
    fig, axes = plt.subplots(1, len(robot_types), figsize=(16, 6), sharey=True)
    
    # If only one robot type, wrap axes in a list
    if len(robot_types) == 1:
        axes = [axes]
    
    # Plot each robot's joint distributions
    for i, robot in enumerate(robot_types):
        robot_joint_df = joint_df[joint_df["Robot"] == robot]
        
        # Create violin plot for this robot - use hue instead of palette directly
        sns.violinplot(
            x="Joint", 
            y="Angle (rad)", 
            hue="Joint",
            data=robot_joint_df,
            palette="viridis",
            inner="quartile",
            ax=axes[i],
            legend=False
        )
        
        # Add horizontal line at 0
        axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Set titles and labels
        axes[i].set_title(f"{robot.upper()}")
        axes[i].set_xlabel("Joint")
        
        # Only add y-label for the first subplot
        if i == 0:
            axes[i].set_ylabel("Joint Angle (radians)")
        else:
            axes[i].set_ylabel("")
    
    # Add overall title
    fig.suptitle("Joint Angle Distributions by Robot Type", fontsize=22, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    
    # Apply enhanced styling
    set_figure_style(fig)
    
    # Save the combined figure (PNG only)
    plt.savefig(
        os.path.join(results_dir, "joint_distributions.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    
    # Create a combined correlation heatmap for joint angles
    # Create a figure with grid layout
    fig, axes = plt.subplots(1, len(robot_types), figsize=(16, 6), squeeze=False)
    
    # Create a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Create correlation heatmaps for each robot
    for i, robot in enumerate(robot_types):
        # Filter data for this robot
        robot_data = stats[stats["Robot"] == robot]
        
        # Extract the joint configurations
        joint_configs = np.vstack(robot_data["Sol. Config"].values)
        
        # Get number of joints for this robot
        num_joints = joint_configs.shape[1]
        
        # Calculate correlation matrix with handling for potential NaN values
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        
        # Calculate means and standard deviations
        means = np.mean(joint_configs, axis=0)
        stds = np.std(joint_configs, axis=0)
        
        # Replace zero standard deviations with epsilon
        stds[stds < epsilon] = epsilon
        
        # Calculate correlation matrix manually to avoid divide by zero warnings
        corr_matrix = np.zeros((num_joints, num_joints))
        for j in range(num_joints):
            for k in range(num_joints):
                if j == k:
                    corr_matrix[j, k] = 1.0
                else:
                    # Calculate correlation coefficient
                    cov_jk = np.mean((joint_configs[:, j] - means[j]) * 
                                     (joint_configs[:, k] - means[k]))
                    corr_matrix[j, k] = cov_jk / (stds[j] * stds[k])
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Draw the heatmap without annotations
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            annot=False,
            ax=axes[0, i],
            cbar=False
        )
        
        # Set title
        axes[0, i].set_title(f"{robot.upper()}")
        
        # Set joint labels using theta with subscript notation
        joint_labels = [f"θ{subscripts[j+1]}" for j in range(num_joints)]
        axes[0, i].set_xticklabels(joint_labels)
        axes[0, i].set_yticklabels(joint_labels)
    
    # Add a single colorbar for all subplots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    
    # Add colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Correlation')
    
    # Add overall title
    fig.suptitle("Joint Angle Correlations by Robot Type", fontsize=22, fontweight='bold')
    
    # Adjust layout - use a different approach to avoid the warning
    plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
    
    # Apply enhanced styling
    set_figure_style(fig)
    
    # Save the combined correlation heatmap (PNG only)
    plt.savefig(
        os.path.join(results_dir, "joint_correlations.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_error_vs_joints(stats, results_dir):
    """
    Create a plot showing error distribution by number of joints.
    """
    # First, extract the number of joints from the solution configurations
    stats["Num Joints"] = stats["Sol. Config"].apply(lambda x: len(x))
    
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot position error by number of joints - use hue instead of palette directly
    sns.boxplot(
        x="Num Joints", 
        y="Err. Position", 
        hue="Num Joints",
        data=stats, 
        ax=ax1,
        palette="Blues",
        legend=False
    )
    ax1.set_title("Position Error by Number of Joints")
    ax1.set_xlabel("Number of Joints")
    ax1.set_ylabel("Position Error (mm)")
    
    # Plot rotation error by number of joints - use hue instead of palette directly
    sns.boxplot(
        x="Num Joints", 
        y="Err. Rotation", 
        hue="Num Joints",
        data=stats, 
        ax=ax2,
        palette="Oranges",
        legend=False
    )
    ax2.set_title("Rotation Error by Number of Joints")
    ax2.set_xlabel("Number of Joints")
    ax2.set_ylabel("Rotation Error (degrees)")
    
    # Apply enhanced styling
    set_figure_style(fig)
    plt.tight_layout()
    
    # Save plot (PNG only)
    plt.savefig(
        os.path.join(results_dir, "error_vs_joints.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def plot_comparative_summary(stats, results_dir):
    """
    Create a comprehensive summary comparing robot performance.
    """
    # Calculate aggregate statistics by robot type
    robot_stats = stats.groupby("Robot").agg({
        "Err. Position": ["mean", "median", "min", "max", "std"],
        "Err. Rotation": ["mean", "median", "min", "max", "std"],
        "Sol. Time": ["mean", "std"]
    })
    
    # Convert solution time to milliseconds
    robot_stats["Sol. Time", "mean"] *= 1000
    robot_stats["Sol. Time", "std"] *= 1000
    
    # Create a radar chart to compare robots across multiple metrics
    metrics = [
        ("Pos. Error", "Err. Position", "mean", False),  # Name, column, stat, invert flag (not used anymore)
        ("Rot. Error", "Err. Rotation", "mean", False),  # Higher error = bigger shape
        ("Solution Time", "Sol. Time", "mean", False),
        ("Pos. Variability", "Err. Position", "std", False),
        ("Rot. Variability", "Err. Rotation", "std", False),
    ]
    
    # Prepare data for radar chart
    robot_types = sorted(robot_stats.index.tolist())
    num_robots = len(robot_types)
    num_metrics = len(metrics)
    
    # Convert to numpy array
    raw_values = np.zeros((num_robots, num_metrics))
    
    # Fill in values for each robot and metric
    for i, robot in enumerate(robot_types):
        for j, (_, col, stat, _) in enumerate(metrics):
            raw_values[i, j] = robot_stats.loc[robot, (col, stat)]
    
    # Debug: Print raw values
    print("\nRaw values for radar chart:")
    # Calculate max values needed for debug output
    max_values = np.max(raw_values, axis=0)
    
    for i, robot in enumerate(robot_types):
        print(f"{robot}:")
        for j, (name, _, _, _) in enumerate(metrics):
            raw_val = raw_values[i, j]
            log_val = np.log1p(raw_val) if raw_val > 0 else 0
            log_max = np.log1p(max_values[j]) if max_values[j] > 0 else 1
            norm_val = log_val / log_max if log_max > 0 else 0
            
            print(f"  {name}: {raw_val:.4f}")
            print(f"    (Log value: {log_val:.4f}, Normalized: {norm_val:.4f})")
    
    # Create the radar chart using raw values
    fig = plt.figure(figsize=(12, 10))
    radar_ax = fig.add_subplot(111, polar=True)
    
    # Set radar chart angles
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Calculate axis limits for each metric
    max_values = np.max(raw_values, axis=0)
    
    # Remove the radial grid and tick labels (hiding the values)
    radar_ax.set_yticklabels([])
    radar_ax.set_rgrids([])  # This removes the circular grid lines and tick labels
    
    # Only keep the angular grid lines
    radar_ax.grid(True, axis='x')
    radar_ax.grid(False, axis='y')
    
    # Function to rescale a value to display on radar chart with logarithmic scaling
    def scale_for_radar(value, j):
        # Avoid division by zero or negative values (for log)
        if max_values[j] == 0 or value <= 0:
            return 0
        
        # Apply logarithmic scaling before normalizing
        # Add 1 to avoid log(0) and to make 0 map to 0
        log_value = np.log1p(value)
        log_max = np.log1p(max_values[j])
        
        # Normalize on log scale from 0 to 1
        return log_value / log_max
            
    # Plot each robot with scaled values
    for i, robot in enumerate(robot_types):
        # Scale values for display
        display_values = []
        for j, (_, _, _, _) in enumerate(metrics):
            display_values.append(scale_for_radar(raw_values[i, j], j))
        
        # Close the loop
        display_values_closed = display_values + [display_values[0]]
        
        radar_ax.plot(angles, display_values_closed, linewidth=2, label=robot)
        radar_ax.fill(angles, display_values_closed, alpha=0.1)
    
    # Set metric labels with bold text
    radar_ax.set_xticks(angles[:-1])
    radar_ax.set_xticklabels([m[0] for m in metrics], fontweight='bold', fontsize=14)
    
    # Add legend
    radar_ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set title
    plt.title("Robot Error Metrics Comparison)", fontsize=22, fontweight='bold', y=1.1)
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "robot_comparison_radar.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    
    # Create a summary bar chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot position error
    pos_data = robot_stats["Err. Position", "mean"].reset_index()
    pos_data.columns = ["Robot", "Mean Position Error (mm)"]
    sns.barplot(x="Robot", y="Mean Position Error (mm)", hue="Robot", data=pos_data, ax=ax1, palette="Blues", legend=False)
    ax1.set_title("Position Error Comparison")
    
    # Plot rotation error
    rot_data = robot_stats["Err. Rotation", "mean"].reset_index()
    rot_data.columns = ["Robot", "Mean Rotation Error (deg)"]
    sns.barplot(x="Robot", y="Mean Rotation Error (deg)", hue="Robot", data=rot_data, ax=ax2, palette="Oranges", legend=False)
    ax2.set_title("Rotation Error Comparison")
    
    # Plot solution time
    time_data = robot_stats["Sol. Time", "mean"].reset_index()
    time_data.columns = ["Robot", "Mean Solution Time (ms)"]
    sns.barplot(x="Robot", y="Mean Solution Time (ms)", hue="Robot", data=time_data, ax=ax3, palette="Greens", legend=False)
    ax3.set_title("Solution Time Comparison")
    
    # Apply styling
    set_figure_style(fig)
    plt.tight_layout()
    
    # Save the bar charts (PNG only)
    plt.savefig(
        os.path.join(results_dir, "robot_comparison_bars.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def main(args):
    # Create output directory if it doesn't exist
    results_dir = f"{sys.path[0]}/results/{args.id}/"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading data from {results_dir}/results.pkl")
    # Load the pickle file
    data = pd.read_pickle(f"{results_dir}/results.pkl")
    stats = data.reset_index()
    
    # Process error metrics
    stats["Err. Position"] = stats["Err. Position"]*1000  # Convert to mm
    stats["Err. Rotation"] = stats["Err. Rotation"]*(180/np.pi)  # Convert to degrees
    
    # Filter outliers if needed
    if args.filter_outliers:
        q_pos = stats["Err. Position"].quantile(0.99)
        q_rot = stats["Err. Rotation"].quantile(0.99)
        stats = stats.drop(stats[stats["Err. Position"] > q_pos].index)
        stats = stats.drop(stats[stats["Err. Rotation"] > q_rot].index)
        print(f"Filtered outliers: position > {q_pos:.2f}mm, rotation > {q_rot:.2f}deg")
    
    print("Generating advanced visualizations...")
    
    # Create directory for advanced plots
    advanced_dir = os.path.join(results_dir, "advanced_analysis")
    os.makedirs(advanced_dir, exist_ok=True)
    
    # Generate all plots
    print("1. Creating error distribution plots...")
    plot_error_distribution(stats, advanced_dir)
    
    print("2. Creating error relationship plots...")
    plot_error_relationship(stats, advanced_dir)
    
    print("3. Creating solution time vs accuracy plots...")
    plot_solution_time_vs_accuracy(stats, advanced_dir)
    
    print("4. Creating joint configuration analysis...")
    plot_joint_configuration_analysis(stats, advanced_dir)
    
    print("5. Creating error vs joints plots...")
    plot_error_vs_joints(stats, advanced_dir)
    
    print("6. Creating comparative summary...")
    plot_comparative_summary(stats, advanced_dir)
    
    print(f"All advanced visualizations saved to {advanced_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # General settings
    parser.add_argument("--id", type=str, default="test_experiment", help="Name of the folder with experiment data")
    parser.add_argument("--filter_outliers", type=bool, default=True, help="Filter outlier data points")
    
    args = parser.parse_args()
    main(args)