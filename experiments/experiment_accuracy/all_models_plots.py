def generate_latex_table(data):
    """Generate a LaTeX table of statistics for each model averaged across all robots"""
    # Group by Model and calculate statistics
    models = data['Model'].unique()
    
    # Create a DataFrame to store the results
    results = pd.DataFrame(index=models, columns=[
        'Pos_mean', 'Pos_min', 'Pos_max', 'Pos_Q1', 'Pos_Q3',
        'Rot_mean', 'Rot_min', 'Rot_max', 'Rot_Q1', 'Rot_Q3'
    ])
    
    # Calculate statistics for each model
    for model in models:
        model_data = data[data['Model'] == model]
        
        # Position error statistics
        results.loc[model, 'Pos_mean'] = model_data['Err. Position (mm)'].mean()
        results.loc[model, 'Pos_min'] = model_data['Err. Position (mm)'].min()
        results.loc[model, 'Pos_max'] = model_data['Err. Position (mm)'].max()
        results.loc[model, 'Pos_Q1'] = model_data['Err. Position (mm)'].quantile(0.25)
        results.loc[model, 'Pos_Q3'] = model_data['Err. Position (mm)'].quantile(0.75)
        
        # Rotation error statistics
        results.loc[model, 'Rot_mean'] = model_data['Err. Rotation (deg)'].mean()
        results.loc[model, 'Rot_min'] = model_data['Err. Rotation (deg)'].min()
        results.loc[model, 'Rot_max'] = model_data['Err. Rotation (deg)'].max()
        results.loc[model, 'Rot_Q1'] = model_data['Err. Rotation (deg)'].quantile(0.25)
        results.loc[model, 'Rot_Q3'] = model_data['Err. Rotation (deg)'].quantile(0.75)
    
    # Sort results by model name alphabetically
    results = results.sort_index()
    
    # Generate LaTeX table
    latex_table = "\\begin{tabular}{lrrrrrrrrrr}\n"
    latex_table += "\\toprule\n"
    latex_table += " & \\multicolumn{5}{c}{Err. Pos. [mm]} & \\multicolumn{5}{c}{Err. Rot. [deg]} \\\\\n"
    latex_table += " & mean & min & max & Q$_{1}$ & Q$_{3}$ & mean & min & max & Q$_{1}$ & Q$_{3}$ \\\\\n"
    latex_table += "Model &  &  &  &  &  &  &  &  &  & \\\\\n"
    latex_table += "\\midrule\n"
    
    # Add each model's data as a row
    for model in results.index:
        row = f"{model} & "
        row += f"{results.loc[model, 'Pos_mean']:.1f} & "
        row += f"{results.loc[model, 'Pos_min']:.1f} & "
        row += f"{results.loc[model, 'Pos_max']:.1f} & "
        row += f"{results.loc[model, 'Pos_Q1']:.1f} & "
        row += f"{results.loc[model, 'Pos_Q3']:.1f} & "
        row += f"{results.loc[model, 'Rot_mean']:.1f} & "
        row += f"{results.loc[model, 'Rot_min']:.1f} & "
        row += f"{results.loc[model, 'Rot_max']:.1f} & "
        row += f"{results.loc[model, 'Rot_Q1']:.1f} & "
        row += f"{results.loc[model, 'Rot_Q3']:.1f} \\\\\n"
        latex_table += row
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"
    
    # Save the LaTeX table to a file
    os.makedirs('results/tables', exist_ok=True)
    with open('results/tables/model_statistics.tex', 'w') as f:
        f.write(latex_table)
    
    print("LaTeX table saved to 'results/tables/model_statistics.tex'")
    
    return latex_table#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path

def main():
    # Set up better styling for plots with increased font sizes
    plt.style.use('seaborn-v0_8-darkgrid')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16  # Increased from 14
    rcParams['axes.titlesize'] = 22  # Increased from 20
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.labelsize'] = 20  # Increased from 18
    rcParams['xtick.labelsize'] = 18  # Increased from 16
    rcParams['ytick.labelsize'] = 18  # Increased from 16
    rcParams['legend.fontsize'] = 16  # Increased from 14
    rcParams['figure.titlesize'] = 26  # Increased from 24
    rcParams['figure.titleweight'] = 'bold'
    
    # Find all results.pkl files in results directory and subdirectories
    results_files = glob.glob(os.path.join("results", "**", "results.pkl"), recursive=True)
    
    if not results_files:
        print("No results.pkl files found in ./results/**/results.pkl")
        return
    
    print(f"Found {len(results_files)} results.pkl files")
    
    # Load all data
    all_data = []
    for file_path in results_files:
        try:
            # Extract model name from the directory name
            full_model_name = os.path.basename(os.path.dirname(file_path))
            
            # Process the model name to keep only the first two terms and replace separators
            model_parts = full_model_name.split('-')
            if len(model_parts) >= 2:
                model_name = '-'.join(model_parts[:2])
            else:
                model_name = full_model_name
            
            # Replace underscores and hyphens with spaces
            model_name = model_name.replace('_', ' ').replace('-', ' ')
            
            # Load the data
            data = pd.read_pickle(file_path)
            
            # Add model name as a column
            data['Model'] = model_name
            
            # Append to our collected data
            all_data.append(data)
            
            print(f"Loaded data from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        print("No data could be loaded from the results files")
        return
    
    # Combine all dataframes
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Convert errors to appropriate units
    combined_data['Err. Position (mm)'] = combined_data['Err. Position'] * 1000  # Convert to mm
    combined_data['Err. Rotation (deg)'] = combined_data['Err. Rotation'] * 180 / np.pi  # Convert to degrees
    
    # Get unique robot types and models
    robot_types = combined_data['Robot'].unique()
    models = combined_data['Model'].unique()
    
    print(f"Robot types found: {robot_types}")
    print(f"Models found: {models}")
    
    # Create output directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    
    # Create the two consolidated plots
    create_consolidated_plots(combined_data, robot_types)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(combined_data)
    print("\nLaTeX Table Preview:")
    print(latex_table)

def create_consolidated_plots(data, robot_types):
    """Create two consolidated plots: one for position error and one for rotation error"""
    # Sort the data alphabetically by model
    data = data.sort_values(by=['Model'])
    
    # Create position error plot
    plt.figure(figsize=(20, 10))
    
    # Create boxplot with alphabetically ordered models
    position_ax = sns.boxplot(x='Robot', y='Err. Position (mm)', hue='Model', data=data, palette='Set3', order=sorted(robot_types))
    
    plt.title('Position Error Comparison Across Models and Robots', fontsize=24)
    plt.xlabel('Robot Type', fontsize=20)
    plt.ylabel('Position Error (mm)', fontsize=20)
    
    # Don't rotate x-axis labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Place the legend inside the plot with a box
    handles, labels = position_ax.get_legend_handles_labels()
    # Sort the legend labels alphabetically
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    
    position_ax.legend(sorted_handles, sorted_labels, title='Model', 
                      loc='upper right', frameon=True, fancybox=True, shadow=True,
                      title_fontsize=18)
    
    plt.tight_layout()
    plt.savefig('results/plots/position_error_comparison.png', dpi=300)
    plt.close()
    
    # Create rotation error plot
    plt.figure(figsize=(20, 10))
    
    # Create boxplot with alphabetically ordered models
    rotation_ax = sns.boxplot(x='Robot', y='Err. Rotation (deg)', hue='Model', data=data, palette='Set3', order=sorted(robot_types))
    
    plt.title('Rotation Error Comparison Across Models and Robots', fontsize=24)
    plt.xlabel('Robot Type', fontsize=20)
    plt.ylabel('Rotation Error (degrees)', fontsize=20)
    
    # Don't rotate x-axis labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Place the legend inside the plot with a box
    handles, labels = rotation_ax.get_legend_handles_labels()
    # Sort the legend labels alphabetically
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    
    rotation_ax.legend(sorted_handles, sorted_labels, title='Model', 
                      loc='upper right', frameon=True, fancybox=True, shadow=True,
                      title_fontsize=18)
    
    plt.tight_layout()
    plt.savefig('results/plots/rotation_error_comparison.png', dpi=300)
    plt.close()
    
    print("Consolidated plots saved to 'results/plots/' directory")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()