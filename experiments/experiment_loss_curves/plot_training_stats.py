#!/usr/bin/env python3
import os
import argparse
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Plot training statistics from TensorBoard event files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--script_path', type=str, default=None, help='Path to save the plots (defaults to current directory)')
    return parser.parse_args()

def find_event_files(model_path):
    """Find all event files in subdirectories of model_path."""
    model_dirs = {}
    for root, dirs, files in os.walk(model_path):
        event_files = glob.glob(os.path.join(root, "events.out.tfevents.*"))
        if event_files:
            # Use the last directory name as the model name
            model_name = os.path.basename(root)
            model_dirs[model_name] = event_files
    return model_dirs

def read_tensorboard_data(event_file):
    """Read TensorBoard event file and extract metrics."""
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # Get available tags (metrics)
    tags = ea.Tags()['scalars']
    
    print(f"Available tags in {os.path.basename(event_file)}:")
    for tag in tags:
        print(f"  - {tag}")
    
    # Extract data for each tag
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [event.step for event in events],
            'values': [event.value for event in events],
            'wall_time': [event.wall_time for event in events]
        }
    
    return data

def format_title(model_name):
    """Format model name for plot title."""
    # Replace underscores and hyphens with spaces
    title = model_name.replace('_', ' ').replace('-', ' ')
    
    # Capitalize each word
    title = ' '.join(word.capitalize() for word in title.split())
    
    # Remove the last word if it's "Model"
    if title.endswith(" Model"):
        title = title[:-6]
        
    return title

def plot_metrics(model_data, model_name, output_dir):
    """Plot training and validation metrics for a model as subplots."""
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
    
    # Define metrics to plot
    metrics = {
        'rec_pos_l': 'Pose Loss',
        'kl_l': 'KL Divergence Loss',
        'total_l': 'Total Loss'
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find train and validation metrics
    train_metrics = {k: v for k, v in model_data.items() if 'train' in k}
    val_metrics = {k: v for k, v in model_data.items() if 'val' in k}
    
    # Debug: Print available metrics to identify naming patterns
    print(f"Available metrics for {model_name}:")
    for k in model_data.keys():
        print(f"  - {k}")
    
    # Format title
    formatted_title = format_title(model_name)
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(formatted_title, fontsize=20, fontweight='bold')
    
    # Store final loss values
    final_losses = {
        'training': {},
        'validation': {}
    }
    
    # Plot each metric in its own subplot
    for i, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Find the train and val metrics for this key
        train_key = next((k for k in train_metrics if metric_key in k), None)
        val_key = next((k for k in val_metrics if metric_key in k), None)
        
        # Debug: print what we found for this metric
        print(f"For {metric_name}:")
        print(f"  - Training key: {train_key}")
        print(f"  - Validation key: {val_key}")
        
        if train_key:
            # Convert steps to epochs (steps are 1-based from TensorBoard)
            epochs = [int(step) for step in model_data[train_key]['steps']]
            values = model_data[train_key]['values']
            
            # Debug: print range of values
            if values:
                print(f"  - Training value range: {min(values):.4f} to {max(values):.4f}")
                print(f"  - Final training value: {values[-1]:.4f}")
            
            ax.plot(epochs, values, label='Training')
            
            # Store final value (last value in the sequence)
            if values:
                final_losses['training'][metric_key] = values[-1]
        
        if val_key:
            epochs = [int(step) for step in model_data[val_key]['steps']]
            values = model_data[val_key]['values']
            
            # Debug: print range of values
            if values:
                print(f"  - Validation value range: {min(values):.4f} to {max(values):.4f}")
                print(f"  - Final validation value: {values[-1]:.4f}")
            
            ax.plot(epochs, values, label='Validation')
            
            # Store final value (last value in the sequence)
            if values:
                final_losses['validation'][metric_key] = values[-1]
        
        ax.set_title(metric_name)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set integer ticks on x-axis
        if len(epochs) > 0:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"{model_name}_training_metrics.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)
    
    print(f"Plot saved: {os.path.join(output_dir, filename)}")
    
    # Return training time information for the summary plot
    training_time = None
    total_loss_key = next((k for k in train_metrics if 'total_l' in k), None)
    if total_loss_key and len(model_data[total_loss_key]['wall_time']) >= 2:
        # Calculate total training time in hours
        start_time = model_data[total_loss_key]['wall_time'][0]
        end_time = model_data[total_loss_key]['wall_time'][-1]
        training_time = (end_time - start_time) / 3600  # Convert to hours
    
    # Debug: Print final losses for this model
    print("Final loss values:")
    for phase in ['training', 'validation']:
        print(f"  {phase.capitalize()}:")
        for metric_key, metric_name in metrics.items():
            value = final_losses.get(phase, {}).get(metric_key, None)
            if value is not None:
                print(f"    - {metric_name}: {value:.4f}")
            else:
                print(f"    - {metric_name}: N/A")
    
    return {
        'model_name': model_name, 
        'formatted_title': formatted_title,
        'training_time': training_time,
        'final_losses': final_losses
    }

def plot_training_time_summary(model_stats, output_dir):
    """Plot a summary of training times for all models."""
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
    
    # Sort models by training time
    model_stats.sort(key=lambda x: x['training_time'] if x['training_time'] is not None else 0)
    
    # Create figure for the bar plot
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    model_names = [stat['formatted_title'] for stat in model_stats]
    training_times = [stat['training_time'] for stat in model_stats]
    
    # Create bar plot
    bars = plt.bar(model_names, training_times, color='cornflowerblue')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Training Time (hours)')
    plt.title('Training Time Comparison', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid and adjust layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    filename = "training_time_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    print(f"Training time comparison plot saved: {os.path.join(output_dir, filename)}")

def generate_latex_table(model_stats, output_dir):
    """Generate a LaTeX table with final loss values for all models."""
    # Define metrics to include in table
    metrics = {
        'rec_pos_l': 'Pose Loss',
        'kl_l': 'KL Divergence Loss',
        'total_l': 'Total Loss'
    }
    
    # Start building the LaTeX table content
    latex_content = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Final Loss Values for All Models}",
        "\\label{tab:final_loss_values}",
        "\\begin{tabular}{l|cccccc}",
        "\\hline",
        "\\multirow{2}{*}{Model} & \\multicolumn{3}{c}{Training} & \\multicolumn{3}{c}{Validation} \\\\",
        "\\cline{2-7}",
        " & Pose Loss & KL Loss & Total Loss & Pose Loss & KL Loss & Total Loss \\\\",
        "\\hline"
    ]
    
    # Add data rows for each model
    for stat in model_stats:
        model_name = stat['formatted_title']
        final_losses = stat.get('final_losses', {'training': {}, 'validation': {}})
        
        # Debug print for table generation
        print(f"Adding to table - Model: {model_name}")
        print(f"Final losses data: {final_losses}")
        
        # Format values for each cell, using N/A if not available
        row_values = []
        for phase in ['training', 'validation']:
            for metric_key in metrics.keys():
                value = final_losses.get(phase, {}).get(metric_key, None)
                if value is not None:
                    # For KL Loss, we may need to handle different scales
                    if metric_key == 'kl_l' and value > 1000:
                        # Format with scientific notation for very large values
                        row_values.append(f"{value:.2e}")
                    else:
                        row_values.append(f"{value:.4f}")
                else:
                    row_values.append("N/A")
        
        # Add the row to the table
        latex_content.append(f"{model_name} & " + " & ".join(row_values) + " \\\\")
    
    # Complete the table
    latex_content.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    # Write to file
    table_filename = os.path.join(output_dir, "final_loss_values_table.tex")
    with open(table_filename, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX table with final loss values saved: {table_filename}")

def main():
    args = parse_args()
    
    # Find event files in model directories
    model_dirs = find_event_files(args.model_path)
    
    if not model_dirs:
        print(f"No TensorBoard event files found in {args.model_path}")
        return
    
    print(f"Found {len(model_dirs)} models with TensorBoard event files")
    
    # Determine output directory for plots
    if args.script_path:
        output_dir = os.path.join(args.script_path, "results")
    else:
        output_dir = os.path.join(os.getcwd(), "results")
        
    print(f"Plots will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model and collect training time data
    model_stats = []
    
    for model_name, event_files in model_dirs.items():
        print(f"\nProcessing model: {model_name}")
        
        # Process all event files for this model (there might be multiple)
        all_data = {}
        for event_file in event_files:
            print(f"  Reading event file: {os.path.basename(event_file)}")
            data = read_tensorboard_data(event_file)
            
            # Merge data from this file
            for tag, values in data.items():
                if tag not in all_data:
                    all_data[tag] = values
                else:
                    # If the tag already exists, append new data
                    # (assuming steps are different)
                    all_data[tag]['steps'].extend(values['steps'])
                    all_data[tag]['values'].extend(values['values'])
                    all_data[tag]['wall_time'].extend(values['wall_time'])
                    
                    # Sort by steps to maintain order
                    sorted_indices = np.argsort(all_data[tag]['steps'])
                    all_data[tag]['steps'] = [all_data[tag]['steps'][i] for i in sorted_indices]
                    all_data[tag]['values'] = [all_data[tag]['values'][i] for i in sorted_indices]
                    all_data[tag]['wall_time'] = [all_data[tag]['wall_time'][i] for i in sorted_indices]
        
        # Plot metrics for this model - save directly to output_dir
        stats = plot_metrics(all_data, model_name, output_dir)
        model_stats.append(stats)
    
    # Plot training time summary if we have data
    if model_stats:
        plot_training_time_summary(model_stats, output_dir)
        # Generate and save the LaTeX table with final loss values
        generate_latex_table(model_stats, output_dir)
    
    print(f"\nAll plots and LaTeX table have been saved to {output_dir}")

if __name__ == "__main__":
    main()
