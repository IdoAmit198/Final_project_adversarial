#!/usr/bin/env python3
"""
Script to generate improved similarity plots from existing CSV files.
This script reads the CSV files created by the perturbations_similarity function
and creates high-quality visualization plots.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_matrix_from_csv(csv_path):
    """
    Load similarity matrix from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        tuple: (matrix, epsilon_labels)
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # First row contains epsilon labels (skip first empty cell)
    epsilon_labels = [float(x) for x in rows[0][1:]]
    
    # Extract the matrix values
    matrix = []
    for row in rows[1:]:
        matrix.append([float(x) for x in row[1:]])
    
    matrix = np.array(matrix)
    
    # Skip epsilon 0 (first column and row) if it exists
    if len(epsilon_labels) > 0 and epsilon_labels[0] == 0.0:
        epsilon_labels = epsilon_labels[1:]  # Remove first epsilon (0)
        matrix = matrix[1:, 1:]  # Remove first row and column
    
    return matrix, epsilon_labels


def create_improved_plot(matrix, epsilon_labels, metric_name, output_path, normalized=False):
    """
    Create an improved visualization of the similarity matrix.
    
    Args:
        matrix (np.ndarray): Similarity matrix
        epsilon_labels (list): List of epsilon values
        metric_name (str): Name of the metric (FID, Cosine, MSE)
        output_path (str): Path to save the plot
        normalized (bool): Whether this is a normalized matrix
    """
    # Make matrix an upper diagonal matrix
    matrix_plot = np.triu(matrix, k=1)
    # Where the matrix is zero, fill with NaN for better visualization
    matrix_plot[matrix_plot == 0] = np.nan
    
    # Create larger figure with better spacing - increased size for better text visibility
    plt.figure(figsize=(16, 12))
    im = plt.imshow(matrix_plot, cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(f'{metric_name} Score', fontsize=14, fontweight='bold')
    
    # Determine font size based on matrix size and maximum value
    max_val = np.nanmax(matrix_plot)
    if max_val > 100:
        font_size = 8  # Smaller font for large values
    elif max_val > 10:
        font_size = 9
    else:
        font_size = 11
    
    # Add text annotations with adaptive formatting
    for (i, j), value in np.ndenumerate(matrix_plot):
        if i >= j or np.isnan(value):
            continue
        # Use white text for dark backgrounds, black for light backgrounds
        text_color = 'white' if value < (np.nanmax(matrix_plot) + np.nanmin(matrix_plot)) / 2 else 'black'
        
        # Format the value text based on magnitude
        if value >= 100:
            text = f'{value:.1f}'  # 1 decimal place for large values
        else:
            text = f'{value:.2f}'  # 2 decimal places for smaller values
            
        plt.text(j, i, text, ha='center', va='center', 
                color=text_color, fontsize=font_size, fontweight='bold')
    
    # Set title and labels
    title_suffix = "Normalized " if normalized else ""
    plt.title(f'Perturbations Similarity {title_suffix}({metric_name})', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epsilon', fontsize=16, fontweight='bold')
    plt.ylabel('Epsilon', fontsize=16, fontweight='bold')
    
    # Format epsilon labels for better readability
    formatted_labels = [f'{eps:.3f}' if eps < 1 else f'{int(eps)}' for eps in epsilon_labels]
    plt.xticks(ticks=np.arange(len(epsilon_labels)), labels=formatted_labels, 
               fontsize=14, rotation=45)
    plt.yticks(ticks=np.arange(len(epsilon_labels)), labels=formatted_labels, fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def plot_from_csv_files(base_dir, dataset, output_dir=None):
    """
    Generate plots from all CSV files in the specified directory.
    
    Args:
        base_dir (str): Base directory containing the CSV files
        dataset (str): Dataset name (e.g., 'imagenet', 'cifar10')
        output_dir (str): Optional custom output directory
    """
    if output_dir is None:
        output_dir = f"plots/similarity/{dataset}"
    
    # Define the directories and metrics
    directories = ['non-normalized', 'normalized']
    metrics = ['FID', 'Cosine', 'MSE']
    
    for directory in directories:
        csv_dir = os.path.join(base_dir, dataset, directory)
        plot_dir = os.path.join(output_dir, directory)
        
        # Create output directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        
        if not os.path.exists(csv_dir):
            print(f"Directory {csv_dir} does not exist, skipping...")
            continue
        
        for metric in metrics:
            csv_file = f"perturbations_{metric}_matrix.csv"
            csv_path = os.path.join(csv_dir, csv_file)
            
            if not os.path.exists(csv_path):
                print(f"CSV file {csv_path} does not exist, skipping...")
                continue
            
            try:
                # Load matrix from CSV
                matrix, epsilon_labels = load_matrix_from_csv(csv_path)
                
                # Create improved plot
                output_filename = f"perturbations_similarity_{metric}"
                if directory == 'normalized':
                    output_filename += "_normalized"
                output_filename += ".png"
                
                output_path = os.path.join(plot_dir, output_filename)
                
                create_improved_plot(
                    matrix, epsilon_labels, metric.upper(), output_path, 
                    normalized=(directory == 'normalized')
                )
                
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")


def create_comparison_plot(base_dir, dataset, output_dir=None):
    """
    Create a comparison plot showing all metrics side by side.
    
    Args:
        base_dir (str): Base directory containing the CSV files
        dataset (str): Dataset name
        output_dir (str): Optional custom output directory
    """
    if output_dir is None:
        output_dir = f"plots/similarity/{dataset}"
    
    metrics = ['fid', 'cosine', 'mse']
    directories = ['non-normalized', 'normalized']
    
    for directory in directories:
        csv_dir = os.path.join(base_dir, dataset, directory)
        
        if not os.path.exists(csv_dir):
            continue
        
        # Create subplot figure - increased size for better visibility
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Perturbations Similarity Comparison {"(Normalized)" if directory == "normalized" else ""}', 
                     fontsize=22, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            csv_file = f"perturbations_{metric}_matrix.csv"
            csv_path = os.path.join(csv_dir, csv_file)
            
            if not os.path.exists(csv_path):
                continue
            
            try:
                matrix, epsilon_labels = load_matrix_from_csv(csv_path)
                matrix_plot = np.triu(matrix, k=1)
                matrix_plot[matrix_plot == 0] = np.nan
                
                ax = axes[idx]
                im = ax.imshow(matrix_plot, cmap='viridis', interpolation='nearest')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.ax.tick_params(labelsize=12)
                
                # Determine font size based on maximum value
                max_val = np.nanmax(matrix_plot)
                if max_val > 100:
                    font_size = 7  # Smaller font for large values in comparison
                elif max_val > 10:
                    font_size = 8
                else:
                    font_size = 9
                
                # Add text annotations
                for (i, j), value in np.ndenumerate(matrix_plot):
                    if i >= j or np.isnan(value):
                        continue
                    text_color = 'white' if value < (np.nanmax(matrix_plot) + np.nanmin(matrix_plot)) / 2 else 'black'
                    
                    # Format the value text based on magnitude
                    if value >= 100:
                        text = f'{value:.1f}'  # 1 decimal place for large values
                    else:
                        text = f'{value:.2f}'  # 2 decimal places for smaller values
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           color=text_color, fontsize=font_size, fontweight='bold')
                
                ax.set_title(f'{metric.upper()}', fontsize=16, fontweight='bold')
                ax.set_xlabel('Epsilon', fontsize=14)
                ax.set_ylabel('Epsilon', fontsize=14)
                
                formatted_labels = [f'{eps:.3f}' if eps < 1 else f'{int(eps)}' for eps in epsilon_labels]
                ax.set_xticks(np.arange(len(epsilon_labels)))
                ax.set_xticklabels(formatted_labels, fontsize=12, rotation=45)
                ax.set_yticks(np.arange(len(epsilon_labels)))
                ax.set_yticklabels(formatted_labels, fontsize=12)
                
            except Exception as e:
                print(f"Error processing {metric} for comparison plot: {e}")
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, directory, f"perturbations_similarity_comparison{'_normalized' if directory == 'normalized' else ''}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate improved similarity plots from CSV files')
    parser.add_argument('--base_dir', type=str, default='plots/similarity', 
                       help='Base directory containing CSV files (default: plots/similarity)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., imagenet, cifar10)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Custom output directory (default: plots/similarity/{dataset})')
    parser.add_argument('--comparison', action='store_true',
                       help='Also create comparison plots showing all metrics side by side')
    
    args = parser.parse_args()
    
    print(f"Generating plots for dataset: {args.dataset}")
    print(f"Reading CSV files from: {args.base_dir}")
    
    # Generate individual plots
    plot_from_csv_files(args.base_dir, args.dataset, args.output_dir)
    
    # Generate comparison plots if requested
    if args.comparison:
        print("Creating comparison plots...")
        create_comparison_plot(args.base_dir, args.dataset, args.output_dir)
    
    print("Plot generation completed!")


if __name__ == "__main__":
    main()
