#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a summary table as PNG image from test results.
Run this after run_all_tests.sh completes.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
LOGS_DIR = Path("/home/hjh/Mangrove_segmentation/logs")
OUTPUT_DIR = Path("/home/hjh/Mangrove_segmentation/final_results")

# Model display names and encoder mappings
MODEL_DISPLAY_NAMES = {
    'UnetPlusPlus': 'UNet++',
    'MAnet': 'MAnet',
    'PAN': 'PAN',
    'Segformer': 'SegFormer',
    'FPN': 'FPN',
    'DPT': 'DPT',
    'UperNet': 'UPerNet',
}

ENCODER_DISPLAY_NAMES = {
    'resnet34': 'resnet34',
    'resnet50': 'resnet50',
    'tu-pvt_v2_b2': 'pvtv2-b2',
    'mit_b2': 'mit-b2',
    'tu-vit_base_patch16_224.augreg_in21k': 'vit-b16',
    'tu-swin_tiny_patch4_window7_224': 'swin-t',
}

# Model order for display
MODEL_ORDER = ['UNet++', 'MAnet', 'PAN', 'SegFormer', 'FPN', 'DPT', 'UPerNet']


def get_experiment_info(log_dir):
    """Extract experiment info from config."""
    config_path = log_dir / 'config.yaml'
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['model']['name']
    encoder_name = config['model']['args'].get('encoder_name', 'resnet34')
    
    # Determine selection method from directory name
    dir_name = log_dir.name
    if '_MVI_' in dir_name:
        selection_method = 'MVI-based'
    elif '_MF_' in dir_name:
        selection_method = 'MF-based'
    elif '_ACE_' in dir_name:
        selection_method = 'ACE-based'
    else:
        selection_method = 'Unknown'
    
    return {
        'model_name': model_name,
        'encoder_name': encoder_name,
        'selection_method': selection_method,
        'log_dir': str(log_dir),
    }


def collect_all_results():
    """Collect results from all experiments."""
    results = []
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()])
    
    for exp_dir in exp_dirs:
        # Get experiment info
        info = get_experiment_info(exp_dir)
        if info is None:
            continue
        
        # Check if results exist
        summary_path = exp_dir / 'results' / 'summary_results.csv'
        if not summary_path.exists():
            print(f"  Skipping {exp_dir.name} - no results found")
            continue
        
        # Load results
        summary_df = pd.read_csv(summary_path)
        mean_row = summary_df[summary_df['metric'] == 'mean'].iloc[0]
        
        # Get display names
        model_display = MODEL_DISPLAY_NAMES.get(info['model_name'], info['model_name'])
        encoder_display = ENCODER_DISPLAY_NAMES.get(info['encoder_name'], info['encoder_name'])
        
        results.append({
            'Selection Method': info['selection_method'],
            'Model': model_display,
            'Encoder': encoder_display,
            'IoU(%)': mean_row['iou'] * 100,
            'F1(%)': mean_row['f1'] * 100,
            'Pre(%)': mean_row['precision'] * 100,
            'Rec(%)': mean_row['recall'] * 100,
            'Acc(%)': mean_row['accuracy'] * 100,
        })
        print(f"  Loaded results from {exp_dir.name}")
    
    return pd.DataFrame(results)


def create_table_image(df, output_path, title="Segmentation Results"):
    """Create a publication-quality table image matching the reference style."""
    
    if df.empty:
        print("No results to display")
        return
    
    # Sort by Selection Method and Model
    df = df.copy()
    
    # Define sort order
    method_order = {'MVI-based': 0, 'MF-based': 1, 'ACE-based': 2}
    model_order = {m: i for i, m in enumerate(MODEL_ORDER)}
    
    df['method_order'] = df['Selection Method'].map(lambda x: method_order.get(x, 99))
    df['model_order'] = df['Model'].map(lambda x: model_order.get(x, 99))
    df = df.sort_values(['method_order', 'model_order'])
    df = df.drop(columns=['method_order', 'model_order'])
    
    # Find best and second best values for each metric within each selection method
    metrics = ['IoU(%)', 'F1(%)', 'Pre(%)', 'Rec(%)', 'Acc(%)']
    best_values = {}
    second_best_values = {}
    
    for method in df['Selection Method'].unique():
        method_df = df[df['Selection Method'] == method]
        best_values[method] = {}
        second_best_values[method] = {}
        for metric in metrics:
            sorted_vals = method_df[metric].sort_values(ascending=False)
            if len(sorted_vals) >= 1:
                best_values[method][metric] = sorted_vals.iloc[0]
            if len(sorted_vals) >= 2:
                second_best_values[method][metric] = sorted_vals.iloc[1]
    
    # Create figure with proper sizing
    n_rows = len(df)
    fig_height = max(4, n_rows * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    ax.axis('tight')
    
    # Prepare cell text and formatting
    columns = df.columns.tolist()
    cell_text = []
    
    # Track selection method groups
    prev_method = None
    method_start_rows = {}
    
    for idx, (_, row) in enumerate(df.iterrows()):
        row_text = []
        method = row['Selection Method']
        
        if method != prev_method:
            method_start_rows[method] = idx
        
        for col in columns:
            val = row[col]
            
            if col in metrics:
                # Format numeric values
                is_best = abs(val - best_values[method].get(col, -999)) < 0.01
                is_second = abs(val - second_best_values[method].get(col, -999)) < 0.01
                
                if is_best:
                    text = f"**{val:.2f}**"  # Will be bolded
                elif is_second:
                    text = f"__{val:.2f}__"  # Will be underlined
                else:
                    text = f"{val:.2f}"
                row_text.append(text)
            elif col == 'Selection Method':
                # Only show method name for first row of each group
                if method != prev_method:
                    row_text.append(val)
                else:
                    row_text.append('')
            else:
                row_text.append(str(val))
        
        cell_text.append(row_text)
        prev_method = method
    
    # Create the table
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    
    # Style header row
    for j, col in enumerate(columns):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        cell.set_height(0.08)
    
    # Style data cells
    prev_method = None
    for i, (_, row) in enumerate(df.iterrows()):
        method = row['Selection Method']
        
        for j, col in enumerate(columns):
            cell = table[(i + 1, j)]
            cell_text_val = cell_text[i][j]
            
            # Background colors
            if col == 'Selection Method':
                cell.set_facecolor('#ecf0f1')
            elif col in metrics:
                if '**' in cell_text_val:
                    cell.set_facecolor('#d5f5e3')  # Light green for best
                elif '__' in cell_text_val:
                    cell.set_facecolor('#ebf5fb')  # Light blue for second
                else:
                    cell.set_facecolor('white')
            else:
                cell.set_facecolor('white')
            
            # Format text (remove markers and apply styling)
            if '**' in cell_text_val:
                clean_text = cell_text_val.replace('**', '')
                cell.get_text().set_text(clean_text)
                cell.get_text().set_fontweight('bold')
            elif '__' in cell_text_val:
                clean_text = cell_text_val.replace('__', '')
                cell.get_text().set_text(clean_text)
                # Underline effect - matplotlib doesn't support underline directly
                # Use italic as alternative visual indicator
                cell.get_text().set_style('italic')
            
            # Add horizontal line between method groups
            if method != prev_method and i > 0:
                cell.set_linewidth(2)
                cell.visible_edges = 'BTLR'
        
        prev_method = method
    
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nTable saved to: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("Generating Results Table")
    print("="*60)
    
    # Collect all results
    print("\nCollecting results from all experiments...")
    results_df = collect_all_results()
    
    if results_df.empty:
        print("\nNo results found!")
        print("Please run 'bash run_all_tests.sh' first to generate test results.")
        return
    
    # Save raw results to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / 'all_experiments_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to: {csv_path}")
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Format for display
    display_df = results_df.copy()
    for col in ['IoU(%)', 'F1(%)', 'Pre(%)', 'Rec(%)', 'Acc(%)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    print(display_df.to_string(index=False))
    
    # Create table image
    table_path = OUTPUT_DIR / 'experiments_comparison_table.png'
    create_table_image(results_df, table_path)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()

