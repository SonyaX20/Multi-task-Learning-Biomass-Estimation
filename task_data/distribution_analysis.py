"""
Distribution Analysis with Threshold Detection

Analyzes value distributions and finds threshold based on log10 histogram.
Only performs visualization and threshold detection, no data modification.
"""

import sys
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'task_gen_chm' / 'chm_stacked_treesat_new'
BAND_MAP = {'chm': 0, 'vv': 1, 'vh': 2, 'vv/vh': 3, 'vv/vh_new': 4}

def collect_band_values(band_idx, sample_size=None):
    """
    Collect all values from specified band across all files.
    
    Args:
        band_idx: Band index (0-3)
        sample_size: Number of values to sample per file (None = all)
    
    Returns:
        numpy array of all values
    """
    tif_files = sorted(list(DATA_DIR.glob('*.tif')))
    
    if not tif_files:
        print(f"No TIF files found in {DATA_DIR}")
        return None
    
    print(f"Collecting values from {len(tif_files)} files...")
    
    all_values = []
    
    for file_path in tqdm(tif_files, desc="Reading files"):
        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                
                if data.shape[0] <= band_idx:
                    continue
                
                band_data = data[band_idx].astype(np.float32).flatten()
                
                # Remove NaN, Inf, and -9999 values
                # For VV, VH, and VV/VH_new bands, exclude -9999 from statistics
                if band_idx in [1, 2, 4]:  # VV, VH, and VV/VH_new bands
                    valid_mask = np.isfinite(band_data) & (band_data != -9999.0)
                else:
                    valid_mask = np.isfinite(band_data)
                
                valid_data = band_data[valid_mask]
                
                if len(valid_data) > 0:
                    # Sample if requested
                    if sample_size and len(valid_data) > sample_size:
                        valid_data = np.random.choice(valid_data, sample_size, replace=False)
                    
                    all_values.extend(valid_data)
        
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    return np.array(all_values) if all_values else None

def find_threshold_from_log_histogram(values, num_bins=500):
    """
    Find threshold based on log10 histogram where frequency first drops below 10^2.
    Only for CHM band.
    
    Args:
        values: Array of values
        num_bins: Number of histogram bins
    
    Returns:
        threshold value, counts, bin_edges
    """
    if values is None or len(values) == 0:
        return None, None, None
    
    # Create histogram
    counts, bin_edges = np.histogram(values, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find first bin where count < 10^2 (100)
    threshold_idx = None
    for i, count in enumerate(counts):
        if count < 100:
            threshold_idx = i
            break
    
    if threshold_idx is not None:
        threshold = bin_edges[threshold_idx]
    else:
        threshold = values.max()  # No threshold found, use max value
    
    return threshold, counts, bin_edges

def get_optimal_bins_and_range(values, band_name):
    """
    Get optimal bin count and range for different bands.
    
    Args:
        values: Array of values
        band_name: Name of the band
    
    Returns:
        num_bins, x_range, y_range
    """
    if values is None or len(values) == 0:
        return 100, (0, 1), (0, 1)
    
    # Calculate data range
    data_min, data_max = values.min(), values.max()
    data_range = data_max - data_min
    
    # Band-specific settings
    if band_name.lower() == 'chm':
        # CHM: Tree height, typically 0-100m
        num_bins = min(200, max(50, int(data_range / 0.5)))  # 0.5m bins
        x_range = (0, min(100, data_max * 1.1))  # Focus on 0-100m range
        y_range = (0, None)  # Auto y-range
        
    elif band_name.lower() in ['vv', 'vh']:
        # VV/VH: SAR backscatter, typically -30 to 10 dB
        num_bins = min(200, max(100, int(data_range / 0.1)))  # 0.1 dB bins for higher precision
        x_range = (data_min * 1.05, data_max * 1.05)  # Include full range including negatives
        y_range = (0, None)  # Auto y-range
        
    elif band_name.lower() in ['vv/vh', 'vv/vh_new']:
        # VV/VH ratio: can have negative values, need higher precision
        num_bins = min(300, max(100, int(data_range / 0.05)))  # 0.05 ratio bins for higher precision
        x_range = (data_min * 1.1, data_max * 1.1)  # Include negative values
        y_range = (0, None)  # Auto y-range
        
    else:
        # Default settings
        num_bins = 100
        x_range = (data_min * 0.95, data_max * 1.05)
        y_range = (0, None)
    
    return num_bins, x_range, y_range

def plot_histogram_with_threshold(values, band_name, threshold, counts, bin_edges):
    """Plot histogram with optimal settings for each band."""
    # Get optimal settings for this band
    num_bins, x_range, y_range = get_optimal_bins_and_range(values, band_name)
    
    # Create new histogram with optimal bins
    optimal_counts, optimal_bin_edges = np.histogram(values, bins=num_bins, range=x_range)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Linear scale histogram
    ax1.hist(values, bins=optimal_bin_edges, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_title(f'{band_name.upper()} - Value Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_range)
    
    # Add zero line for reference if data spans negative values
    if x_range[0] < 0:
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Draw threshold line (only for CHM)
    if threshold is not None and band_name.lower() == 'chm':
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
        ax1.legend()
    
    # Plot 2: Log scale histogram
    ax2.hist(values, bins=optimal_bin_edges, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_title(f'{band_name.upper()} - Value Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_ylabel('Frequency (log scale)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_range)
    
    # Add zero line for reference if data spans negative values
    if x_range[0] < 0:
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Draw threshold line in log plot (only for CHM)
    if threshold is not None and band_name.lower() == 'chm':
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        ax2.axhline(y=100, color='orange', linestyle=':', alpha=0.7, label='10^2 reference')
        ax2.legend()
    
    # Add statistics
    stats_text = f"Total values: {len(values):,}\n"
    stats_text += f"Min: {values.min():.4f}\n"
    stats_text += f"Max: {values.max():.4f}\n"
    stats_text += f"Mean: {values.mean():.4f}\n"
    if threshold is not None and band_name.lower() == 'chm':
        stats_text += f"Threshold: {threshold:.4f}\n"
        values_above = np.sum(values > threshold)
        stats_text += f"Values > threshold: {values_above:,} ({values_above/len(values)*100:.2f}%)"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Fix filename for bands with special characters
    safe_filename = band_name.replace('/', '_').replace('\\', '_')
    output_filename = f'{safe_filename}_distribution.png'
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    plt.close()

def print_threshold_report(band_name, values, threshold):
    """Print threshold detection report (only for CHM)"""
    if values is None or len(values) == 0:
        print("No valid values found!")
        return
    
    print(f"\n{band_name.upper()} Statistics:")
    print(f"  Total values: {len(values):,}")
    print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"  Mean: {values.mean():.4f}")
    
    # Only show threshold for CHM
    if threshold is not None and band_name.lower() == 'chm':
        values_above = np.sum(values > threshold)
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Values > threshold: {values_above:,} ({values_above/len(values)*100:.2f}%)")
    elif band_name.lower() != 'chm':
        print("  (No threshold detection for this band)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python distribution_analysis.py <BAND_NAME>")
        print("Available bands: chm, vv, vh, vv/vh, vv/vh_new")
        sys.exit(1)
    
    band_name = sys.argv[1]
    
    if band_name not in BAND_MAP:
        print(f"Error: Unknown band '{band_name}'")
        print("Available bands: chm, vv, vh, vv/vh, vv/vh_new")
        sys.exit(1)
    
    band_idx = BAND_MAP[band_name]
    
    # Collect values
    values = collect_band_values(band_idx, sample_size=1000)
    
    if values is None or len(values) == 0:
        print("No valid values found!")
        return
    
    # Find threshold (only for CHM)
    if band_name.lower() == 'chm':
        threshold, counts, bin_edges = find_threshold_from_log_histogram(values, num_bins=500)
    else:
        threshold, counts, bin_edges = None, None, None
    
    # Print report
    print_threshold_report(band_name, values, threshold)
    
    # Plot histogram
    plot_histogram_with_threshold(values, band_name, threshold, counts, bin_edges)

if __name__ == '__main__':
    main()

