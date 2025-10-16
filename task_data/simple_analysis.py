"""
Simple analyzer for stacked CHM and Sentinel data.
Analyzes 4 bands: CHM, VV, VH, VV/VH ratio.
Reports: min, max, NaN ratio for each band.
"""

import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'task_gen_chm' / 'chm_stacked_treesat'
BAND_NAMES = ['CHM', 'VV', 'VH', 'VV/VH']

def analyze_files():
    """Analyze all stacked TIF files"""
    tif_files = sorted(list(DATA_DIR.glob('*.tif')))
    
    if not tif_files:
        print(f"No TIF files found in {DATA_DIR}")
        return
    
    print(f"Analyzing {len(tif_files)} files...\n")
    
    # Initialize statistics for each band
    band_stats = {i: {'mins': [], 'maxs': [], 'nan_ratios': []} for i in range(4)}
    
    # Process each file
    for file_path in tqdm(tif_files, desc="Processing"):
        try:
            with rasterio.open(file_path) as src:
                data = src.read()  # Shape: (bands, height, width)
                
                if data.shape[0] != 4:
                    continue
                
                # Analyze each band
                for band_idx in range(4):
                    band_data = data[band_idx].astype(np.float32)
                    
                    # Calculate NaN ratio
                    total = band_data.size
                    nan_count = np.isnan(band_data).sum()
                    nan_ratio = nan_count / total
                    
                    # Get valid data
                    valid_data = band_data[~np.isnan(band_data) & np.isfinite(band_data)]
                    
                    if len(valid_data) > 0:
                        band_stats[band_idx]['mins'].append(float(valid_data.min()))
                        band_stats[band_idx]['maxs'].append(float(valid_data.max()))
                    
                    band_stats[band_idx]['nan_ratios'].append(nan_ratio)
        
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    for band_idx, band_name in enumerate(BAND_NAMES):
        stats = band_stats[band_idx]
        
        print(f"\n[{band_name}]")
        print(f"  Files analyzed: {len(stats['nan_ratios'])}")
        
        if stats['mins']:
            print(f"  Global min: {np.min(stats['mins']):.4f}")
            print(f"  Global max: {np.max(stats['maxs']):.4f}")
            print(f"  Mean: {np.mean([np.mean(stats['mins']), np.mean(stats['maxs'])]):.4f}")
        else:
            print(f"  Global min: N/A")
            print(f"  Global max: N/A")
        
        avg_nan = np.mean(stats['nan_ratios']) * 100
        max_nan = np.max(stats['nan_ratios']) * 100
        print(f"  Avg NaN ratio: {avg_nan:.2f}%")
        print(f"  Max NaN ratio: {max_nan:.2f}%")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    analyze_files()
