#!/usr/bin/env python3
"""
TIF File Statistics Checker

Analyzes all TIF files in a directory and reports:
- Min/Max values for each band
- Invalid value types (NaN, 0, -9999)
- Number of files with invalid values
- File count and band information
"""

import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm
import argparse


def analyze_tif_file(file_path):
    """Analyze a single TIF file and return statistics."""
    try:
        with rasterio.open(file_path) as src:
            stats = {
                'file': os.path.basename(file_path),
                'bands': src.count,
                'width': src.width,
                'height': src.height,
                'dtype': str(src.dtypes[0]),
                'nodata': src.nodata,
                'crs': str(src.crs) if src.crs else 'None',
                'band_stats': []
            }
            
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
                
                # Get valid data (exclude NaN)
                valid_mask = ~np.isnan(data)
                valid_data = data[valid_mask]
                
                band_stat = {
                    'band': band_idx,
                    'total_pixels': data.size,
                    'valid_pixels': int(np.sum(valid_mask)),
                    'nan_pixels': int(np.sum(np.isnan(data))),
                    'zero_pixels': int(np.sum(data == 0)),
                    'neg9999_pixels': int(np.sum(data == -9999)),
                }
                
                if len(valid_data) > 0:
                    band_stat.update({
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data))
                    })
                else:
                    band_stat.update({
                        'min': None,
                        'max': None,
                        'mean': None,
                        'std': None
                    })
                
                stats['band_stats'].append(band_stat)
            
            return stats
            
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'error': str(e)
        }


def check_directory(directory_path):
    """Check all TIF files in a directory."""
    # Find all TIF files
    tif_files = glob.glob(os.path.join(directory_path, '*.tif'))
    
    if not tif_files:
        print(f"No TIF files found in {directory_path}")
        return
    
    print(f"Found {len(tif_files)} TIF files in {directory_path}")
    print("=" * 80)
    
    # Analyze all files
    all_stats = []
    files_with_invalid = 0
    
    for file_path in tqdm(tif_files, desc="Analyzing files"):
        stats = analyze_tif_file(file_path)
        all_stats.append(stats)
        
        # Check if file has invalid values
        if 'error' not in stats:
            has_invalid = any(
                band['nan_pixels'] > 0 or 
                band['zero_pixels'] > 0 or 
                band['neg9999_pixels'] > 0
                for band in stats['band_stats']
            )
            if has_invalid:
                files_with_invalid += 1
    
    # Print summary statistics
    print(f"\nSUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total files: {len(tif_files)}")
    print(f"Files with invalid values: {files_with_invalid}")
    print(f"Files without invalid values: {len(tif_files) - files_with_invalid}")
    
    # Aggregate statistics across all files
    all_mins = []
    all_maxs = []
    all_means = []
    total_nan = 0
    total_zero = 0
    total_neg9999 = 0
    total_pixels = 0
    total_valid_pixels = 0
    
    for stats in all_stats:
        if 'error' not in stats:
            for band in stats['band_stats']:
                if band['min'] is not None:
                    all_mins.append(band['min'])
                    all_maxs.append(band['max'])
                    all_means.append(band['mean'])
                
                total_nan += band['nan_pixels']
                total_zero += band['zero_pixels']
                total_neg9999 += band['neg9999_pixels']
                total_pixels += band['total_pixels']
                total_valid_pixels += band['valid_pixels']
    
    if all_mins:
        print(f"\nGLOBAL STATISTICS (across all bands and files):")
        print(f"  Overall minimum: {min(all_mins):.6f}")
        print(f"  Overall maximum: {max(all_maxs):.6f}")
        print(f"  Overall mean: {np.mean(all_means):.6f}")
        print(f"  Overall std: {np.std(all_means):.6f}")
    
    print(f"\nINVALID VALUE SUMMARY:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Valid pixels: {total_valid_pixels:,}")
    print(f"  NaN pixels: {total_nan:,} ({total_nan/total_pixels*100:.2f}%)")
    print(f"  Zero pixels: {total_zero:,} ({total_zero/total_pixels*100:.2f}%)")
    print(f"  -9999 pixels: {total_neg9999:,} ({total_neg9999/total_pixels*100:.2f}%)")
    
    # Show files with most invalid values
    invalid_files = []
    for stats in all_stats:
        if 'error' not in stats:
            total_invalid = sum(
                band['nan_pixels'] + band['zero_pixels'] + band['neg9999_pixels']
                for band in stats['band_stats']
            )
            if total_invalid > 0:
                invalid_files.append((stats['file'], total_invalid))
    
    if invalid_files:
        invalid_files.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTOP 10 FILES WITH MOST INVALID VALUES:")
        for filename, count in invalid_files[:10]:
            print(f"  {filename}: {count:,} invalid pixels")
    
    # Show files with errors
    error_files = [stats for stats in all_stats if 'error' in stats]
    if error_files:
        print(f"\nFILES WITH ERRORS ({len(error_files)}):")
        for stats in error_files[:10]:
            print(f"  {stats['file']}: {stats['error']}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more errors")


def main():
    parser = argparse.ArgumentParser(description='Analyze TIF files in a directory')
    parser.add_argument('directory', help='Directory path to analyze')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Analyze only first N files (for testing)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        return
    
    if args.sample:
        print(f"Analyzing first {args.sample} files only...")
        # Modify the glob to limit files
        tif_files = glob.glob(os.path.join(args.directory, '*.tif'))[:args.sample]
        # Temporarily modify the function to work with limited files
        original_glob = glob.glob
        def limited_glob(pattern):
            if '*.tif' in pattern:
                return original_glob(pattern)[:args.sample]
            return original_glob(pattern)
        glob.glob = limited_glob
    
    check_directory(args.directory)


if __name__ == "__main__":
    main()
