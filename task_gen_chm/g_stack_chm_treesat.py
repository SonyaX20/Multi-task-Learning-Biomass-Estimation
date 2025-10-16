"""
Stack CHM and 60m sentinel data files together.
"""

import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm
from pathlib import Path


def stack_chm_and_60m_files(chm_path, sentinel_path, output_path):
    """Stack CHM and Sentinel-2 60m files together."""
    # Read CHM file
    with rasterio.open(chm_path) as chm_src:
        chm_data = chm_src.read(1)
        chm_meta = chm_src.meta.copy()
        chm_transform = chm_src.transform
        chm_bounds = chm_src.bounds
        chm_crs = chm_src.crs
        chm_shape = chm_data.shape
    
    # Read Sentinel-2 file
    with rasterio.open(sentinel_path) as sentinel_src:
        sentinel_data = sentinel_src.read()  # Read all bands
        sentinel_meta = sentinel_src.meta.copy()
        sentinel_transform = sentinel_src.transform
        sentinel_bounds = sentinel_src.bounds
        sentinel_crs = sentinel_src.crs
        sentinel_shape = (sentinel_src.height, sentinel_src.width)
        num_sentinel_bands = sentinel_src.count
    
    # Verify alignment
    if chm_transform != sentinel_transform:
        raise ValueError(f"Transforms don't match!\nCHM: {chm_transform}\nSentinel: {sentinel_transform}")
    
    if chm_shape != sentinel_shape:
        raise ValueError(f"Shapes don't match!\nCHM: {chm_shape}\nSentinel: {sentinel_shape}")
    
    if chm_crs != sentinel_crs:
        raise ValueError(f"CRS don't match!\nCHM: {chm_crs}\nSentinel: {sentinel_crs}")
    
    # Stack the data: CHM as first band, then all Sentinel bands
    # Total bands = 1 (CHM) + num_sentinel_bands
    total_bands = 1 + num_sentinel_bands
    
    # Create stacked array
    stacked_data = np.zeros((total_bands, chm_shape[0], chm_shape[1]), dtype=np.float32)
    
    # Band 1: CHM data
    stacked_data[0] = chm_data.astype(np.float32)
    
    # Bands 2-N: Sentinel data
    for i in range(num_sentinel_bands):
        stacked_data[i + 1] = sentinel_data[i].astype(np.float32)
    
    # Prepare output metadata
    out_meta = chm_meta.copy()
    out_meta.update({
        'count': total_bands,
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'lzw'
    })
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write stacked file
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        for band_idx in range(total_bands):
            dst.write(stacked_data[band_idx], band_idx + 1)
        
        # Set band descriptions
        band_names = ['chm', 'vv', 'vh', 'vv/vh']
        for i in range(min(total_bands, len(band_names))):
            dst.set_band_description(i + 1, band_names[i])
    
    return True


def stack_all_files(chm_dir, sentinel_dir, output_dir):
    """Stack all matching files from CHM and Sentinel directories."""
    # Get all CHM files
    chm_files = glob.glob(os.path.join(chm_dir, '*.tif'))
    
    if not chm_files:
        return 0, 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process statistics
    success_count = 0
    error_count = 0
    
    for chm_path in tqdm(chm_files, desc="Stacking"):
        try:
            filename = os.path.basename(chm_path)
            sentinel_path = os.path.join(sentinel_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Check if corresponding Sentinel file exists
            if not os.path.exists(sentinel_path):
                error_count += 1
                continue
            
            # Stack files
            stack_chm_and_60m_files(chm_path, sentinel_path, output_path)
            success_count += 1
            
        except Exception as e:
            error_count += 1
    
    return success_count, error_count




if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chm_dir = os.path.join(script_dir, 'chm_cropped_downsampled')
    sentinel_dir = os.path.join(os.path.dirname(script_dir), '60m')
    output_dir = os.path.join(script_dir, 'chm_stacked_treesat')
    
    # Check directories exist
    if not os.path.exists(chm_dir):
        print(f"Error: CHM directory not found: {chm_dir}")
        exit(1)
    
    if not os.path.exists(sentinel_dir):
        print(f"Error: Sentinel directory not found: {sentinel_dir}")
        exit(1)
    
    # Stack all files
    success_count, error_count = stack_all_files(
        chm_dir=chm_dir,
        sentinel_dir=sentinel_dir,
        output_dir=output_dir
    )
    
    # Print summary
    print(f"\nStacking complete:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Output: {output_dir}")

