#!/usr/bin/env python3
"""
Add VV-VH Band to Stacked Data

Processes chm_stacked_treesat files and adds a new VV-VH band.
Outputs 5-band files to chm_stacked_treesat_new folder.
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


def process_stacked_file(input_path, output_path):
    """
    Process single stacked file and add VV-VH band.
    
    Args:
        input_path: Path to input 4-band file
        output_path: Path to output 5-band file
    
    Returns:
        bool: Success status
    """
    try:
        with rasterio.open(input_path) as src:
            # Read all 4 bands
            data = src.read()  # Shape: (4, H, W)
            profile = src.profile.copy()
        
        # Convert to float32
        data = data.astype(np.float32)
        
        # Get VV (band 1) and VH (band 2)
        vv = data[1]  # VV band
        vh = data[2]  # VH band
        
        # Create VV-VH band
        vv_vh_new = np.full_like(vv, np.nan, dtype=np.float32)
        
        # Calculate VV-VH where both VV and VH are valid (not -9999)
        valid_mask = (vv != -9999.0) & (vh != -9999.0)
        vv_vh_new[valid_mask] = vv[valid_mask] - vh[valid_mask]
        
        # Create 5-band array
        data_5band = np.zeros((5, data.shape[1], data.shape[2]), dtype=np.float32)
        data_5band[:4] = data  # Original 4 bands
        data_5band[4] = vv_vh_new  # New VV-VH band
        
        # Update profile for 5 bands
        profile.update({
            'count': 5,
            'dtype': 'float32',
            'nodata': np.nan
        })
        
        # Save 5-band file
        with rasterio.open(output_path, 'w', **profile) as dst:
            for band_idx in range(5):
                dst.write(data_5band[band_idx], band_idx + 1)
            
            # Set band descriptions
            band_names = ['chm', 'vv', 'vh', 'vv/vh', 'vv/vh_new']
            for i, name in enumerate(band_names):
                dst.set_band_description(i + 1, name)
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return False


def main():
    # Configuration paths
    input_dir = Path(__file__).parent.parent / 'task_gen_chm' / 'chm_stacked_treesat'
    output_dir = Path(__file__).parent.parent / 'task_gen_chm' / 'chm_stacked_treesat_new'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all TIF files
    tif_files = sorted(list(input_dir.glob('*.tif')))
    
    if not tif_files:
        print(f"No TIF files found in {input_dir}")
        return
    
    print(f"Processing {len(tif_files)} files...")
    
    # Process files
    success_count = 0
    error_count = 0
    
    for tif_file in tqdm(tif_files, desc="Adding VV-VH band"):
        output_path = output_dir / tif_file.name
        
        if process_stacked_file(tif_file, output_path):
            success_count += 1
        else:
            error_count += 1
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
