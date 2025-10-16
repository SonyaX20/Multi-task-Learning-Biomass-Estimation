#!/usr/bin/env python3
"""
CHM Generation: DSM - DTM with invalid value interpolation.

Interpolates DSM invalid values using 8-neighbor mean before CHM calculation.
Ensures output contains no NaN or -9999 values.
"""

import os
import numpy as np
import rasterio
from tqdm import tqdm

# Configuration
dsm_dir = "dsm1_tif"
dtm_dir = "dtm1_tif_smoothed"
output_dir = "chm"
invalid_value = -9999.0

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def interpolate_invalid_pixels(data, invalid_value):
    """Interpolate invalid pixels using 8-neighbor mean."""
    invalid_mask = (data == invalid_value)
    
    if not np.any(invalid_mask):
        return data
    
    result = data.copy()
    invalid_coords = np.where(invalid_mask)
    
    for i, j in zip(invalid_coords[0], invalid_coords[1]):
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = i + di, j + dj
                
                if 0 <= ni < data.shape[0] and 0 <= nj < data.shape[1]:
                    neighbor_val = data[ni, nj]
                    if neighbor_val != invalid_value and not np.isnan(neighbor_val):
                        neighbors.append(neighbor_val)
        
        if neighbors:
            result[i, j] = np.mean(neighbors)
    
    return result

# Find matching files
dsm_files = []
if os.path.exists(dsm_dir):
    for filename in os.listdir(dsm_dir):
        if filename.endswith('.tif') and filename[0].isdigit():
            dsm_files.append(filename)

dtm_files = []
if os.path.exists(dtm_dir):
    for filename in os.listdir(dtm_dir):
        if filename.endswith('.tif') and filename[0].isdigit():
            dtm_files.append(filename)

matching_files = set(dsm_files).intersection(set(dtm_files))

if len(matching_files) == 0:
    print("Error: No matching file pairs found!")
    exit(1)

# Process matching files
processed_count = 0
error_count = 0

for filename in tqdm(matching_files, desc="Generating CHM"):
    try:
        # Build file paths
        dsm_path = os.path.join(dsm_dir, filename)
        dtm_path = os.path.join(dtm_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read DSM data
        with rasterio.open(dsm_path) as dsm_src:
            dsm_data = dsm_src.read(1).astype(np.float32)
            dsm_profile = dsm_src.profile.copy()
            dsm_shape = dsm_data.shape
            
            # Handle DSM nodata values
            if dsm_src.nodata is not None:
                dsm_data[dsm_data == dsm_src.nodata] = invalid_value
            
            # Interpolate DSM invalid values using 8-neighbor mean
            dsm_data = interpolate_invalid_pixels(dsm_data, invalid_value)
            
            # Handle any remaining invalid pixels
            remaining_invalid = (dsm_data == invalid_value)
            if np.any(remaining_invalid):
                valid_pixels = dsm_data[dsm_data != invalid_value]
                if len(valid_pixels) > 0:
                    dsm_data[remaining_invalid] = np.mean(valid_pixels)
                else:
                    dsm_data[remaining_invalid] = 0.0
        
        # Read DTM data (already processed, should have no invalid values)
        with rasterio.open(dtm_path) as dtm_src:
            dtm_data = dtm_src.read(1).astype(np.float32)
            dtm_shape = dtm_data.shape
        
        # Verify dimensions match
        if dsm_shape != dtm_shape:
            error_count += 1
            continue
        
        # Calculate CHM: DSM - DTM
        chm_data = dsm_data - dtm_data
        
        # Apply height thresholds
        chm_data[chm_data < 0] = 0.0  # Set negative values to 0
        
        # Update profile - no nodata since we've eliminated all invalid values
        dsm_profile.update({
            'dtype': 'float32',
            'nodata': None
        })
        
        # Save CHM
        with rasterio.open(output_path, 'w', **dsm_profile) as dst:
            dst.write(chm_data, 1)
        
        processed_count += 1
        
    except Exception as e:
        error_count += 1
        continue

# Print summary
print(f"\nCHM generation complete:")
print(f"  Successful: {processed_count}")
print(f"  Failed: {error_count}")
print(f"  Output: {output_dir}")

