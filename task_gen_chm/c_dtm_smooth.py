#!/usr/bin/env python3
"""
DTM Smoothing with 10m Mean Filter

Applies 10m×10m mean filter to DTM data for smoothing.
Handles -9999 invalid values by interpolating from 8-neighbor pixels before smoothing.
"""

import os
import glob
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
from scipy.ndimage import binary_dilation

# Configuration
input_dir = "dtm1_tif"
output_dir = "dtm1_tif_smoothed"
filter_size_m = 10.0  # 10m square filter
invalid_value = -9999.0  # Invalid value to handle

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all TIF files in the input directory
tif_files = glob.glob(os.path.join(input_dir, "*.tif"))

processed_count = 0


def interpolate_invalid_pixels(data, invalid_value):
    """
    Interpolate invalid pixels (-9999) using 8-neighbor mean.
    
    Args:
        data: 2D numpy array
        invalid_value: Value to treat as invalid
    
    Returns:
        data with invalid pixels interpolated
    """
    # Create mask for invalid pixels
    invalid_mask = (data == invalid_value)
    
    if not np.any(invalid_mask):
        return data  # No invalid pixels to fix
    
    # Create a copy to work with
    result = data.copy()
    
    # Get coordinates of invalid pixels
    invalid_coords = np.where(invalid_mask)
    
    for i, j in zip(invalid_coords[0], invalid_coords[1]):
        # Get 8-neighbor values
        neighbors = []
        
        # Check all 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip center pixel
                
                ni, nj = i + di, j + dj
                
                # Check bounds
                if 0 <= ni < data.shape[0] and 0 <= nj < data.shape[1]:
                    neighbor_val = data[ni, nj]
                    # Only use valid neighbors (not invalid_value)
                    if neighbor_val != invalid_value and not np.isnan(neighbor_val):
                        neighbors.append(neighbor_val)
        
        # If we have valid neighbors, use their mean
        if neighbors:
            result[i, j] = np.mean(neighbors)
        else:
            # If no valid neighbors, keep original value (will be handled later)
            pass
    
    return result

for tif_file in tif_files:
    try:
        filename = os.path.basename(tif_file)
        output_path = os.path.join(output_dir, filename)
        
        # Read DTM data
        with rasterio.open(tif_file) as src:
            dtm_data = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
            
            # Convert to float32 for processing
            dtm_data = dtm_data.astype(np.float32)
            
            # Step 1: Interpolate invalid pixels (-9999) using 8-neighbor mean
            dtm_interpolated = interpolate_invalid_pixels(dtm_data, invalid_value)
            
            # Step 2: Handle any remaining invalid pixels (those with no valid neighbors)
            remaining_invalid = (dtm_interpolated == invalid_value)
            if np.any(remaining_invalid):
                # For pixels with no valid neighbors, use a larger neighborhood
                # or set to a reasonable default (e.g., mean of all valid pixels)
                valid_pixels = dtm_interpolated[dtm_interpolated != invalid_value]
                if len(valid_pixels) > 0:
                    default_value = np.mean(valid_pixels)
                    dtm_interpolated[remaining_invalid] = default_value
                else:
                    # If no valid pixels at all, set to 0
                    dtm_interpolated[remaining_invalid] = 0.0
            
            # Step 3: Apply smoothing filter
            # Calculate filter size in pixels
            pixel_size = abs(transform[0])
            filter_size_pixels = int(filter_size_m / pixel_size)
            
            # Ensure odd kernel size
            if filter_size_pixels % 2 == 0:
                filter_size_pixels += 1
            
            # Apply uniform filter for smoothing
            smoothed_dtm = uniform_filter(dtm_interpolated, size=filter_size_pixels, 
                                        mode='constant', cval=0.0)
            
            # Update profile - no nodata since we've eliminated all invalid values
            profile['nodata'] = None
            profile['dtype'] = 'float32'
            
            # Save smoothed DTM
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(smoothed_dtm.astype(np.float32), 1)
            
            processed_count += 1
            
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

print(f"DTM smoothing complete: {processed_count} files processed")