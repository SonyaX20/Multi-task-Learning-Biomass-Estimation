#!/usr/bin/env python3
"""
DTM Smoothing with 10m Mean Filter

This script applies a 10m×10m mean filter to Digital Terrain Model (DTM) data for smoothing 
and gap filling. The convolution-based mean filter performs spatial averaging within a 
10m×10m window to achieve DTM smoothing, with the primary purpose of filling data gaps 
and ensuring terrain model continuity.

Technical Background:
- Mean Filter: A convolution filter that replaces each pixel with the average value of 
  pixels within the specified neighborhood window
- Purpose: Unlike linear interpolation which is mainly used for resampling or scale 
  transformation, the mean filter effectively smoothes spatial data by reducing noise 
  and filling small gaps through neighborhood averaging
- Implementation: For 1m resolution DTM data, a 10m filter corresponds to an 11×11 pixel 
  kernel (covering 10m distance from center)
- Benefits: Ensures continuous terrain surface, reduces small-scale irregularities, 
  and provides a reliable ground reference for subsequent CHM calculation

Input: DTM files in TIF format from dtm1_tif/ directory
Output: Smoothed DTM files saved to dtm1_smoothed/ directory with same filenames
"""

import os
import glob
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

# Configuration
input_dir = "dtm1_tif"
output_dir = "dtm1_smoothed"
filter_size_m = 10.0  # 10m square filter

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all TIF files in the input directory
pattern = os.path.join(input_dir, "*.tif")
tif_files = glob.glob(pattern)

print(f"Found {len(tif_files)} DTM files to process")
print(f"Applying {filter_size_m}m mean filter for terrain smoothing...")

processed_count = 0

for tif_file in tif_files:
    try:
        # Extract filename for output
        filename = os.path.basename(tif_file)
        output_path = os.path.join(output_dir, filename)
        
        # Read DTM data
        with rasterio.open(tif_file) as src:
            dtm_data = src.read(1)  # Read first band
            profile = src.profile.copy()
            transform = src.transform
            nodata = src.nodata
            
            # Calculate pixel size from transform
            pixel_size = abs(transform[0])  # Assuming square pixels
            
            # Calculate filter size in pixels
            filter_size_pixels = int(filter_size_m / pixel_size)
            
            # Ensure odd kernel size for symmetric filtering
            if filter_size_pixels % 2 == 0:
                filter_size_pixels += 1
            
            if processed_count % 600 == 0:
                print(f"Processing {filename}: {dtm_data.shape}, pixel size: {pixel_size:.2f}m, "
                    f"filter kernel: {filter_size_pixels}×{filter_size_pixels} pixels")
                print("...")
            
            # Check for NaN values and print filename if found
            if np.isnan(dtm_data).any():
                print(f"Warning: NaN values detected in {filename}")
            
            # Simple approach: replace NaN with 0 for processing
            dtm_data = np.nan_to_num(dtm_data, nan=0.0)
            
            # Apply uniform mean filter
            smoothed_dtm = uniform_filter(dtm_data.astype(np.float64), 
                                        size=filter_size_pixels, mode='reflect')
            
            # Convert back to original data type
            smoothed_dtm = smoothed_dtm.astype(profile['dtype'])
            
            # Save smoothed DTM
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(smoothed_dtm, 1)
            
            processed_count += 1
            
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

print(f"\nDTM smoothing completed!")
print(f"Successfully processed: {processed_count} files")
print(f"Output directory: {output_dir}")
print(f"Filter applied: {filter_size_m}m mean filter for terrain continuity and gap filling")