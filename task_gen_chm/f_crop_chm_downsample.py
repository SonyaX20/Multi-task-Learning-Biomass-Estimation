#!/usr/bin/env python3
"""
CHM Cropping with Perfect Alignment

Crops CHM data to match 60m template files exactly.
Stops processing if invalid values (NaN or -9999) are detected.
"""

import os
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
import rasterio.windows
from rasterio.merge import merge
from tqdm import tqdm

# Configuration
treesat_mapping_file = "treesat_to_tiles.json"
chm_dir = "chm"
template_60m_dir = "../60m"
output_dir = "chm_cropped_downsampled"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load treesat-to-tiles mapping
with open(treesat_mapping_file, "r") as f:
    treesat_mapping = json.load(f)

# Load 60m folder tif files as templates
template_info = {}
if not os.path.exists(template_60m_dir):
    print(f"ERROR: Template directory not found: {template_60m_dir}")
    exit(1)

for filename in os.listdir(template_60m_dir):
    if filename.endswith('.tif'):
        treesat_id = filename.replace('.tif', '')
        template_path = os.path.join(template_60m_dir, filename)
        
        try:
            with rasterio.open(template_path) as src:
                template_info[treesat_id] = {
                    'path': template_path,
                    'bounds': src.bounds,
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height
                }
        except Exception as e:
            print(f"Warning: Could not read template {filename}: {e}")

# Pre-load CHM file paths
chm_file_map = {}
if not os.path.exists(chm_dir):
    print(f"ERROR: CHM directory not found: {chm_dir}")
    exit(1)

for filename in os.listdir(chm_dir):
    if filename.endswith('.tif'):
        tile_id = filename.replace('.tif', '')
        chm_path = os.path.join(chm_dir, filename)
        chm_file_map[tile_id] = chm_path


def crop_and_align_single_tile(tile_id, template_info, output_path, treesat_id):
    """
    Crop CHM from a single tile and align to 60m template file exactly.
    
    Process:
    1. Read template to get exact bounds/CRS/transform (EPSG:32632)
    2. Read CHM source (EPSG:25832)
    3. Check for invalid values (NaN or -9999) - stop if found
    4. Reproject CHM to match template exactly
    5. Output has identical georeferencing as template
    """
    try:
        chm_path = chm_file_map.get(tile_id)
        if not chm_path:
            return False, f"CHM tile {tile_id} not found"
        
        # Get template information (EPSG:32632)
        target_bounds = template_info['bounds']
        target_crs = template_info['crs']
        target_transform = template_info['transform']
        target_width = template_info['width']
        target_height = template_info['height']
        
        # Read source CHM (EPSG:25832)
        with rasterio.open(chm_path) as src:
            src_crs = src.crs
            
            # Calculate window in source CRS that covers target bounds
            # We need to reproject target bounds from 32632 to source CRS 25832
            from rasterio.warp import transform_bounds
            
            # Transform target bounds to source CRS
            src_bounds = transform_bounds(target_crs, src_crs, *target_bounds)
            
            # Get window in source raster
            try:
                window = from_bounds(*src_bounds, src.transform)
            except Exception as e:
                return False, f"Window calculation failed: {e}"
            
            # Read data with window
            # Add some buffer to ensure coverage
            window_floored = window.round_offsets(op='floor').round_lengths(op='ceil')
            
            try:
                src_data = src.read(1, window=window_floored)
            except Exception as e:
                return False, f"Reading window failed: {e}"
            
            # Convert to float32 for processing
            src_data = src_data.astype(np.float32)
            
            # Check for invalid values - if found, stop processing
            has_nan = np.any(np.isnan(src_data))
            has_neg9999 = np.any(src_data == -9999)
            
            if has_nan or has_neg9999:
                return False, f"Invalid values detected: NaN={has_nan}, -9999={has_neg9999}"
            
            # Create output array with target dimensions
            dst_data = np.full((target_height, target_width), 0.0, dtype=np.float32)
            
            # Get transform for the windowed read
            src_window_transform = rasterio.windows.transform(window_floored, src.transform)
            
            # Reproject to match template exactly
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_window_transform,
                src_crs=src_crs,
                src_nodata=None,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=None,
                resampling=Resampling.bilinear
            )
            
            # Prepare output metadata
            out_meta = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'nodata': None,
                'width': target_width,
                'height': target_height,
                'count': 1,
                'crs': target_crs,
                'transform': target_transform,
                'compress': 'lzw'
            }
            
            # Save cropped and aligned CHM
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(dst_data, 1)
            
            return True, ""
            
    except Exception as e:
        import traceback
        return False, f"Error: {str(e)}\n{traceback.format_exc()}"


def mosaic_and_align(tile_ids, template_info, output_path, treesat_id):
    """
    Mosaic multiple CHM tiles and align to 60m template exactly.
    
    Process:
    1. Mosaic CHM tiles (in EPSG:25832)
    2. Check for invalid values (NaN or -9999) - stop if found
    3. Reproject mosaic to match template (EPSG:32632)
    4. Output has identical georeferencing as template
    """
    try:
        # Get CHM file paths
        chm_paths = []
        for tile_id in tile_ids:
            chm_path = chm_file_map.get(tile_id)
            if not chm_path:
                return False, f"CHM tile {tile_id} not found"
            chm_paths.append(chm_path)
        
        # Open all CHM files
        src_files = [rasterio.open(path) for path in chm_paths]
        
        # Create mosaic
        mosaic, mosaic_transform = merge(src_files)
        mosaic = mosaic[0]  # Get first band
        
        # Get CRS and nodata from first file
        src_crs = src_files[0].crs
        src_nodata = src_files[0].nodata
        src_dtype = src_files[0].dtypes[0]
        
        # Close source files
        for src in src_files:
            src.close()
        
        # Convert to float32 for processing
        mosaic = mosaic.astype(np.float32)
        
        # Check for invalid values - if found, stop processing
        has_nan = np.any(np.isnan(mosaic))
        has_neg9999 = np.any(mosaic == -9999)
        
        if has_nan or has_neg9999:
            return False, f"Invalid values detected in mosaic: NaN={has_nan}, -9999={has_neg9999}"
        
        # Get template information
        target_bounds = template_info['bounds']
        target_crs = template_info['crs']
        target_transform = template_info['transform']
        target_width = template_info['width']
        target_height = template_info['height']
        
        # Create output array with target dimensions
        dst_data = np.full((target_height, target_width), 0.0, dtype=np.float32)
        
        # Reproject mosaic to match template exactly
        reproject(
            source=mosaic,
            destination=dst_data,
            src_transform=mosaic_transform,
            src_crs=src_crs,
            src_nodata=None,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=None,
            resampling=Resampling.bilinear
        )
        
        # Prepare output metadata
        out_meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': None,
            'width': target_width,
            'height': target_height,
            'count': 1,
            'crs': target_crs,
            'transform': target_transform,
            'compress': 'lzw'
        }
        
        # Save aligned CHM
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(dst_data, 1)
        
        return True, ""
        
    except Exception as e:
        import traceback
        return False, f"Mosaicking error: {str(e)}\n{traceback.format_exc()}"


# Process each treesat sample
successful_crops = 0
failed_crops = 0
missing_template = 0
missing_chm_tiles = 0
all_values = []

for treesat_id, info in tqdm(treesat_mapping.items(), desc="Processing"):
    # Check if we have template for this treesat
    if treesat_id not in template_info:
        missing_template += 1
        continue
    
    # Get template information
    template = template_info[treesat_id]
    
    # Get CHM tile IDs from mapping
    dtm_tiles = info.get("dtm_tiles", [])
    if not dtm_tiles:
        missing_chm_tiles += 1
        continue
    
    tile_ids = [tile["tile_id"] for tile in dtm_tiles]
    
    # Check if all required CHM tiles exist
    missing_tiles = [tid for tid in tile_ids if tid not in chm_file_map]
    if missing_tiles:
        missing_chm_tiles += 1
        continue
    
    # Create output filename
    output_filename = f"{treesat_id}.tif"
    output_path = os.path.join(output_dir, output_filename)
    
    # Process based on number of tiles
    if len(tile_ids) == 1:
        success, _ = crop_and_align_single_tile(tile_ids[0], template, output_path, treesat_id)
    else:
        success, _ = mosaic_and_align(tile_ids, template, output_path, treesat_id)
    
    if success:
        successful_crops += 1
        # Collect statistics from output file
        try:
            with rasterio.open(output_path) as src:
                data = src.read(1)
                all_values.extend(data.flatten().tolist())
        except:
            pass
    else:
        failed_crops += 1

# Print summary
total_samples = len(treesat_mapping)
success_rate = successful_crops / total_samples * 100 if total_samples > 0 else 0

print(f"\nProcessing complete:")
print(f"  Total samples: {total_samples}")
print(f"  Successful: {successful_crops}")
print(f"  Failed: {failed_crops}")
print(f"  Missing template: {missing_template}")
print(f"  Missing CHM tiles: {missing_chm_tiles}")
print(f"  Success rate: {success_rate:.1f}%")

# Print statistics
if all_values:
    all_values = np.array(all_values)
    print(f"\nCHM Statistics:")
    print(f"  Min height: {all_values.min():.2f}m")
    print(f"  Max height: {all_values.max():.2f}m")
    print(f"  Mean height: {all_values.mean():.2f}m")
    print(f"  Valid pixels: {len(all_values):,}")

print(f"  Output: {output_dir}")

