"""
Problem Definition:
Given two GeoJSON files, one representing DGM1/DOM1 polygons (lgln-opengeodata-dgm1.geojson or lgln-opengeodata-dom1.geojson) and the other representing 60m grid polygons (bb_60m.GeoJSON),
the goal is to find all DGM1/DOM1 polygons that intersect with at least one 60m polygon (using bounding box intersection), and to output a list of these polygons with their tile_id and dtm1/dsm1 link.

Solution Approach and Method:
1. Read both GeoJSON files using geopandas.
2. Build an R-tree spatial index for the DGM1/DOM1 polygons to accelerate spatial queries.
3. For each small grid polygon, use the spatial index to find candidate DGM1/DOM1 polygons whose bounding boxes may intersect with it, and check for actual bounding box intersection.
4. Record which DGM1/DOM1 polygons intersect with at least one small grid polygon.
5. Output a JSON file listing the tile_id and dtm1/dsm1 link for each such polygon.

Note: Changed from containment to intersection because 60m polygons (60m x 60m) may span across boundaries of DGM1/DOM1 polygons (1000m x 1000m).

Expected Output:
A JSON file (dtm1_links.json or dsm1_links.json) containing a list of UNIQUE DGM1/DOM1 tiles. Each object contains:
- tile_id: The unique identifier for the DGM1/DOM1 tile
- dtm1/dsm1: The download link for the tile
- treesat_count: Number of 60m grid polygons that intersect with this tile
- treesat_samples: List of all IMG_IDs from 60m polygons that intersect with this tile

Usage:
python a_detect_cover_60m.py --type dtm  # Generate DTM links (uses dgm1.geojson)
python a_detect_cover_60m.py --type dsm  # Generate DSM links (uses dom1.geojson)
"""
import geopandas as gpd
from shapely.geometry import box
from rtree import index
from tqdm import tqdm
import json
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate DTM or DSM links for 60m grid polygons')
parser.add_argument('--type', choices=['dtm', 'dsm'], required=True,
                    help='Type of data to process: dtm (uses dgm1.geojson) or dsm (uses dom1.geojson)')
args = parser.parse_args()

# Set file names and field names based on type
if args.type == 'dtm':
    input_file = "lgln-opengeodata-dgm1.geojson"
    output_file = "dtm1_links.json"
    link_field = "dtm1"
    data_type_name = "DGM1"
elif args.type == 'dsm':
    input_file = "lgln-opengeodata-dom1.geojson"
    output_file = "dsm1_links.json"
    link_field = "dsm1"
    data_type_name = "DOM1"

print(f"Processing {args.type.upper()} data...")
print(f"Input file: {input_file}")
print(f"Output file: {output_file}")

# 1. Read GeoJSON files
print("Reading GeoJSON files...")
gdf_big = gpd.read_file(input_file)
gdf_small = gpd.read_file("bb_60m.GeoJSON")

print(f"{data_type_name} polygons: {len(gdf_big)}")
print(f"60m grid polygons: {len(gdf_small)}")

# 2. Build R-tree index for DGM1/DOM1 polygons
print("Building spatial index...")
big_bboxes = [poly.bounds for poly in gdf_big.geometry]
idx = index.Index()
for i, bbox in enumerate(big_bboxes):
    idx.insert(i, bbox)

# 3. Record which DGM1/DOM1 polygons contain which 60m grid polygons
big_to_small = {i: [] for i in range(len(gdf_big))}
small_grids_found = set()  # Track which small grids have been assigned to at least one DGM1/DOM1 polygon

# 4. For each 60m grid polygon, find all DGM1/DOM1 polygons that intersect with it
print("Processing spatial intersection...")

for j, small_poly in tqdm(enumerate(gdf_small.geometry), total=len(gdf_small)):
    small_bbox = small_poly.bounds
    
    # Find all candidate DGM1/DOM1 polygons that might intersect with this small grid
    candidate_big = list(idx.intersection(small_bbox))
    
    found_in_any_big = False
    for i in candidate_big:
        big_bbox = big_bboxes[i]
        
        # Check for bounding box intersection (overlap)
        # Two rectangles intersect if they overlap in both X and Y dimensions
        if (small_bbox[0] < big_bbox[2] and small_bbox[2] > big_bbox[0] and
            small_bbox[1] < big_bbox[3] and small_bbox[3] > big_bbox[1]):
            
            big_to_small[i].append(j)
            found_in_any_big = True
    
    if found_in_any_big:
        small_grids_found.add(j)

# Debug output
print(f"Small grids successfully assigned: {len(small_grids_found)} / {len(gdf_small)}")
if len(small_grids_found) < len(gdf_small):
    missed_count = len(gdf_small) - len(small_grids_found)  
    print(f"WARNING: {missed_count} 60m polygons were not assigned to any {data_type_name} polygon!")
    
    # Show some examples of missed 60m polygons
    missed_indices = set(range(len(gdf_small))) - small_grids_found
    print("Examples of missed 60m polygons (showing first 5):")
    for idx, missed_idx in enumerate(list(missed_indices)[:5]):
        missed_bbox = gdf_small.geometry.iloc[missed_idx].bounds
        print(f"  60m polygon {missed_idx}: bbox = {missed_bbox}")

# 5. Collect and output results - only unique DGM1/DOM1 tiles
result = []
found_big_count = 0
dgm1_polygons_with_content = set()

for i, small_idxs in big_to_small.items():
    if small_idxs:
        found_big_count += 1
        dgm1_polygons_with_content.add(i)
        props = gdf_big.iloc[i].properties if "properties" in gdf_big.columns else gdf_big.iloc[i]
        
        # Extract IMG_IDs from all intersecting 60m polygons
        img_ids = []
        for small_idx in small_idxs:
            small_props = gdf_small.iloc[small_idx]
            if "IMG_ID" in small_props:
                img_ids.append(small_props["IMG_ID"])
        
        # Create only ONE entry per unique DGM1/DOM1 tile
        result.append({
            "tile_id": props["tile_id"],
            link_field: props["dgm1"] if args.type == 'dtm' else props["dom1"],
            "treesat_count": len(img_ids),  # Number of 60m grids intersecting this tile
            "treesat_samples": img_ids  # List of all IMG_IDs intersecting this tile
        })

unique_result = result  # No need for deduplication since each DGM1/DOM1 tile appears only once

with open(output_file, "w") as f:
    json.dump(unique_result, f, indent=2)

print(f"Number of unique {data_type_name} tiles found: {len(unique_result)}")
print(f"Total number of 60m polygons covered: {len(small_grids_found)}")
print(f"Coverage ratio: {len(small_grids_found) / len(gdf_small) * 100:.1f}%")
print(f"Total {data_type_name}-60m intersections: {sum(item['treesat_count'] for item in unique_result)}")
print(f"Output saved to: {output_file}")