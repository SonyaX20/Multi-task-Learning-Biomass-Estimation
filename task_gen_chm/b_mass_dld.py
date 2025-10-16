"""
This script downloads all GeoTIFF files listed in treesat_to_tiles.json.
For each unique tile, it downloads the file from the provided URL and saves it using the tile_id as the filename.
The script supports retrying failed downloads and skips files that already exist and are larger than 10KB (to avoid re-downloading corrupted or incomplete files).

Note: This script now downloads UNIQUE tiles only (not per-treesat), using treesat_to_tiles.json.
"""
import os
import json
import requests
from tqdm import tqdm
import argparse

def load_and_extract_tiles(target):
    """Load treesat_to_tiles.json and extract unique tiles for the specified target"""
    json_path = "treesat_to_tiles.json"
    print(f"Loading {json_path}...")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File {json_path} not found!")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} treesat entries from {json_path}")
    
    # Extract unique tiles for the target (dtm1 or dsm1)
    tile_key = "dtm_tiles" if target == "dtm1" else "dsm_tiles"
    unique_tiles = {}
    
    for treesat_id, info in data.items():
        tiles = info.get(tile_key, [])
        for tile in tiles:
            tile_id = tile["tile_id"]
            url = tile["url"]
            if tile_id not in unique_tiles:
                unique_tiles[tile_id] = url
    
    print(f"Found {len(unique_tiles)} unique {target.upper()} tiles to download")
    return unique_tiles

def download_file(url, save_path, max_retries=3):
    """Download a file from URL with retry mechanism"""
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"Download failed: {url}, retrying ({attempt+1}/{max_retries}), error: {e}")
    return False

def download_files(target, unique_tiles):
    """Download files for the specified target using tile_id as filename"""
    save_dir = f"{target}_tif"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting downloads for {target} files...")
    print(f"Save directory: {save_dir}")
    
    successful_downloads = 0
    skipped_files = 0
    failed_downloads = 0
    
    for tile_id, url in tqdm(unique_tiles.items(), desc=f"Downloading {target} files"):
        # Use tile_id as filename
        filename = os.path.join(save_dir, f"{tile_id}.tif")
        
        # Skip files that already exist and are likely valid (size > 10KB)
        if os.path.exists(filename) and os.path.getsize(filename) > 10000:
            skipped_files += 1
            continue
            
        success = download_file(url, filename)
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1
            print(f"Failed to download after retries: {tile_id}")
    
    print(f"\nDownload summary for {target}:")
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Skipped files: {skipped_files}")
    print(f"  Failed downloads: {failed_downloads}")
    print(f"  Total unique tiles: {len(unique_tiles)}")

def main():
    parser = argparse.ArgumentParser(description='Download DSM or DTM files using tile_id as filenames')
    parser.add_argument('target', choices=['dsm1', 'dtm1'], 
                       help='Target to download: dsm1 for DSM files, dtm1 for DTM files')
    parser.add_argument('--both', action='store_true', 
                       help='Download both DSM and DTM files')
    
    args = parser.parse_args()
    
    if args.both:
        print("Downloading both DSM and DTM files...")
        for target in ['dsm1', 'dtm1']:
            try:
                unique_tiles = load_and_extract_tiles(target)
                download_files(target, unique_tiles)
                print(f"\n{'='*50}\n")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                continue
    else:
        try:
            unique_tiles = load_and_extract_tiles(args.target)
            download_files(args.target, unique_tiles)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    print("All download tasks completed.")

if __name__ == "__main__":
    main()
