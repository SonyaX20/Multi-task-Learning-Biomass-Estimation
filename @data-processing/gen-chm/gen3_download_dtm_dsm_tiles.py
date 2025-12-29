"""Download unique DTM/DSM GeoTIFF tiles for TreeSat samples.

This script reads the per-TreeSat mapping produced by
``gen2_reverse_mapping.py`` (``treesat_to_tiles_<res>m.json``) and
extracts the **unique** DTM/DSM tiles that are needed. Each tile is
then downloaded once and stored as ``<tile_id>.tif``.

Inputs (relative to repository root):
- data-processing/dtm-dsm-treesat-links/treesat_to_tiles_60m.json
- data-processing/dtm-dsm-treesat-links/treesat_to_tiles_200m.json

Outputs (under ``data/``):
- data/dtm1_tif/<tile_id>.tif
- data/dsm1_tif/<tile_id>.tif

Example usage:
  python data-processing/gen-chm/gen3_download_dtm_dsm_tiles.py dtm1 --resolution 200
  python data-processing/gen-chm/gen3_download_dtm_dsm_tiles.py dsm1 --resolution 60
  python data-processing/gen-chm/gen3_download_dtm_dsm_tiles.py dtm1 --resolution 60 --both
"""

import argparse
import json
import os
from typing import Dict

import requests
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download unique DTM/DSM tiles referenced in "
            "treesat_to_tiles_<res>m.json."
        )
    )
    parser.add_argument(
        "target",
        choices=["dsm1", "dtm1"],
        help=(
            "Target product to download: 'dsm1' for DSM tiles, 'dtm1' for DTM tiles. "
            "Ignored if --both is given."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[60, 200],
        required=True,
        help="TreeSat grid resolution in meters (60 or 200).",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Download both DSM and DTM tiles.",
    )
    return parser.parse_args()


def get_project_root() -> str:
    """Return the repository root inferred from this file location."""
    # This file lives in: <root>/data-processing/gen-chm/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_processing_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(data_processing_dir)
    return project_root


def load_and_extract_tiles(
    project_root: str, resolution: int, target: str
) -> Dict[str, str]:
    """Load treesat_to_tiles_<res>m.json and extract unique tiles for target.

    target: "dtm1" or "dsm1".
    Returns mapping tile_id -> url.
    """

    links_dir = os.path.join(
        project_root,
        "data-processing",
        "dtm-dsm-treesat-links",
    )
    json_path = os.path.join(links_dir, f"treesat_to_tiles_{resolution}m.json")

    print(f"Loading {json_path}...")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File {json_path} not found!")

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} TreeSat entries from {json_path}")

    tile_key = "dtm_tiles" if target == "dtm1" else "dsm_tiles"
    unique_tiles: Dict[str, str] = {}

    for treesat_id, info in data.items():
        tiles = info.get(tile_key, [])
        for tile in tiles:
            tile_id = tile["tile_id"]
            url = tile["url"]
            if tile_id not in unique_tiles:
                unique_tiles[tile_id] = url

    print(f"Found {len(unique_tiles)} unique {target.upper()} tiles to download")
    return unique_tiles


def download_file(url: str, save_path: str, max_retries: int = 3) -> bool:
    """Download a file from URL with a simple retry mechanism."""

    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"Download failed: {url}, retrying "
                f"({attempt + 1}/{max_retries}), error: {exc}"
            )
    return False


def download_files(project_root: str, target: str, unique_tiles: Dict[str, str]) -> None:
    """Download files for the specified target using tile_id as filename."""

    save_dir = os.path.join(project_root, "data", f"{target}_tif")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting downloads for {target} files...")
    print(f"Save directory: {save_dir}")

    successful_downloads = 0
    skipped_files = 0
    failed_downloads = 0

    for tile_id, url in tqdm(unique_tiles.items(), desc=f"Downloading {target} files"):
        filename = os.path.join(save_dir, f"{tile_id}.tif")

        # Skip files that already exist and are likely valid (size > 10KB)
        if os.path.exists(filename) and os.path.getsize(filename) > 10_000:
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


def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    targets = ["dsm1", "dtm1"] if args.both else [args.target]

    for target in targets:
        print("=" * 70)
        print(f"Processing target: {target} at {args.resolution} m resolution")
        try:
            unique_tiles = load_and_extract_tiles(project_root, args.resolution, target)
            download_files(project_root, target, unique_tiles)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            continue

    print("All download tasks completed.")


if __name__ == "__main__":
    main()
