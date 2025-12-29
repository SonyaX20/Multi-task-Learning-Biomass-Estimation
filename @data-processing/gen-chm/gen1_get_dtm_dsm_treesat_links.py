"""
Compute coverage of DGM1/DOM1 elevation tiles by TreeSat sampling grids.

For each DGM1/DOM1 tile that intersects at least one TreeSat grid cell
(60 m or 200 m), this script records the tile ID, the download URL, and
all intersecting TreeSat sample IDs.

Inputs (relative to the repository root):
- data/lgln-opengeodata/lgln-opengeodata-dgm1.geojson
- data/lgln-opengeodata/lgln-opengeodata-dom1.geojson
- data/treesat_ori/geojson/bb_60m.GeoJSON
- data/treesat_ori/geojson/bb_200m.GeoJSON

Outputs (in this folder):
- query_result_60m/dtm1_links.json or dsm1_links.json
- query_result_200m/dtm1_links.json or dsm1_links.json

Example usage:
  python get_dtm_dsm_treesat_links.py --type dtm --resolution 200
  python get_dtm_dsm_treesat_links.py --type dsm --resolution 60
"""

import argparse
import json
import os

import geopandas as gpd
from rtree import index
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute intersections between DGM1/DOM1 elevation tiles and "
            "TreeSat sampling grids (60 m or 200 m)."
        )
    )
    parser.add_argument(
        "--type",
        choices=["dtm", "dsm"],
        required=True,
        help="Elevation product to process: 'dtm' (DGM1) or 'dsm' (DOM1).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[60, 200],
        required=True,
        help="TreeSat grid resolution in meters (60 or 200).",
    )
    return parser.parse_args()


def get_project_root() -> str:
    """Return the repository root inferred from this file location."""
    # This file lives in: <root>/data-processing/gen-chm/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_processing_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(data_processing_dir)
    return project_root


def main() -> None:
    args = parse_args()

    project_root = get_project_root()

    # Configure big-tile (DGM1/DOM1) input and link field
    if args.type == "dtm":
        big_path = os.path.join(
            project_root,
            "data",
            "lgln-opengeodata",
            "lgln-opengeodata-dgm1.geojson",
        )
        link_field = "dtm1"
        data_type_name = "DGM1"
        link_source_field = "dgm1"
        output_basename = f"dtm1_links_{args.resolution}m"
    else:  # args.type == "dsm"
        big_path = os.path.join(
            project_root,
            "data",
            "lgln-opengeodata",
            "lgln-opengeodata-dom1.geojson",
        )
        link_field = "dsm1"
        data_type_name = "DOM1"
        link_source_field = "dom1"
        output_basename = f"dsm1_links_{args.resolution}m"

    # Configure TreeSat grid input based on resolution
    small_filename = f"bb_{args.resolution}m.GeoJSON"
    small_path = os.path.join(
        project_root,
        "data",
        "treesat_ori",
        "geojson",
        small_filename,
    )

    # Configure output directory, separated by resolution
    output_dir = os.path.join(project_root, "data-processing", "dtm-dsm-treesat-links")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{output_basename}.json")

    print(f"Processing {args.type.upper()} data at {args.resolution} m resolution...")
    print(f"  Big tiles (DGM1/DOM1): {big_path}")
    print(f"  TreeSat grid: {small_path}")
    print(f"  Output JSON: {output_file}")

    # 1. Read GeoJSON files
    print("Reading GeoJSON files...")
    gdf_big = gpd.read_file(big_path)
    gdf_small = gpd.read_file(small_path)

    print(f"  {data_type_name} polygons: {len(gdf_big)}")
    print(f"  TreeSat grid polygons ({args.resolution} m): {len(gdf_small)}")

    # 2. Build R-tree index for DGM1/DOM1 polygons (bounding boxes)
    print("Building spatial index over big tiles...")
    big_bboxes = [poly.bounds for poly in gdf_big.geometry]
    idx = index.Index()
    for i, bbox in enumerate(big_bboxes):
        idx.insert(i, bbox)

    # 3. Mapping from big-tile index to list of small-grid indices
    big_to_small: dict[int, list[int]] = {i: [] for i in range(len(gdf_big))}
    small_grids_found: set[int] = set()

    # 4. For each TreeSat polygon, find all intersecting DGM1/DOM1 tiles
    print("Processing spatial intersections (bounding-box overlap)...")
    for j, small_poly in tqdm(
        enumerate(gdf_small.geometry), total=len(gdf_small), desc="TreeSat cells"
    ):
        small_bbox = small_poly.bounds

        # Candidate big tiles whose bounding boxes intersect the small bbox
        candidate_big = list(idx.intersection(small_bbox))

        found_in_any_big = False
        for i in candidate_big:
            big_bbox = big_bboxes[i]

            # Bounding-box intersection test in X and Y
            if (
                small_bbox[0] < big_bbox[2]
                and small_bbox[2] > big_bbox[0]
                and small_bbox[1] < big_bbox[3]
                and small_bbox[3] > big_bbox[1]
            ):
                big_to_small[i].append(j)
                found_in_any_big = True

        if found_in_any_big:
            small_grids_found.add(j)

    # Debug summary about coverage
    print(
        f"  TreeSat grids assigned to at least one {data_type_name} tile: "
        f"{len(small_grids_found)} / {len(gdf_small)}"
    )
    if len(small_grids_found) < len(gdf_small):
        missed_count = len(gdf_small) - len(small_grids_found)
        print(
            f"WARNING: {missed_count} TreeSat polygons (" f"{args.resolution} m) were not assigned to any {data_type_name} tile!"
        )
        missed_indices = set(range(len(gdf_small))) - small_grids_found
        print("  Examples of missed TreeSat polygons (first 5):")
        for idx_missed, missed_idx in enumerate(list(missed_indices)[:5]):
            missed_bbox = gdf_small.geometry.iloc[missed_idx].bounds
            print(f"    TreeSat polygon {missed_idx}: bbox = {missed_bbox}")

    # 5. Collect and output results: one entry per DGM1/DOM1 tile with content
    result: list[dict] = []

    for i, small_idxs in big_to_small.items():
        if not small_idxs:
            continue

        props = gdf_big.iloc[i]

        # Collect TreeSat IMG_IDs for all intersecting polygons
        img_ids: list[str] = []
        for small_idx in small_idxs:
            small_props = gdf_small.iloc[small_idx]
            if "IMG_ID" in small_props:
                img_ids.append(small_props["IMG_ID"])

        if not img_ids:
            continue

        result.append(
            {
                "tile_id": props["tile_id"],
                link_field: props[link_source_field],
                "treesat_count": len(img_ids),
                "treesat_samples": img_ids,
            }
        )

    unique_result = result  # One entry per tile by construction

    with open(output_file, "w") as f:
        json.dump(unique_result, f, indent=2)

    total_small_covered = len(small_grids_found)
    total_intersections = sum(item["treesat_count"] for item in unique_result)

    print(f"  Number of {data_type_name} tiles with TreeSat coverage: {len(unique_result)}")
    print(f"  Total TreeSat polygons covered: {total_small_covered}")
    print(
        f"  Coverage ratio: {total_small_covered / len(gdf_small) * 100:.1f}% of "
        f"{len(gdf_small)} TreeSat cells"
    )
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    main()
