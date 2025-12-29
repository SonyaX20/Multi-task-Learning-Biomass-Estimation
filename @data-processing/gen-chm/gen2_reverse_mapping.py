"""Reverse mapping from elevation tiles to TreeSat samples.

This script reads the per-tile link files produced by
``get_dtm_dsm_treesat_links.py`` (for a given TreeSat grid resolution)
and creates a reverse mapping from TreeSat sample ID to the list of
DTM/DSM tiles (with URLs) that contribute to that sample.

example: 
python data-processing/gen-chm-200/a1_reverse_mapping.py --resolution 200
"""

import argparse
import json
import os
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse mapping from DTM/DSM tile -> TreeSat samples to "
            "TreeSat sample -> contributing tiles."
        )
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

    # This file currently lives in: <root>/data-processing/gen-chm-200/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_processing_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(data_processing_dir)
    return project_root


def main() -> None:
    args = parse_args()

    project_root = get_project_root()
    links_dir = os.path.join(
        project_root,
        "data-processing",
        "dtm-dsm-treesat-links",
    )

    dtm_links_path = os.path.join(links_dir, f"dtm1_links_{args.resolution}m.json")
    dsm_links_path = os.path.join(links_dir, f"dsm1_links_{args.resolution}m.json")

    output_file = os.path.join(
        links_dir,
        f"treesat_to_tiles_{args.resolution}m.json",
    )

    print("Reversing tile_id->treesat mapping to treesat->tile_ids mapping...")
    print(f"  Resolution: {args.resolution} m")
    print(f"  DTM links: {dtm_links_path}")
    print(f"  DSM links: {dsm_links_path}")
    print(f"  Output:    {output_file}")

    # Load DTM and DSM links
    print("\nLoading dtm1 links...")
    with open(dtm_links_path, "r") as f:
        dtm_data = json.load(f)

    print("Loading dsm1 links...")
    with open(dsm_links_path, "r") as f:
        dsm_data = json.load(f)

    print(f"    DTM tiles: {len(dtm_data)}")
    print(f"    DSM tiles: {len(dsm_data)}")

    # Create reverse mapping: treesat_id -> list of tile descriptors
    treesat_to_dtm_tiles = defaultdict(list)
    treesat_to_dsm_tiles = defaultdict(list)

    # Process DTM data
    print("\nProcessing DTM mappings...")
    for entry in dtm_data:
        tile_id = entry["tile_id"]
        dtm_url = entry["dtm1"]
        treesat_samples = entry["treesat_samples"]

        for treesat_id in treesat_samples:
            treesat_to_dtm_tiles[treesat_id].append(
                {
                    "tile_id": tile_id,
                    "url": dtm_url,
                }
            )

    # Process DSM data
    print("Processing DSM mappings...")
    for entry in dsm_data:
        tile_id = entry["tile_id"]
        dsm_url = entry["dsm1"]
        treesat_samples = entry["treesat_samples"]

        for treesat_id in treesat_samples:
            treesat_to_dsm_tiles[treesat_id].append(
                {
                    "tile_id": tile_id,
                    "url": dsm_url,
                }
            )

    # Combine DTM and DSM information
    print("\nCombining DTM and DSM information...")
    treesat_mapping: dict[str, dict] = {}

    all_treesat_ids = set(treesat_to_dtm_tiles.keys()) | set(
        treesat_to_dsm_tiles.keys()
    )

    for treesat_id in all_treesat_ids:
        dtm_tiles = treesat_to_dtm_tiles.get(treesat_id, [])
        dsm_tiles = treesat_to_dsm_tiles.get(treesat_id, [])

        treesat_mapping[treesat_id] = {
            "treesat_id": treesat_id,
            "dtm_tiles": dtm_tiles,
            "dsm_tiles": dsm_tiles,
            "dtm_tile_count": len(dtm_tiles),
            "dsm_tile_count": len(dsm_tiles),
            "requires_mosaic": len(dtm_tiles) > 1 or len(dsm_tiles) > 1,
        }

    # Save to JSON file
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(treesat_mapping, f, indent=2)

    # Statistics
    print(f"    Total unique treesat samples: {len(treesat_mapping)}")

    # Count how many treesat samples require mosaicking
    single_tile = sum(1 for v in treesat_mapping.values() if not v["requires_mosaic"])
    multi_tile = sum(1 for v in treesat_mapping.values() if v["requires_mosaic"])

    print(
        f"    Single-tile treesat samples: {single_tile} "
        f"({single_tile/len(treesat_mapping)*100:.1f}%)"
    )
    print(
        f"    Multi-tile treesat samples (require mosaic): {multi_tile} "
        f"({multi_tile/len(treesat_mapping)*100:.1f}%)"
    )

    # Distribution of tile counts
    from collections import Counter

    dtm_tile_counts = Counter(v["dtm_tile_count"] for v in treesat_mapping.values())
    dsm_tile_counts = Counter(v["dsm_tile_count"] for v in treesat_mapping.values())

    print(f"    DTM tile count distribution:")
    for count in sorted(dtm_tile_counts.keys()):
        print(f"      {count} tile(s): {dtm_tile_counts[count]} treesat samples")

    print(f"    DSM tile count distribution:")
    for count in sorted(dsm_tile_counts.keys()):
        print(f"      {count} tile(s): {dsm_tile_counts[count]} treesat samples")

    # Show examples of multi-tile treesat samples
    print(f"\nExamples of treesat samples requiring mosaic (first 5):")
    multi_tile_examples = [
        v for v in treesat_mapping.values() if v["requires_mosaic"]
    ][
        :5
    ]
    for example in multi_tile_examples:
        print(
            f"  {example['treesat_id']}: {example['dtm_tile_count']} DTM tiles, "
            f"{example['dsm_tile_count']} DSM tiles",
        )
        if example["dtm_tiles"]:
            tile_ids = [t["tile_id"] for t in example["dtm_tiles"]]
            print(f"    DTM tile_ids: {', '.join(tile_ids)}")

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()

