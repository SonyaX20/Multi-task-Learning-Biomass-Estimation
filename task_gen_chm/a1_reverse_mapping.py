"""
Reverse Mapping: From tile_id->treesat to treesat->tile_ids

This script reverses the mapping in dtm1_links.json and dsm1_links.json.
Instead of tile_id -> list of treesat_samples, we create:
treesat_id -> list of tile_ids (with download links)

This is crucial because:
1. A 60m treesat grid may span multiple DGM1/DOM1 tiles
2. For CHM generation, we need to know which tiles to download and mosaic for each treesat
3. Makes the download and cropping process much simpler

Output:
- treesat_to_tiles.json: Mapping of treesat_id -> DTM and DSM tile information
"""

import json
from collections import defaultdict

print("Reversing tile_id->treesat mapping to treesat->tile_ids mapping...")
print("=" * 70)

# Load DTM and DSM links
print("Loading dtm1_links.json...")
with open("dtm1_links.json", "r") as f:
    dtm_data = json.load(f)

print("Loading dsm1_links.json...")
with open("dsm1_links.json", "r") as f:
    dsm_data = json.load(f)

print(f"DTM tiles: {len(dtm_data)}")
print(f"DSM tiles: {len(dsm_data)}")

# Create reverse mapping: treesat_id -> list of (tile_id, dtm_url)
treesat_to_dtm_tiles = defaultdict(list)
treesat_to_dsm_tiles = defaultdict(list)

# Process DTM data
print("\nProcessing DTM mappings...")
for entry in dtm_data:
    tile_id = entry["tile_id"]
    dtm_url = entry["dtm1"]
    treesat_samples = entry["treesat_samples"]
    
    for treesat_id in treesat_samples:
        treesat_to_dtm_tiles[treesat_id].append({
            "tile_id": tile_id,
            "url": dtm_url
        })

# Process DSM data
print("Processing DSM mappings...")
for entry in dsm_data:
    tile_id = entry["tile_id"]
    dsm_url = entry["dsm1"]
    treesat_samples = entry["treesat_samples"]
    
    for treesat_id in treesat_samples:
        treesat_to_dsm_tiles[treesat_id].append({
            "tile_id": tile_id,
            "url": dsm_url
        })

# Combine DTM and DSM information
print("\nCombining DTM and DSM information...")
treesat_mapping = {}

all_treesat_ids = set(treesat_to_dtm_tiles.keys()) | set(treesat_to_dsm_tiles.keys())

for treesat_id in all_treesat_ids:
    dtm_tiles = treesat_to_dtm_tiles.get(treesat_id, [])
    dsm_tiles = treesat_to_dsm_tiles.get(treesat_id, [])
    
    treesat_mapping[treesat_id] = {
        "treesat_id": treesat_id,
        "dtm_tiles": dtm_tiles,
        "dsm_tiles": dsm_tiles,
        "dtm_tile_count": len(dtm_tiles),
        "dsm_tile_count": len(dsm_tiles),
        "requires_mosaic": len(dtm_tiles) > 1 or len(dsm_tiles) > 1
    }

# Save to JSON file
output_file = "treesat_to_tiles.json"
print(f"\nSaving to {output_file}...")
with open(output_file, "w") as f:
    json.dump(treesat_mapping, f, indent=2)

# Statistics
print("\n" + "=" * 70)
print("STATISTICS")
print("=" * 70)
print(f"Total unique treesat samples: {len(treesat_mapping)}")

# Count how many treesat samples require mosaicking
single_tile = sum(1 for v in treesat_mapping.values() if not v["requires_mosaic"])
multi_tile = sum(1 for v in treesat_mapping.values() if v["requires_mosaic"])

print(f"Single-tile treesat samples: {single_tile} ({single_tile/len(treesat_mapping)*100:.1f}%)")
print(f"Multi-tile treesat samples (require mosaic): {multi_tile} ({multi_tile/len(treesat_mapping)*100:.1f}%)")

# Distribution of tile counts
from collections import Counter
dtm_tile_counts = Counter(v["dtm_tile_count"] for v in treesat_mapping.values())
dsm_tile_counts = Counter(v["dsm_tile_count"] for v in treesat_mapping.values())

print(f"\nDTM tile count distribution:")
for count in sorted(dtm_tile_counts.keys()):
    print(f"  {count} tile(s): {dtm_tile_counts[count]} treesat samples")

print(f"\nDSM tile count distribution:")
for count in sorted(dsm_tile_counts.keys()):
    print(f"  {count} tile(s): {dsm_tile_counts[count]} treesat samples")

# Show examples of multi-tile treesat samples
print(f"\nExamples of treesat samples requiring mosaic (first 5):")
multi_tile_examples = [v for v in treesat_mapping.values() if v["requires_mosaic"]][:5]
for example in multi_tile_examples:
    print(f"  {example['treesat_id']}: {example['dtm_tile_count']} DTM tiles, {example['dsm_tile_count']} DSM tiles")
    if example['dtm_tiles']:
        tile_ids = [t['tile_id'] for t in example['dtm_tiles']]
        print(f"    DTM tile_ids: {', '.join(tile_ids)}")

print(f"\nOutput saved to: {output_file}")


