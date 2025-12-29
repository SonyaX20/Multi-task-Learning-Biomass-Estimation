"""Reproject CHM tiles to TreeSat grid, then optionally stack with S1.

For each TreeSat sample, this script can:
- read the CHM tiles contributing to that sample (from treesat_to_tiles_<res>m.json),
- mosaic them in the native CHM CRS (DGM1/DOM1 grid),
- reproject the mosaic onto the TreeSat Sentinel-1 grid (60 m or 200 m),
  using **max** resampling instead of mean/bilinear, and
- write a single-band CHM GeoTIFF aligned to the TreeSat raster;

and, if requested, additionally:
- stack the reprojected CHM with Sentinel-1 vv, vh, and vv/vh ratio
  into a 4-band patch per TreeSat sample.

Inputs (relative to repository root):
- data/chm_standard/<tile_id>.tif
- @data-processing/dtm-dsm-treesat-links/treesat_to_tiles_60m.json
- @data-processing/dtm-dsm-treesat-links/treesat_to_tiles_200m.json
- Sentinel-1 TreeSat rasters (templates), e.g.
  - data/treesatai_data/s1/60m/<treesat_id>.tif
  - data/treesatai_data/s1/200m/<treesat_id>.tif

Outputs (relative to repository root):
- data/chm_60m/<treesat_id>.tif
- data/chm_200m/<treesat_id>.tif
- data/stacked_treesat_60m/<treesat_id>.tif (if stacking enabled)
- data/stacked_treesat_200m/<treesat_id>.tif (if stacking enabled)

Example usage (from repo root):
  python @data-processing/gen-chm/gen5_chm_reproj_maxpool_crop_stack.py --resolution 200
  python @data-processing/gen-chm/gen5_chm_reproj_maxpool_crop_stack.py --resolution 60 \
      --file_name Pinus_sylvestris_4_297957_BI_NLF
  python @data-processing/gen-chm/gen5_chm_reproj_maxpool_crop_stack.py \
    --resolution 200 \
    --stack

Debug usage (from repo root, ETH 10 m comparison tiles only):
  python @data-processing/gen-chm/gen5_chm_reproj_maxpool_crop_stack.py \
    --resolution 60 \
    --debug_tiles
"""

import argparse
import json
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from scipy.ndimage import maximum_filter
from tqdm import tqdm


NODATA_VALUE = -9999.0
CHM_UPPER_LIMIT = 1000

# ---------------------------------------------------------------------------
# Filesystem configuration (all paths are relative to the repository root)
# ---------------------------------------------------------------------------

# Directory with per-tile 1 m CHM rasters (standard LGLN CHM tiles).
# Default: "data/chm_standard"
CHM_TILE_DIR_REL = os.path.join("data", "chm_standard")

# Directory with TreeSat-to-tile mapping JSONs (treesat_to_tiles_<res>m.json).
# Default: "@data-processing/dtm-dsm-treesat-links"
TREESAT_MAPPING_DIR_REL = os.path.join("@data-processing", "dtm-dsm-treesat-links")

# Root directory with TreeSat Sentinel-1 rasters used as CHM reprojection
# templates. Files live under "{resolution}m" subfolders.
# Default root: "data/treesatai_data/s1"
S1_TEMPLATE_DIR_ROOT_REL = os.path.join("data", "treesatai_data", "s1")

# Output directory pattern for per-sample CHM rasters on the TreeSat grid.
# Examples: "data/chm_60m", "data/chm_200m"
CHM_TREESAT_OUT_DIR_PATTERN = os.path.join("data", "chm_{res}m")

# Output directory pattern for stacked S1+CHM rasters on the TreeSat grid.
# Examples: "data/stacked_treesat_60m", "data/stacked_treesat_200m"
STACKED_TREESAT_OUT_DIR_PATTERN = os.path.join("data", "stacked_treesat_{res}m")

# Directory with ETH 10 m CHM rasters used as reprojection targets in debug
# mode. Files: "ETH_CHM_10m_<tile_id>.tif".
# Default: "data/ETH_chm_10m"
ETH_CHM_10M_DIR_REL = os.path.join("data", "ETH_chm_10m")

# Output directory for debug CHM rasters reprojected to the ETH 10 m grid.
# Files: "<tile_id>.tif".
# Default: "data/chm_debug_tiles"
CHM_DEBUG_TILES_DIR_REL = os.path.join("data", "chm_debug_tiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproject CHM tiles to TreeSat grid at 60 m or 200 m resolution "
            "using max-based downsampling, and crop per TreeSat sample."
        )
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[60, 200],
        required=True,
        help="TreeSat grid resolution in meters (60 or 200).",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        help=(
            "After generating CHM_<res>m patches, also stack them with S1 "
            "bands (vv, vh, vv/vh, chm) into 4-band rasters."
        ),
    )
    parser.add_argument(
        "--debug_tiles",
        action="store_true",
        help=(
            "Debug mode: only process a fixed set of CHM tiles and reproject "
            "them to a 10 m target grid using a 25x25 max filter before "
            "reprojection, writing results to a debug directory."
        ),
    )
    return parser.parse_args()


def get_project_root() -> str:
    """Return the repository root inferred from this file location."""

    # This file lives in: <root>/data-processing/gen-chm/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_processing_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(data_processing_dir)
    return project_root


def load_treesat_mapping(project_root: str, resolution: int) -> Dict:
    """Load TreeSat-to-tile mapping JSON for the given resolution.

    Uses the global ``TREESAT_MAPPING_DIR_REL`` and expects a file named
    ``treesat_to_tiles_<res>m.json`` inside that directory.
    """

    mapping_path = os.path.join(
        project_root,
        TREESAT_MAPPING_DIR_REL,
        f"treesat_to_tiles_{resolution}m.json",
    )

    print(f"Loading treesat mapping from: {mapping_path}")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    print(f"  Loaded {len(mapping)} TreeSat entries")
    return mapping


def load_templates(project_root: str, resolution: int) -> Dict[str, Dict]:
    """Load TreeSat Sentinel-1 templates for a given resolution.

    Uses the global ``S1_TEMPLATE_DIR_ROOT_REL`` and resolution-specific
    subfolder (e.g. ``60m`` or ``200m``).
    """

    template_dir = os.path.join(
        project_root,
        S1_TEMPLATE_DIR_ROOT_REL,
        f"{resolution}m",
    )

    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    print(f"Loading TreeSat templates from: {template_dir}")
    template_info: Dict[str, Dict] = {}

    for filename in os.listdir(template_dir):
        if not filename.endswith(".tif"):
            continue
        treesat_id = filename[:-4]
        template_path = os.path.join(template_dir, filename)
        try:
            with rasterio.open(template_path) as src:
                template_info[treesat_id] = {
                    "path": template_path,
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "meta": src.meta.copy(),
                }
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not read template {filename}: {exc}")

    print(f"  Loaded {len(template_info)} template rasters")
    return template_info


def load_chm_tile_map(project_root: str) -> Dict[str, str]:
    """Index available 1 m CHM tiles.

    Uses the global ``CHM_TILE_DIR_REL`` directory and returns a mapping
    from tile ID (filename without ``.tif``) to absolute file path.
    """

    chm_dir = os.path.join(project_root, CHM_TILE_DIR_REL)

    if not os.path.exists(chm_dir):
        raise FileNotFoundError(f"CHM directory not found: {chm_dir}")

    print(f"Indexing CHM tiles in: {chm_dir}")
    chm_file_map: Dict[str, str] = {}

    for filename in os.listdir(chm_dir):
        if not filename.endswith(".tif"):
            continue
        tile_id = filename[:-4]
        chm_file_map[tile_id] = os.path.join(chm_dir, filename)

    print(f"  Found {len(chm_file_map)} CHM tiles")
    return chm_file_map


def mosaic_chm_tiles(tile_ids: List[str], chm_file_map: Dict[str, str]) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    chm_paths: List[str] = []
    for tile_id in tile_ids:
        path = chm_file_map.get(tile_id)
        if path is None:
            raise FileNotFoundError(f"CHM tile not found for tile_id={tile_id}")
        chm_paths.append(path)

    src_files = [rasterio.open(p) for p in chm_paths]
    try:
        mosaic, mosaic_transform = merge(src_files)
        mosaic_data = mosaic[0].astype(np.float32)
        src_crs = src_files[0].crs
    finally:
        for src in src_files:
            src.close()

    return mosaic_data, mosaic_transform, src_crs


def reproject_to_template_max(
    mosaic_data: np.ndarray,
    mosaic_transform: rasterio.Affine,
    src_crs: rasterio.crs.CRS,
    template: Dict,
    ) -> Tuple[np.ndarray, Dict]:
    """Reproject CHM mosaic onto template grid.

    Processing steps:
    - clip CHM values to [0.0, 41.5983] (excluding nodata);
    - apply a 20x20 maximum filter in the native grid;
    - reproject to the template grid using bilinear resampling.
    """

    dst_meta = template["meta"].copy()
    width = template["width"]
    height = template["height"]
    dst_transform = template["transform"]
    dst_crs = template["crs"]

    # 1) Clip CHM values to global percentile range [0.0, 41.5983]
    chm = mosaic_data.astype(np.float32).copy()
    nodata_mask = chm == NODATA_VALUE
    valid_mask = ~nodata_mask
    chm[valid_mask & (chm < 0.0)] = 0.0
    chm[valid_mask & (chm > 41.5983)] = 41.5983

    # 2) Apply 20x20 max filter in source (1 m) grid
    filtered = maximum_filter(chm, size=20, mode="nearest")

    # 3) Reproject to template grid with bilinear resampling
    dst_data = np.full((height, width), NODATA_VALUE, dtype=np.float32)

    reproject(
        source=filtered,
        destination=dst_data,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        src_nodata=NODATA_VALUE,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=NODATA_VALUE,
        resampling=Resampling.bilinear,
    )

    # Prepare single-band float32 profile
    dst_meta.update(
        {
            "count": 1,
            "dtype": "float32",
            "nodata": NODATA_VALUE,
            "compress": "lzw",
        }
    )

    return dst_data, dst_meta

def stack_chm_and_s1(
    chm_path: str,
    s1_path: str,
    out_path: str,
    # vv_bounds: tuple[float | None, float | None],
    # vh_bounds: tuple[float | None, float | None],
    # ratio_bounds: tuple[float | None, float | None],
    ) -> None:
    """Stack CHM and Sentinel-1 bands into a 4-band GeoTIFF."""

    with rasterio.open(chm_path) as chm_src:
        chm = chm_src.read(1).astype(np.float32)
        chm_transform = chm_src.transform
        chm_crs = chm_src.crs

    with rasterio.open(s1_path) as s1_src:
        s1_data = s1_src.read().astype(np.float32)
        s1_meta = s1_src.meta.copy()
        s1_transform = s1_src.transform
        s1_crs = s1_src.crs

    # Sanity checks on grid alignment
    if chm_crs != s1_crs:
        raise ValueError(f"CRS mismatch between CHM ({chm_crs}) and S1 ({s1_crs})")

    if chm_transform != s1_transform:
        raise ValueError("Transform mismatch between CHM and S1 grids")

    if chm.shape != (s1_src.height, s1_src.width):
        raise ValueError("Shape mismatch between CHM and S1 grids")

    if s1_src.count < 2:
        raise ValueError(f"Sentinel file has fewer than 2 bands: {s1_path}")

    # Clean CHM: treat -9999 as nodata; enforce plausible height range
    nodata_mask = chm == NODATA_VALUE
    valid_mask = ~nodata_mask

    if np.any(valid_mask):
        cleaned = chm.copy()

        neg_mask = valid_mask & (cleaned < 0.0)
        cleaned[neg_mask] = 0.0

        upper_clip_mask = (
            valid_mask
            & (cleaned > CHM_UPPER_LIMIT)
        )
        cleaned[upper_clip_mask] = CHM_UPPER_LIMIT

        cleaned[nodata_mask] = 0.0
        chm = cleaned

    vv = s1_data[0]
    vh = s1_data[1]
    vv_vh = s1_data[2]

    # Clip S1 bands to global percentile ranges before stacking
    s1_nodata = -9999.0

    def _clip_band_inplace(band: np.ndarray, low: float, high: float) -> None:
        mask = (band != s1_nodata) & ~np.isnan(band)
        band[mask & (band < low)] = low
        band[mask & (band > high)] = high

    _clip_band_inplace(vv, -18.0383, 5.5330)
    _clip_band_inplace(vh, -26.1871, -1.7555)
    _clip_band_inplace(vv_vh, -1.7393, 1.4362)

    stacked = np.zeros((4, vv.shape[0], vv.shape[1]), dtype=np.float32)
    stacked[0] = vv
    stacked[1] = vh
    stacked[2] = vv_vh
    stacked[3] = chm

    out_meta = s1_meta.copy()
    out_meta.update(
        {
            "count": 4,
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "lzw",
        }
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(out_path, "w", **out_meta) as dst:
        band_names = ["vv", "vh", "vv/vh", "chm"]
        for i in range(4):
            dst.write(stacked[i], i + 1)
            dst.set_band_description(i + 1, band_names[i])


def run_stacking_pipeline(
    project_root: str,
    resolution: int,
    chm_out_dir: str,
) -> None:
    """Run CHM+S1 stacking over all CHM patches in ``chm_out_dir``.

    Uses global ``S1_TEMPLATE_DIR_ROOT_REL`` and
    ``STACKED_TREESAT_OUT_DIR_PATTERN`` for input/output locations.
    """

    s1_dir = os.path.join(
        project_root,
        S1_TEMPLATE_DIR_ROOT_REL,
        f"{resolution}m",
    )

    out_dir_rel = STACKED_TREESAT_OUT_DIR_PATTERN.format(res=resolution)
    out_dir = os.path.join(project_root, out_dir_rel)

    if not os.path.exists(chm_out_dir):
        raise FileNotFoundError(f"CHM directory not found for stacking: {chm_out_dir}")
    if not os.path.exists(s1_dir):
        raise FileNotFoundError(f"Sentinel directory not found: {s1_dir}")

    chm_files = glob(os.path.join(chm_out_dir, "*.tif"))
    if not chm_files:
        print(f"No CHM files found for stacking in: {chm_out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    success_count = 0
    error_count = 0

    for chm_path in tqdm(chm_files, desc="Stacking CHM + S1"):
        filename = os.path.basename(chm_path)
        s1_path = os.path.join(s1_dir, filename)
        out_path = os.path.join(out_dir, filename)

        if not os.path.exists(s1_path):
            error_count += 1
            continue

        try:
            stack_chm_and_s1(
                chm_path,
                s1_path,
                out_path,
            )
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            print(f"Failed for {filename}: {exc}")

    print("\nStacking complete:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Output directory: {out_dir}")


def run_debug_chm_tiles(
    project_root: str,
    ) -> None:
    """Debug helper: process a fixed set of CHM tiles against ETH CHM 10 m grid.

    Uses global directories:
    - ``CHM_TILE_DIR_REL`` for input 1 m CHM tiles;
    - ``ETH_CHM_10M_DIR_REL`` for ETH 10 m CHM templates;
    - ``CHM_DEBUG_TILES_DIR_REL`` for debug outputs.
    """

    from scipy.ndimage import maximum_filter

    chm_dir = os.path.join(project_root, CHM_TILE_DIR_REL)
    eth_dir = os.path.join(project_root, ETH_CHM_10M_DIR_REL)
    out_dir = os.path.join(project_root, CHM_DEBUG_TILES_DIR_REL)

    if not os.path.exists(chm_dir):
        raise FileNotFoundError(f"CHM directory not found for debug mode: {chm_dir}")
    if not os.path.exists(eth_dir):
        raise FileNotFoundError(f"ETH CHM directory not found for debug mode: {eth_dir}")

    os.makedirs(out_dir, exist_ok=True)

    tile_filenames: List[str] = [
        "326625868.tif",
        "325685714.tif",
        "324665881.tif",
        "325415729.tif",
        "325395731.tif",
        "325695707.tif",
        "325795870.tif",
        "325845870.tif",
        "325865737.tif",
        "326135728.tif",
    ]

    processed = 0
    failed = 0
    missing_src = 0
    missing_tmpl = 0

    for fname in tile_filenames:
        src_path = os.path.join(chm_dir, fname)
        tmpl_path = os.path.join(eth_dir, f"ETH_CHM_10m_{fname}")

        if not os.path.exists(src_path):
            print(f"[DEBUG] CHM tile not found: {src_path}")
            missing_src += 1
            continue
        if not os.path.exists(tmpl_path):
            print(f"[DEBUG] ETH CHM template not found: {tmpl_path}")
            missing_tmpl += 1
            continue

        try:
            with rasterio.open(src_path) as src:
                src_data = src.read(1).astype(np.float32)
                src_transform = src.transform
                src_crs = src.crs
                src_nodata = src.nodata if src.nodata is not None else NODATA_VALUE

            src_data = src_data.copy()
            src_data[src_data == src_nodata] = NODATA_VALUE

            # Clip CHM to [0.0, 41.5983] before filtering
            chm = src_data
            nodata_mask = chm == NODATA_VALUE
            valid_mask = ~nodata_mask
            chm[valid_mask & (chm < 0.0)] = 0.0
            chm[valid_mask & (chm > 41.5983)] = 41.5983

            # Apply 20x20 max filter in native (1 m) grid
            filtered = maximum_filter(chm, size=20, mode="nearest")

            with rasterio.open(tmpl_path) as tmpl:
                dst_meta = tmpl.meta.copy()
                dst_transform = tmpl.transform
                dst_crs = tmpl.crs
                height = tmpl.height
                width = tmpl.width

            dst_data = np.full((height, width), NODATA_VALUE, dtype=np.float32)

            reproject(
                source=filtered,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=NODATA_VALUE,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=NODATA_VALUE,
                resampling=Resampling.bilinear,
            )

            dst_meta.update(
                {
                    "count": 1,
                    "dtype": "float32",
                    "nodata": NODATA_VALUE,
                    "compress": "lzw",
                }
            )

            out_path = os.path.join(out_dir, fname)

            with rasterio.open(out_path, "w", **dst_meta) as dst:
                dst.write(dst_data, 1)

            processed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[DEBUG] Failed for {fname}: {exc}")

    print("\nDebug CHM reproject (tiles -> ETH 10 m) complete:")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Missing CHM tiles: {missing_src}")
    print(f"  Missing ETH CHM templates: {missing_tmpl}")
    print(f"  Output directory: {out_dir}")


def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    if args.debug_tiles:
        print("===== DEBUG: ETH 10 m CHM tiles =====")
        run_debug_chm_tiles(
            project_root=project_root,
        )
        return

    print("===== 1. Load mappings, templates, and CHM index =====")
    treesat_mapping = load_treesat_mapping(project_root, args.resolution)
    template_info = load_templates(project_root, args.resolution)
    chm_file_map = load_chm_tile_map(project_root)

    print("===== 2. Prepare CHM output directory =====")
    out_dir_rel = CHM_TREESAT_OUT_DIR_PATTERN.format(res=args.resolution)
    out_dir = os.path.join(project_root, out_dir_rel)
    os.makedirs(out_dir, exist_ok=True)

    successful = 0
    failed = 0
    missing_template = 0
    missing_chm_tiles = 0

    print("===== 3. Reproject CHM tiles to TreeSat grid =====")
    for treesat_id, info in tqdm(
        treesat_mapping.items(), desc=f"Reproject+filter+reproject CHM ({args.resolution}m)"
    ):
        template = template_info.get(treesat_id)
        if template is None:
            missing_template += 1
            continue

        dtm_tiles = info.get("dtm_tiles", [])
        if not dtm_tiles:
            missing_chm_tiles += 1
            continue

        tile_ids = [t["tile_id"] for t in dtm_tiles]

        # Check that all CHM tiles exist
        missing = [tid for tid in tile_ids if tid not in chm_file_map]
        if missing:
            missing_chm_tiles += 1
            continue

        out_path = os.path.join(out_dir, f"{treesat_id}.tif")
        if os.path.exists(out_path):
            continue

        try:
            mosaic_data, mosaic_transform, src_crs = mosaic_chm_tiles(
                tile_ids, chm_file_map
            )
            chm_reproj, out_meta = reproject_to_template_max(
                mosaic_data, mosaic_transform, src_crs, template
            )

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(chm_reproj, 1)

            successful += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed for {treesat_id}: {exc}")

    print("\nReprojection + filtering + resampling complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Missing template: {missing_template}")
    print(f"  Missing CHM tiles: {missing_chm_tiles}")
    print(f"  Output directory: {out_dir}")

    if args.stack:
        print("===== 4. Stack CHM with Sentinel-1 =====")
        run_stacking_pipeline(
            project_root=project_root,
            resolution=args.resolution,
            chm_out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
