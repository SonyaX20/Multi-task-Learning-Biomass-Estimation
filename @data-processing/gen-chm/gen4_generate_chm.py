"""Generate canopy height model (CHM) rasters from DSM/DTM GeoTIFF pairs.

For each matching DSM/DTM tile, this script:
- fills nodata regions iteratively using neighboring pixels, and
- computes CHM = DSM - DTM, preserving nodata where either input is nodata.

Default locations (relative to repository root):
- DSM tiles: data/dsm1_tif/<tile_id>.tif
- DTM tiles: data/dtm1_tif/<tile_id>.tif
- CHM output: data/chm/<tile_id>.tif

Example usage (from repo root):
  python data-processing/gen-chm/gen4_generate_chm.py
  python data-processing/gen-chm/gen4_generate_chm.py --filename 325465699.tif

python data-processing/gen-chm/gen4_generate_chm.py \
  --dsm_dir data/dsm1_tif \
  --dtm_dir data/dtm1_tif \
  --out_dir data/chm \
  --max_iter 5 \
  --nodata -9999
"""

import argparse
import os
from glob import glob

import numpy as np
import rasterio
from tqdm import tqdm


def fill_nodata_iterative(data, nodata_value: float = -9999.0, max_iter: int = 5) -> np.ndarray:
    """Iteratively fill nodata pixels using the mean of 8-connected neighbors.

    Any pixel equal to ``nodata_value`` is considered missing. At each
    iteration, missing pixels with at least one valid neighbor are
    replaced by the mean of their valid neighbors. The process stops
    when no more pixels can be filled or ``max_iter`` iterations are
    reached.
    """

    arr = data.astype(np.float32).copy()
    mask = arr == nodata_value

    for _ in range(max_iter):
        if not mask.any():
            break

        values = arr.copy()
        values[mask] = 0.0
        valid = (~mask).astype(np.float32)

        neighbor_sum = np.zeros_like(arr, dtype=np.float32)
        neighbor_cnt = np.zeros_like(arr, dtype=np.float32)

        # up
        neighbor_sum[1:] += values[:-1]
        neighbor_cnt[1:] += valid[:-1]
        # down
        neighbor_sum[:-1] += values[1:]
        neighbor_cnt[:-1] += valid[1:]
        # left
        neighbor_sum[:, 1:] += values[:, :-1]
        neighbor_cnt[:, 1:] += valid[:, :-1]
        # right
        neighbor_sum[:, :-1] += values[:, 1:]
        neighbor_cnt[:, :-1] += valid[:, 1:]
        # up-left
        neighbor_sum[1:, 1:] += values[:-1, :-1]
        neighbor_cnt[1:, 1:] += valid[:-1, :-1]
        # up-right
        neighbor_sum[1:, :-1] += values[:-1, 1:]
        neighbor_cnt[1:, :-1] += valid[:-1, 1:]
        # down-left
        neighbor_sum[:-1, 1:] += values[1:, :-1]
        neighbor_cnt[:-1, 1:] += valid[1:, :-1]
        # down-right
        neighbor_sum[:-1, :-1] += values[1:, 1:]
        neighbor_cnt[:-1, :-1] += valid[1:, 1:]

        fillable = mask & (neighbor_cnt > 0)
        arr[fillable] = (neighbor_sum[fillable] / neighbor_cnt[fillable]).astype(
            np.float32
        )

        mask = arr == nodata_value

    arr[mask] = nodata_value
    return arr


def get_project_root() -> str:
    """Return the repository root inferred from this file location."""

    # This file lives in: <root>/data-processing/gen-chm/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_processing_dir = os.path.dirname(this_dir)
    project_root = os.path.dirname(data_processing_dir)
    return project_root


def process_pair(
    dsm_path: str,
    dtm_path: str,
    out_path: str,
    nodata_value: float = -9999.0,
    max_iter: int = 5,
) -> None:
    """Generate a single CHM raster from a DSM/DTM pair."""

    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
        if dsm_src.shape != dtm_src.shape or dsm_src.transform != dtm_src.transform:
            raise ValueError(f"Mismatch between DSM and DTM: {dsm_path} vs {dtm_path}")

        profile = dsm_src.profile
        dsm_nodata = dsm_src.nodata if dsm_src.nodata is not None else nodata_value
        dtm_nodata = dtm_src.nodata if dtm_src.nodata is not None else nodata_value

        dsm = dsm_src.read(1).astype(np.float32)
        dtm = dtm_src.read(1).astype(np.float32)

    dsm[dsm == dsm_nodata] = nodata_value
    dtm[dtm == dtm_nodata] = nodata_value

    dsm_filled = fill_nodata_iterative(dsm, nodata_value=nodata_value, max_iter=max_iter)
    dtm_filled = fill_nodata_iterative(dtm, nodata_value=nodata_value, max_iter=max_iter)

    if (dsm_filled == nodata_value).any():
        print(
            "Warning: DSM still contains nodata after filling: "
            f"{os.path.basename(dsm_path)}"
        )

    if (dtm_filled == nodata_value).any():
        print(
            "Warning: DTM still contains nodata after filling: "
            f"{os.path.basename(dtm_path)}"
        )

    chm = dsm_filled - dtm_filled
    invalid = (dsm_filled == nodata_value) | (dtm_filled == nodata_value)
    chm[invalid] = nodata_value

    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=nodata_value,
        compress="lzw",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(chm.astype(np.float32), 1)


def resolve_path(root: str, path: str) -> str:
    """Return absolute path; treat ``path`` as relative to ``root`` if not absolute."""

    return path if os.path.isabs(path) else os.path.join(root, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CHM from DSM/DTM GeoTIFF pairs."
    )
    parser.add_argument(
        "--dsm_dir",
        default="data/dsm1_tif",
        help="Directory containing DSM GeoTIFF tiles (relative to repo root).",
    )
    parser.add_argument(
        "--dtm_dir",
        default="data/dtm1_tif",
        help="Directory containing DTM GeoTIFF tiles (relative to repo root).",
    )
    parser.add_argument(
        "--out_dir",
        default="data/chm_standard",
        help="Output directory for CHM tiles (relative to repo root).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=5,
        help="Max iterations for nodata filling.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=-9999.0,
        help="Nodata value to treat as missing.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Only process the DSM/DTM pair with this filename (e.g. 325465699.tif)",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    dsm_dir = resolve_path(project_root, args.dsm_dir)
    dtm_dir = resolve_path(project_root, args.dtm_dir)
    out_dir = resolve_path(project_root, args.out_dir)

    dsm_files = sorted(glob(os.path.join(dsm_dir, "*.tif")))
    dtm_files = sorted(glob(os.path.join(dtm_dir, "*.tif")))

    dsm_dict = {os.path.basename(p): p for p in dsm_files}
    dtm_dict = {os.path.basename(p): p for p in dtm_files}

    common_names = sorted(set(dsm_dict.keys()) & set(dtm_dict.keys()))
    if not common_names:
        print("No matching DSM/DTM pairs found.")
        return

    if args.filename:
        target_name = args.filename if args.filename.endswith(".tif") else f"{args.filename}.tif"
        if target_name not in dsm_dict or target_name not in dtm_dict:
            print(f"Specified filename not found in both directories: {target_name}")
            return
        names_to_process = [target_name]
        print(f"Processing single DSM/DTM pair: {target_name}")
    else:
        names_to_process = common_names
        print(f"Found {len(common_names)} DSM/DTM pairs.")

    for name in tqdm(names_to_process, desc="Generating CHM"):
        dsm_path = dsm_dict[name]
        dtm_path = dtm_dict[name]
        out_path = os.path.join(out_dir, name)

        try:
            process_pair(
                dsm_path,
                dtm_path,
                out_path,
                nodata_value=args.nodata,
                max_iter=args.max_iter,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed for {name}: {exc}")


if __name__ == "__main__":
    main()
