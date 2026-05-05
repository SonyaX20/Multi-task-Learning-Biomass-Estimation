#!/usr/bin/env python3

import os
import warnings

# Set environment variables BEFORE importing rasterio to suppress PROJ warnings
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'
os.environ['PROJ_NETWORK'] = 'OFF'
os.environ['CPL_LOG'] = '/dev/null'

import argparse
import csv
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from tqdm import tqdm

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger('rasterio').setLevel(logging.ERROR)

_EE_IMPORT_ERROR: Optional[str] = None


def _earth_engine_available() -> bool:
    global _EE_IMPORT_ERROR
    if _EE_IMPORT_ERROR is not None:
        return False
    try:
        import ee  # noqa: F401
        return True
    except Exception as e:
        _EE_IMPORT_ERROR = str(e)
        return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    # Allow passing repo-relative "@data/..." style paths from any CWD.
    if p.startswith('@'):
        return _repo_root() / p
    return path


def _reservoir_update(
    sample_x: List[float],
    sample_y: List[float],
    new_x: np.ndarray,
    new_y: np.ndarray,
    cap: int,
    rng: random.Random,
    seen: int,
) -> int:
    """Reservoir sampling update.

    Keeps a uniform sample of size `cap` from a stream of paired observations.

    Returns:
        Updated seen count.
    """
    if cap <= 0:
        return seen

    n = int(new_x.size)
    if n == 0:
        return seen

    # Fast path: fill up reservoir
    if len(sample_x) < cap:
        take = min(cap - len(sample_x), n)
        sample_x.extend(new_x[:take].astype(float, copy=False).tolist())
        sample_y.extend(new_y[:take].astype(float, copy=False).tolist())
        seen += take
        new_x = new_x[take:]
        new_y = new_y[take:]
        n = int(new_x.size)
        if n == 0:
            return seen

    # Reservoir replacement
    for i in range(n):
        seen += 1
        j = rng.randint(0, seen - 1)
        if j < cap:
            sample_x[j] = float(new_x[i])
            sample_y[j] = float(new_y[i])

    return seen


def _iter_tile_pairs(
    chm_tiles_dir: Path,
    meta_tiles_dir: Path,
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for pred_file in sorted(chm_tiles_dir.glob('*.tif')):
        tile_id = pred_file.stem
        ref_file = meta_tiles_dir / f"Meta_CHM_1m_{tile_id}.tif"
        if ref_file.exists():
            pairs.append((pred_file, ref_file))
    return pairs


def _download_meta_tile_to_temp(
    tile_id: str,
    bounds: Tuple[float, float, float, float],
    crs: str,
    out_path: Path,
    scale: float = 1.0,
    ee_source: str = 'meta',
    ee_band: str = '',
    value_scale: float = 1.0,
    value_offset: float = 0.0,
    ee_resample: str = 'none',
    debug: bool = False,
) -> bool:
    """Best-effort on-demand download of a Meta CHM tile via Google Earth Engine.

    Notes:
        - Requires `earthengine-api` (`ee`) and an authenticated session.
        - This downloads only the current tile bounds and writes it to `out_path`.
        - If anything fails, returns False.
    """
    if not _earth_engine_available():
        return False

    import ee  # type: ignore

    try:
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()

        if ee_source == 'meta':
            meta_chm: 'ee.Image' = ee.ImageCollection(
                'projects/meta-forest-monitoring-okw37/assets/CanopyHeight'
            ).mosaic()
        elif ee_source == 'meta_sat_io':
            # Community-catalog curated Meta/WRI canopy height collection.
            # See: projects/sat-io/open-datasets/facebook/meta-canopy-height
            meta_chm = ee.ImageCollection(
                'projects/sat-io/open-datasets/facebook/meta-canopy-height'
            ).mosaic()
        elif ee_source == 'gedi_monthly':
            # GEDI L2A monthly raster; choose a canopy top height relative metric.
            # Common choices: rh98 / rh100.
            band = ee_band.strip() if ee_band.strip() else 'rh98'
            meta_chm = (
                ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
                .select(band)
                .median()
            )
        elif ee_source == 'nasa_jpl_2005':
            # Very coarse (~1km) global canopy height product.
            meta_chm = ee.Image('NASA/JPL/global_forest_canopy_height_2005')
        else:
            raise ValueError(f"Unknown ee_source '{ee_source}'")

        meta_chm = meta_chm.toFloat()

        if value_scale != 1.0 or value_offset != 0.0:
            meta_chm = meta_chm.multiply(float(value_scale)).add(float(value_offset))

        # Important: EE often uses nearest-neighbor behavior unless resample() is set
        # before reprojection. This can make values look integer even if the source
        # has higher precision. Keep default 'none' to preserve native pixels.
        if ee_resample and ee_resample.lower() != 'none':
            meta_chm = meta_chm.resample(ee_resample.lower())

        xmin, ymin, xmax, ymax = bounds
        # EE is picky about CRS strings; use explicit EPSG format and ee.Projection.
        proj = ee.Projection(crs)
        region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj=proj, geodesic=False)

        # Force projection/scale for export consistency
        img = meta_chm.reproject(crs=proj, scale=scale).clip(region)

        url = img.getDownloadURL(
            {
                'scale': scale,
                'crs': crs,
                'region': region,
                'format': 'GEO_TIFF',
            }
        )

        import requests

        def _looks_like_zip(buf: bytes) -> bool:
            return buf.startswith(b'PK\x03\x04')

        def _looks_like_tif(buf: bytes) -> bool:
            # TIFF magic numbers: II*\x00 or MM\x00*
            return buf.startswith(b'II*\x00') or buf.startswith(b'MM\x00*')

        def _download_once() -> bytes:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            return r.content

        # In practice EE sometimes returns:
        # - a ZIP containing a GeoTIFF
        # - a raw GeoTIFF
        # - an HTML/text error payload (quota, invalid request, etc.)
        # We sniff content and handle all cases.
        last_err: Optional[str] = None
        content: Optional[bytes] = None
        for attempt in range(1, 4):
            try:
                content = _download_once()
                if content is None or len(content) < 16:
                    raise ValueError('Empty/too-small download')
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(min(10.0, 1.5 ** attempt))

        if content is None:
            raise RuntimeError(last_err or 'Unknown download error')

        head = content[:64]

        if _looks_like_zip(head):
            import io
            import zipfile

            with zipfile.ZipFile(io.BytesIO(content)) as z:
                tif_names = [n for n in z.namelist() if n.lower().endswith('.tif')]
                if not tif_names:
                    logger.error(f'No tif found in EE zip download for tile {tile_id}')
                    return False
                with z.open(tif_names[0]) as src, open(out_path, 'wb') as dst:
                    dst.write(src.read())
        elif _looks_like_tif(head):
            with open(out_path, 'wb') as dst:
                dst.write(content)
        else:
            # Likely HTML/text error. Log a short preview to help diagnose.
            preview = head.decode('utf-8', errors='replace')
            logger.error(f"EE download for tile {tile_id} returned unexpected payload (first bytes): {preview!r}")
            return False

        if not (out_path.exists() and out_path.stat().st_size > 0):
            return False

        # Ensure downloaded raster is float. Also apply scale/offset if present.
        # GeoTIFFs can store values as integers + scale/offset metadata.
        try:
            with rasterio.open(out_path) as src:
                dtype = str(src.dtypes[0])
                try:
                    scale0 = float(src.scales[0]) if src.scales else 1.0
                except Exception:
                    scale0 = 1.0
                try:
                    offset0 = float(src.offsets[0]) if src.offsets else 0.0
                except Exception:
                    offset0 = 0.0

                if debug:
                    logger.info(
                        f"Downloaded meta tile {tile_id}: dtype={dtype}, scale={scale0}, offset={offset0}, nodata={src.nodata}"
                    )
                    try:
                        rx, ry = src.res
                        logger.info(
                            f"Downloaded meta tile {tile_id}: crs={src.crs}, res=({rx:.3f},{ry:.3f}), bounds={src.bounds}"
                        )
                    except Exception:
                        pass

                # Always rewrite to float32 so downstream reads are consistent.
                data_raw = src.read(1)
                data = data_raw.astype(np.float32, copy=False)
                if scale0 != 1.0 or offset0 != 0.0:
                    data = data * scale0 + offset0

                if debug:
                    finite = np.isfinite(data)
                    if np.any(finite):
                        frac = np.abs(data[finite] - np.round(data[finite]))
                        non_int_ratio = float(np.mean(frac > 1e-6))
                        logger.info(
                            f"Downloaded meta tile {tile_id}: non-integer ratio (|x-round(x)|>1e-6)={non_int_ratio:.4f}"
                        )

                profile = src.profile.copy()
                profile.update(dtype='float32', count=1)

                # Some EE downloads can miss CRS tags even though the pixels are in the
                # requested CRS. If CRS is missing, enforce it so downstream WarpedVRT
                # alignment works reliably.
                if src.crs is None:
                    try:
                        profile.update(crs=CRS.from_string(crs))
                    except Exception:
                        # If CRS parsing fails, keep profile as-is.
                        pass

            tmp_float = out_path.with_suffix(out_path.suffix + '.float.tmp')
            with rasterio.open(tmp_float, 'w', **profile) as dst:
                dst.write(data, 1)
            tmp_float.replace(out_path)
        except Exception as e:
            logger.warning(f"Downloaded tile {tile_id}: failed to enforce float dtype ({e}). Proceeding with original.")

        return True
    except Exception as e:
        logger.error(f'Failed to download meta tile {tile_id} (crs={crs}): {e}')
        return False


def _compare_tile_pair_streaming(
    pred_file: Path,
    ref_file: Path,
    valid_min: float,
    valid_max: Optional[float],
    diff_clip: Optional[float],
    sample_cap: int,
    seed: int,
    resampling: Resampling,
) -> Optional[Dict]:
    try:
        rng = random.Random(seed)
        local_sample_pred: List[float] = []
        local_sample_ref: List[float] = []
        seen = 0

        sum_squared_errors = 0.0
        sum_errors = 0.0
        sum_ref = 0.0
        sum_ref_squared = 0.0
        total_valid = 0

        with rasterio.open(pred_file) as pred_src, rasterio.open(ref_file) as ref_src:
            pred_nodata = pred_src.nodata
            ref_nodata = ref_src.nodata

            same_grid = (
                pred_src.crs == ref_src.crs
                and pred_src.transform == ref_src.transform
                and pred_src.width == ref_src.width
                and pred_src.height == ref_src.height
            )

            if same_grid:
                ref_reader = ref_src
            else:
                ref_reader = WarpedVRT(
                    ref_src,
                    crs=pred_src.crs,
                    transform=pred_src.transform,
                    width=pred_src.width,
                    height=pred_src.height,
                    resampling=resampling,
                )

            try:
                for _, window in pred_src.block_windows(1):
                    pred = pred_src.read(1, window=window, masked=False).astype(np.float32, copy=False)
                    ref = ref_reader.read(1, window=window, masked=False).astype(np.float32, copy=False)

                    valid = np.isfinite(pred) & np.isfinite(ref)
                    if pred_nodata is not None:
                        valid &= pred != pred_nodata
                    if ref_nodata is not None:
                        valid &= ref != ref_nodata

                    valid &= (pred >= valid_min) & (ref >= valid_min)
                    if valid_max is not None:
                        valid &= (pred <= valid_max) & (ref <= valid_max)

                    if not np.any(valid):
                        continue

                    pv = pred[valid]
                    rv = ref[valid]

                    if diff_clip is not None:
                        keep = np.abs(pv - rv) <= diff_clip
                        pv = pv[keep]
                        rv = rv[keep]

                    if pv.size == 0:
                        continue

                    err = pv - rv
                    n = int(pv.size)

                    total_valid += n
                    sum_squared_errors += float(np.sum(err ** 2))
                    sum_errors += float(np.sum(err))
                    sum_ref += float(np.sum(rv))
                    sum_ref_squared += float(np.sum(rv ** 2))

                    if sample_cap > 0:
                        # keep sampling work bounded per-window
                        k = min(2000, n)
                        idx = np.linspace(0, n - 1, k, dtype=int)
                        seen = _reservoir_update(
                            local_sample_pred,
                            local_sample_ref,
                            pv[idx],
                            rv[idx],
                            cap=sample_cap,
                            rng=rng,
                            seen=seen,
                        )
            finally:
                if isinstance(ref_reader, WarpedVRT):
                    ref_reader.close()

        if total_valid == 0:
            return None

        rmse = float(np.sqrt(sum_squared_errors / total_valid))
        bias = float(sum_errors / total_valid)

        mean_ref = sum_ref / total_valid
        ss_tot = sum_ref_squared - 2 * mean_ref * sum_ref + total_valid * (mean_ref ** 2)
        ss_res = sum_squared_errors
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            'tile_id': pred_file.stem,
            'pred_file': str(pred_file),
            'ref_file': str(ref_file),
            'n_valid': int(total_valid),
            'rmse': rmse,
            'r2': r2,
            'bias': bias,
            'sum_squared_errors': float(sum_squared_errors),
            'sum_errors': float(sum_errors),
            'sum_ref': float(sum_ref),
            'sum_ref_squared': float(sum_ref_squared),
            'sample_pred': np.array(local_sample_pred) if sample_cap > 0 else np.array([]),
            'sample_ref': np.array(local_sample_ref) if sample_cap > 0 else np.array([]),
        }
    except Exception:
        return None


def _plot_scatter(sample_ref: np.ndarray, sample_pred: np.ndarray, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    if sample_ref.size == 0:
        logger.warning('No samples collected for scatter plot')
        return

    ax.scatter(sample_ref, sample_pred, s=8, alpha=0.6)

    min_plot = 0.0
    max_plot = float(max(sample_ref.max(), sample_pred.max()))
    ax.plot([min_plot, max_plot], [min_plot, max_plot], linestyle='--', color='k', linewidth=1.0)
    ax.set_xlim(min_plot, max_plot)
    ax.set_ylim(min_plot, max_plot)
    ax.set_xlabel('Meta CHM (m)')
    ax.set_ylabel('CHM tile (m)')
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate @data/chm tiles vs Meta CHM tiles with streaming (windowed) metrics and lightweight sampling.'
    )

    parser.add_argument('--chm-tiles', type=str, required=True, help='Directory with CHM tiles: <tile_id>.tif')
    parser.add_argument('--meta-tiles', type=str, required=True, help='Directory with Meta tiles: Meta_CHM_1m_<tile_id>.tif')
    parser.add_argument('--out-csv', type=str, required=True, help='Output CSV for per-tile metrics')
    parser.add_argument('--out-scatter', type=str, default=None, help='Optional scatter plot output PNG')

    parser.add_argument('--subset-percent', type=float, default=100.0, help='Process only x%% of tile pairs (1-100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subset selection and sampling')
    parser.add_argument('--workers', type=int, default=8, help='Thread workers')

    parser.add_argument('--valid-min', type=float, default=0.0, help='Minimum valid CHM value')
    parser.add_argument('--valid-max', type=float, default=None, help='Optional maximum valid CHM value')
    parser.add_argument('--diff-clip', type=float, default=0.0, help='Clip outliers by |pred-ref| <= diff_clip; set <=0 to disable')

    parser.add_argument(
        '--default-epsg',
        type=int,
        default=25832,
        help='Fallback EPSG code used when a CHM tile has missing/unknown CRS (default: 25832). Use 0 to disable fallback.',
    )

    parser.add_argument('--sample-cap', type=int, default=1000, help='Global sample cap for scatter plot (<=0 disables)')

    parser.add_argument(
        '--download-missing-meta',
        action='store_true',
        help='If Meta tile is missing locally, attempt an on-demand download to temp dir and delete after processing.',
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=None,
        help='Temp directory for downloaded tiles (defaults to system temp in @data/chm-validation/tmp_meta_tiles)',
    )

    parser.add_argument(
        '--keep-downloaded-meta',
        action='store_true',
        help='Do not delete temporary downloaded Meta tiles after processing (debugging).',
    )
    parser.add_argument(
        '--force-redownload',
        action='store_true',
        help='Always re-download reference tiles even if a temp tile path already exists.',
    )
    parser.add_argument(
        '--debug-download',
        action='store_true',
        help='Print dtype/scale/offset information for downloaded Meta tiles.',
    )
    parser.add_argument(
        '--ee-value-scale',
        type=float,
        default=1.0,
        help='Optional scaling applied to EE Meta CHM values before download (default 1.0).',
    )
    parser.add_argument(
        '--ee-value-offset',
        type=float,
        default=0.0,
        help='Optional offset applied to EE Meta CHM values before download (default 0.0).',
    )
    parser.add_argument(
        '--ee-source',
        type=str,
        default='meta',
        choices=['meta', 'meta_sat_io', 'gedi_monthly', 'nasa_jpl_2005'],
        help='Earth Engine source for reference CHM. Default meta (projects/meta-forest-monitoring-okw37/assets/CanopyHeight).',
    )
    parser.add_argument(
        '--ee-band',
        type=str,
        default='',
        help="Band to use for --ee-source gedi_monthly (default rh98 if empty).",
    )
    parser.add_argument(
        '--ee-scale',
        type=float,
        default=1.0,
        help='EE export pixel size in meters (default 1.0). Set explicitly for coarse products like GEDI (~25m).',
    )
    parser.add_argument(
        '--ee-resample',
        type=str,
        default='none',
        choices=['none', 'bilinear', 'bicubic'],
        help='Optional EE resampling applied before reprojection/download. Default none (native pixels).',
    )

    args = parser.parse_args()

    if args.download_missing_meta and not _earth_engine_available():
        logger.error("--download-missing-meta was requested, but Earth Engine (ee) is not available in this Python environment.")
        logger.error(f"Import error: {_EE_IMPORT_ERROR}")
        logger.error("Fix options:")
        logger.error("  1) Install dependencies in the environment you are using to run the script: earthengine-api and requests")
        logger.error("  2) Or disable downloads and point --meta-tiles to an existing local folder (e.g. '@data/Meta_chm_1m')")
        return

    chm_tiles_dir = _resolve_repo_path(args.chm_tiles)
    meta_tiles_dir = _resolve_repo_path(args.meta_tiles)
    out_csv = _resolve_repo_path(args.out_csv)
    out_scatter = _resolve_repo_path(args.out_scatter) if args.out_scatter else None

    rng = random.Random(args.seed)

    if not chm_tiles_dir.exists():
        logger.error(f"CHM tiles directory does not exist: {chm_tiles_dir}")
        return

    # meta_tiles_dir can be empty / non-existent if download-on-demand is enabled.
    if meta_tiles_dir.exists():
        pairs = _iter_tile_pairs(chm_tiles_dir, meta_tiles_dir)
    else:
        pairs = []

    # If we want download-on-demand, we also include tiles whose meta file is missing
    missing_pairs: List[Tuple[Path, Path]] = []
    if args.download_missing_meta:
        for pred_file in sorted(chm_tiles_dir.glob('*.tif')):
            tile_id = pred_file.stem
            ref_file = meta_tiles_dir / f"Meta_CHM_1m_{tile_id}.tif"
            if not ref_file.exists():
                missing_pairs.append((pred_file, ref_file))

    all_pairs = pairs + missing_pairs
    if not all_pairs:
        n_chm = len(list(chm_tiles_dir.glob('*.tif')))
        logger.error('No tiles found')
        logger.error(f"  CHM tiles found under: {chm_tiles_dir} -> {n_chm} tif")
        logger.error(f"  Meta tiles dir: {meta_tiles_dir} (exists={meta_tiles_dir.exists()})")
        logger.error("  Tip: if you run from a subfolder, pass paths like '@data/chm' so they resolve correctly")
        return

    if args.subset_percent < 100.0:
        n_subset = max(1, int(len(all_pairs) * args.subset_percent / 100.0))
        all_pairs = rng.sample(all_pairs, n_subset)
        logger.info(f'Using {args.subset_percent}% subset: {n_subset} tile pairs')

    diff_clip = args.diff_clip if args.diff_clip and args.diff_clip > 0 else None

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Global aggregation
    lock = threading.Lock() if args.workers > 1 else None
    sum_squared_errors = 0.0
    sum_errors = 0.0
    sum_ref = 0.0
    sum_ref_squared = 0.0
    total_valid = 0
    n_success = 0
    n_failed = 0
    n_skipped_missing_epsg = 0

    # Global reservoir sample
    sample_pred: List[float] = []
    sample_ref: List[float] = []
    sample_seen = 0

    temp_dir = _resolve_repo_path(args.temp_dir) if args.temp_dir else (_repo_root() / '@data/chm-validation/tmp_meta_tiles')
    if args.download_missing_meta:
        temp_dir.mkdir(parents=True, exist_ok=True)

    def _process_pair(pred_file: Path, ref_file: Path) -> Optional[Dict]:
        tmp_path: Optional[Path] = None
        try:
            if not ref_file.exists():
                if not args.download_missing_meta:
                    return None

                with rasterio.open(pred_file) as src:
                    bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                    epsg = src.crs.to_epsg() if src.crs is not None else None
                    default_epsg = args.default_epsg if args.default_epsg and args.default_epsg > 0 else None
                    if epsg is None:
                        epsg = default_epsg
                    if epsg is None:
                        nonlocal n_skipped_missing_epsg
                        if lock:
                            with lock:
                                n_skipped_missing_epsg += 1
                        else:
                            n_skipped_missing_epsg += 1
                        logger.error(
                            f"Tile {pred_file.stem}: missing/unknown CRS (src.crs={src.crs}). "
                            "Provide --default-epsg <code> to enable EE downloads."
                        )
                        return None
                    crs = f"EPSG:{int(epsg)}"

                # Include ee_source in filename to avoid accidentally reusing tiles from a
                # previous run with a different EE source.
                tmp_path = temp_dir / f"{str(args.ee_source)}_{pred_file.stem}.tif"

                if args.force_redownload and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

                ok = _download_meta_tile_to_temp(
                    tile_id=pred_file.stem,
                    bounds=bounds,
                    crs=crs,
                    out_path=tmp_path,
                    scale=float(args.ee_scale),
                    ee_source=str(args.ee_source),
                    ee_band=str(args.ee_band),
                    value_scale=float(args.ee_value_scale),
                    value_offset=float(args.ee_value_offset),
                    ee_resample=str(args.ee_resample),
                    debug=bool(args.debug_download),
                )
                if not ok:
                    return None

                ref_to_use = tmp_path
            else:
                ref_to_use = ref_file

            result = _compare_tile_pair_streaming(
                pred_file=pred_file,
                ref_file=ref_to_use,
                valid_min=args.valid_min,
                valid_max=args.valid_max,
                diff_clip=diff_clip,
                sample_cap=max(0, int(args.sample_cap)),
                seed=args.seed,
                resampling=Resampling.bilinear,
            )
            return result
        finally:
            if (not args.keep_downloaded_meta) and tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['tile_id', 'pred_file', 'ref_file', 'n_valid', 'rmse', 'r2', 'bias'],
        )
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_pair = {
                executor.submit(_process_pair, pred_file, ref_file): (pred_file, ref_file)
                for pred_file, ref_file in all_pairs
            }

            for future in tqdm(as_completed(future_to_pair), total=len(all_pairs), desc='Tile validation'):
                result = future.result()

                if result is None:
                    if lock:
                        with lock:
                            n_failed += 1
                    else:
                        n_failed += 1
                    continue

                writer.writerow(
                    {
                        'tile_id': result['tile_id'],
                        'pred_file': result['pred_file'],
                        'ref_file': result['ref_file'],
                        'n_valid': result['n_valid'],
                        'rmse': result['rmse'],
                        'r2': result['r2'],
                        'bias': result['bias'],
                    }
                )

                def _accumulate_one() -> None:
                    nonlocal sum_squared_errors, sum_errors, sum_ref, sum_ref_squared, total_valid, n_success, sample_seen
                    sum_squared_errors += result['sum_squared_errors']
                    sum_errors += result['sum_errors']
                    sum_ref += result['sum_ref']
                    sum_ref_squared += result['sum_ref_squared']
                    total_valid += result['n_valid']
                    n_success += 1

                    if args.sample_cap and args.sample_cap > 0 and result['sample_ref'].size > 0:
                        sample_seen = _reservoir_update(
                            sample_pred,
                            sample_ref,
                            result['sample_pred'],
                            result['sample_ref'],
                            cap=int(args.sample_cap),
                            rng=rng,
                            seen=sample_seen,
                        )

                if lock:
                    with lock:
                        _accumulate_one()
                else:
                    _accumulate_one()

    if total_valid == 0:
        logger.error('No valid pixels across processed tiles')
        return

    overall_rmse = float(np.sqrt(sum_squared_errors / total_valid))
    overall_bias = float(sum_errors / total_valid)
    mean_ref = sum_ref / total_valid
    ss_tot = sum_ref_squared - 2 * mean_ref * sum_ref + total_valid * (mean_ref ** 2)
    ss_res = sum_squared_errors
    overall_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    logger.info('\nValidation summary:')
    logger.info(f'  Tiles processed: {n_success}/{len(all_pairs)}')
    logger.info(f'  Failed tiles: {n_failed}')
    if n_skipped_missing_epsg > 0:
        logger.info(f'  Skipped tiles (missing EPSG/CRS): {n_skipped_missing_epsg}')
    logger.info(f'  Total valid pixels: {total_valid:,}')
    logger.info(f'  Overall RMSE: {overall_rmse:.3f} m')
    logger.info(f'  Overall R²: {overall_r2:.3f}')
    logger.info(f'  Overall Bias: {overall_bias:.3f} m')

    if out_scatter is not None and args.sample_cap and args.sample_cap > 0:
        _plot_scatter(
            sample_ref=np.array(sample_ref, dtype=float),
            sample_pred=np.array(sample_pred, dtype=float),
            out_path=out_scatter,
            title=f'Meta vs CHM tiles (n_sample={len(sample_ref)})',
        )
        logger.info(f'Saved scatter plot to {out_scatter}')


if __name__ == '__main__':
    main()
