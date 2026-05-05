#!/usr/bin/env python3
"""
Compare predicted CHM (band 4 from stacked_treesat_60m) with Meta/ETH CHM products.
Calculate pixel-wise RMSE and generate scatter plots.
"""

import os
import warnings

# Set environment variables BEFORE importing rasterio to suppress PROJ warnings
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'
os.environ['PROJ_NETWORK'] = 'OFF'
os.environ['CPL_LOG'] = '/dev/null'  # Suppress GDAL/PROJ warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import csv

# Suppress warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chm_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress rasterio warnings in logger
logging.getLogger('rasterio').setLevel(logging.ERROR)

# Configure GDAL to suppress PROJ warnings
try:
    from osgeo import gdal
    gdal.SetConfigOption('CPL_LOG', '/dev/null')
    gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')
except ImportError:
    pass


class CHMComparator:
    """Compare predicted CHM with reference CHM products."""
    
    def __init__(self, 
                 predicted_dir: str,
                 meta_dir: str,
                 eth_dir: str,
                 output_dir: str = None):
        """
        Initialize the CHM comparator.
        
        Args:
            predicted_dir: Directory containing stacked TIF files with CHM in band 4
            meta_dir: Directory containing Meta CHM products
            eth_dir: Directory containing ETH CHM products
            output_dir: Directory for output plots and results
        """
        self.predicted_dir = Path(predicted_dir)
        self.meta_dir = Path(meta_dir)
        self.eth_dir = Path(eth_dir)
        self.output_dir = Path(output_dir) if output_dir else Path('chm_comparison_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Predicted CHM directory: {self.predicted_dir}")
        logger.info(f"Meta CHM directory: {self.meta_dir}")
        logger.info(f"ETH CHM directory: {self.eth_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def read_predicted_chm(self, file_path: Path) -> Tuple[np.ndarray, dict]:
        """
        Read band 4 (CHM) from stacked TIF file.
        
        Args:
            file_path: Path to stacked TIF file
            
        Returns:
            Tuple of (CHM array, metadata dict)
        """
        with rasterio.open(file_path) as src:
            # Read band 4 (CHM)
            chm = src.read(4)
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'shape': chm.shape
            }
        return chm, metadata
    
    def read_reference_chm(self, file_path: Path) -> np.ndarray:
        """
        Read reference CHM from Meta or ETH product.
        
        Args:
            file_path: Path to reference CHM file
            
        Returns:
            CHM array
        """
        with rasterio.open(file_path) as src:
            chm = src.read(1)
        return chm

    def _accumulate_window_stats(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        valid_min: float,
        valid_max: Optional[float],
        pred_nodata: Optional[float],
        ref_nodata: Optional[float],
        diff_clip: Optional[float],
        rng: random.Random,
        sample_cap: int,
        sample_predicted: List[float],
        sample_reference: List[float],
        total_valid_pixels_so_far: int,
    ) -> Tuple[Dict, int]:
        pred = pred.astype(np.float32, copy=False)
        ref = ref.astype(np.float32, copy=False)

        valid_mask = np.isfinite(pred) & np.isfinite(ref)

        if pred_nodata is not None:
            valid_mask &= (pred != pred_nodata)
        if ref_nodata is not None:
            valid_mask &= (ref != ref_nodata)

        valid_mask &= (pred >= valid_min) & (ref >= valid_min)
        if valid_max is not None:
            valid_mask &= (pred <= valid_max) & (ref <= valid_max)

        if not np.any(valid_mask):
            return {
                'n_valid': 0,
                'sum_squared_errors': 0.0,
                'sum_errors': 0.0,
                'sum_ref': 0.0,
                'sum_ref_squared': 0.0,
                'pred_sample': [],
                'ref_sample': [],
            }, total_valid_pixels_so_far

        pred_valid = pred[valid_mask]
        ref_valid = ref[valid_mask]

        if diff_clip is not None:
            diff_mask = np.abs(pred_valid - ref_valid) <= diff_clip
            pred_valid = pred_valid[diff_mask]
            ref_valid = ref_valid[diff_mask]

        n_valid = int(pred_valid.size)
        if n_valid == 0:
            return {
                'n_valid': 0,
                'sum_squared_errors': 0.0,
                'sum_errors': 0.0,
                'sum_ref': 0.0,
                'sum_ref_squared': 0.0,
                'pred_sample': [],
                'ref_sample': [],
            }, total_valid_pixels_so_far

        errors = pred_valid - ref_valid

        stats = {
            'n_valid': n_valid,
            'sum_squared_errors': float(np.sum(errors ** 2)),
            'sum_errors': float(np.sum(errors)),
            'sum_ref': float(np.sum(ref_valid)),
            'sum_ref_squared': float(np.sum(ref_valid ** 2)),
            'pred_sample': [],
            'ref_sample': [],
        }

        # Sampling (for optional scatter plots). Keep it cheap.
        if sample_cap > 0:
            desired = min(1000, n_valid)
            if len(sample_predicted) < sample_cap:
                take = min(desired, sample_cap - len(sample_predicted))
                stats['pred_sample'] = pred_valid[:take].tolist()
                stats['ref_sample'] = ref_valid[:take].tolist()
            else:
                # Reservoir: try a few random replacements
                k = min(100, desired)
                idx = np.linspace(0, n_valid - 1, k, dtype=int)
                pred_s = pred_valid[idx]
                ref_s = ref_valid[idx]
                for i in range(int(k)):
                    j = rng.randint(0, total_valid_pixels_so_far + n_valid - 1)
                    if j < sample_cap:
                        sample_predicted[j] = float(pred_s[i])
                        sample_reference[j] = float(ref_s[i])

        return stats, total_valid_pixels_so_far + n_valid

    def compare_tile_pair(
        self,
        pred_file: Path,
        ref_file: Path,
        valid_min: float = 2.0,
        valid_max: Optional[float] = None,
        diff_clip: Optional[float] = 10.0,
        sample_cap: int = 0,
        seed: int = 42,
        resampling: Resampling = Resampling.bilinear,
    ) -> Optional[Dict]:
        """Compare two single-band rasters by iterating over blocks/windows."""
        try:
            rng = random.Random(seed)
            sample_predicted: List[float] = []
            sample_reference: List[float] = []

            sum_squared_errors = 0.0
            sum_errors = 0.0
            sum_ref = 0.0
            sum_ref_squared = 0.0
            total_valid_pixels = 0

            with rasterio.open(pred_file) as pred_src, rasterio.open(ref_file) as ref_src:
                pred_nodata = pred_src.nodata
                ref_nodata = ref_src.nodata

                # If grids mismatch, read reference through a VRT aligned to predicted.
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
                        pred = pred_src.read(1, window=window, masked=False)
                        ref = ref_reader.read(1, window=window, masked=False)

                        window_stats, total_valid_pixels = self._accumulate_window_stats(
                            pred=pred,
                            ref=ref,
                            valid_min=valid_min,
                            valid_max=valid_max,
                            pred_nodata=pred_nodata,
                            ref_nodata=ref_nodata,
                            diff_clip=diff_clip,
                            rng=rng,
                            sample_cap=sample_cap,
                            sample_predicted=sample_predicted,
                            sample_reference=sample_reference,
                            total_valid_pixels_so_far=total_valid_pixels,
                        )

                        sum_squared_errors += window_stats['sum_squared_errors']
                        sum_errors += window_stats['sum_errors']
                        sum_ref += window_stats['sum_ref']
                        sum_ref_squared += window_stats['sum_ref_squared']

                        if window_stats['pred_sample']:
                            sample_predicted.extend(window_stats['pred_sample'])
                            sample_reference.extend(window_stats['ref_sample'])
                            if len(sample_predicted) > sample_cap > 0:
                                sample_predicted[:] = sample_predicted[:sample_cap]
                                sample_reference[:] = sample_reference[:sample_cap]
                finally:
                    if isinstance(ref_reader, WarpedVRT):
                        ref_reader.close()

            if total_valid_pixels == 0:
                return None

            rmse = float(np.sqrt(sum_squared_errors / total_valid_pixels))
            bias = float(sum_errors / total_valid_pixels)

            mean_ref = sum_ref / total_valid_pixels
            ss_tot = sum_ref_squared - 2 * mean_ref * sum_ref + total_valid_pixels * (mean_ref ** 2)
            ss_res = sum_squared_errors
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            return {
                'tile_id': pred_file.stem,
                'pred_file': str(pred_file),
                'ref_file': str(ref_file),
                'n_valid': int(total_valid_pixels),
                'rmse': rmse,
                'r2': r2,
                'bias': bias,
                'sum_squared_errors': float(sum_squared_errors),
                'sum_errors': float(sum_errors),
                'sum_ref': float(sum_ref),
                'sum_ref_squared': float(sum_ref_squared),
                'sample_pred': np.array(sample_predicted) if sample_cap > 0 else np.array([]),
                'sample_ref': np.array(sample_reference) if sample_cap > 0 else np.array([]),
            }
        except Exception:
            return None

    def compare_meta_tiles(
        self,
        chm_tiles_dir: str,
        meta_tiles_dir: str,
        output_csv: str,
        subset_percent: float = 100.0,
        random_seed: int = 42,
        workers: int = 8,
        valid_min: float = 2.0,
        valid_max: Optional[float] = None,
        diff_clip: Optional[float] = 10.0,
        sample_cap: int = 0,
    ) -> Optional[Dict]:
        """Compare @data/chm/<tile>.tif against @data/Meta_chm_1m/Meta_CHM_1m_<tile>.tif."""
        chm_tiles_dir = str(chm_tiles_dir)
        meta_tiles_dir = str(meta_tiles_dir)
        pred_dir = Path(chm_tiles_dir)
        ref_dir = Path(meta_tiles_dir)

        pred_files = list(pred_dir.glob('*.tif'))
        pairs: List[Tuple[Path, Path]] = []
        for pred_file in pred_files:
            tile_id = pred_file.stem
            ref_file = ref_dir / f"Meta_CHM_1m_{tile_id}.tif"
            if ref_file.exists():
                pairs.append((pred_file, ref_file))

        if not pairs:
            logger.error("No matching tile pairs found")
            return None

        if subset_percent < 100.0:
            rng = random.Random(random_seed)
            n_subset = max(1, int(len(pairs) * subset_percent / 100.0))
            pairs = rng.sample(pairs, n_subset)
            logger.info(f"Using {subset_percent}% subset: {n_subset} tiles (seed={random_seed})")

        lock = threading.Lock()
        sum_squared_errors = 0.0
        sum_errors = 0.0
        sum_ref = 0.0
        sum_ref_squared = 0.0
        total_valid_pixels = 0
        n_success = 0
        n_failed = 0

        # For optional scatter plot
        sample_predicted: List[float] = []
        sample_reference: List[float] = []
        if sample_cap <= 0:
            sample_cap = 0

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'tile_id',
                    'pred_file',
                    'ref_file',
                    'n_valid',
                    'rmse',
                    'r2',
                    'bias',
                ],
            )
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_pair = {
                    executor.submit(
                        self.compare_tile_pair,
                        pred_file,
                        ref_file,
                        valid_min,
                        valid_max,
                        diff_clip,
                        sample_cap,
                        random_seed,
                    ): (pred_file, ref_file)
                    for pred_file, ref_file in pairs
                }

                for future in tqdm(as_completed(future_to_pair), total=len(pairs), desc='META Tile Validation'):
                    result = future.result()
                    if result is None:
                        with lock:
                            n_failed += 1
                        continue

                    writer.writerow({
                        'tile_id': result['tile_id'],
                        'pred_file': result['pred_file'],
                        'ref_file': result['ref_file'],
                        'n_valid': result['n_valid'],
                        'rmse': result['rmse'],
                        'r2': result['r2'],
                        'bias': result['bias'],
                    })

                    with lock:
                        sum_squared_errors += result['sum_squared_errors']
                        sum_errors += result['sum_errors']
                        sum_ref += result['sum_ref']
                        sum_ref_squared += result['sum_ref_squared']
                        total_valid_pixels += result['n_valid']
                        n_success += 1

                        if sample_cap > 0 and result['sample_pred'].size > 0:
                            if len(sample_predicted) < sample_cap:
                                take = min(sample_cap - len(sample_predicted), int(result['sample_pred'].size))
                                sample_predicted.extend(result['sample_pred'][:take].tolist())
                                sample_reference.extend(result['sample_ref'][:take].tolist())

        if total_valid_pixels == 0:
            logger.error('No valid pixels found across tiles')
            return None

        overall_rmse = float(np.sqrt(sum_squared_errors / total_valid_pixels))
        overall_bias = float(sum_errors / total_valid_pixels)
        mean_ref = sum_ref / total_valid_pixels
        ss_tot = sum_ref_squared - 2 * mean_ref * sum_ref + total_valid_pixels * (mean_ref ** 2)
        ss_res = sum_squared_errors
        overall_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        logger.info("\nMETA Tile Validation Summary:")
        logger.info(f"  Tiles processed: {n_success}/{len(pairs)}")
        logger.info(f"  Total valid pixels: {total_valid_pixels:,}")
        logger.info(f"  Overall RMSE: {overall_rmse:.3f} m")
        logger.info(f"  Overall R²: {overall_r2:.3f}")
        logger.info(f"  Overall Bias: {overall_bias:.3f} m")
        if n_failed > 0:
            logger.warning(f"  Failed tiles: {n_failed}")

        return {
            'n_tiles': len(pairs),
            'n_success': n_success,
            'n_failed': n_failed,
            'total_pixels': int(total_valid_pixels),
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'overall_bias': overall_bias,
            'all_predicted': np.array(sample_predicted) if sample_cap > 0 else np.array([]),
            'all_reference': np.array(sample_reference) if sample_cap > 0 else np.array([]),
        }
    
    def process_single_file(self, pred_file: Path, ref_file: Path, 
                           valid_threshold: float = 2.0) -> Optional[Dict]:
        """
        Process a single file pair and return statistics for online aggregation.
        
        Args:
            pred_file: Path to predicted CHM file
            ref_file: Path to reference CHM file
            valid_threshold: Minimum valid CHM value (default 2.0m)
            
        Returns:
            Dictionary with statistics or None if failed
        """
        try:
            # Read CHM data
            pred_chm, _ = self.read_predicted_chm(pred_file)
            ref_chm = self.read_reference_chm(ref_file)
            
            # Ensure same shape
            if pred_chm.shape != ref_chm.shape:
                return None
            
            # Create valid mask: both predicted and reference must be > threshold
            valid_mask = (pred_chm > valid_threshold) & (ref_chm > valid_threshold)
            valid_mask &= ~np.isnan(pred_chm) & ~np.isnan(ref_chm)
            valid_mask &= ~np.isinf(pred_chm) & ~np.isinf(ref_chm)
            
            if valid_mask.sum() == 0:
                return None
            
            # Extract valid pixels
            pred_valid = pred_chm[valid_mask]
            ref_valid = ref_chm[valid_mask]
            
            # Filter out pixels where absolute difference > 10m (outliers)
            diff_mask = np.abs(pred_valid - ref_valid) <= 10.0
            pred_valid = pred_valid[diff_mask]
            ref_valid = ref_valid[diff_mask]
            
            n_valid = len(pred_valid)
            
            if n_valid == 0:
                return None
            
            # Calculate statistics for online aggregation
            errors = pred_valid - ref_valid
            
            stats = {
                'n_valid': n_valid,
                'sum_squared_errors': float(np.sum(errors ** 2)),
                'sum_errors': float(np.sum(errors)),
                'sum_pred': float(np.sum(pred_valid)),
                'sum_ref': float(np.sum(ref_valid)),
                'sum_pred_squared': float(np.sum(pred_valid ** 2)),
                'sum_ref_squared': float(np.sum(ref_valid ** 2)),
                'pred_sample': pred_valid[:1000].tolist(),  # Sample for plot
                'ref_sample': ref_valid[:1000].tolist()
            }
            
            return stats
            
        except Exception as e:
            return None
    
    def find_matching_files(self, product: str = 'meta') -> List[Tuple[Path, Path]]:
        """
        Find matching predicted and reference CHM files.
        
        Args:
            product: 'meta' or 'eth'
            
        Returns:
            List of (predicted_path, reference_path) tuples
        """
        if product == 'meta':
            ref_dir = self.meta_dir
            ref_suffix = '_meta_10m.tif'
        else:
            ref_dir = self.eth_dir
            ref_suffix = '_eth_10m.tif'
        
        matching_pairs = []
        
        # Get all reference files
        ref_files = list(ref_dir.glob(f'*{ref_suffix}'))
        logger.info(f"Found {len(ref_files)} {product.upper()} reference files")
        
        for ref_file in ref_files:
            # Extract IMG_ID from reference filename
            # Format: <IMG_ID>_meta_10m.tif or <IMG_ID>_eth_10m.tif
            img_id = ref_file.name.replace(ref_suffix, '')
            
            # Find corresponding predicted file
            pred_file = self.predicted_dir / f"{img_id}.tif"
            
            if pred_file.exists():
                matching_pairs.append((pred_file, ref_file))
            else:
                logger.debug(f"No predicted file found for {img_id}")
        
        logger.info(f"Found {len(matching_pairs)} matching pairs for {product.upper()}")
        return matching_pairs
    
    def compare_product(self, 
                       product: str = 'meta',
                       subset_percent: float = 100.0,
                       random_seed: int = 42,
                       workers: int = 8) -> Dict:
        """
        Compare predicted CHM with reference product using multithreading.
        
        Args:
            product: 'meta' or 'eth'
            subset_percent: Percentage of files to process (1-100)
            random_seed: Random seed for subset selection
            workers: Number of parallel threads
            
        Returns:
            Dictionary with comparison results (only overall RMSE, R², Bias, and plot data)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing with {product.upper()} CHM")
        logger.info(f"{'='*60}\n")
        
        # Find matching files
        matching_pairs = self.find_matching_files(product)
        
        if len(matching_pairs) == 0:
            logger.error(f"No matching files found for {product}")
            return None
        
        # Sample subset if requested
        if subset_percent < 100.0:
            random.seed(random_seed)
            n_subset = max(1, int(len(matching_pairs) * subset_percent / 100.0))
            matching_pairs = random.sample(matching_pairs, n_subset)
            logger.info(f"Using {subset_percent}% subset: {n_subset} files (seed={random_seed})")
        
        # Online statistics for overall metrics (thread-safe)
        lock = threading.Lock()
        sum_squared_errors = 0.0
        sum_errors = 0.0
        sum_pred = 0.0
        sum_ref = 0.0
        sum_pred_squared = 0.0
        sum_ref_squared = 0.0
        total_valid_pixels = 0
        
        # For scatter plot: sample pixels
        sample_predicted = []
        sample_reference = []
        max_sample_pixels = 100000
        
        n_success = 0
        n_failed = 0
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_file, pred_file, ref_file): 
                            (pred_file, ref_file) for pred_file, ref_file in matching_pairs}
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_file), total=len(matching_pairs), 
                             desc=f"{product.upper()} Comparison"):
                stats = future.result()
                
                if stats is not None:
                    with lock:
                        # Update online statistics
                        sum_squared_errors += stats['sum_squared_errors']
                        sum_errors += stats['sum_errors']
                        sum_pred += stats['sum_pred']
                        sum_ref += stats['sum_ref']
                        sum_pred_squared += stats['sum_pred_squared']
                        sum_ref_squared += stats['sum_ref_squared']
                        total_valid_pixels += stats['n_valid']
                        n_success += 1
                        
                        # Sample pixels for scatter plot
                        if len(sample_predicted) < max_sample_pixels:
                            sample_predicted.extend(stats['pred_sample'])
                            sample_reference.extend(stats['ref_sample'])
                        else:
                            # Reservoir sampling
                            for i in range(min(len(stats['pred_sample']), 100)):
                                j = random.randint(0, total_valid_pixels - 1)
                                if j < max_sample_pixels:
                                    sample_predicted[j] = stats['pred_sample'][i]
                                    sample_reference[j] = stats['ref_sample'][i]
                else:
                    with lock:
                        n_failed += 1
        
        # Calculate overall statistics
        if total_valid_pixels == 0:
            logger.error("No valid pixels found")
            return None
        
        # Overall RMSE
        overall_rmse = np.sqrt(sum_squared_errors / total_valid_pixels)
        
        # Overall Bias
        overall_bias = sum_errors / total_valid_pixels
        
        # Overall R²
        mean_ref = sum_ref / total_valid_pixels
        ss_tot = sum_ref_squared - 2 * mean_ref * sum_ref + total_valid_pixels * (mean_ref ** 2)
        ss_res = sum_squared_errors
        
        if ss_tot > 0:
            overall_r2 = 1.0 - (ss_res / ss_tot)
        else:
            overall_r2 = 0.0
        
        results = {
            'product': product,
            'n_files': len(matching_pairs),
            'n_success': n_success,
            'n_failed': n_failed,
            'total_pixels': total_valid_pixels,
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'overall_bias': overall_bias,
            'all_predicted': np.array(sample_predicted[:max_sample_pixels]),
            'all_reference': np.array(sample_reference[:max_sample_pixels])
        }
        
        # Print summary
        logger.info(f"\n{product.upper()} Comparison Summary:")
        logger.info(f"  Files processed: {n_success}/{len(matching_pairs)}")
        logger.info(f"  Total valid pixels: {total_valid_pixels:,}")
        logger.info(f"  Overall RMSE: {overall_rmse:.3f} m")
        logger.info(f"  Overall R²: {overall_r2:.3f}")
        logger.info(f"  Overall Bias: {overall_bias:.3f} m")
        if n_failed > 0:
            logger.warning(f"  Failed files: {n_failed}")
        
        return results
    
    def plot_scatter_comparison(self, 
                               results: Dict,
                               max_value: float = None,
                               save_path: str = None):
        """
        Create scatter plot with marginal histograms.
        
        Args:
            results: Results dictionary from compare_product()
            max_value: Maximum value for plot axes
            save_path: Path to save the plot
        """
        y_true = results['all_reference']
        y_pred = results['all_predicted']
        product = results['product'].upper()
        
        # Calculate metrics
        rmse = results['overall_rmse']
        r2 = results['overall_r2']
        
        # Determine plot range
        if max_value is None:
            min_plot = 0.0
            max_plot = max(float(y_true.max()), float(y_pred.max()))
        else:
            min_plot = 0.0
            max_plot = float(max_value)
        
        # Create bins for histograms
        bins = np.linspace(min_plot, max_plot, 50)
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 0.8), height_ratios=(0.8, 4), 
                             wspace=0.05, hspace=0.05)
        
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_joint = fig.add_subplot(gs[1, 0])
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_joint)
        
        # Top histogram (reference)
        ax_histx.hist(y_true, bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.5)
        ax_histx.set_ylabel("Count")
        ax_histx.tick_params(labelbottom=False)
        ax_histx.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_histx.set_xlim(min_plot, max_plot)
        
        # Right histogram (predicted)
        ax_histy.hist(y_pred, bins=bins, orientation="horizontal", color="#1f77b4", 
                     edgecolor="black", linewidth=0.5)
        ax_histy.tick_params(labelleft=False)
        ax_histy.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax_histy.set_ylim(min_plot, max_plot)
        
        # Calculate regression line
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        x_line = np.linspace(min_plot, max_plot, 100)
        y_reg = slope * x_line + intercept
        
        # Main scatter plot with hexbin
        ax_joint.hexbin(y_true, y_pred, gridsize=200, cmap="turbo", mincnt=1)
        ax_joint.plot([min_plot, max_plot], [min_plot, max_plot], 
                     linestyle="--", color="k", linewidth=1.0, label="1:1 line")
        ax_joint.plot(x_line, y_reg, linestyle="--", color="red", linewidth=1.0, 
                     label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
        ax_joint.set_xlim(min_plot, max_plot)
        ax_joint.set_ylim(min_plot, max_plot)
        ax_joint.set_xlabel(f"{product} CHM (m)")
        ax_joint.set_ylabel("Predicted CHM (m)")
        ax_joint.text(0.5, 0.95, f"R² = {r2:.3f}, RMSE = {rmse:.3f} m", 
                     transform=ax_joint.transAxes, ha="center", va="top",
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_joint.legend(loc='lower right', fontsize=8)
        
        plt.suptitle(f"Predicted vs {product} CHM Comparison", y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save comparison results to file.
        
        Args:
            results: Results dictionary from compare_product()
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            filename = f"{results['product']}_comparison_results.txt"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write(f"{results['product'].upper()} CHM Comparison Results\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Files Processed: {results['n_success']}/{results['n_files']}\n")
            f.write(f"Failed Files: {results['n_failed']}\n\n")
            f.write(f"Overall Statistics (all pixels):\n")
            f.write(f"  Total valid pixels: {results['total_pixels']:,}\n")
            f.write(f"  RMSE: {results['overall_rmse']:.3f} m\n")
            f.write(f"  R²: {results['overall_r2']:.3f}\n")
            f.write(f"  Bias: {results['overall_bias']:.3f} m\n")
        
        logger.info(f"Saved results to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Compare predicted CHM with Meta/ETH CHM products'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='patch',
        choices=['patch', 'meta_tiles'],
        help='Run mode: patch (existing pipeline) or meta_tiles (@data/chm vs Meta_CHM_1m tiles)'
    )
    parser.add_argument(
        '--predicted',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/stacked_treesat_60m',
        help='Directory containing predicted CHM files (stacked TIFs)'
    )

    parser.add_argument(
        '--chm-tiles',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/chm',
        help='Directory containing CHM tiles (single-band GeoTIFFs)'
    )
    parser.add_argument(
        '--meta-tiles',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/Meta_chm_1m',
        help='Directory containing Meta_CHM_1m tiles (Meta_CHM_1m_<tile>.tif)'
    )
    parser.add_argument(
        '--tile-csv',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/chm-validation/meta_tile_validation.csv',
        help='CSV output path for per-tile metrics (meta_tiles mode)'
    )
    parser.add_argument(
        '--meta',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/chm-validation/meta_chm_10m',
        help='Directory containing Meta CHM products'
    )
    parser.add_argument(
        '--eth',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/chm-validation/eth_chm_10m',
        help='Directory containing ETH CHM products'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/chm-validation/comparison_results',
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--products',
        nargs='+',
        default=['meta', 'eth'],
        choices=['meta', 'eth'],
        help='Products to compare'
    )
    parser.add_argument(
        '--subset',
        type=float,
        default=100.0,
        help='Percentage of files to process (1-100)'
    )

    parser.add_argument(
        '--valid-min',
        type=float,
        default=2.0,
        help='Minimum valid CHM height (m)'
    )
    parser.add_argument(
        '--valid-max',
        type=float,
        default=None,
        help='Optional maximum valid CHM height (m)'
    )
    parser.add_argument(
        '--diff-clip',
        type=float,
        default=10.0,
        help='Optional outlier clip: keep pixels where |pred-ref| <= diff_clip (set <=0 to disable)'
    )
    parser.add_argument(
        '--sample-cap',
        type=int,
        default=0,
        help='Max number of sampled pixels kept for scatter plot (0 disables sampling)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for subset sampling'
    )
    parser.add_argument(
        '--max-value',
        type=float,
        default=None,
        help='Maximum value for plot axes'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel threads for processing'
    )
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = CHMComparator(
        predicted_dir=args.predicted,
        meta_dir=args.meta,
        eth_dir=args.eth,
        output_dir=args.output
    )

    diff_clip = args.diff_clip if args.diff_clip and args.diff_clip > 0 else None

    if args.mode == 'meta_tiles':
        results = comparator.compare_meta_tiles(
            chm_tiles_dir=args.chm_tiles,
            meta_tiles_dir=args.meta_tiles,
            output_csv=args.tile_csv,
            subset_percent=args.subset,
            random_seed=args.seed,
            workers=args.workers,
            valid_min=args.valid_min,
            valid_max=args.valid_max,
            diff_clip=diff_clip,
            sample_cap=args.sample_cap,
        )

        if results is None:
            logger.error('META tile validation failed')
            return

        if args.sample_cap > 0 and not args.no_plot and results['all_reference'].size > 0:
            plot_path = comparator.output_dir / 'meta_tiles_scatter_plot.png'
            comparator.plot_scatter_comparison(
                results={
                    'product': 'meta_tiles',
                    'overall_rmse': results['overall_rmse'],
                    'overall_r2': results['overall_r2'],
                    'all_reference': results['all_reference'],
                    'all_predicted': results['all_predicted'],
                },
                max_value=args.max_value,
                save_path=str(plot_path),
            )

        logger.info("\nComparison complete!")
        return
    
    # Compare each product
    for product in args.products:
        logger.info(f"\nProcessing {product.upper()} comparison...")
        
        # Run comparison
        results = comparator.compare_product(
            product=product,
            subset_percent=args.subset,
            random_seed=args.seed,
            workers=args.workers
        )
        
        if results is None:
            logger.error(f"Failed to compare {product}")
            continue
        
        # Save results
        comparator.save_results(results)
        
        # Generate plot
        if not args.no_plot:
            plot_path = comparator.output_dir / f"{product}_scatter_plot.png"
            comparator.plot_scatter_comparison(
                results=results,
                max_value=args.max_value,
                save_path=str(plot_path)
            )
    
    logger.info("\nComparison complete!")


if __name__ == '__main__':
    main()
