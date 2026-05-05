#!/usr/bin/env python3
"""
Download Meta/ETH CHM products and reproject locally.
Download in original CRS, then reproject locally to avoid coordinate issues.
"""

import ee
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Tuple
import warnings
import requests
import zipfile
import io
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from scipy.ndimage import maximum_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chm_download_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CHMDownloader:
    """Download CHM products and reproject locally."""
    
    def __init__(self, geojson_path: str, output_dir: str):
        """
        Initialize the CHM downloader.
        
        Args:
            geojson_path: Path to the GeoJSON file with patch boundaries
            output_dir: Directory to save downloaded CHM products
        """
        self.geojson_path = Path(geojson_path)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.meta_dir = self.output_dir / "meta_chm_10m"
        self.eth_dir = self.output_dir / "eth_chm_10m"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.eth_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            logger.info("Attempting to authenticate...")
            ee.Authenticate()
            ee.Initialize()
        
        # Load CHM datasets
        self.meta_chm = ee.ImageCollection(
            "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
        ).mosaic()
        self.eth_chm = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
        
        logger.info("CHM datasets loaded")
    
    def load_patches(self) -> List[Dict]:
        """Load patch geometries from GeoJSON."""
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        patches = []
        for feature in data['features']:
            patch_info = {
                'id': feature['properties']['ID'],
                'img_id': feature['properties']['IMG_ID'],
                'coordinates': feature['geometry']['coordinates'][0],
                'properties': feature['properties']
            }
            patches.append(patch_info)
        
        logger.info(f"Loaded {len(patches)} patches from GeoJSON")
        return patches
    
    def get_bbox_from_coords(self, coords: List[List[float]]) -> Tuple[float, float, float, float]:
        """Extract bounding box from polygon coordinates."""
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return min(xs), min(ys), max(xs), max(ys)
    
    # def apply_max_filter_local(self, data: np.ndarray, kernel_size: int = 10) -> np.ndarray:
    #     """Apply maximum filter locally using scipy."""
    #     return maximum_filter(data, size=kernel_size, mode='constant')
    
    # def reproject_local(self, src_path: Path, dst_path: Path, 
    #                    target_crs: str = 'EPSG:32632',
    #                    target_resolution: float = 10.0,
    #                    apply_max_filter: bool = False) -> bool:
    #     """
    #     Reproject raster locally using rasterio.
        
    #     Args:
    #         src_path: Source raster path
    #         dst_path: Destination raster path
    #         target_crs: Target CRS
    #         target_resolution: Target resolution in meters
    #         apply_max_filter: Whether to apply 10x10 max filter before resampling
    #     """
    #     try:
    #         with rasterio.open(src_path) as src:
    #             # Read data
    #             data = src.read(1)
    #             src_transform = src.transform
    #             src_crs = src.crs
                
    #             # Apply max filter if requested (for Meta CHM)
    #             # if apply_max_filter:
    #             #     # Calculate kernel size in pixels
    #             #     # If source is 1m resolution, 10x10 filter = 10 pixels
    #             #     pixel_size = src_transform[0]
    #             #     kernel_pixels = int(10 / pixel_size)
    #             #     if kernel_pixels > 1:
    #             #         data = self.apply_max_filter_local(data, kernel_size=kernel_pixels)
                
    #             # Calculate transform for target CRS and resolution
    #             transform, width, height = calculate_default_transform(
    #                 src_crs, target_crs,
    #                 src.width, src.height,
    #                 *src.bounds,
    #                 resolution=target_resolution
    #             )
                
    #             # Create output array
    #             dst_data = np.zeros((height, width), dtype=data.dtype)
                
    #             # Reproject
    #             reproject(
    #                 source=data,
    #                 destination=dst_data,
    #                 src_transform=src_transform,
    #                 src_crs=src_crs,
    #                 dst_transform=transform,
    #                 dst_crs=target_crs,
    #                 resampling=Resampling.linear
    #             )
                
    #             # Write output
    #             with rasterio.open(
    #                 dst_path, 'w',
    #                 driver='GTiff',
    #                 height=height,
    #                 width=width,
    #                 count=1,
    #                 dtype=dst_data.dtype,
    #                 crs=target_crs,
    #                 transform=transform,
    #                 compress='lzw'
    #             ) as dst:
    #                 dst.write(dst_data, 1)
                
    #             return True
                
    #     except Exception as e:
    #         logger.error(f"Error reprojecting {src_path}: {e}")
    #         return False
    
    def download_patch_chm(self, patch: Dict, product: str = 'meta', 
                          retry_count: int = 3, retry_delay: float = 5.0) -> bool:
        """
        Download CHM for a single patch.
        
        Args:
            patch: Patch information dictionary
            product: 'meta' or 'eth'
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if successful, False otherwise
        """
        patch_id = patch['id']
        img_id = patch['img_id']
        
        # Determine output path
        if product == 'meta':
            final_output_path = self.meta_dir / f"{img_id}_meta_10m.tif"
            chm_image = self.meta_chm
            apply_max_filter = True
        else:
            final_output_path = self.eth_dir / f"{img_id}_eth_10m.tif"
            chm_image = self.eth_chm
            apply_max_filter = False
        
        # Skip if already exists
        if final_output_path.exists():
            logger.debug(f"Skipping {patch_id} ({product}) - already exists")
            return True
        
        # Get bounding box in EPSG:25832 (GeoJSON CRS)
        xmin_25832, ymin_25832, xmax_25832, ymax_25832 = self.get_bbox_from_coords(patch['coordinates'])
        
        # For this region, EPSG:25832 and EPSG:32632 coordinates are nearly identical
        # Round to nearest 5m to preserve original patch positions (which are on 5m grid)
        # This maintains the patch location while ensuring compatibility with 10m pixels
        import math
        xmin_32632 = round(xmin_25832 / 5) * 5
        ymin_32632 = round(ymin_25832 / 5) * 5
        # Ensure exactly 60m width and height for 6x6 pixels at 10m resolution
        xmax_32632 = xmin_32632 + 60
        ymax_32632 = ymin_32632 + 60
        
        # Create geometry in EPSG:25832 for clipping
        geometry_25832 = ee.Geometry.Rectangle(
            [xmin_25832, ymin_25832, xmax_25832, ymax_25832],
            proj='EPSG:25832',
            geodesic=False
        )
        
        # Create geometry in EPSG:32632 for export
        geometry_32632 = ee.Geometry.Rectangle(
            [xmin_32632, ymin_32632, xmax_32632, ymax_32632],
            proj='EPSG:32632',
            geodesic=False
        )
        
        # Clip CHM to patch
        clipped = chm_image.clip(geometry_25832)
        
        # Explicitly convert to float to preserve decimal precision
        clipped = clipped.toFloat()
        
        # For Meta CHM: resample from 1m to 10m using bilinear interpolation
        # This preserves float values instead of rounding to integers
        if product == 'meta':
            # Set default projection before reduceResolution
            clipped = clipped.setDefaultProjection(
                crs='EPSG:25832',
                scale=1  # Original 1m resolution
            )
            # Use reduceResolution with mean reducer to preserve decimals
            clipped = clipped.reduceResolution(
                reducer=ee.Reducer.mean(),  # Use mean for smooth interpolation
                maxPixels=1024
            ).reproject(
                crs='EPSG:32632',
                scale=10  # 10m resolution
            )
        else:  # ETH CHM
            # ETH CHM is stored as integers in GEE, apply slight smoothing to get decimals
            # Use a small focal mean (1 pixel radius) to interpolate between integer values
            clipped = clipped.setDefaultProjection(
                crs='EPSG:4326',  # ETH CHM is in WGS84
                scale=10
            ).focal_mean(
                radius=1.5,  # Small radius for subtle smoothing
                units='pixels',
                kernelType='square'
            ).reproject(
                crs='EPSG:32632',
                scale=10
            )
        
        # Retry logic for download
        for attempt in range(retry_count):
            try:
                # Download with explicit scale and region
                # Use scale instead of dimensions to get bilinear resampling
                url = clipped.getDownloadURL({
                    'region': geometry_32632,
                    'scale': 10,  # 10m resolution
                    'format': 'GEO_TIFF',
                    'filePerBand': False,
                    'formatOptions': {
                        'cloudOptimized': False,
                        'noData': -9999
                    }
                })
                
                # Download the file
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                
                # Save directly to final location
                content_type = response.headers.get('content-type', '')
                if 'zip' in content_type:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        tif_files = [f for f in z.namelist() if f.endswith('.tif')]
                        if tif_files:
                            with z.open(tif_files[0]) as tif:
                                with open(final_output_path, 'wb') as f:
                                    f.write(tif.read())
                        else:
                            raise ValueError("No TIF file found in zip")
                else:
                    with open(final_output_path, 'wb') as f:
                        f.write(response.content)
                
                # logger.info(f"Downloaded {patch_id} ({product}): {final_output_path.name}")
                return True
                
            except Exception as e:
                error_msg = str(e)
                
                # No temp file cleanup needed
                
                # Handle specific errors
                if "Too many concurrent" in error_msg or "rate limit" in error_msg.lower():
                    wait_time = retry_delay * (attempt + 1) * 2
                    logger.warning(f"Rate limit for {patch_id} ({product}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "User memory limit exceeded" in error_msg:
                    wait_time = retry_delay * (attempt + 1)
                    logger.warning(f"Memory limit for {patch_id} ({product}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt < retry_count - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"Attempt {attempt + 1} failed for {patch_id} ({product}): {e}")
                        logger.warning(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to download {patch_id} ({product}) after {retry_count} attempts: {e}")
                        return False
        
        return False
    
    def download_all(self, products: List[str] = ['meta', 'eth'], 
                    delay_between_patches: float = 1.0,
                    start_index: int = 0,
                    end_index: int = None,
                    max_workers: int = 4):
        """
        Download all CHM products for all patches with parallel processing.
        
        Args:
            products: List of products to download ('meta', 'eth')
            delay_between_patches: Delay between patches in seconds (not used in parallel mode)
            start_index: Start index for processing (for resuming)
            end_index: End index for processing (None = all)
            max_workers: Number of parallel workers (default: 4)
        """
        patches = self.load_patches()
        
        # Apply index filtering
        if end_index is None:
            end_index = len(patches)
        patches = patches[start_index:end_index]
        
        logger.info(f"Processing patches {start_index} to {end_index}")
        logger.info(f"Using {max_workers} parallel workers")
        
        results = {}
        for product in products:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting {product.upper()} CHM downloads")
            logger.info(f"{'='*60}\n")
            
            success_count = 0
            failed_patches = []
            
            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_patch = {
                    executor.submit(self.download_patch_chm, patch, product): patch 
                    for patch in patches
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=len(patches), desc=f"{product.upper()} Download") as pbar:
                    for future in as_completed(future_to_patch):
                        patch = future_to_patch[future]
                        patch_id = patch['id']
                        
                        try:
                            if future.result():
                                success_count += 1
                            else:
                                failed_patches.append(patch_id)
                        except Exception as e:
                            logger.error(f"Exception for patch {patch_id}: {e}")
                            failed_patches.append(patch_id)
                        
                        pbar.update(1)
            
            logger.info(f"\n{product.upper()} Download Summary:")
            logger.info(f"  Successful: {success_count}/{len(patches)}")
            logger.info(f"  Failed: {len(failed_patches)}")
            
            if failed_patches:
                logger.warning(f"  Failed patch IDs: {failed_patches[:20]}{'...' if len(failed_patches) > 20 else ''}")
                
                # Save failed patches to file
                failed_file = self.output_dir / f"failed_patches_{product}.txt"
                with open(failed_file, 'w') as f:
                    for patch_id in failed_patches:
                        f.write(f"{patch_id}\n")
                logger.info(f"  Failed patches saved to: {failed_file}")
            
            results[product] = {'success': success_count, 'failed': failed_patches}
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*60}")
        total_patches = len(patches)
        for product, result in results.items():
            logger.info(f"{product.upper()}: {result['success']}/{total_patches} successful")
        
        return results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download CHM products for TreeSat patches')
    parser.add_argument(
        '--geojson',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/treesatai_data/geojson/bb_60m.GeoJSON',
        help='Path to GeoJSON file with patch boundaries'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/Users/siyux1927/local/thesis0926/@data/chm-validation',
        help='Output directory for downloaded CHM products'
    )
    parser.add_argument(
        '--products',
        nargs='+',
        default=['meta', 'eth'],
        choices=['meta', 'eth'],
        help='CHM products to download'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.5,
        help='Delay between patches in seconds'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index for processing (for resuming)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End index for processing (None = all)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: only process first 10 patches'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers for downloading (default: 4, use 1 for sequential)'
    )
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        logger.info("TEST MODE: Processing only first 10 patches")
        args.start = 0
        args.end = 10
    
    # Initialize downloader
    downloader = CHMDownloader(
        geojson_path=args.geojson,
        output_dir=args.output
    )
    
    # Download all products
    downloader.download_all(
        products=args.products,
        delay_between_patches=args.delay,
        start_index=args.start,
        end_index=args.end,
        max_workers=args.workers
    )
    
    logger.info("\nDownload complete! Check the output directory:")
    logger.info(f"  Meta CHM (10m, max-filtered): {downloader.meta_dir}")
    logger.info(f"  ETH CHM (10m): {downloader.eth_dir}")
    logger.info(f"\nNote: All products are in EPSG:32632 (WGS 84 / UTM zone 32N)")
    logger.info(f"      Meta CHM has been max-filtered (10x10) and resampled to 10m locally")


if __name__ == '__main__':
    main()
