"""Visualization utilities for DTM/DSM/CHM tiles.

Currently provides a triplet visualization of DTM, DSM, and CHM for
selected tile IDs. Each visualization function has an ``on`` parameter
that can be used to enable/disable plotting.

Figures are saved under ``plots/data-insight`` relative to the project
root.
"""

import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# Hard-coded list of interesting tile IDs (filenames) from 1 to 10
TILE_IDS: List[str] = [
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


def _get_project_root() -> str:
    """Infer repository root from this file location."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_processing_dir = os.path.dirname(this_dir)
    return os.path.dirname(data_processing_dir)


def _ensure_output_dir() -> str:
    """Create and return the output directory for plots."""
    out_dir = os.path.join(_get_project_root(), "@plots", "data-insight")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_raster(path: str) -> np.ndarray:
    """Load the first band of a GeoTIFF as float32."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def vis_dtm_dsm_chm(
    on: bool = True,
    indices: Iterable[int] | None = None,
    ) -> None:
    """Visualize DTM, DSM, and CHM triplets for selected tile IDs."""
    if not on:
        return

    root = _get_project_root()
    dtm_dir = os.path.join(root, "@data", "dtm1_tif")
    dsm_dir = os.path.join(root, "@data", "dsm1_tif")
    chm_dir = os.path.join(root, "@data", "chm_60m")

    if indices is None:
        indices = [1, 2, 3, 4]

    chosen = [TILE_IDS[idx - 1] for idx in indices]
    n = len(chosen)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 8))
    if n == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["DTM", "DSM", "CHM"]

    for col, tile_name in enumerate(chosen):
        dtm = _load_raster(os.path.join(dtm_dir, tile_name))
        dsm = _load_raster(os.path.join(dsm_dir, tile_name))
        chm = _load_raster(os.path.join(chm_dir, tile_name))

        data_arrays = [dtm, dsm, chm]
        cmaps = ["gray", "gray", "viridis"]
        images = []

        for row in range(3):
            im = axes[row, col].imshow(data_arrays[row], cmap=cmaps[row])
            images.append(im)

        axes[0, col].set_title(os.path.splitext(tile_name)[0], fontsize=9, pad=0.1)
        for row in range(3):
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].text(
                    -0.05, 0.5, row_labels[row],
                    transform=axes[row, col].transAxes,
                    fontsize=10, va="center", ha="right",
                    rotation=90
                )
            # Add thin colorbar with 5 ticks from min to max
            vmin = np.nanmin(data_arrays[row])
            vmax = np.nanmax(data_arrays[row])
            images[row].set_clim(vmin, vmax)
            cbar = fig.colorbar(images[row], ax=axes[row, col], fraction=0.046, pad=0.02)
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    fig.suptitle("LGLN Patch", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02, wspace=0.2, top=0.95)
    plt.savefig(os.path.join(_ensure_output_dir(), "dtm_dsm_chm.png"), dpi=150)
    plt.close(fig)


def vis_chm_lgln_meta(
    on: bool = True,
    indices: Iterable[int] | None = None,
) -> None:
    """Visualize CHM comparison between LGLN and Meta."""
    if not on:
        return

    root = _get_project_root()
    chm_dir = os.path.join(root, "@data", "chm_60m")
    meta_chm_dir = os.path.join(root, "@data", "Meta_chm_1m")

    if indices is None:
        indices = [1, 2, 3, 4]

    chosen = [TILE_IDS[idx - 1] for idx in indices]
    n = len(chosen)

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 5.5),
                              gridspec_kw={'height_ratios': [1, 1]})
    if n == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["CHM", "Meta"]

    # Collect metrics for each column
    metrics_list = []

    for col, tile_name in enumerate(chosen):
        chm_lgln = _load_raster(os.path.join(chm_dir, tile_name))
        chm_meta = _load_raster(os.path.join(meta_chm_dir, "Meta_CHM_1m_" + tile_name))

        data_arrays = [chm_lgln, chm_meta]
        images = []

        for row in range(2):
            im = axes[row, col].imshow(data_arrays[row], cmap="viridis", aspect='auto')
            images.append(im)

        axes[0, col].set_title(os.path.splitext(tile_name)[0], fontsize=15, pad=2)
        for row in range(2):
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].text(
                    -0.05, 0.5, row_labels[row],
                    transform=axes[row, col].transAxes,
                    fontsize=15, va="center", ha="right",
                    rotation=90
                )
            vmin = np.nanmin(data_arrays[row])
            vmax = np.nanmax(data_arrays[row])
            images[row].set_clim(vmin, vmax)
            cbar = fig.colorbar(images[row], ax=axes[row, col], fraction=0.046, pad=0.02)
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

        # Calculate metrics between LGLN and Meta
        lgln_flat = chm_lgln.flatten()
        meta_flat = chm_meta.flatten()
        valid_mask = ~np.isnan(lgln_flat) & ~np.isnan(meta_flat)
        lgln_valid = lgln_flat[valid_mask]
        meta_valid = meta_flat[valid_mask]

        if len(lgln_valid) > 0:
            mae = np.mean(np.abs(lgln_valid - meta_valid))
            rmse = np.sqrt(np.mean((lgln_valid - meta_valid) ** 2))
            corr = np.corrcoef(lgln_valid, meta_valid)[0, 1]
            metrics_text = f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nCorr={corr:.3f}"
        else:
            metrics_text = "No valid data"

        metrics_list.append(metrics_text)

    fig.suptitle("", fontsize=15, y=0.99)
    plt.subplots_adjust(hspace=0.1, wspace=0.2, top=0.92, bottom=0.16)

    # Add metrics text below each column using figure coordinates
    for col, metrics_text in enumerate(metrics_list):
        # Get the position of the bottom axis in figure coordinates
        bbox = axes[1, col].get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, 0.02, metrics_text, fontsize=15, va="bottom", ha="center",
                 linespacing=1.2)

    plt.savefig(os.path.join(_ensure_output_dir(), "chm_lgln_meta.png"), dpi=150)
    plt.close(fig)


def _max_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample array using max pooling with given factor."""
    h, w = arr.shape
    new_h = h // factor
    new_w = w // factor
    trimmed = arr[:new_h * factor, :new_w * factor]
    reshaped = trimmed.reshape(new_h, factor, new_w, factor)
    return np.nanmax(reshaped, axis=(1, 3))


def vis_chm_lgln_eth(
    on: bool = True,
    indices: Iterable[int] | None = None,
    ) -> None:
    """Visualize CHM comparison between LGLN (downsampled to 10m with 5x5 smoothing) and ETH."""
    if not on:
        return

    root = _get_project_root()
    chm_dir = os.path.join(root, "@data", "chm_debug_tiles")
    eth_chm_dir = os.path.join(root, "@data", "ETH_chm_10m")

    if indices is None:
        indices = [1, 2, 3, 4, 5, 6]

    chosen = [TILE_IDS[idx - 1] for idx in indices]
    n = len(chosen)

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 5.5),
                              gridspec_kw={'height_ratios': [1, 1]})
    if n == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["CHM", "ETH"]

    # Collect metrics for each column
    metrics_list = []

    for col, tile_name in enumerate(chosen):
        chm_lgln = _load_raster(os.path.join(chm_dir, tile_name))
        chm_eth = _load_raster(os.path.join(eth_chm_dir, "ETH_CHM_10m_" + tile_name))

        # Downsample LGLN from 1m to 10m using max pooling (factor=10)
        # chm_lgln_10m = _max_downsample(chm_lgln, factor=10)
        
        # Apply 5x5 smoothing (mean filter)
        from scipy.ndimage import uniform_filter
        # chm_lgln_10m_smooth = uniform_filter(chm_lgln_10m, size=3, mode='nearest')
        chm_lgln_10m = chm_lgln
        chm_lgln_10m_smooth = chm_lgln_10m

        data_arrays = [chm_lgln_10m_smooth, chm_eth]
        images = []

        for row in range(2):
            im = axes[row, col].imshow(data_arrays[row], cmap="viridis", aspect='auto')
            images.append(im)

        axes[0, col].set_title(os.path.splitext(tile_name)[0], fontsize=15, pad=2)
        for row in range(2):
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].text(
                    -0.05, 0.5, row_labels[row],
                    transform=axes[row, col].transAxes,
                    fontsize=15, va="center", ha="right",
                    rotation=90
                )
            vmin = np.nanmin(data_arrays[row])
            vmax = np.nanmax(data_arrays[row])
            images[row].set_clim(vmin, vmax)
            cbar = fig.colorbar(images[row], ax=axes[row, col], fraction=0.046, pad=0.02)
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

        # Calculate metrics between LGLN 10m smoothed and ETH
        # Need to align sizes - use minimum common size
        min_h = min(chm_lgln_10m_smooth.shape[0], chm_eth.shape[0])
        min_w = min(chm_lgln_10m_smooth.shape[1], chm_eth.shape[1])
        lgln_cropped = chm_lgln_10m_smooth[:min_h, :min_w]
        eth_cropped = chm_eth[:min_h, :min_w]

        lgln_flat = lgln_cropped.flatten()
        eth_flat = eth_cropped.flatten()
        valid_mask = ~np.isnan(lgln_flat) & ~np.isnan(eth_flat)
        lgln_valid = lgln_flat[valid_mask]
        eth_valid = eth_flat[valid_mask]

        if len(lgln_valid) > 0:
            mae = np.mean(np.abs(lgln_valid - eth_valid))
            rmse = np.sqrt(np.mean((lgln_valid - eth_valid) ** 2))
            corr = np.corrcoef(lgln_valid, eth_valid)[0, 1]
            metrics_text = f"MAE={mae:.2f}\nRMSE={rmse:.2f}\nCorr={corr:.3f}"
        else:
            metrics_text = "No valid data"

        metrics_list.append(metrics_text)

    fig.suptitle("", fontsize=15, y=0.1)
    plt.subplots_adjust(hspace=0.1, wspace=0.2, top=0.92, bottom=0.16)

    # Add metrics text below each column using figure coordinates
    for col, metrics_text in enumerate(metrics_list):
        # Get the position of the bottom axis in figure coordinates
        bbox = axes[1, col].get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, 0.02, metrics_text, fontsize=15, va="bottom", ha="center",
                 linespacing=1.2)

    plt.savefig(os.path.join(_ensure_output_dir(), "chm_lgln_eth.png"), dpi=150)
    plt.close(fig)


def vis_stacked_s1_chm(
    on: bool = True,
    indices: Iterable[int] | None = None,
    resolution: int = 60,
    ) -> None:
    """Visualize stacked S1 (vv, vh, vv/vh) and CHM patches."""
    if not on:
        return

    stacked_dir = os.path.join(
        _get_project_root(), "@data", f"stacked_treesat_{resolution}m"
    )
    all_files = sorted(f for f in os.listdir(stacked_dir) if f.endswith(".tif"))

    if indices is None:
        indices = [1, 2, 3]

    chosen_files = [all_files[idx - 1] for idx in indices]
    n = len(chosen_files)

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 10))
    if n == 1:
        axes = axes.reshape(4, 1)

    row_labels = ["VV", "VH", "VV/VH", "CHM"]

    for col, fname in enumerate(chosen_files):
        with rasterio.open(os.path.join(stacked_dir, fname)) as src:
            data = src.read().astype(np.float32)

        # Add sample name as column title
        axes[0, col].set_title(os.path.splitext(fname)[0], fontsize=9)

        for row in range(4):
            cmap = "viridis" if row == 3 else "gray"
            im = axes[row, col].imshow(data[row], cmap=cmap)
            axes[row, col].axis("off")
            
            # Add colorbar for each image
            vmin = np.nanmin(data[row])
            vmax = np.nanmax(data[row])
            im.set_clim(vmin, vmax)
            cbar = fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}' if x < 1 else f'{int(x)}'))
            
            # Add row label at the beginning of each row
            if col == 0:
                axes[row, col].text(
                    -0.05, 0.5, row_labels[row],
                    transform=axes[row, col].transAxes,
                    fontsize=10, va="center", ha="right",
                    rotation=90
                )

    fig.suptitle(f"VV, VH, VV/VH (Sentinel-1) and CHM, 10m resolution", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(_ensure_output_dir(), f"stacked_s1_chm_{resolution}m.png"), dpi=150
    )
    plt.close(fig)

def analyze_stacked_bands(
    on: bool = True,
    resolution: int = 60,
    ) -> None:
    """Analyze statistics for each band in stacked S1+CHM data."""
    if not on:
        return

    root = _get_project_root()
    stacked_dir = os.path.join(root, "@data", f"stacked_treesat_{resolution}m")

    if not os.path.isdir(stacked_dir):
        print(f"Directory not found: {stacked_dir}")
        return

    files = sorted([f for f in os.listdir(stacked_dir) if f.endswith(".tif")])
    if not files:
        print(f"No .tif files found in {stacked_dir}")
        return

    band_names = ["VV", "VH", "VV/VH", "CHM"]
    percentiles = [0.0001,0.001, 0.01, 0.1, 1, 2, 5, 95, 98, 99, 99.9, 99.99, 99.999, 99.9999]
    nodata_val = -9999

    # Collect all data per band
    all_data = {i: [] for i in range(4)}
    nodata_counts = {i: 0 for i in range(4)}
    total_pixels = {i: 0 for i in range(4)}
    files_with_nodata = {i: 0 for i in range(4)}

    for fname in files:
        with rasterio.open(os.path.join(stacked_dir, fname)) as src:
            data = src.read().astype(np.float32)

        for band in range(4):
            band_data = data[band].ravel()
            total_pixels[band] += len(band_data)

            # Check for nodata (both -9999 and NaN)
            nodata_mask = (band_data == nodata_val) | np.isnan(band_data)

            nodata_count = np.sum(nodata_mask)
            nodata_counts[band] += nodata_count
            if nodata_count > 0:
                files_with_nodata[band] += 1

            # Store valid data
            valid_data = band_data[~nodata_mask]
            all_data[band].append(valid_data)

    # Print statistics for each band
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    for band in range(4):
        combined = np.concatenate(all_data[band])
        # Filter out any remaining NaN values
        combined = combined[~np.isnan(combined)]
        
        print(f"\n--- Band {band}: {band_names[band]} ---")
        print(f"NoData value: {nodata_val}")
        print(f"Files with NoData: {files_with_nodata[band]}/{len(files)} ({100*files_with_nodata[band]/len(files):.2f}%)")
        print(f"NoData pixels: {nodata_counts[band]}/{total_pixels[band]} ({100*nodata_counts[band]/total_pixels[band]:.4f}%)")
        
        if len(combined) > 0:
            print(f"Min: {np.min(combined):.4f}")
            print(f"Max: {np.max(combined):.4f}")
            print(f"Mean: {np.mean(combined):.4f}")
            print(f"Std: {np.std(combined):.4f}")
            
            pct_values = np.percentile(combined, percentiles)
            for p, v in zip(percentiles, pct_values):
                print(f"Percentile {p}: {v:.4f}")

            # Filter data for plotting
            plot_data = combined.copy()
            if band == 2:  # VV/VH band: filter out values < -10 and > 10
                plot_data = plot_data[(plot_data >= -3) & (plot_data <= 3)]
                print(f"Plotting {len(plot_data)} values after filtering [-10, 10]")
            elif band == 3:  # CHM band: filter out 0 values
                plot_data = plot_data[plot_data > 0.5]
                print(f"Plotting {len(plot_data)} values after removing zeros")

            # Plot histogram
            ax = axes[band]
            ax.hist(plot_data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_title(f"{band_names[band]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # se:
            print("No valid data!")

    fig.su# ptitle(f"Histograms for Bands", fontsize=12)
    plt# .tight_layout()
    plt.savefig(
        os.path.join(_ensure_output_dir(), f"stacked_band_histograms_{resolution}m.png"), dpi=150
    )
    plt.close(fig)
    print(f"\nHistogram saved to {_ensure_output_dir()}/stacked_band_histograms_{resolution}m.png")

if __name__ == "__main__":
    # vis_dtm_dsm_chm(on=True, indices=[1, 2, 3, 6])
    vis_chm_lgln_eth(on=True, indices=[1, 2, 3, 6])
    vis_chm_lgln_meta(on=True, indices=[1, 2, 3, 6])
    # vis_stacked_s1_chm(on=True, indices=[1, 20, 70], resolution=60)
    # analyze_stacked_bands(on=True, resolution=60)