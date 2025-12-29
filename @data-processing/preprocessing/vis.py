"""Visualization utilities for training npy arrays (S1+CHM).

This module works with the outputs of the preprocessing pipeline in
``data/training-data-<resolution>m`` and provides three main helpers:

1. vis_sample_grid: show one sample from train/val/test
   as 3 rows (splits) x 4 columns (bands: VV, VH, VV/VH, CHM).
2. vis_histograms: for each of train/val/test, plot one histogram
   per band (4 subplots per figure, 3 figures total).
3. vis_class_distribution: bar chart comparing class counts (primary
   class per sample) across train/val/test.

The visual style roughly follows the functions in
``data-processing/gen-chm/vis.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


NODATA_VALUE = -9999.0
BAND_LABELS = ["VV", "VH", "VV/VH", "CHM"]
SPLITS = ["train", "val", "test"]


def _get_project_root() -> Path:
    """Return the repository root inferred from this file location."""

    # This file lives in: <root>/data-processing/preprocessing/
    return Path(__file__).resolve().parents[2]


def _get_training_dir(resolution: int) -> Path:
    root = _get_project_root()
    return root / "@data" / "training-data-60m"


def _ensure_output_dir(resolution: int) -> Path:
    out_dir = _get_project_root() / "plots" / "training-data-insight"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_split_x(split: str, resolution: int) -> np.ndarray:
    data_dir = _get_training_dir(resolution)
    arr = np.load(data_dir / f"{split}_x.npy")
    return arr


def _load_split_y(split: str, resolution: int) -> np.ndarray:
    data_dir = _get_training_dir(resolution)
    arr = np.load(data_dir / f"{split}_y.npy")
    return arr


def _load_classes(resolution: int) -> List[str]:
    data_dir = _get_training_dir(resolution)
    arr = np.load(data_dir / "classes.npy", allow_pickle=True)
    classes = arr.tolist()
    return classes


def vis_sample_grid(
    on: bool = True,
    resolution: int = 60,
    train_idx: int = 0,
    val_idx: int = 0,
    test_idx: int = 0,
) -> None:
    """Visualize one sample from train/val/test as 3x4 grid.

    Rows:   train, val, test
    Columns: VV, VH, VV/VH, CHM
    """

    if not on:
        return

    out_dir = _ensure_output_dir(resolution)

    train_x = _load_split_x("train", resolution)
    val_x = _load_split_x("val", resolution)
    test_x = _load_split_x("test", resolution)

    splits_data = [
        ("Train", train_x, train_idx),
        ("Val", val_x, val_idx),
        ("Test", test_x, test_idx),
    ]

    for name, arr, idx in splits_data:
        if arr.ndim != 4:
            raise ValueError(
                f"Expected {name} X to have shape (N, 4, H, W), got {arr.shape}"
            )
        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(
                f"Index {idx} out of range for {name} split with {arr.shape[0]} samples"
            )

    fig, axes = plt.subplots(3, 4, figsize=(4 * 3.2, 3 * 3.0))
    if axes.ndim == 1:
        axes = axes.reshape(3, 4)

    row_labels = ["Train", "Val", "Test"]

    for row, (row_name, arr, idx) in enumerate(splits_data):
        sample = arr[idx]  # (4, H, W)

        for col in range(4):
            ax = axes[row, col]
            band = sample[col]
            cmap = "gray" if col < 3 else "viridis"

            im = ax.imshow(band, cmap=cmap)
            ax.axis("off")

            # Column titles: band names
            if row == 0:
                ax.set_title(BAND_LABELS[col], fontsize=9)

            # Row labels: split names
            if col == 0:
                ax.text(
                    -0.05,
                    0.5,
                    row_labels[row],
                    transform=ax.transAxes,
                    fontsize=10,
                    va="center",
                    ha="right",
                    rotation=90,
                )

            # Per-image colorbar with ~5 ticks
            vmin = float(np.nanmin(band))
            vmax = float(np.nanmax(band))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = 0.0, 1.0
            im.set_clim(vmin, vmax)
            
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
            cbar.ax.tick_params(labelsize=9)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}' if x < 1 else f'{int(x)}'))


    fig.suptitle(f"Train/Val/Test samples ({resolution} m)", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.05, wspace=0.2)

    out_path = out_dir / f"sample_grid_{resolution}m.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _prepare_hist_data_band(values: np.ndarray, band_index: int) -> np.ndarray:
    """Clean and optionally clip values for histogram plotting.

    - Remove NaN and very negative nodata-like values.
    - For CHM band, drop zeros (background).
    - For VV/VH band, clip to [-3, 3].
    """

    vals = values.astype(np.float32).ravel()
    mask = np.isfinite(vals) & (vals > NODATA_VALUE + 1.0)
    vals = vals[mask]

    if vals.size == 0:
        return vals

    if band_index == 2:
        # VV/VH: filter to [-3, 3]
        vals = vals[(vals >= -3) & (vals <= 3)]
        return vals

    if band_index == 3:
        # CHM: drop zeros (no canopy)
        vals = vals[vals > 0.5]
        return vals

    return vals


def vis_histograms(
    on: bool = True,
    resolution: int = 60,
    bins: int = 50,
) -> None:
    """Plot histograms per band for train/val/test.

    For each split, a 1x4 grid of histograms (one per band) is saved
    as a separate PNG under the training-data directory.
    """

    if not on:
        return

    out_dir = _ensure_output_dir(resolution)

    for split in SPLITS:
        X = _load_split_x(split, resolution)
        if X.ndim != 4 or X.shape[1] < 4:
            raise ValueError(
                f"Expected {split}_x.npy to have shape (N, 4, H, W), got {X.shape}"
            )

        fig, axes = plt.subplots(1, 4, figsize=(16, 3))

        for b in range(4):
            ax = axes[b]
            vals = _prepare_hist_data_band(X[:, b, :, :], band_index=b)

            if vals.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue

            ax.hist(vals, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_title(BAND_LABELS[b])
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        fig.suptitle(f"{split.capitalize()} histograms ({resolution} m)", fontsize=12)
        plt.tight_layout()

        out_path = out_dir / f"{split}_histograms_{resolution}m.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def vis_class_distribution(
    on: bool = True,
    resolution: int = 60,
) -> None:
    """Plot bar chart of primary-class counts for train/val/test.

    Primary class per sample is taken as the first positive entry in
    the multi-hot label vector (equivalent to argmax if labels are 0/1).
    """

    if not on:
        return

    out_dir = _ensure_output_dir(resolution)

    classes = _load_classes(resolution)
    num_classes = len(classes)

    split_counts: Dict[str, np.ndarray] = {}

    for split in SPLITS:
        Y = _load_split_y(split, resolution)
        if Y.ndim != 2 or Y.shape[1] != num_classes:
            raise ValueError(
                f"Expected {split}_y.npy to have shape (N, {num_classes}), got {Y.shape}"
            )

        # Determine primary class per sample (first positive label)
        sums = Y.sum(axis=1)
        valid_mask = sums > 0.0
        if not np.any(valid_mask):
            counts = np.zeros(num_classes, dtype=int)
        else:
            Y_valid = Y[valid_mask]
            primary_idx = np.argmax(Y_valid, axis=1)
            counts = np.bincount(primary_idx, minlength=num_classes)

        split_counts[split] = counts

    indices = np.arange(num_classes)
    width = 0.3

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.bar(indices - width, split_counts["train"], width, label="Train", edgecolor="black", linewidth=0.8)
    ax.bar(indices, split_counts["val"], width, label="Val", edgecolor="black", linewidth=0.8)
    ax.bar(indices + width, split_counts["test"], width, label="Test", edgecolor="black", linewidth=0.8)

    ax.set_xticks(indices)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Number of samples (primary class)")
    ax.set_title(f"Class distribution by split")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_xlim(-0.5, num_classes-0.5)

    plt.tight_layout(pad=0)

    out_path = out_dir / f"class_distribution_{resolution}m.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Convenience entry-point for quick visualization at 60 m.

    This will generate:
      - sample_grid_60m.png
      - train_histograms_60m.png, val_histograms_60m.png, test_histograms_60m.png
      - class_distribution_60m.png
    under data/training-data-60m/vis.
    """

    resolution = 60
    vis_sample_grid(on=True, resolution=resolution)
    vis_histograms(on=True, resolution=resolution)
    vis_class_distribution(on=True, resolution=resolution)


if __name__ == "__main__":
    main()

