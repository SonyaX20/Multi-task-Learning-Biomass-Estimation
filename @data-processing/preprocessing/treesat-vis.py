"""Additional visualization utilities for TreeSat training data.

This module focuses on per-class CHM distributions using the
preprocessed numpy arrays under ``@data/training-data-60m``.

The main helper is:

  - vis_chm_histograms_by_primary_class:
      For a given split (default: train), determine the *primary class*
      per sample from the multi-hot label vector, collect all CHM pixels
      whose sample has that primary class, and plot one CHM histogram
      per class in a multi-panel figure.

The visual style and directory layout follow ``preprocessing/vis.py``.
"""

from __future__ import annotations

from math import ceil
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from vis import (
    _get_training_dir,
    _ensure_output_dir,
    _load_split_y,
    _load_classes,
    _prepare_hist_data_band,
)


def _load_split_x_mmap(split: str, resolution: int) -> np.memmap:
    """Memory-mapped loader for X arrays.

    Only used here because we mainly need the CHM band and want to keep
    memory usage modest even for large training sets.
    """

    data_dir = _get_training_dir(resolution)
    arr = np.load(data_dir / f"{split}_x.npy", mmap_mode="r")
    return arr


def _primary_class_indices(Y: np.ndarray) -> np.ndarray:
    """Return primary-class index per sample, or -1 if no label.

    Primary class is defined as the first positive entry in the
    multi-hot label vector (equivalent to argmax for 0/1 labels).
    """

    if Y.ndim != 2:
        raise ValueError(f"Expected Y to have shape (N, C), got {Y.shape}")

    N, _ = Y.shape
    primary = np.full(N, -1, dtype=int)

    sums = Y.sum(axis=1)
    valid_mask = sums > 0.0
    if not np.any(valid_mask):
        return primary

    Y_valid = Y[valid_mask]
    primary_idx = np.argmax(Y_valid, axis=1)
    primary[valid_mask] = primary_idx
    return primary


def vis_chm_histograms_by_primary_class(
    on: bool = True,
    resolution: int = 60,
    splits: tuple[str, ...] = ("train", "val", "test"),
    bins: int = 25,
    max_pixels_per_class: int | None = 200_000,
) -> None:
    """Plot CHM histograms per primary class in one multi-panel figure.

    By default, this function combines pixels from ``train``, ``val``,
    and ``test`` splits. For each class:

      1. For each requested split, loads X (S1+CHM) and Y (multi-label
         classes).
      2. Determines the primary class per sample (first positive label).
      3. For each class, gathers all CHM pixels for samples whose
         primary class equals that class index across all splits.
      4. Applies the same CHM cleaning logic as ``vis._prepare_hist_data_band``.
      5. Optionally subsamples per-class pixels to ``max_pixels_per_class``
         for memory efficiency.
      6. Plots one histogram per class in a grid of subplots.
    """

    if not on:
        return

    if isinstance(splits, str):
        splits = (splits,)

    for s in splits:
        if s not in {"train", "val", "test"}:
            raise ValueError("splits must be drawn from {'train', 'val', 'test'}")

    out_dir = _ensure_output_dir(resolution)

    classes: List[str] = _load_classes(resolution)
    num_classes = len(classes)

    # Collect per-class values across all splits
    vals_per_class: list[list[np.ndarray]] = [[] for _ in range(num_classes)]

    rng = np.random.default_rng(42)

    for split in splits:
        Y = _load_split_y(split, resolution)

        if Y.ndim != 2 or Y.shape[1] != num_classes:
            raise ValueError(
                f"Expected {split}_y.npy to have shape (N, {num_classes}), got {Y.shape}"
            )

        primary = _primary_class_indices(Y)
        N = primary.shape[0]

        # Load X via memmap and extract only CHM band (index 3)
        X = _load_split_x_mmap(split, resolution)
        if X.ndim != 4 or X.shape[0] != N or X.shape[1] < 4:
            raise ValueError(
                f"Expected {split}_x.npy to have shape (N, >=4, H, W), got {X.shape}"
            )

        chm = X[:, 3, :, :]  # memmap view, not a full copy yet

        for cls_idx in range(num_classes):
            sample_mask = primary == cls_idx
            if not np.any(sample_mask):
                continue

            chm_cls = chm[sample_mask]  # shape: (n_cls, H, W)
            vals = _prepare_hist_data_band(chm_cls, band_index=3)

            if vals.size == 0:
                continue

            # Optional per-split subsampling to control memory / rendering cost
            if max_pixels_per_class is not None and vals.size > max_pixels_per_class:
                idx = rng.choice(vals.size, size=max_pixels_per_class, replace=False)
                vals = vals[idx]

            vals_per_class[cls_idx].append(vals)

    # Prepare figure layout
    cols = 5
    rows = int(ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for cls_idx, cls_name in enumerate(classes):
        row = cls_idx // cols
        col = cls_idx % cols
        ax = axes[row, col]
        if not vals_per_class[cls_idx]:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        vals = np.concatenate(vals_per_class[cls_idx])

        # Optional global subsampling per class
        if max_pixels_per_class is not None and vals.size > max_pixels_per_class:
            idx = rng.choice(vals.size, size=max_pixels_per_class, replace=False)
            vals = vals[idx]

        ax.hist(vals, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(cls_name, fontsize=17)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.tick_params(axis="both", which="major", labelsize=13)

        if col == 0 and row == 1:
            ax.set_ylabel("Count", fontsize=17)
        else:
            ax.set_ylabel("")

        if row == rows - 1 and col == 2:
            ax.set_xlabel("CHM", fontsize=17)

    # Turn off any unused axes
    for idx in range(num_classes, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_axis_off()

    split_str = ",".join(splits)
    fig.suptitle(
        f"",
        fontsize=12,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)

    out_path = out_dir / f"chm_histograms_by_class_{'_'.join(splits)}_{resolution}m.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    vis_chm_histograms_by_primary_class(on=True, resolution=60)
