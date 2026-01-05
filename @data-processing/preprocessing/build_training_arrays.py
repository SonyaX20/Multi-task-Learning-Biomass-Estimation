"""Build numpy training arrays for stacked S1+CHM data.

Currently supports 60m and 200m resolutions via the ``resolution``
argument. For labels, we reuse the 60m stratified labels produced by
labels_area_filter_and_split.py, assuming filenames are identical
across resolutions.

Inputs (relative to repo root):
  - data/treesatai_data/s1_60m_stratified/classes.json
  - data/treesatai_data/s1_60m_stratified/labels_{train,val,test}_filtered.json
  - data/stacked_treesat_<res>m/*.tif  (4-band stacked VV, VH, VV/VH, CHM)

Outputs (relative to repo root):
  - data/training-data-<res>m/
      train_x.npy, train_y.npy
      val_x.npy,   val_y.npy
      test_x.npy,  test_y.npy
      classes.npy
      *_filenames.npy

No intermediate GeoTIFFs are written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from scipy.ndimage import convolve


NODATA_VALUE = -9999.0


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _fill_nodata_bilinear(
    arr: np.ndarray,
    nodata: float = NODATA_VALUE,
    max_iter: int = 5,
) -> np.ndarray:
    """Fill NaN/-9999 using a local bilinear-style kernel.

    We iteratively update only nodata pixels using a 3x3 kernel that
    gives higher weight to the center and its direct neighbours:

        [1, 2, 1]
        [2, 4, 2]
        [1, 2, 1]

    The value at each nodata pixel is replaced by a normalized weighted
    average of valid neighbours. Any remaining nodata after the loop is
    set to 0.0.
    """

    data = arr.astype(np.float32).copy()
    mask = np.isnan(data) | (data == nodata)

    if not mask.any():
        return np.nan_to_num(data, nan=0.0)

    kernel = np.array(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        dtype=np.float32,
    )
    kernel /= kernel.sum()

    for _ in range(max_iter):
        if not mask.any():
            break

        values = data.copy()
        values[mask] = 0.0
        valid = (~mask).astype(np.float32)

        num = convolve(values, kernel, mode="nearest")
        den = convolve(valid, kernel, mode="nearest")

        update_mask = mask & (den > 0)
        data[update_mask] = num[update_mask] / den[update_mask]

        mask = np.isnan(data) | (data == nodata)

    # Replace any remaining nodata with zeros
    data[mask] = 0.0
    return np.nan_to_num(data, nan=0.0)


def _build_split_arrays(
    stacked_dir: Path,
    labels: Dict[str, List[str]],
    class_to_idx: Dict[str, int],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Build X, y arrays for a given split.

    X: (N, 4, H, W)
    y: (N, num_classes)
    """

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    names: List[str] = []

    for fname, lbls in labels.items():
        tif_path = stacked_dir / fname
        if not tif_path.exists():
            # Skip samples without stacked data
            continue

        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)  # (bands, H, W)

        if data.shape[0] < 4:
            raise ValueError(f"Expected at least 4 bands in {tif_path}, got {data.shape[0]}")

        # Global percentile-based outlier clipping per band
        # Band order: 0=VV, 1=VH, 2=VV/VH, 3=CHM
        thresholds = [
            (-21.7151, 8.7294),   # VV
            (-32.2846, 0.7015),   # VH
            (-9.0865, 7.1253),    # VV/VH
            (0.0, 44.7409),       # CHM
        ]

        for b, (low, high) in enumerate(thresholds):
            if b >= data.shape[0]:
                break
            band = data[b]
            mask = (band < low) | (band > high)
            band[mask] = NODATA_VALUE
            data[b] = band

        # Bilinear-style fill for VV, VH, VV/VH bands
        for b in range(3):
            band = data[b]
            band = _fill_nodata_bilinear(band, nodata=NODATA_VALUE)
            data[b] = band

        # CHM band (index 3) is assumed to be already cleaned; just replace NaN
        data[3] = np.nan_to_num(data[3], nan=0.0)

        y = np.zeros(len(class_to_idx), dtype=np.float32)
        for lbl in lbls:
            idx = class_to_idx.get(lbl)
            if idx is not None:
                y[idx] = 1.0

        xs.append(data)
        ys.append(y)
        names.append(fname)

    if not xs:
        raise RuntimeError(f"No samples found in {stacked_dir} for given labels.")

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    return X, Y, names


def build_training_arrays(resolution: int = 60) -> None:
    if resolution not in (60, 200):
        raise ValueError("resolution must be 60 or 200")

    root = get_project_root()

    # Labels: always re-use 60m stratified labels (filenames are shared).
    labels_dir = root / "@data" / "treesatai_data" / "s1_60m_stratified"
    stacked_dir = root / "@data" / f"stacked_treesat_{resolution}m"
    out_dir = root / "@data" / f"training-data-{resolution}m"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = _load_json(labels_dir / "classes.json")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_labels = _load_json(labels_dir / "labels_train_filtered.json")
    val_labels = _load_json(labels_dir / "labels_val_filtered.json")
    test_labels = _load_json(labels_dir / "labels_test_filtered.json")

    train_x, train_y, train_names = _build_split_arrays(stacked_dir, train_labels, class_to_idx)
    val_x, val_y, val_names = _build_split_arrays(stacked_dir, val_labels, class_to_idx)
    test_x, test_y, test_names = _build_split_arrays(stacked_dir, test_labels, class_to_idx)

    np.save(out_dir / "train_x.npy", train_x)
    np.save(out_dir / "train_y.npy", train_y)
    np.save(out_dir / "val_x.npy", val_x)
    np.save(out_dir / "val_y.npy", val_y)
    np.save(out_dir / "test_x.npy", test_x)
    np.save(out_dir / "test_y.npy", test_y)
    np.save(out_dir / "classes.npy", np.array(classes, dtype=object))

    # Also save filenames for traceability
    np.save(out_dir / "train_filenames.npy", np.array(train_names, dtype=object))
    np.save(out_dir / "val_filenames.npy", np.array(val_names, dtype=object))
    np.save(out_dir / "test_filenames.npy", np.array(test_names, dtype=object))

    print(f"Training arrays saved to: {out_dir}")


if __name__ == "__main__":
    build_training_arrays(resolution=60)
