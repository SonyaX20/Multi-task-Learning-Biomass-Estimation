"""Area-based label filtering and stratified train/val/test split for 60m TreeSat.

This reads the multi-label JSON
  data/treesatai_data/labels/TreeSatBA_v9_60m_multi_labels.json
applies an area threshold per class, and performs a stratified
8:1:1 train/val/test split based on a primary label.

Outputs (relative to repo root):
  data/treesatai_data/s1_60m_stratified/
    labels_train_filtered.json
    labels_val_filtered.json
    labels_test_filtered.json
    classes.json
    label_stats.json

The filtering logic follows the original
01_build_s1_60m_area_filtered_labels.py, but no pre-defined
splits are used; instead, a new stratified split is created.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


AREA_THRESHOLD = 0.07
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass
class SplitResult:
    train: Dict[str, List[str]]
    val: Dict[str, List[str]]
    test: Dict[str, List[str]]
    classes: List[str]


def get_project_root() -> Path:
    """Return the repository root inferred from this file location."""

    # This file lives in: <root>/data-processing/preprocessing/
    return Path(__file__).resolve().parents[2]


def filter_labels_by_threshold(
    labels_dict: Dict[str, List[Tuple[str, float]]],
    area_threshold: float = AREA_THRESHOLD,
) -> Dict[str, List[str]]:
    """Replicate TreeSat filter_labels_by_threshold logic.

    Input format:
        {filename: [[label, area], [label, area], ...]}
    Output format:
        {filename: [label, label, ...]}
    """

    filtered: Dict[str, List[str]] = {}
    for img, entries in labels_dict.items():
        for lbl, area in entries:
            if area > area_threshold:
                filtered.setdefault(img, []).append(lbl)
    return filtered


def _load_full_labels() -> Dict[str, List[Tuple[str, float]]]:
    root = get_project_root()
    labels_path = (
        root
        / "@data"
        / "treesatai_data"
        / "labels"
        / "TreeSatBA_v9_60m_multi_labels.json"
    )

    with labels_path.open("r") as f:
        full_labels = json.load(f)

    return full_labels


def _load_split_lists() -> tuple[list[str], list[str]]:
    root = get_project_root()
    base = root / "@data" / "treesatai_data"

    def _read_list(path: Path) -> list[str]:
        with path.open("r") as f:
            return [line.strip() for line in f if line.strip()]

    train_list = _read_list(base / "train_filenames.lst")
    test_list = _read_list(base / "test_filenames.lst")
    return train_list, test_list


def _build_primary_labels(filtered: Dict[str, List[str]]) -> Dict[str, str]:
    """Pick a primary label per sample for stratification.

    We simply take the first label in the filtered list, which
    corresponds to the largest-area class after filtering.
    """

    primary: Dict[str, str] = {}
    for fn, labels in filtered.items():
        if not labels:
            continue
        primary[fn] = labels[0]
    return primary


def _stratified_split_train_val(
    primary_labels: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Stratified train/val split within a predefined training set."""

    if not np.isclose(train_ratio + val_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    rng = np.random.default_rng(seed)

    per_class: Dict[str, List[str]] = defaultdict(list)
    for fn, cls in primary_labels.items():
        per_class[cls].append(fn)

    train_files: List[str] = []
    val_files: List[str] = []

    for cls, fns in per_class.items():
        fns = fns.copy()
        rng.shuffle(fns)
        n = len(fns)
        if n == 0:
            continue

        n_train = int(round(n * train_ratio))
        if n_train == 0 and n > 0:
            n_train = 1
        if n_train > n:
            n_train = n
        n_val = n - n_train

        train_files.extend(fns[:n_train])
        val_files.extend(fns[n_train:])

    return train_files, val_files


def _stratified_split(
    primary_labels: Dict[str, str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified 8:1:1 split based on primary label.

    Returns lists of filenames for train, val, and test.
    """

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)

    # Group filenames by primary class
    per_class: Dict[str, List[str]] = defaultdict(list)
    for fn, cls in primary_labels.items():
        per_class[cls].append(fn)

    train_files: List[str] = []
    val_files: List[str] = []
    test_files: List[str] = []

    for cls, fns in per_class.items():
        fns = fns.copy()
        rng.shuffle(fns)
        n = len(fns)
        if n == 0:
            continue

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least one sample in train if possible
        if n_train == 0 and n > 0:
            n_train = 1
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_files.extend(fns[:n_train])
        val_files.extend(fns[n_train : n_train + n_val])
        test_files.extend(fns[n_train + n_val :])

    return train_files, val_files, test_files


def build_area_filtered_labels_and_splits() -> SplitResult:
    """Filter labels by area and create stratified 8:1:1 splits.

    This function performs the full pipeline and writes JSON outputs
    under data/treesatai_data/s1_60m_stratified/.
    """

    root = get_project_root()
    out_dir = root / "@data" / "treesatai_data" / "s1_60m_stratified"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_labels = _load_full_labels()
    filtered_all = filter_labels_by_threshold(full_labels, AREA_THRESHOLD)

    train_list, test_list = _load_split_lists()

    # Restrict to original TreeSat train/test splits
    filtered_train = {
        fn: labels for fn, labels in filtered_all.items() if fn in train_list
    }
    filtered_test = {fn: labels for fn, labels in filtered_all.items() if fn in test_list}

    # Build primary-label map for stratification within original train split
    primary_train = _build_primary_labels(filtered_train)

    # Desired global ratios are TRAIN_RATIO:VAL_RATIO:TEST_RATIO (e.g. 0.8:0.1:0.1).
    # We keep the original TreeSat test split fixed, and only split the original
    # train set into new train/val such that within (train+val) we have 8:1.
    total_tv = TRAIN_RATIO + VAL_RATIO
    inner_train_ratio = TRAIN_RATIO / total_tv
    inner_val_ratio = VAL_RATIO / total_tv

    train_files, val_files = _stratified_split_train_val(
        primary_train,
        train_ratio=inner_train_ratio,
        val_ratio=inner_val_ratio,
    )

    # Build split dictionaries
    train_filtered = {
        fn: filtered_train[fn] for fn in train_files if fn in filtered_train
    }
    val_filtered = {fn: filtered_train[fn] for fn in val_files if fn in filtered_train}
    test_filtered = filtered_test

    # Compute class sets and counts
    class_counter_train: Counter[str] = Counter()
    class_counter_val: Counter[str] = Counter()
    class_counter_test: Counter[str] = Counter()

    for labels in train_filtered.values():
        class_counter_train.update(labels)
    for labels in val_filtered.values():
        class_counter_val.update(labels)
    for labels in test_filtered.values():
        class_counter_test.update(labels)

    classes = sorted(class_counter_train.keys())

    # Save filtered label files
    with (out_dir / "labels_train_filtered.json").open("w") as f:
        json.dump(train_filtered, f, indent=2, sort_keys=True)

    with (out_dir / "labels_val_filtered.json").open("w") as f:
        json.dump(val_filtered, f, indent=2, sort_keys=True)

    with (out_dir / "labels_test_filtered.json").open("w") as f:
        json.dump(test_filtered, f, indent=2, sort_keys=True)

    with (out_dir / "classes.json").open("w") as f:
        json.dump(classes, f, indent=2)

    stats = {
        "area_threshold": AREA_THRESHOLD,
        "n_total_samples": len(train_filtered) + len(val_filtered) + len(test_filtered),
        "n_train": len(train_filtered),
        "n_val": len(val_filtered),
        "n_test": len(test_filtered),
        "classes": classes,
        "class_counts_train": dict(class_counter_train),
        "class_counts_val": dict(class_counter_val),
        "class_counts_test": dict(class_counter_test),
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }
    with (out_dir / "label_stats.json").open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print("Area-filtered labels and stratified splits saved to:")
    print(out_dir / "labels_train_filtered.json")
    print(out_dir / "labels_val_filtered.json")
    print(out_dir / "labels_test_filtered.json")
    print(out_dir / "classes.json")
    print(out_dir / "label_stats.json")

    return SplitResult(
        train=train_filtered,
        val=val_filtered,
        test=test_filtered,
        classes=classes,
    )


if __name__ == "__main__":
    build_area_filtered_labels_and_splits()
