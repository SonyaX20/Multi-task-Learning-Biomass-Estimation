"""Run the full preprocessing pipeline for TreeSat S1+CHM.

This orchestrates the following steps in order:

1. Area filtering + stratified 8:1:1 split of labels (60m labels)
   - labels_area_filter_and_split.build_area_filtered_labels_and_splits
2. Class weights computation (normalized inverse-frequency and others)
   - compute_class_weights.compute_class_weights
3. Building numpy training arrays from stacked S1+CHM data
   - build_training_arrays.build_training_arrays

Usage (from repo root):

  python data-processing/preprocessing/run_all_preprocessing.py --resolution 60
  python data-processing/preprocessing/run_all_preprocessing.py --resolution 200

Note: step (1) and (2) are resolution-agnostic and operate on 60m
label JSON; step (3) chooses between stacked_treesat_60m and
stacked_treesat_200m and writes to training-data-<res>m/.
"""

from __future__ import annotations

import argparse

from labels_area_filter_and_split import build_area_filtered_labels_and_splits
from compute_class_weights import compute_class_weights
from build_training_arrays import build_training_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run label filtering, class-weight computation, and training "
            "array building for stacked S1+CHM at 60m or 200m."
        )
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[60, 200],
        default=60,
        help="Target grid resolution in meters (60 or 200).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/3] Building area-filtered labels and stratified splits (60m labels)...")
    build_area_filtered_labels_and_splits()

    print("\n[2/3] Computing class weights (60m labels)...")
    compute_class_weights()

    print(
        f"\n[3/3] Building numpy training arrays from stacked S1+CHM "
        f"({args.resolution}m)..."
    )
    build_training_arrays(resolution=args.resolution)

    print("\nPreprocessing pipeline completed.")


if __name__ == "__main__":
    main()
