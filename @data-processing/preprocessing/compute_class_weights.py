"""Compute class weights for 60m TreeSat classification.

This replicates the logic of 02_compute_s1_60m_class_weights.py but
uses the outputs of labels_area_filter_and_split.py:

  data/treesatai_data/s1_60m_stratified/
    classes.json
    labels_train_filtered.json

The main output is a JSON file:
  data/treesatai_data/s1_60m_stratified/class_weights.json

which contains, among others, the normalized inverse-frequency
weights under the key
  "class_weights_invfreq_normalized".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compute_class_weights() -> Path:
    root = get_project_root()
    out_dir = root / "@data" / "treesatai_data" / "s1_60m_stratified"

    classes_path = out_dir / "classes.json"
    train_labels_path = out_dir / "labels_train_filtered.json"

    if not classes_path.exists() or not train_labels_path.exists():
        raise FileNotFoundError(
            "Expected classes.json and labels_train_filtered.json in "
            "data/treesatai_data/s1_60m_stratified/. Run "
            "labels_area_filter_and_split.py first."
        )

    with classes_path.open("r") as f:
        classes = json.load(f)

    with train_labels_path.open("r") as f:
        train_labels = json.load(f)

    # Count label frequency in training set
    counts = {c: 0 for c in classes}
    for labels in train_labels.values():
        for lbl in labels:
            if lbl not in counts:
                counts[lbl] = 0
                classes.append(lbl)
            counts[lbl] += 1

    # Convert to array in the same order as classes
    class_imbal_weights = [int(counts[c]) for c in classes]

    counts_array = np.array(class_imbal_weights, dtype=float)
    zero_mask = counts_array == 0
    if zero_mask.any():
        print("Warning: some classes have zero samples in the training set:")
        for c, is_zero in zip(classes, zero_mask):
            if is_zero:
                print(f"  - {c}")
        min_nonzero = counts_array[~zero_mask].min()
        counts_array[zero_mask] = min_nonzero

    total = counts_array.sum()
    freq = counts_array / total

    # 1) Normalized inverse-frequency weights (original behavior)
    inv_freq = 1.0 / freq
    inv_freq_norm = inv_freq / inv_freq.sum()

    # 2) Median frequency balancing (median(freq) / freq)
    median_freq = np.median(freq)
    median_freq_bal = median_freq / freq
    median_freq_bal_norm = median_freq_bal / median_freq_bal.sum()

    # 3) Inverse sqrt frequency
    inv_sqrt_freq = 1.0 / np.sqrt(freq)
    inv_sqrt_freq_norm = inv_sqrt_freq / inv_sqrt_freq.sum()

    # 4) Effective-number-based weights
    beta_99 = 0.99
    eff_num_99 = (1.0 - np.power(beta_99, counts_array)) / (1.0 - beta_99)
    eff_w_99 = 1.0 / eff_num_99
    eff_w_99_norm = eff_w_99 / eff_w_99.sum()

    beta_999 = 0.999
    eff_num_999 = (1.0 - np.power(beta_999, counts_array)) / (1.0 - beta_999)
    eff_w_999 = 1.0 / eff_num_999
    eff_w_999_norm = eff_w_999 / eff_w_999.sum()

    out = {
        "classes": classes,
        "class_imbal_weights": class_imbal_weights,
        "class_freq": freq.tolist(),
        "class_weights_invfreq_normalized": inv_freq_norm.tolist(),
        "class_weights_median_freq_normalized": median_freq_bal_norm.tolist(),
        "class_weights_inv_sqrt_freq_normalized": inv_sqrt_freq_norm.tolist(),
        "class_weights_effective_num_beta_0_99_normalized": eff_w_99_norm.tolist(),
        "class_weights_effective_num_beta_0_999_normalized": eff_w_999_norm.tolist(),
    }

    out_path = out_dir / "class_weights.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print("Class weights saved to:", out_path)
    print("Classes:", classes)
    print("Raw counts:", class_imbal_weights)
    print("Class frequencies:")
    print(freq)
    print("Normalized inverse-frequency weights:")
    print(inv_freq_norm)

    return out_path


if __name__ == "__main__":
    compute_class_weights()
