# Training Pipeline for Multi-Task Forest Parameter Extraction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Training scripts and experiments for multi-task learning on the TreeSatAI-CHM dataset. Implements MLP and U-Net baselines (single-task) and a multi-task U-Net with uncertainty weighting for joint species classification and height regression from Sentinel-1 SAR.

## Structure

```
@training/
├── documentations/          # Experiment logs and results
│   ├── mlp-unet-baseline.md
│   └── multitask.md
│
├── jupyter-notebooks/       # Interactive analysis
│   └── treesatai_chm_mtl.ipynb
│
├── baseline_clf.py          # Single-task classification baseline
├── baseline_reg.py          # Single-task regression baseline
├── treesat_mtl.py          # Multi-task training script
└── README.md               # This file
```

## Dataset Configuration

- **Input:** 4-channel Sentinel-1 patches (VV, VH, VV/VH, CHM) at 6×6 pixels
- **Targets:**
  - Classification: 15 tree genera (multi-label)
  - Regression: canopy height in metres
- **Splits:** Train (45,343) / Val (5,038) / Test (5,038)
- **Class weighting:** Inverse frequency for imbalanced species

## Model Architectures

### 1. MLP Baseline

```python
Input (4×6×6) → Flatten (144) 
→ FC(512) → BN → ReLU → Dropout(0.3)
→ FC(512) → BN → ReLU → Dropout(0.3)
→ FC(512) → BN → ReLU → Dropout(0.3)
→ Output (15 for cls / 36 for reg)
```

3 hidden layers, 512 units each, batch norm, dropout 0.3. Softplus output for height regression (non-negative).

### 2. U-Net Baseline

```python
Encoder:  6×6 → 3×3 → 1×1 (channels: 4 → 64 → 128 → 256)
Decoder:  1×1 → 3×3 → 6×6 (with skip connections)
```

Two downsampling stages (limited by 6×6 patch size). Strided convolutions for downsampling; global average pooling for classification head.

### 3. Multi-Task U-Net

```python
Shared Encoder: 6×6 → 3×3 → 1×1
├─ Classification Head: GAP → FC(256) → Output(15)
└─ Regression Head: Decoder → Conv(1) → Softplus
```

Hard parameter sharing with asymmetric decoders. Uncertainty-weighted loss for automatic task balancing.

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | — |
| Learning rate | 1e-4 | Warmup 1e-5 → 1e-4 |
| Scheduler | CosineAnnealingLR | T_max = epochs |
| Batch size | 32 | — |
| Epochs | 100 | Early stopping (patience=10) |
| Weight decay | 1e-4 | — |

### Loss Functions

**Classification:**
```python
L_cls = BCEWithLogitsLoss(pos_weight=class_weights)
```

**Regression:**
```python
L_reg = MSELoss()
```

**Multi-task (uncertainty weighting):**
```python
L_total = (1/(2σ²_cls)) * L_cls + (1/2)log(σ²_cls)
        + (1/(2σ²_reg)) * L_reg + (1/2)log(σ²_reg)
```

σ²_cls and σ²_reg are learnable uncertainty parameters.

## Results

### Baseline

| Model | Task | Metric | Performance |
|-------|------|--------|-------------|
| MLP | Classification | F1-Score | 0.52 ± 0.02 |
| MLP | Regression | RMSE | 8.2 ± 0.3 m |
| MLP | Regression | R² | 0.07 |
| U-Net | Classification | F1-Score | 0.54 ± 0.01 |
| U-Net | Regression | RMSE | 7.9 ± 0.2 m |
| U-Net | Regression | R² | 0.09 |

### Multi-Task

| Loss Weighting | Classification F1 | Regression RMSE | Regression R² |
|----------------|-------------------|-----------------|---------------|
| Fixed (1:1) | 0.53 | 8.0 m | 0.08 |
| Uncertainty | 0.53 | 8.1 m | 0.08 |

Multi-task performance matches single-task at 50% parameter reduction. Gradient cosine similarity of 0.3–0.6 in early encoder layers indicates compatible feature learning. Uncertainty weighting converges to σ²_cls ≈ 0.8, σ²_reg ≈ 1.2. Absolute performance is limited by the 6×6 patch size, C-band insensitivity to canopy height, and class imbalance.
