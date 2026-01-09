# Training Pipeline for Multi-Task Forest Parameter Extraction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **TL;DR:** Training scripts and experiments for multi-task learning on TreeSatAI-CHM dataset. Implements MLP and U-Net baselines for single-task learning, plus multi-task U-Net with uncertainty weighting for joint species classification and height regression from Sentinel-1 SAR imagery.

## Overview

This directory contains all training code, experiment configurations, and analysis notebooks for the TreeSatAI-CHM multi-task learning experiments. The goal is to investigate whether joint training on species classification and canopy height regression can leverage shared SAR feature representations to improve performance over single-task baselines.

**Key components:**
- **Baseline models**: MLP and U-Net for single-task classification/regression
- **Multi-task models**: Shared encoder with task-specific decoders
- **Loss strategies**: Fixed weighting vs. uncertainty-based automatic balancing
- **Analysis tools**: Gradient alignment, loss dynamics, qualitative visualization

## Repository Structure

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

## Experimental Setup

### Dataset Configuration

- **Input**: 4-channel Sentinel-1 patches (VV, VH, VV/VH, CHM) at 6×6 pixels
- **Targets**: 
  - Classification: 15 tree genera (multi-label)
  - Regression: Canopy height in meters
- **Splits**: Train (45,343) / Val (5,038) / Test (5,038)
- **Class weighting**: Inverse frequency weighting for imbalanced species

### Model Architectures

#### 1. MLP Baseline

Simple fully-connected architecture for ablation studies:

```python
Input (4×6×6) → Flatten (144) 
→ FC(512) → BN → ReLU → Dropout(0.3)
→ FC(512) → BN → ReLU → Dropout(0.3)
→ FC(512) → BN → ReLU → Dropout(0.3)
→ Output (15 for cls / 36 for reg)
```

**Key features:**
- 3 hidden layers with 512 units each
- Batch normalization for stable training
- Dropout (0.3) for regularization
- Softplus activation for height regression (non-negative outputs)

#### 2. U-Net Baseline

Encoder-decoder architecture adapted for small patches:

```python
Encoder:  6×6 → 3×3 → 1×1 (channels: 4 → 64 → 128 → 256)
Decoder:  1×1 → 3×3 → 6×6 (with skip connections)
```

**Key features:**
- Only 2 downsampling stages (limited by 6×6 patch size)
- Strided convolutions for learnable downsampling
- Skip connections preserve spatial details
- Global average pooling for classification
- Full spatial output for regression

#### 3. Multi-Task U-Net

Shared encoder with task-specific decoders:

```python
Shared Encoder: 6×6 → 3×3 → 1×1
├─ Classification Head: GAP → FC(256) → Output(15)
└─ Regression Head: Decoder → Conv(1) → Softplus
```

**Key features:**
- Hard parameter sharing (single encoder)
- Asymmetric decoders (classification uses bottleneck, regression uses full decoder)
- Uncertainty-weighted loss for automatic task balancing
- Gradient analysis for task compatibility monitoring

## Training Configuration

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW | - |
| **Learning rate** | 1e-4 | With warmup (1e-5 → 1e-4) |
| **Scheduler** | CosineAnnealingLR | T_max = epochs |
| **Batch size** | 32 | - |
| **Epochs** | 100 | With early stopping (patience=10) |
| **Weight decay** | 1e-4 | L2 regularization |

### Loss Functions

**Classification:**
```python
L_cls = BCEWithLogitsLoss(pos_weight=class_weights)
```

**Regression:**
```python
L_reg = MSELoss()  # RMSE computed as sqrt(MSE)
```

**Multi-task (Uncertainty Weighting):**
```python
L_total = (1/(2σ²_cls)) * L_cls + (1/2)log(σ²_cls)
        + (1/(2σ²_reg)) * L_reg + (1/2)log(σ²_reg)
```
where σ²_cls and σ²_reg are learnable uncertainty parameters.

## Results Summary

### Baseline Performance

| Model | Task | Metric | Performance |
|-------|------|--------|-------------|
| **MLP** | Classification | F1-Score | 0.52 ± 0.02 |
| **MLP** | Regression | RMSE | 8.2 ± 0.3 m |
| **MLP** | Regression | R² | 0.07 |
| **U-Net** | Classification | F1-Score | 0.54 ± 0.01 |
| **U-Net** | Regression | RMSE | 7.9 ± 0.2 m |
| **U-Net** | Regression | R² | 0.09 |

### Multi-Task Performance

| Loss Weighting | Classification F1 | Regression RMSE | Regression R² |
|----------------|-------------------|-----------------|---------------|
| **Fixed (1:1)** | 0.53 | 8.0 m | 0.08 |
| **Uncertainty** | 0.53 | 8.1 m | 0.08 |

### Key Outcomes

**Multi-task learning achieves comparable performance** to single-task baselines while using a single shared encoder (50% parameter reduction)

**Gradient analysis shows positive task alignment** in early encoder layers (cosine similarity 0.3-0.6), suggesting compatible feature learning

**Uncertainty weighting converges to reasonable balance** (σ²_cls ≈ 0.8, σ²_reg ≈ 1.2), automatically handling loss scale differences

**Absolute performance is moderate** due to:
- Small patch size (6×6 pixels) limits spatial context
- C-band SAR has limited sensitivity to canopy height
- Class imbalance in species distribution

**Height estimation remains challenging** (R² ≈ 0.08-0.09):
- SAR backscatter saturates in dense forests
- 10m resolution CHM labels contain uncertainty
- Limited penetration of C-band into canopy