---
marp: true
---

# Experiment Log – MTL-UNet / MultiTaskTreeNet

---

## 0. Metadata

- **Experiment ID**: `mtl_unet_YYYYMMDD_xxx`
- **Date**: 
- **Researcher**: 
- **Code version**: (git commit / branch / tag)
- **Script / Notebook**: [model_mtl/mtl_unet.py](cci:7://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:0:0-0:0) + (train script)
- **Dataset version**: (e.g. `treesat_60m_stra_vX`)
- **Notes**: (one-line summary of experiment)

---

## 1. Objectives

## 1. Objectives

- **Primary goal**:  
  Compare a UNet-based multi-task model and a multilayer perceptron (MLP)-based multi-task model for (i) multi-label forest classification and (ii) canopy height model (CHM) regression from Sentinel-1 6×6 patches.

- **Secondary goals**:  
  - Quantify to what extent the poor CHM performance of the current UNet baseline is due to architectural limitations vs. data/label noise.
  - Evaluate whether a simple MLP baseline on patch-level features can match or outperform UNet for classification while remaining competitive for CHM.
  - Analyze error patterns for UNet vs. MLP with respect to forest structural attributes (e.g., low vs. high canopy, dense vs. sparse stands).

- **Hypotheses**:  
  - H1: For CHM regression, UNet should benefit from modeling local spatial context and therefore outperform an MLP that ignores spatial structure, unless the effective spatial signal in 6×6 patches is weak.  
  - H2: For multi-label forest classification, a well-regularized MLP on normalized patch features can achieve performance comparable to UNet, given the small spatial footprint.  
  - H3: If the MLP substantially improves CHM accuracy relative to the current UNet, this suggests that optimization or regularization issues, rather than spatial inductive bias, are limiting the UNet baseline.

---

## 2. Pre-research / Background

- **Related experiments / baselines**:
  - `mtl_unet_base`: shared UNet encoder for S1 6×6 patches, separate decoders for classification and CHM regression, fixed or uncertainty-weighted task losses.
  - `mlp_patch_flat`: shared MLP on flattened S1 6×6 patches with two task-specific heads (classification logits + per-pixel CHM map), trained on exactly the same splits as `mtl_unet_base`.

- **Key design choices from literature or prior work**:
  - Shared UNet encoder for S1/S2 6×6 patches
  - Multi-task loss with uncertainty weighting (`use_uncertainty_loss`)
  - CHM clipping to `[0.5, 50] m`
  - Remote-sensing studies on tree species classification show that multilayer perceptrons (MLPs) can be competitive with more complex models when they are provided with rich spectral–structural features and sufficient training data. For example, Sumsion et al. used an MLP on airborne hyperspectral and LiDAR-derived features to classify individual tree crowns, and found that MLP performance was comparable to or better than support vector machines and random forests when the feature space was high dimensional and carefully normalized (Sumsion et al., 2019).
  - For forest vertical structure retrieval, MLPs have been used to map PolInSAR or LiDAR waveform parameters to canopy height and cover. A recent study in Remote Sensing reported normalized RMSE values on the order of 10–15% for canopy height and canopy cover when regressing these biophysical variables from SAR/PolInSAR-derived descriptors using a shallow MLP (Remote Sensing, 2019, 11(4):381). These approaches typically rely on:
    - summarizing spatial information into per-plot or per-crown features (e.g. waveform metrics, texture measures, spectral indices),
    - strict normalization or standardization of all input features,
    - relatively small MLPs (1–3 hidden layers with a few hundred units) with dropout and/or L2 regularization to control overfitting.
  - Compared with convolutional architectures such as UNet, MLPs discard explicit spatial inductive biases but are simpler to optimize and serve as strong baselines when (i) spatial context is limited (small patches like 6×6) or (ii) the main predictive signal lies in aggregated spectral statistics rather than fine-scale patterns.

- **Questions this run is meant to answer**:
  - Does a carefully tuned MLP baseline on patch-level S1 features close the performance gap in CHM regression compared to the current UNet, or does UNet’s spatial modeling remain advantageous?  
  - For multi-label forest classification, is the additional complexity of a UNet justified over an MLP in terms of F1 score and calibration, given the same inputs and data splits?

**References**

- Sumsion, G. R., Bradshaw, M. S., Hill, K. T., Pinto, L. D. G., & Piccolo, S. R. *Remote sensing tree classification with a multilayer perceptron*. PeerJ, 2019.  
- *Retrieval of Forest Vertical Structure from PolInSAR Data by Machine Learning*. Remote Sensing, 2019, 11(4):381.

---

## 3. TODO / Plan

- **High-level steps**
  - [ ] Prepare data & config
  - [ ] Run training
  - [ ] Run evaluation (train/val/test)
  - [ ] Generate plots (curves, t-SNE, etc.)
  - [ ] Summarize insights and decide next steps

- **Concrete tasks**
  - [ ] Set `enable_self_attention` = ...
  - [ ] Set `enable_cross_attention` = ...
  - [ ] Set `use_uncertainty_loss` / `cls_weight` / `reg_weight`
  - [ ] Log seed and all hyperparameters
  - [ ] Save metrics JSON + figures into `results/mtl_unet/...`

---

## 4. Setup

### 4.1 Data

- **Base directory**:  
  `BASE_DIR = "/content/drive/MyDrive/data/treesat_60m_stra/"` (or local equivalent)

- **Input**:
  - Source: `train_images.npy`, `val_images.npy`, `test_images.npy`
  - Shape: `(N, C, H, W)` with `C ≥ 4`, `H = 6`, `W = 6`
  - Channels used for model input: `images[:, 1:4, ...]` (3 bands)
  - Random seed: [set_seed(42)](cci:1://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:38:0-46:42) (or new seed if different)

- **Targets**:
  - **Classification**: multi-label vectors from `*_labels.npy` (shape `(N, num_classes)`)
  - **Regression (CHM)**:
    - Source: `images[:, 0, ...]`
    - Thresholding: `CHM_MIN_THRESHOLD = 0.5`, `CHM_MAX_THRESHOLD = 50.0`
    - Invalid values set to `NaN` before loss

- **Splits**:
  - Train: `len(train_ds) = ...`
  - Val: `len(val_ds) = ...`
  - Test: `len(test_ds) = ...`

### 4.2 Model Configuration

- **Model type**:  
  - [MultiTaskTreeNet](cci:2://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:338:0-608:55) | [MultiTaskTransposedUNet](cci:2://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:612:0-867:55) | [UNetRegressor](cci:2://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:143:0-192:18) | [UNetClassifier](cci:2://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:194:0-249:21)

- **Shared encoder**:
  - `base_channels = ...`
  - Encoder blocks: [DoubleConv](cci:2://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:251:0-265:34), pooling: `MaxPool2d` / strided conv
  - Bottleneck depth: ...

- **Task heads**:
  - Classification head enabled: `enable_cls_decoder = ...`
  - Regression head enabled: `enable_reg_decoder = ...`
  - Number of classes: `num_classes = ...`

- **Attention / interaction**:
  - Self-attention: `enable_self_attention = ...`
  - Cross-task attention: `enable_cross_attention = ...`
  - Cross-att module: [CrossTaskAttentionMultiScale](cci:2://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:281:0-335:35)
  - Channel weights: `alpha_cls`, `alpha_reg` (initialized to ones)

- **Loss weighting**:
  - Uncertainty-based loss: `use_uncertainty_loss = ...`
  - If enabled: `log_sigma_cls`, `log_sigma_reg` initialized to 0
  - If disabled: fixed `cls_weight = ...`, `reg_weight = ...`
  - Task weight bounds: `[min_task_weight = 0.1, max_task_weight = 5.0]`

### 4.3 Training Hyperparameters

- **Hardware**:
  - Device: `cuda` / `cpu`
  - GPU model / RAM: ...

- **Optimization**:
  - Optimizer: (e.g. Adam, params)
  - Learning rate: ...
  - Weight decay: ...
  - LR schedule: ...

- **Training details**:
  - Batch size: ...
  - # Epochs (max): ...
  - Early stopping: (metric, patience)
  - Gradient clipping: (yes/no, value)
  - Class weights / focal loss (if used): ...

- **Logging & saving**:
  - Metrics file: `.../MultiTask_s1_mtl_report.json` (or similar)
  - Checkpoints: path & naming scheme
  - Random seed(s): ...

---

## 5. Results

### 5.1 Training Dynamics

- **Summary table (final & best epoch)**:

| Metric                 | Task | Best Epoch | Train | Val   | Test |
|------------------------|------|-----------:|------:|------:|-----:|
| Loss                   | both |           |       |       |      |
| Classification loss    | cls  |           |       |       |      |
| Regression loss        | reg  |           |       |       |      |
| Macro F1               | cls  |           |       |       |      |
| R²                     | reg  |           |       |       |      |
| RMSE                   | reg  |           |       |       |      |

- **Curves**:
  - `![Loss curves](path/to/loss_curves.png)`
  - `![Task weights over epochs](path/to/task_weight_curves.png)`  
    (w_cls, w_reg from [get_task_weights()](cci:1://file:///Users/siyux1927/local/thesis0926/model_mtl/mtl_unet.py:592:4-608:55))

### 5.2 Classification Metrics

- **Overall metrics**:

| Metric            | Train | Val  | Test |
|-------------------|------:|-----:|-----:|
| Macro F1          |       |      |      |
| Micro F1          |       |      |      |
| Weighted F1       |       |      |      |
| Precision         |       |      |      |
| Recall            |       |      |      |
| mAP (avg_prec)    |       |      |      |
| Accuracy (if any) |       |      |      |

- **Per-class F1 (Val/Test)**:

| Class ID / Name | F1 (Val) | F1 (Test) | Support (Test) |
|-----------------|---------:|----------:|----------------|
| ...             |          |           |                |

- **Confusion analysis**:
  - Multi-label confusion matrix: `multilabel_confusion_matrix`
  - Standard confusion matrix (if using single-label view)

  `![Confusion matrix](path/to/confusion_matrix.png)`

- **PR or ROC curves (optional)**:
  - `![PR curves](path/to/pr_curves.png)`

### 5.3 Regression (CHM) Metrics

- **Global metrics**:

| Metric | Train | Val  | Test |
|--------|------:|-----:|-----:|
| R²     |       |      |      |
| RMSE   |       |      |      |
| MAE    |       |      |      |

- **Plots**:
  - `![Pred vs. GT scatter (test)](path/to/scatter_chm_test.png)`
  - `![Residual histogram](path/to/residual_hist.png)`
  - `![CHM distribution](path/to_chm_hist.png)`

### 5.4 Qualitative / Representation Analysis

- **t-SNE / feature visualization**:
  - t-SNE on penultimate features for classification:
    `![t-SNE embeddings](path/to/tsne_cls.png)`
  - Color by dominant class / vegetation type.

- **Patch examples**:
  - Input S1/S2 patch, GT labels, predicted labels, GT CHM, predicted CHM:
    `![Qualitative examples](path/to/qualitative_examples.png)`

- **Attention maps (if inspected)**:
  - `![Cross-task attention maps](path/to/cross_attention.png)`

### 5.5 Ablation / Comparison (optional)

- **Comparison to reference runs**:

| Exp ID           | Self-Att | Cross-Att | Loss Weighting     | Macro F1 (Val) | R² (Test) | Notes                |
|------------------|----------|----------|--------------------|----------------|-----------|----------------------|
| mtl_unet_base    | off      | off      | fixed (1.0, 0.5)   |                |           |                      |
| mtl_unet_self    | on       | off      | fixed              |                |           |                      |
| mtl_unet_cross   | on       | on       | uncertainty-based  |                |           |                      |

---

## 6. Insights & Next Steps

- **Did the hypotheses hold?**
  - ...

- **Key observations**:
  - **Classification**: (e.g. where F1 improved / degraded, which classes benefited)
  - **Regression (CHM)**: (e.g. bias in low/high CHM ranges, saturation issues)
  - **Multi-task interaction**:
    - Changes in task weights over time
    - Evidence that cross-task attention helped/hurt

- **Potential causes / diagnostics**:
  - Data issues (class imbalance, CHM outliers)
  - Model capacity (under/overfitting signs)
  - Optimization issues (learning rate, instability)

- **Actionable next steps**:
  - [ ] Try different `base_channels` / depth
  - [ ] Adjust CHM thresholds or loss for `NaN` regions
  - [ ] Tune `cls_weight` / `reg_weight` or prior on `log_sigma_*`
  - [ ] Additional attention variants or regularization
  - [ ] Design next experiment ID: `...`

---