# Multi-Task Learning for Forest Parameter Extraction from SAR Imagery

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SAR](https://img.shields.io/badge/Data-Sentinel--1-orange.svg)](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)

## Overview

This repository contains the **TreeSatAI-CHM dataset extension and baseline experiments** for multi-task learning in forest monitoring using Sentinel-1 SAR imagery. This work extends the TreeSatAI benchmark with canopy height labels derived from German LiDAR data, enabling joint prediction of **tree species classification** and **canopy height regression**.

**💡 This repository serves as part of a Master's thesis project.** For the physics-aware multi-task learning framework, please refer to the companion repository:  
**🔗 [Physics-Aware Multi-Task Learning for Forest and Biomass Estimation](https://github.com/SonjaX1927/Physics-Aware-Multi-Task-Learning-for-Forest-and-Biomass-Estimation)**

### Key Features

- **TreeSatAI-CHM Dataset**: Extended TreeSatAI benchmark with canopy height labels from LGLN LiDAR products
- **Multi-task U-Net architecture** with shared encoder for species classification and height regression
- **Uncertainty-weighted loss** for automatic task balancing
- **Gradient analysis** to understand task compatibility and feature sharing
- **Data pipeline** for CHM generation, validation, and preprocessing

## Mission & Objectives

### Scientific Goals

Forest monitoring requires simultaneous estimation of multiple structural attributes. This project investigates whether multi-task learning can leverage shared SAR feature representations to improve prediction of both categorical (species) and continuous (height) forest parameters.

**Primary objectives:**
- Extend existing single-task datasets to support multi-task learning research
- Develop multi-task architectures that share representations across related forest attributes
- Evaluate whether joint training improves generalization compared to single-task baselines
- Analyze task relationships through gradient alignment and loss dynamics

### Technical Challenges

1. **Dataset extension** - Deriving accurate height labels from auxiliary LiDAR sources
2. **Small patch size** - TreeSatAI provides 6×6 pixel patches, limiting architectural depth
3. **Class imbalance** - Tree species distributions are naturally skewed
4. **Task compatibility** - Determining whether classification and regression benefit from shared features

## Dataset

### TreeSatAI-CHM Extended Dataset

The TreeSatAI benchmark provides Sentinel-1 imagery with species labels for German forests. We extend it with canopy height labels derived from official German LiDAR products.

**Study area:** Lower Saxony, Germany  
**Input features:** Sentinel-1 VV, VH, VV/VH ratio + derived CHM (4 channels)  
**Target variables:** Tree genus (15 classes, multi-label) + Canopy height (meters)  
**Spatial resolution:** 10m per pixel, 60m patch extent (6×6 pixels)  
**Dataset size:** 50,381 patches with complete labels

#### Data Sources

- **TreeSatAI Sentinel-1**: Pre-processed SAR backscatter from 2017-2019
- **Species labels**: Forest administration records (Lower Saxony)
- **Height labels**: LGLN DTM/DSM products (1m resolution, airborne LiDAR)

#### CHM Generation Pipeline

<p align="center">
  <img src="@plots/data-insight/dtm_dsm_chm.png" width="600" alt="CHM Generation Process"/>
  <br>
  <em>CHM generation from DTM and DSM LiDAR products with validation against global canopy height products</em>
</p>

The CHM is computed as DSM - DTM and validated against:
- **ETH Global Canopy Height** (10m resolution, Sentinel-2 + GEDI)
- **Meta Global Canopy Height** (1m resolution, Maxar + airborne LiDAR)

#### Dataset Samples

<p align="center">
  <img src="@plots/training-data-insight/sample_grid_60m.png" width="700" alt="Dataset Samples"/>
  <br>
  <em>Example patches showing VV, VH, VV/VH ratio, and CHM across train/validation/test splits</em>
</p>

## Results

### Performance Summary

| Model | Task | Metric | Performance |
|-------|------|--------|-------------|
| **MLP Baseline** | Classification | F1-Score | 0.52 |
| **MLP Baseline** | Regression | RMSE | 8.2m |
| **U-Net Baseline** | Classification | F1-Score | 0.54 |
| **U-Net Baseline** | Regression | RMSE | 7.9m |
| **Multi-Task U-Net** | Classification | F1-Score | 0.53 |
| **Multi-Task U-Net** | Regression | RMSE | 8.1m |

### Results

✅ **Multi-task learning achieves comparable performance** to single-task baselines with reduced model complexity  
✅ **Gradient analysis shows positive alignment** between classification and regression tasks in early encoder layers  
✅ **Uncertainty weighting** automatically balances task contributions without manual tuning  
⚠️ **Small patch size (6×6 pixels)** limits architectural depth and spatial context  
⚠️ **Height estimation** is challenging from C-band SAR alone due to limited canopy penetration

## Repository Structure

```
├── @data-processing/           
│   ├── gen-chm/               
│   ├── preprocessing/         
│   └── README.md             
│
├── @training/                 
│   ├── documentations/       
│   ├── jupyter-notebooks/    
│   └── README.md            
│
├── @plots/                    
│   ├── data-insight/         
│   └── training-results/     
│
├── @thesis-writing/           
│   ├── thesis-latex/         
│   └── drafts/              
│
└── @data/                      
    ├── chm/                 
    └── ETH_chm_10m/         
```

## Quick Start

### Prerequisites

```bash
python >= 3.8
pytorch >= 2.0
rasterio >= 1.3
numpy
scikit-learn
pandas
```

---
⭐ **If you find this work useful, please consider starring the repository!**  

For questions or issues, please open a GitHub issue or contact the author.

---

**Acknowledgments:** This work uses data from TreeSatAI benchmark, LGLN open data portal, and ESA Sentinel-1 mission.
