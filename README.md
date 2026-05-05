# Multi-Task Learning for Forest Parameter Extraction from SAR Imagery

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SAR](https://img.shields.io/badge/Data-Sentinel--1-orange.svg)](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)

This repository contains the TreeSatAI-CHM dataset extension and baseline experiments for multi-task learning in forest monitoring using Sentinel-1 SAR. It extends the TreeSatAI benchmark with canopy height labels derived from German LiDAR data, supporting joint prediction of tree species classification and canopy height regression.

Part of a Master's thesis project. The physics-aware multi-task learning framework is in the companion repository: [Physics-Aware Multi-Task Learning for Forest and Biomass Estimation](https://github.com/SonjaX1927/Physics-Aware-Multi-Task-Learning-for-Forest-and-Biomass-Estimation)

**Contents:**
- TreeSatAI-CHM dataset with canopy height labels from LGLN LiDAR products
- Multi-task U-Net with shared encoder for species classification and height regression
- Uncertainty-weighted loss for task balancing
- Gradient analysis for task compatibility evaluation
- Data pipeline for CHM generation and preprocessing

## Dataset

The dataset is hosted on Hugging Face: [siyux1927/treesatai-chm](https://huggingface.co/datasets/siyux1927/treesatai-chm)

The TreeSatAI benchmark provides Sentinel-1 imagery with species labels for German forests. This work adds canopy height labels from official German LiDAR products (LGLN).

**Study area:** Lower Saxony, Germany  
**Input features:** Sentinel-1 VV, VH, VV/VH ratio + CHM (4 channels)  
**Target variables:** Tree genus (15 classes, multi-label) + canopy height (metres)  
**Spatial resolution:** 10 m/pixel, 60 m patch extent (6×6 pixels)  
**Dataset size:** 50,381 patches with complete labels

**Data sources:**
- TreeSatAI Sentinel-1: SAR backscatter from 2017–2019
- Species labels: Forest administration records, Lower Saxony
- Height labels: LGLN DTM/DSM (1 m resolution, airborne LiDAR)

**CHM generation:**

<p align="center">
  <img src="@plots/data-insight/dtm_dsm_chm.png" width="600" alt="CHM Generation Process"/>
  <br>
  <em>CHM generation from DTM and DSM, validated against ETH and Meta global canopy height products</em>
</p>

CHM = DSM − DTM, validated against ETH Global Canopy Height (10 m) and Meta Global Canopy Height (1 m).

**Dataset samples:**

<p align="center">
  <img src="@plots/training-data-insight/sample_grid_60m.png" width="700" alt="Dataset Samples"/>
  <br>
  <em>Example patches: VV, VH, VV/VH ratio, and CHM across train/validation/test splits</em>
</p>

## Results

| Model | Task | Metric | Performance |
|-------|------|--------|-------------|
| MLP Baseline | Classification | F1-Score | 0.52 |
| MLP Baseline | Regression | RMSE | 8.2 m |
| U-Net Baseline | Classification | F1-Score | 0.54 |
| U-Net Baseline | Regression | RMSE | 7.9 m |
| Multi-Task U-Net | Classification | F1-Score | 0.53 |
| Multi-Task U-Net | Regression | RMSE | 8.1 m |

Multi-task learning matches single-task baselines at roughly half the parameter count. Gradient analysis shows positive task alignment (cosine similarity 0.3–0.6) in early encoder layers. Height estimation remains difficult due to C-band saturation in dense forest and the limited spatial context of 6×6 patches.

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

## Prerequisites

```bash
python >= 3.8
pytorch >= 2.0
rasterio >= 1.3
numpy
scikit-learn
pandas
```

**Acknowledgments:** TreeSatAI benchmark, LGLN open data portal, ESA Sentinel-1 mission.
