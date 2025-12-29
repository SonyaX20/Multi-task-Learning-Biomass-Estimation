## Table of Contents

## 1. Introduction (6-8 pages)
### 1.1 Motivation and Background
### 1.2 Research Objectives
### 1.3 Scope and Limitations
- geographic scope: german forests (TreeSatAI dataset coverage)
- parameters: tree species classification, height estimation, biomass prediction
- sensor: Sentinel-1 SAR (VV, VH polarizations)
- exclusions: temporal analysis, other sensor fusion
### 1.4 Thesis Structure
- overview & timeline

## 2. Theoretical Background (12-15 pages)
### 2.1 Synthetic Aperture Radar Fundamentals
- SAR imaging principles, geometry
- backscatter mechanisms in vegetated areas
- Sentinel-1 mission specifications
### 2.2 Forest Biophysical Parameters
- tree species characteristics and spectral/structural differences
- forest height: definition, ecological significance, measurement methods
- agb: importance for carbon stocks, allometric relationships, relationships between parameters (height-biomass correlations)
### 2.3 Deep Learning for Remote Sensing
- deep learning in image processing
- U-Net in remote sensing
- other models in remote sening
### 2.4 Multitask Learning
- MTL paradigm: parameter sharing, architectures sharing...
- task weights
- MTL in remote sensing: state of the art

## 3. Related Work (8-10 pages)
### 3.1 SAR-based Forest Parameter Retrieval
### 3.2 Multitask Learning in Remote Sensing
### 3.3 TreeSatAI and Other Benchmark Datasets...
### 3.4 Research Gap Identification
- extending tresatai
- mtl
## 4. Dataset and Data Processing (10-12 pages)

### 4.1 TreeSatAI Dataset Overview
### 4.2 CHM Generations
- flowchart...
- how to generate chm
### 4.3 Data Preprocessing Pipeline
* how to split datasets?
* how to validate the generated chm

## 4. Materials and Methods (15-18 pages)

### 4.1 Study Area and Dataset
- TreeSatAI dataset description
- geographic coverage, forest types
### 4.2 Data Preprocessing
- how to generate chm, flowchart...
- how to split datasets?
* how to validate the generated chm
### 4.3 Model Architecture

#### 4.3.1 Baseline: Single-Task U-Net
- encoder-decoder structure details
- classification head (species) vs. regression heads (height, biomass)
- loss functions: cross-entropy, MSE/MAE
#### 4.3.2 Multitask Architecture
- shared encoder design
- task-specific decoder branches
- parameter sharing
### 4.4 Training Procedure
- multi-task loss 
- task weighting approaches tested
    - equal weighting
    - uncertainty weighting
    - gradient normalization (GradNorm)
- optimizer, learning rate schedule
### 4.5 Evaluation Metrics
* task improvement ratio, multi-task gain
### 4.6 Experimental Setup
- colab
## 5. Experiments and Results (12-15 pages)
### 5.1 Data Exploration and Preprocessing Results
- statistics per class
- correlation analysis: SAR features vs. target variables
- normalization
### 5.2 Single-Task Baseline Performance
### 5.3 Multitask Learning Results

#### 5.3.1 MTL vs. Single Tasks
- performance table across all metrics
#### 5.3.2 Task Weighting Analysis
- comparison of weighting strategies
#### 5.3.3 Task Relationship Analysis
- negative transfer
- gradient conflict analysis
## 6. Conclusion and Future Work (8-10 pages)
* interpret results