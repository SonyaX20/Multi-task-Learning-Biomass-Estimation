
## 4.1 TreeSatAI Dataset Overview

### 4.1.1 Dataset Composition and Structure
- Study region and spatial extent; forest types represented
- Patch definition at 60 m resolution; tiling scheme and coordinate reference systems
- Per-patch content: SAR backscatter bands, CHM, auxiliary metadata fields
- Number of 60 m samples per split (raw counts before filtering)
- File organisation: GeoJSON index (bb_60m), raster directories, HDF5 archives

### 4.1.2 SAR Data Specifications
- Sentinel-1 acquisition characteristics: frequency band, polarizations (VV, VH), orbit geometry
- Temporal coverage and compositing strategy for TreeSat 60 m patches
- Radiometric units (sigma0 / gamma0), scaling conventions and dynamic range
- Native spatial resolution vs. effective patch resolution (60 m grid)
- Known artefacts and noise sources relevant for downstream processing (speckle, layover, shadow)

### 4.1.3 Ground-Truth Labels
- TreeSat label definition BT_GEN: forest type / dominant species categories
- Additional continuous targets: canopy height proxy via CHM, biomass (if available)
- Label sources: forest inventory / reference datasets used to derive BT_GEN and height/biomass
- Label granularity: patch-level labels vs. pixel-level information
- Summary of class distribution and imbalance; rare classes and under-represented conditions

## 4.2 CHM Generation

### 4.2.1 Source Elevation Products and Coverage
- Description of DGM1 (DTM) and DOM1 (DSM) datasets: resolution, provider, acquisition period
- Tile grid layout and naming convention (tile_id), national coverage
- Spatial intersection of TreeSat patches with DTM/DSM tiles; detection of required tiles only
- Handling of patches spanning multiple elevation tiles; motivation for reverse mapping

### 4.2.2 CHM Computation from DSM and DTM
- Preprocessing of DTM: nodata handling (-9999), 8-neighbour interpolation, 10 m mean filter for smoothing
- Preprocessing of DSM: nodata handling and iterative gap filling
- Per-tile CHM calculation: CHM = DSM − DTM at native resolution
- Height constraints: clipping of negative values to 0 m; definition of plausible upper bound
- Storage of intermediate CHM tiles: directory structure, nodata conventions

### 4.2.3 Patch-Level CHM Extraction and Co-Registration
- Use of TreeSat 60 m templates to define patch extents and target CRS
- Cropping and mosaicking CHM tiles for each IMG_ID; handling of multi-tile patches
- Reprojection from elevation CRS (e.g. EPSG:25832) to TreeSat/S1 CRS (e.g. EPSG:32632)
- Downsampling from native resolution to patch grid (bilinear), ensuring exact alignment with SAR grid
- Quality checks: detection of remaining nodata / -9999; statistics on successful vs. failed patches

## 4.3 Data Preprocessing Pipeline

### 4.3.1 SAR Data Preprocessing
- Ingestion of Sentinel-1 60 m mosaics; band ordering (vv, vh)
- Radiometric preprocessing applied before patch extraction (calibration, terrain correction, if applicable)
- Derivation of additional features: vv/vh ratio and other simple transforms
- Global statistics and quantile analysis for each band (vv, vh, vv/vh) to identify valid dynamic range
- Intensity clipping based on robust quantiles to suppress extreme outliers and artefacts

### 4.3.2 Label and CHM Quality Control
- Consistency checks between GeoJSON indices and raster archives (matching IMG_IDs, missing files)
- Screening of CHM tiles and patches for nodata, implausible heights, and extreme outliers
- Use of nan/invalid-value detection tools to quantify fractions of NaN, Inf, and -9999
- Filtering rules for problematic samples (e.g. large nodata fraction, unrealistic CHM distributions)
- Final label set after filtering: per-class counts, removal of ambiguous or low-confidence labels

### 4.3.3 Patch Construction, Stacking, and Augmentation
- Construction of multi-band patches: stacking SAR features and CHM into a unified tensor
- Harmonisation of spatial resolution, CRS, and grid alignment across modalities
- Final normalisation / standardisation strategies applied per band (e.g. mean-std, min-max)
- On-the-fly data augmentation strategies at training time (rotations, flips, intensity jitter)
- Rationale for chosen augmentations in the context of SAR forest mapping and CHM estimation

## 4.4 Dataset Splitting and Validation Strategy

### 4.4.1 Original TreeSat Splits
- Description of original SPLIT field in TreeSat (train / test) and its intended usage
- Limitations of purely random or non-spatial splits for SAR-based forest mapping

### 4.4.2 Spatial Train–Validation Partitioning
- Construction of 60 m patch bounding boxes and overlap-connected components
- Greedy assignment of components to train and validation subsets (target 8:1 ratio)
- Ensuring non-overlap between train and val regions to reduce spatial autocorrelation
- Mapping from patch-level spatial split to final train/val/test sets used in experiments

### 4.4.3 Final Data Splits for Experiments
- Definition of splits per task (species classification, height estimation, biomass prediction)
- Sample counts per split and per class; impact of spatial split on class balance
- Strategy for cross-validation or repeated experiments (if any)
- Alignment between dataset splits and evaluation metrics described in later chapters

## 4.5 Exploratory Data Analysis

### 4.5.1 Global Dataset Statistics
- Summary statistics of SAR bands and CHM (mean, std, quantiles, value ranges)
- Distributions of CHM heights per forest type or BT_GEN class
- Histograms and density plots to reveal skewness and heavy tails

### 4.5.2 Spatial Patterns and Example Patches
- Visual examples of representative patches across forest types and height ranges
- Maps showing spatial distribution of samples, splits, and key variables
- Qualitative assessment of alignment between SAR backscatter patterns and CHM structure

### 4.5.3 Identification of Outliers and Biases
- Detection of outliers in CHM or SAR intensity (extreme values, anomalous patterns)
- Analysis of class imbalance and geographic bias (e.g. certain classes clustered in regions)
- Implications of observed biases for model training and evaluation; notes for later discussion

