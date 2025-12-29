# CHM and Preprocessing Pipeline Updates

## 1. Motivation

The original TreeSatAI benchmark focuses on Sentinel‑1 and Sentinel‑2 inputs. For biomass estimation, we need a more direct representation of forest structure. Canopy Height Models (CHM) derived from LiDAR‑based DTM/DSM products are a natural way to extend the benchmark towards 3D information while staying tied to official geodata.

Early experiments showed that simply subtracting DSM − DTM at 1 m resolution and downsampling to 60 m was not sufficient. The resulting CHM was noisy, sensitive to outliers, and sometimes misaligned with the TreeSat grid. The goal of this update was to design a robust, reproducible pipeline that produces clean CHM patches, aligns them with Sentinel‑1, and integrates smoothly into the existing training workflow.

## 2. Issues in the initial pipeline

The first CHM generation scripts were mainly technical proofs of concept. They reproduced the DSM − DTM logic but left several practical issues open. Small misalignments between the LGLN grid and the TreeSat grid led to artefacts at patch borders. Outliers in CHM values caused very tall trees or spurious spikes to dominate the dynamic range.

Downsampling was also too naïve. Early versions relied on simple max pooling or resampling choices that were not well matched to the TreeSat templates. This occasionally produced CHM patches that looked plausible on average but deviated from independent references such as the ETH 10 m CHM product.

## 3. Final CHM processing design

The updated `gen5_chm_reproj_maxpool_crop_stack.py` script implements a three‑step strategy for each TreeSat sample. First, CHM values are clipped to a fixed percentile range `[0.0, 41.5983]`. Values below zero are set to zero, and values above 41.5983 are truncated. This stabilises the dynamic range and reduces the influence of extreme outliers while preserving normal forest heights.

Second, a `20 × 20` maximum filter is applied on the native 1 m grid. This acts as a local envelope over tree tops and suppresses small gaps and noise in the canopy. By filtering before any reprojection, the operation respects the original high‑resolution geometry.

Third, the filtered CHM is reprojected onto the TreeSat Sentinel‑1 template grid using bilinear resampling. The TreeSat transform and resolution remain fixed; only the resampling strategy was changed. This choice strikes a balance between preserving smooth canopy structure and avoiding blocky artefacts from pure max pooling.

A dedicated debug mode reuses the same clipping and filtering strategy to compare 1 m CHM tiles against the ETH 10 m CHM product. This helped confirm that the chosen parameters yield visually and statistically consistent results.

## 4. Stacking Sentinel‑1 and CHM

The stacking stage combines the processed CHM with the Sentinel‑1 VV, VH, and VV/VH bands on the TreeSat grid. Before stacking, each S1 band is clipped to global percentile ranges derived from the data:

- VV: `[-18.0383, 5.5330]`
- VH: `[-26.1871, -1.7555]`
- VV/VH: `[-1.7393, 1.4362]`

Clipping is applied only to valid pixels and leaves nodata values untouched. This makes the S1 inputs numerically stable and more comparable across sites, while still keeping the original information content. The final stacked rasters are written as four‑band GeoTIFFs `[VV, VH, VV/VH, CHM]` under `data/stacked_treesat_<res>m/`.

## 5. Label processing and dataset splits

For labels, the pipeline follows TreeSatAI as closely as possible while adding a validation split. The multi‑label file `TreeSatBA_v9_60m_multi_labels.json` is filtered with a simple area threshold to remove very small class fragments. This reproduces the original TreeSat area filtering logic.

The official TreeSat train/test file lists (`train_filenames.lst`, `test_filenames.lst`) are then treated as a hard constraint. All subsequent processing respects this 9:1 split: the test set is fixed, and no sample moves between train and test.

Inside the original train split, samples are assigned a primary label (the largest‑area class after filtering), and a stratified split is performed across primary labels. This produces new train and validation subsets inside the original train such that the overall dataset ends up with an approximate 8:1:1 train/val/test ratio. This design keeps comparability with TreeSat while providing a dedicated validation set for model selection.

## 6. From GeoTIFFs to NumPy training tensors

The preprocessing scripts in `@data-processing/preprocessing` bridge the gap between geospatial rasters and deep‑learning‑ready arrays. First, `labels_area_filter_and_split.py` applies area filtering and stratified splitting, writing per‑split label JSONs and basic class statistics. Next, `compute_class_weights.py` computes inverse‑frequency class weights from the final train split.

Finally, `build_training_arrays.py` aligns the filtered labels with the stacked S1+CHM rasters, loads patches from `data/stacked_treesat_<res>m/`, and saves NumPy arrays under `@data/training-data-<res>m/`. This produces `train_x.npy`, `val_x.npy`, `test_x.npy` and their corresponding multi‑hot label arrays, plus `classes.npy` and `*_filenames.npy` for book‑keeping.

## 7. Visual diagnostics and sanity checks

A small visualization module in `preprocessing/vis.py` provides quick feedback on the new pipeline. It plots sample grids of 4‑band patches, histograms for each band and split, and bar charts of class distributions across train, val, and test. These plots were used to verify that clipping and filtering behave as expected and that the new stratified split does not introduce obvious imbalances.

The same tools were also used during debugging of CHM processing, for example to compare LGLN‑derived CHM with ETH and Meta canopy height products. These comparisons motivated the choice of percentile ranges and the 20×20 max filter as a compromise between smoothing and preserving fine‑scale canopy structure.

## 8. Impact on baseline models

The updated pipeline feeds directly into baseline models such as the multi‑task MLP and UNet variants. Because S1 and CHM are now consistently aligned, clipped, and stored in a single tensor, model code can treat them as standard image inputs without additional geospatial handling.

Using the TreeSat‑consistent 8:1:1 split and the computed class weights makes training curves easier to interpret and results more comparable. Overall, these updates trade some implementation complexity in the data pipeline for cleaner inputs, better reproducibility, and a clearer connection between the raw geodata and the final learning task.
