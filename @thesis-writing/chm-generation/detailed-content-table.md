# CHM data generation

## 1. Overview

Canopy height models (CHMs) provide a compact description of vertical forest structure and are widely used in applications such as biomass estimation, habitat mapping, and stand characterisation. In this work, the CHM serves as a physically interpretable target and auxiliary variable in the multi-task learning framework. We derive CHM layers by combining high-resolution national elevation products with radar-based TreeSat patches and then aggregate them to the 60 m and 200 m patch scales used for model training.

The overall processing chain starts from a national index of digital terrain and digital surface model tiles, identifies all tiles intersecting the TreeSat patch grid, and downloads the corresponding GeoTIFFs. At tile level, we pre-process terrain and surface elevation, interpolate nodata areas, and compute CHM as the difference between DSM and DTM. In a second stage, we extract patch-level CHM by cropping or mosaicking the CHM tiles to match the TreeSat patches, reprojecting them to the Sentinel‑1 grid while preserving spatial alignment. Finally, we stack CHM with Sentinel‑1 backscatter features, clean remaining nodata, and package the resulting multi-band patches into HDF5 datasets with spatially aware train/validation/test splits.

*(Figure 1: Conceptual illustration of DTM, DSM and CHM along a vertical profile, highlighting the definition of canopy height as DSM–DTM.)*

## 2. Input datasets

### 2.1 National elevation products (DGM1 / DOM1)

The CHM computation is based on two national elevation products provided by the Landesamt für Geoinformation und Landesvermessung Niedersachsen (LGLN). The digital terrain model DGM1 represents the bare-earth surface at approximately 1 m spatial resolution, whereas the digital surface model DOM1 captures the uppermost reflecting surface, including vegetation and built structures, at the same resolution. Both products are distributed as tiled GeoTIFFs with associated polygon index files (GeoJSON) describing tile footprints and download links.

Following common terminology, the DTM is interpreted as ground elevation, while the DSM represents the top of canopy and other objects (e.g. Earth Lab, “Canopy Height Models, Digital Surface Models & Digital Elevation Models”, accessed 2025). A canopy height model is then defined as the elevation difference between DSM and DTM at each pixel (OpenTopography, “Canopy Height Model Tool”, accessed 2025). The scripts use the naming convention `dtm1` and `dsm1` to refer to these raster tiles.

*(Table 1: Overview of elevation products, including provider, spatial resolution, vertical datum, coordinate reference system (CRS), and tile size.)*

### 2.2 TreeSat patch grid and reference labels

TreeSat operates on patch-based representations of the landscape. Two grids are relevant for CHM processing. The first is a 60 m grid, used mainly for early experiments and for visualisation of CHM at a finer aggregation level. The second is a 200 m grid, which forms the basis of the Sentinel‑1 patches used in the main modelling experiments. Both grids are stored as polygon feature collections with unique patch identifiers (`IMG_ID`) and additional attributes.

Each patch is associated with a forest stand generation or broad forest type label (`BT_GEN`), derived from forest inventory or management data. The GeoJSON files also contain a `SPLIT` attribute indicating the original role of each patch (train or test) in the TreeSat dataset. This attribute is preserved throughout the processing chain and later refined into spatially disjoint train and validation subsets.

*(Table 2: Summary of TreeSat patch grids, listing patch size, number of patches per split, and class distribution of `BT_GEN` for 60 m and 200 m grids.)*

### 2.3 Sentinel‑1 backscatter mosaics

To complement the CHM with active microwave information, we use Sentinel‑1 C‑band synthetic aperture radar (SAR) mosaics at 200 m resolution. For each 200 m patch, the mosaics provide backscatter in VV and VH polarisation, typically aggregated over one or more acquisition periods to reduce speckle noise and seasonal variability. From these bands, a simple ratio feature VV/VH is derived, which has been shown to carry information on vegetation structure and moisture (see, for example, applications in forest height and biomass mapping that integrate Sentinel‑1 backscatter with canopy height information in Geophysical Research Letters, “Mapping Forest Height and Aboveground Biomass by Integrating ICESat‑2 and Sentinel‑1 Data”, 2021).

All Sentinel‑1 mosaics are provided in a common projected CRS (EPSG:32632), with spatial alignment to the 200 m TreeSat grid. These rasters serve both as templates for CHM reprojection and as inputs to the final stacked data cubes.

*(Figure 2: Example Sentinel‑1 VV and VH mosaics over the study area, with TreeSat 200 m patch grid overlaid.)*

## 3. Selection and retrieval of DTM/DSM tiles

### 3.1 Spatial overlay between TreeSat grid and elevation tiles

The first processing stage identifies all DGM1 and DOM1 tiles that are required to cover the TreeSat patches. We use the polygon index of elevation tiles (GeoJSON) and the TreeSat grid (either 60 m or 200 m) and perform an overlay based on bounding-box intersection. To make this operation efficient at national extent, an R‑tree spatial index is built on the tile polygons. For each TreeSat patch, the index is queried for candidate tiles whose bounding boxes intersect the patch, and simple overlap tests on bounding boxes are applied.

The result is a mapping from each tile identifier to the set of TreeSat patch IDs that intersect that tile, along with basic coverage statistics such as the number of patches per tile and the proportion of patches successfully assigned. This step is implemented for both DTM and DSM, and separate link files are produced for each (one for DGM1, one for DOM1).

*(Table 3: Number of DTM and DSM tiles intersecting the TreeSat grid, including the distribution of TreeSat patches per tile and the percentage of patches covered.)*

### 3.2 Reverse mapping to TreeSat-centric tile lists

For subsequent processing it is more convenient to work from a patch-centric perspective: for each TreeSat patch, we need to know which DTM and DSM tiles must be downloaded and mosaicked. Therefore, the tile-to-patch mapping is inverted to obtain, for each `IMG_ID`, two lists of tile descriptors, one for DTM and one for DSM. The scripts also record how many tiles are associated with each patch and mark those cases where mosaicking is required (more than one tile per DTM or DSM).

This reverse mapping is stored in a dedicated JSON file and serves as the authoritative link between TreeSat patches and elevation tiles throughout the pipeline.

### 3.3 Mass download of elevation tiles

Once the required tiles have been identified, we download the corresponding GeoTIFFs using the HTTP links provided in the elevation index. To avoid redundant downloads, we first extract the set of unique tile IDs and only request each tile once, even if it is needed by multiple patches. The downloader performs simple integrity checks: existing files larger than a small threshold (10 kB) are assumed valid and skipped, and failed downloads are retried a few times before being flagged.

The downloaded tiles are stored in separate folders for DTM and DSM. At this point, we have a consistent local copy of all elevation tiles that intersect the TreeSat grid and are ready for CHM computation.

*(Figure 3: Map of DTM/DSM tile footprints overlaid with TreeSat patches, illustrating variable patch–tile relationships and areas where mosaicking is required.)*

## 4. CHM computation at tile level

### 4.1 Pre-processing of DTM and DSM

Real-world elevation products invariably contain nodata regions, artefacts, and small inconsistencies between tiles. Before computing CHM, we therefore pre-process both DTM and DSM tiles. The nodata value used by the provider, −9999, is first identified and converted to an internal nodata representation. We then apply an iterative 8‑neighbour mean interpolation to fill isolated nodata pixels and small gaps. This local scheme exploits the strong spatial autocorrelation of elevation and is applied up to a maximum number of iterations. If, after the interpolation passes, nodata pixels remain, they are either filled with a simple global statistic (mean of valid pixels) or set to zero, depending on the product and script variant.

In addition, the DTM can be smoothed with a mean filter corresponding to a 10 m × 10 m window. For 1 m resolution data this translates into an approximately 11×11 pixel kernel. The purpose of this filter is to reduce small-scale noise, fill narrow gaps, and produce a more continuous ground surface while preserving larger-scale terrain features. Similar smoothing has been recommended in previous work on canopy height modelling from photogrammetric DSMs and DTMs (e.g. in regional CHM products such as those described by Lang et al., Remote Sensing of Environment, 2019, and related studies).

### 4.2 Canopy height model calculation

After pre-processing, CHM is computed at tile level as the simple difference between the DSM and the (optionally smoothed) DTM:

\[\mathrm{CHM}(x, y) = \mathrm{DSM}(x, y) - \mathrm{DTM}(x, y).\]

This definition follows standard practice in LiDAR and photogrammetric CHM generation (Earth Lab, OpenTopography). Negative values may occur due to residual misregistrations or noise; these are truncated to zero, as negative canopy heights have no physical interpretation. The resulting CHM tiles are stored as single-band float32 GeoTIFFs with explicit nodata handling.

At the end of this stage we obtain a seamless, though still tile-based, canopy height surface at native resolution over all areas where both DTM and DSM are available.

*(Figure 4: Example DTM, DSM, and CHM tile triplet, showing the effect of smoothing and the resulting canopy height field.)*

## 5. Patch-level CHM extraction

### 5.1 Cropping and mosaicking in native CRS

The tile-based CHM needs to be translated into patch-based representations consistent with the TreeSat footprints. Using the TreeSat-to-tile mapping, we first determine, for each patch, which CHM tiles contribute to its extent. If all pixels fall within a single tile, we simply crop the CHM tile to the patch footprint. If the patch straddles multiple tiles, we mosaic the required CHM tiles and then crop the mosaic.

These operations are performed in the native coordinate reference system of the elevation data (EPSG:25832). The patch footprint is defined using the bounds of the corresponding 60 m or 200 m patch raster, transformed into this CRS. For robustness, cropping windows are expanded slightly to provide a buffer around the target extent before being refined. During this step, we also enforce basic quality criteria: patches whose CHM crops still contain NaN values or unfilled −9999 nodata are discarded, and the number of such failures is recorded.

*(Table 4: Numbers of TreeSat patches requiring mosaicking vs. single-tile cropping, and number of patches rejected due to residual nodata.)*

### 5.2 Reprojection to TreeSat / Sentinel‑1 grids

To ensure strict spatial alignment with the radar-based inputs, we reproject the cropped CHM patches to the CRS and grid definition of the TreeSat/Sentinel‑1 mosaics (EPSG:32632). Two cases are distinguished. For the 60 m grid, CHM is reprojected and resampled directly to the 60 m patch grid, so that CHM and the 60 m TreeSat raster share the same resolution and extent. For the 200 m grid, we first crop and mosaic CHM at native resolution, then reproject it to EPSG:32632 while keeping the native pixel size. This yields a high-resolution CHM patch in the radar CRS, aligned with the 200 m template only in terms of outer bounds. The actual aggregation to 200 m is deferred to the stacking stage.

Reprojection uses bilinear interpolation, which is appropriate for continuous variables like elevation and canopy height. The scripts explicitly construct the output grids based on the template patch bounds and the desired pixel size, ensuring that subsequent stacking operations can rely on exact matches of CRS, transform, and raster shape.

*(Figure 5: Schematic of CHM cropping and reprojection for a single TreeSat patch, showing original tiles, mosaicked CHM, and the final patch in the radar CRS.)*

## 6. Stacking CHM with Sentinel‑1 features

### 6.1 Resampling CHM to 200 m grid

For the 200 m TreeSat grid, CHM at native resolution is aggregated to the Sentinel‑1 grid by bilinear interpolation. Before resampling, we apply simple physical constraints to CHM values. Pixels marked as nodata (−9999) are excluded from the interpolation. Negative canopy heights are set to zero, and extremely large values are treated conservatively. Based on global analyses of CHM quantiles, we define an upper canopy height threshold and clip all values exceeding this bound to the threshold, while values above a more generous bound (e.g. 50 m) are set to zero to flag likely artefacts.

The resampled CHM patch is thus a smooth representation of mean canopy height at 200 m resolution, aligned perfectly with the Sentinel‑1 mosaic grid. Remaining −9999 values after resampling are converted to zero in order to avoid propagating nodata into the modelling features.

### 6.2 Sentinel‑1 bands and ratio features

The Sentinel‑1 mosaics provide VV and VH backscatter intensity for each 200 m patch. We compute a simple ratio feature, VV/VH, which has been shown to be sensitive to vegetation structure and scattering mechanisms (e.g. in studies combining Sentinel‑1 backscatter with canopy height to upscale forest structure, such as the Geophysical Research Letters article “Mapping Forest Height and Aboveground Biomass by Integrating ICESat‑2 and Sentinel‑1 Data”, 2021, and related work). The ratio is evaluated wherever VH is non-zero, with undefined values set to NaN.

To reduce the influence of extreme values and heavy-tailed distributions, we further clip VV, VH and the VV/VH ratio using global quantiles estimated across the study area. For each band, we compute low-end and high-end quantiles (e.g. 0.01 % and 99.99 %) and restrict the dynamic range accordingly. This approach preserves most of the variability while discarding rare outliers that are likely due to residual speckle, geocoding errors, or unmodelled incidence angle effects.

### 6.3 Cleaning nodata in stacked patches

After stacking the three Sentinel‑1 bands and the 200 m CHM into four-band patches, we perform a final nodata cleaning step. For each band we define nodata consistently: −9999 for VV and VH, and NaN for the VV/VH ratio. We then apply an iterative 8‑neighbour mean interpolation that fills nodata pixels when the surrounding neighbourhood has valid values. This procedure is repeated for a small number of iterations and is applied independently to each band.

At the end of the cleaning, we report which patches still contain nodata. In practice, only a small fraction of patches exhibit persistent gaps, typically in areas near the edge of the mosaics or in regions with poor coverage.

*(Table 5: Summary of value ranges and clipping thresholds for VV, VH, VV/VH, and CHM at 200 m, including global quantiles before and after clipping.)*

*(Figure 6: Example stacked patch showing VV, VH, VV/VH, and CHM bands side by side, illustrating spatial correspondence between backscatter and canopy height.)*

## 7. Dataset assembly and spatial splitting

### 7.1 Original train / test definition

The TreeSat GeoJSONs provide an initial split of patches into train and test sets via the `SPLIT` field. This split was constructed at the patch level without explicit constraints on spatial dependence between patches. In the CHM generation workflow we preserve this original split and use it to distinguish purely held-out test patches from those used in model development.

We also retain the forest generation labels `BT_GEN` as class labels for the supervised learning tasks. Their distribution across train and test is summarised to document potential imbalances and to provide context for later evaluation.

*(Table 6: Counts of patches per `BT_GEN` class and per original `SPLIT` (train/test) for the 200 m grid.)*

### 7.2 Spatial train / validation split

Randomly partitioning spatial data into train and validation sets can lead to overly optimistic estimates of model performance due to spatial autocorrelation. Nearby patches are likely to share environmental conditions and forest structure, so that a model trained on one patch may effectively see very similar data in validation if patches are not spatially separated. To address this, we implement a spatially aware split of the original train set.

We first compute simple bounding boxes for all 200 m train patches and use these to build overlap-connected components. Two patches belong to the same component if their bounding boxes overlap with positive area, directly or via a chain of overlapping neighbours. This connectivity is established using a union–find data structure over a coarse spatial grid index for efficiency. We then greedily assign components to the new train and validation sets such that approximately one ninth of the original train patches fall into validation, while components remain intact. This ensures that there is no spatial overlap between train and validation regions.

This approach is conceptually related to spatial block cross-validation, where spatially contiguous blocks are used to enforce independence between training and testing subsets (see, for example, Frontiers in Remote Sensing, “Choosing blocks for spatial cross-validation: lessons from a marine remote sensing case study”, 2025, and broader discussions of spatial cross-validation in remote-sensing GeoAI workflows). Here we apply a single split rather than repeated cross-validation folds, but the underlying rationale is the same.

*(Figure 7: Map of TreeSat 200 m patches coloured by final split (train, spatial validation, test), illustrating spatial separation between train and validation areas.)*

### 7.3 HDF5 packaging and metadata

The final step assembles the stacked CHM and Sentinel‑1 patches into HDF5 datasets for efficient training and evaluation. For each split (train, spatial validation, test), we create a separate HDF5 file containing:

- an `images` dataset of shape (N, B, H, W) with float32 values, where N is the number of patches, B is the number of bands (4), and H, W are the spatial dimensions,
- a `labels` dataset with integer class indices corresponding to `BT_GEN`,
- auxiliary string datasets for patch filenames and class names.

The files also store basic metadata as HDF5 attributes, including the number of samples, spatial dimensions, number of bands, and number of classes. A label mapping file is written alongside the HDF5 datasets to document the mapping between class names and indices.

*(Table 7: Overview of final HDF5 datasets, listing number of samples per split, patch shape, and number of classes.)*

## 8. Data quality assessment and limitations

### 8.1 Distribution of CHM heights and backscatter

To characterise the resulting CHM and Sentinel‑1 features, we compute global histograms and quantiles for each band. For CHM, we focus on the distribution of positive heights and examine upper quantiles (e.g. 90th, 95th, 99th, 99.9th percentiles) to identify plausible upper bounds for canopy height in the study area. These quantiles inform the choice of clipping thresholds used in the stacking stage. For Sentinel‑1 bands, we evaluate the distribution of backscatter intensity and of the VV/VH ratio, again paying particular attention to the tails of the distributions.

These diagnostics serve two purposes. First, they provide a compact statistical summary of the data, which can be compared to values reported in other studies of forest structure and radar backscatter. Second, they help reveal artefacts or processing errors, such as systematic nodata patterns, extreme outliers, or unexpected multimodality.

*(Figure 8: Histograms of CHM heights and Sentinel‑1 VV, VH, and VV/VH ratio across all patches, with vertical lines marking the chosen clipping thresholds.)*

### 8.2 Coverage, missing tiles, and rejection rates

Throughout the processing chain, we monitor coverage and rejection rates. At the overlay stage we record how many TreeSat patches are successfully assigned to at least one DTM and DSM tile. During CHM cropping and mosaicking we count patches for which required CHM tiles are missing or for which CHM crops still contain nodata after interpolation. In the stacking and nodata cleaning steps, we report how many patches retain nodata in one or more bands.

Summarising these statistics allows us to quantify the effective sample size available for model training at each stage, and to identify systematic coverage gaps. For example, patches near the boundary of the elevation data or in areas with missing Sentinel‑1 coverage may be systematically excluded. Documenting these patterns is important for interpreting model performance and for considering potential biases.

*(Table 8: Summary of coverage and rejection statistics at each processing stage, including numbers and percentages of patches retained or discarded.)*

### 8.3 Sources of uncertainty and potential biases

Several sources of uncertainty are inherent in the CHM generation procedure. The DTM and DSM products themselves contain measurement noise, interpolation artefacts, and occasional misalignments at tile boundaries. Our local nodata interpolation and 10 m smoothing reduce small-scale noise and fill gaps but can also attenuate sharp terrain features and potentially smooth canopy edges. The simple DSM–DTM formulation assumes that both products are perfectly co-registered and share the same vertical reference; any discrepancies will propagate directly into CHM.

When CHM is aggregated to 200 m resolution and combined with Sentinel‑1 backscatter, additional uncertainties arise from resampling, incidence angle effects, and the limited sensitivity of C‑band backscatter to very tall or dense canopies. Spatial splitting mitigates some forms of overfitting but cannot remove all spatial dependencies, especially when training and validation blocks are still relatively close. These limitations should be borne in mind when interpreting model performance and derived biomass estimates.

Despite these caveats, the described pipeline yields a large, spatially consistent dataset of CHM and Sentinel‑1 features that is suitable for training deep learning models and for exploring the joint information content of structural and radiometric descriptors of forests.

### References (selected)

Earth Lab. “Canopy Height Models, Digital Surface Models & Digital Elevation Models – Work With LiDAR Data in Python.” Earth Data Science, University of Colorado Boulder, accessed 2025.

OpenTopography. “OpenTopography Releases Canopy Height Model Tool.” OpenTopography, accessed 2025.

Geophysical Research Letters. “Mapping Forest Height and Aboveground Biomass by Integrating ICESat‑2 and Sentinel‑1 Data.” AGU, 2021.

Frontiers in Remote Sensing. “Choosing blocks for spatial cross-validation: lessons from a marine remote sensing case study.” Frontiers in Remote Sensing, 2025.

Remote Sensing of Environment and related journals: representative studies on regional canopy height modelling from DTM/DSM combinations and LiDAR‑derived CHMs (e.g. Lang et al., 2019, and similar work).

