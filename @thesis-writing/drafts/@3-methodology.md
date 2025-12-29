# 3. Methodology

This chapter describes the methods employed in this research, covering the dataset construction process, the multitask model architecture, and the experimental design. Section 3.1 presents the dataset development, including the derivation of canopy height labels and the assembly of training data. Section 3.2 describes the multitask neural network architecture. Section 3.3 outlines the training procedures and evaluation protocol.


## 3.1 Dataset

A central challenge in this research is the absence of existing datasets that combine SAR imagery with both tree species labels and canopy height annotations. While the TreeSatAI benchmark provides Sentinel-1 imagery with species labels for German forests, it does not include height information. To address this gap, this section describes the construction of an extended dataset that pairs TreeSatAI's Sentinel-1 patches and species labels with canopy height values derived from official German elevation products.

The study area corresponds to the TreeSatAI benchmark coverage in Lower Saxony, Germany. TreeSatAI provides 50,381 image patches at 60 m spatial extent, each containing co-registered aerial imagery, Sentinel-1 SAR data, and Sentinel-2 optical data (Ahlswede et al., 2023). The benchmark covers 20 European tree species grouped into 15 tree genera, with species labels derived from forest administration records. Notably, TreeSatAI employs a spatially distributed sampling strategy across Lower Saxony, capturing diverse tree species characteristics across different forest types and geographic locations. This sampling approach enhances reproducibility and comparability with other studies using the same benchmark. For this research, only the Sentinel-1 component is used, as the objective is to investigate SAR-only multitask learning.

Each Sentinel-1 patch is provided at 10 m resolution and contains two polarization channels: VV (co-polarized) and VH (cross-polarized), both included in the original TreeSatAI dataset. From these, a third feature, the VV/VH ratio, is also provided in the dataset as a simple polarimetric index that correlates with vegetation structure (Dostálová et al., 2021). The resulting three-band SAR input captures both backscatter intensity and polarimetric contrast.

Figure 3.1 illustrates the spatial distribution of LGLN elevation tiles and TreeSatAI patch locations across the study area.

[Figure 3.1: Spatial distribution of elevation data and TreeSatAI samples. (a) LGLN DTM/DSM tile coverage across Lower Saxony. (b) Distribution of TreeSatAI 60 m patch locations, showing the spatially distributed sampling strategy across different forest regions.]

The dataset construction proceeds in two stages. First, a canopy height model (CHM) is generated from digital terrain and surface models and aligned to the TreeSatAI patch grid. Second, the resulting height data is combined with Sentinel-1 backscatter bands, labels are filtered and split into training, validation, and test sets, and the complete dataset is exported as NumPy arrays suitable for deep learning.


### 3.1.1 Canopy Height Model Generation

#### 3.1.1.1 Elevation Data Sources

For canopy height derivation, the elevation data comes from the Lower Saxony State Office for Geoinformation and Land Surveying (LGLN). The LGLN provides two products relevant to CHM derivation:

- **DTM (Digital Terrain Model, DGM in German)**: A 1 m resolution model of bare-earth elevation, derived from airborne LiDAR surveys conducted across Lower Saxony. The DTM represents ground surface elevation with vegetation and buildings removed through point cloud classification and filtering. The product is generated from multiple LiDAR acquisition campaigns and undergoes quality control procedures to ensure consistent vertical accuracy across the coverage area. The data is distributed in the ETRS89/UTM zone 32N coordinate reference system (EPSG:25832) with elevation values referenced to the German height reference system DHHN2016.

- **DSM (Digital Surface Model, DOM in German)**: A 1 m resolution model of the top surface, including trees, buildings, and other above-ground objects. The DSM is derived from the same LiDAR point clouds as the DTM, but uses first-return pulses to capture the uppermost surface rather than ground-classified returns. This product preserves the canopy top structure and is particularly suitable for forestry applications where the highest vegetation points are of interest.

Both products are distributed as tiled GeoTIFF files covering Lower Saxony, with metadata indices in GeoJSON format that specify tile boundaries and download URLs. The tiles follow a regular grid pattern with consistent spatial extent, facilitating automated processing across large areas.


### 3.1.1.2 Generation Procedure

The canopy height model is computed as the difference between the digital surface model and the digital terrain model:

$$\text{CHM} = \text{DSM} - \text{DTM}$$

This approach represents the standard methodology for deriving vegetation height from paired elevation products and has been widely adopted in forestry remote sensing applications (Earth Lab, 2017; NEON, 2023). The resulting CHM represents the height of above-ground objects relative to the terrain surface, which for forested areas corresponds to canopy top height.

The CHM derivation methodology follows established practices in the literature. Shao et al. (2022) demonstrated a similar workflow for regional-scale CHM generation using USGS 3DEP LiDAR data, achieving R² = 0.85 when validated against field measurements. Their approach involved systematic tile processing, quality control, and validation against inventory plots. Stereńczak et al. (2018) evaluated different CHM generation methods from LiDAR point clouds, comparing raw CHM, pit-free CHM, spike-free CHM, and smoothed variants, finding that preprocessing choices significantly affect height estimation accuracy. For DSM-minus-DTM approaches using pre-computed elevation products, as in this study, the key considerations include nodata handling, spatial alignment, and outlier treatment rather than point cloud processing decisions.

The CHM generation process also draws on practices from global canopy height mapping efforts. The ETH Global Canopy Height product (Lang et al., 2023) and the Meta/WRI Global Canopy Height product (Tolan et al., 2024) both rely on LiDAR-derived reference heights for model training and validation, with the understanding that CHM quality directly affects downstream applications. Studies validating these global products against airborne LiDAR have reported RMSE values ranging from 5 to 10 metres depending on forest type and canopy complexity (Moudrý et al., 2024), providing context for acceptable uncertainty levels in derived height labels.

**Tile Selection and Download**

The first step identifies which elevation tiles are needed to cover the TreeSatAI study area. The LGLN tile index is loaded as a GeoJSON file, and a spatial index (R-tree) is constructed on the tile polygons. For each TreeSatAI patch, all tiles whose bounding boxes intersect the patch extent are identified. This process produces a mapping from elevation tiles to TreeSatAI patches, which is then inverted to obtain, for each patch, the list of DTM and DSM tiles required to construct its CHM.

The spatial analysis confirms that the DTM and DSM products share identical tile geometries and spatial coverage. The LGLN archive contains 49,573 tiles in total, of which 6,210 tiles intersect with TreeSatAI patch locations. The coverage analysis achieves 100% overlap with TreeSatAI cells, ensuring that all 50,381 patches can be processed. The unique tile identifiers are extracted and the corresponding files are downloaded via HTTP from the LGLN open data portal. Downloaded tiles are stored as GeoTIFF files, with existing files skipped to enable incremental processing.

**Nodata Interpolation**

Elevation products occasionally contain nodata pixels, typically arising from water bodies, sensor shadows, or processing artifacts. When computing the CHM as DSM minus DTM, nodata pixels in either input propagate to the output: if a pixel is nodata in the DTM but valid in the DSM (or vice versa), the subtraction produces nodata even though partial information exists. This amplification of missing data is undesirable, as it can substantially reduce the usable coverage in forest-edge areas or near water bodies where DTM and DSM acquisition conditions may differ.

To mitigate this issue, nodata values in both DTM and DSM are addressed through iterative neighbourhood interpolation prior to subtraction. This approach, conceptually similar to the inverse distance weighting methods implemented in GDAL's fillnodata utility (GDAL, 2024) and GRASS GIS's r.fill.stats module (GRASS Development Team, 2024), propagates valid values into missing regions from their edges. For each nodata pixel, the algorithm examines its eight immediate neighbours. If at least one valid neighbour exists, the nodata value is replaced by the mean of valid neighbours. This process repeats for five iterations, progressively filling gaps from their boundaries inward.

This neighbourhood-based approach handles small gaps effectively while preserving the spatial structure of the elevation data. For the relatively small gaps typical in the LGLN products (arising from water bodies or isolated sensor dropouts), local averaging provides sufficient accuracy. Pixels that remain nodata after all iterations, typically large water bodies, are retained as nodata in the final CHM.

**CHM Computation and Quality Control**

After interpolation, the CHM is computed by subtracting the DTM from the DSM. Pixels where either input remains nodata produce nodata in the output. The resulting CHM tiles are written as single-band float32 GeoTIFFs with the same spatial metadata as the input elevation products.

The raw CHM values can contain artifacts from several sources. Small inconsistencies between DTM and DSM can produce negative values, which are physically implausible for canopy height. At the upper end, occasional spikes may occur from buildings, power lines, or interpolation errors. To address these issues, two clipping operations are applied:

- Negative CHM values are set to zero, on the assumption that true canopy height cannot be negative.
- Values exceeding the 95th percentile of the CHM distribution across the study area are clipped to that threshold. This percentile-based approach follows practices in forest remote sensing where upper quantiles are used to identify and remove outliers while preserving the main distribution (Potapov et al., 2021; Mulverhill et al., 2022). The specific threshold corresponds to the empirically observed upper bound of realistic canopy heights in the dataset.

Figure 3.2 shows example DTM, DSM, and CHM tiles, illustrating the spatial patterns of terrain, surface elevation, and derived canopy height.

[Figure 3.2: Example DTM, DSM, and CHM tiles for a forested area in Lower Saxony. The DTM (left) shows terrain elevation, the DSM (centre) shows the top surface including tree canopies, and the CHM (right) shows the difference, representing canopy height.]

**Reprojection and Alignment**

The LGLN elevation products use coordinate reference system EPSG:XXXXX, while the TreeSatAI Sentinel-1 patches use EPSG:XXXXX. To combine CHM with SAR data, the CHM must be reprojected and resampled to match the TreeSatAI geometry exactly.

Since the LGLN tiles (approximately XX km × XX km each) are substantially larger than the TreeSatAI patches (60 m × 60 m), each patch requires cropping from the relevant tile(s). The spatial analysis reveals that the majority of TreeSatAI samples fall entirely within a single elevation tile, while some patches near tile boundaries require mosaicking:

- Single-tile samples: 44,191 (87.7%)
- Multi-tile samples (requiring mosaic): 6,190 (12.3%)

The distribution of tile requirements shows:
- 1 tile: 44,191 samples
- 2 tiles: 5,957 samples
- 4 tiles: 233 samples

For patches spanning multiple tiles, the relevant CHM tiles are first mosaicked using rasterio's merge functionality with a "first" strategy for overlapping regions.

A key design decision concerns the resampling method during reprojection from 1 m to 10 m resolution. Standard methods include nearest neighbour, bilinear interpolation, and cubic convolution. However, for canopy height, these averaging-based methods may not be ideal. In forestry applications, height metrics such as the maximum or upper percentiles (90th, 95th) of the height distribution are often more informative than mean height, as they better represent dominant tree height and correlate more strongly with biomass and stand volume (Næsset, 2002). When creating CHM rasters from point clouds, standard practice is to assign each grid cell the maximum height value from contributing points, preserving information about the tallest vegetation (OpenForest4D, 2024; SkyTruth, 2024).

Following this logic, the reprojection uses maximum-value aggregation: each output pixel at 10 m resolution receives the maximum CHM value from the contributing 1 m input pixels. This approach preserves information about the tallest trees within each cell, which is more relevant for characterising forest structure than the spatial average. The reprojection is implemented using rasterio's reproject function with Resampling.max.

The reprojected CHM for each TreeSatAI patch is written as a single-band GeoTIFF file, aligned pixel-for-pixel with the corresponding Sentinel-1 patch. Patches where the CHM contains excessive nodata (more than 50% of pixels) are flagged for exclusion from the training data.

**Validation Against Global Canopy Height Products**

To assess the quality of the derived CHM, comparisons are made against two publicly available global canopy height products:

- **ETH Global Canopy Height** (Lang et al., 2023): A 10 m resolution global product derived from Sentinel-2 imagery and GEDI LiDAR using deep learning. This product represents state-of-the-art optical-based height estimation.
- **Meta Global Canopy Height** (Tolan et al., 2024): A 1 m resolution global product derived from high-resolution Maxar imagery and airborne LiDAR using self-supervised learning.

For the comparison at native resolution (1 m), the LGLN-derived CHM is compared against the Meta product. For the 10 m comparison, the LGLN CHM (after reprojection) is compared against the ETH product. In both cases, a random sample of pixels is extracted from overlapping areas, and three metrics are computed: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Pearson correlation coefficient.

Figure 3.3 shows scatter plots comparing the LGLN CHM against both global products.

[Figure 3.3: Comparison of LGLN-derived CHM against global canopy height products. Left: 1 m resolution comparison with Meta Global Canopy Height, showing MAE, RMSE, and correlation. Right: 10 m resolution comparison with ETH Global Canopy Height.]

Visual inspection of the scatter plots confirms that the LGLN-derived CHM and global products are spatially aligned and capture similar height patterns across the study area. The correlation coefficients indicate positive agreement between the height estimates, with stronger correspondence in areas of moderate canopy height. The RMSE values suggest that the LGLN CHM deviates from the global products by several metres on average, with larger discrepancies occurring at extreme height values.

Interpreting these discrepancies requires caution. The observed differences could arise from several sources: temporal misalignment between data acquisitions, differences in the underlying LiDAR sensors and processing chains, or genuine uncertainty in height estimation. Importantly, no independent ground-truth LiDAR validation is available for the specific TreeSatAI patch locations, making it impossible to definitively attribute the differences to noise in either product. The global products themselves have reported uncertainties of 5-10 m RMSE when validated against airborne LiDAR (Lang et al., 2023; Tolan et al., 2024), suggesting that the observed discrepancies fall within expected ranges for this type of comparison. For the purposes of this study, the LGLN-derived CHM is adopted as the height reference, with the understanding that the labels carry inherent uncertainty that may affect model training and evaluation.




### 3.1.2 Data Preprocessing

The TreeSatAI labels are provided as a JSON file that lists, for each patch, all tree genera present along with their area fractions. This multi-label format reflects the reality that forest patches often contain mixtures of species. However, labels supported by very small areas are unreliable: a genus occupying 2% of a 60 m patch corresponds to only a few trees and may reflect mapping uncertainty rather than meaningful species presence.

**Assembly of the S1+CHM Dataset**

With CHM patches aligned to the TreeSatAI grid, the final step combines the height data with Sentinel-1 backscatter to create the input tensors. For each TreeSatAI patch, four input bands are assembled:

1. **VV**: Co-polarized backscatter coefficient (dB scale)
2. **VH**: Cross-polarized backscatter coefficient (dB scale)
3. **VV/VH**: Ratio of co-polarized to cross-polarized backscatter
4. **CHM**: Canopy height model (metres)

The first three bands come from the TreeSatAI Sentinel-1 archive, while the fourth is the reprojected CHM. All four bands are stacked into a single four-band GeoTIFF file per patch, with bands ordered as (VV, VH, VV/VH, CHM). Basic value checks are applied: any remaining nodata values in the SAR bands are set to a sentinel value, and extreme outliers in the VV/VH ratio are clipped based on global quantiles.

Figure 3.4 shows example four-band patches, illustrating the spatial patterns in each input channel.

[Figure 3.4: Example S1+CHM patches showing all four input bands. VV and VH show backscatter intensity patterns, VV/VH shows polarimetric contrast, and CHM shows canopy height structure.]

**Area-Based Filtering**

Following the approach used in the original TreeSatAI preprocessing, labels are filtered based on area fraction. Only genera occupying more than 7% of the patch area are retained. This threshold balances two considerations: removing spurious labels while retaining ecologically meaningful species mixtures. The specific threshold was chosen based on the TreeSatAI documentation and corresponds to approximately 250 m² of coverage within a 3600 m² patch.

After filtering, each patch retains between one and several labels, with the majority containing one to three genera. The filtered labels form the basis for both training supervision and evaluation.

**Stratified Train/Validation/Test Split**

The original TreeSatAI benchmark provides a predefined 90%/10% train/test split, which is retained in this study to enable direct comparison with published benchmark results. The test set remains unchanged from the original dataset. Within the 90% training portion, stratified sampling is applied to create separate training and validation sets, resulting in a final 8:1:1 split ratio across training, validation, and test sets.

The stratification proceeds as follows. For each patch in the original training portion, a primary label is defined as the genus with the largest area fraction. The patches are then grouped by primary label, and within each group, 1/9 of the samples (approximately 10% of the original training set) are randomly assigned to the validation set while the remainder stays in the training set. This approach ensures that all genera are represented in both training and validation splits, preventing situations where rare genera are absent from validation and thus cannot be monitored during training. The random assignment within strata is performed with a fixed random seed for reproducibility.

**Limitations of the Splitting Strategy**

A known limitation of this splitting approach is that it does not enforce spatial separation between splits. In remote sensing, nearby samples are often more similar than distant ones due to spatial autocorrelation, which arises from continuous environmental gradients, correlated acquisition conditions, and spatial patterns in land cover (Karasiak et al., 2021; Ploton et al., 2020). When spatially correlated samples appear in both training and test sets, model performance can be overestimated because the test samples are not truly independent.

Implementing spatially blocked splits, where entire geographic regions are assigned to different splits, would provide more conservative performance estimates. However, this approach is challenging for the TreeSatAI data because patches are distributed across the study area in a pattern tied to forest administration boundaries rather than a regular grid. Furthermore, spatially blocked splits can severely reduce the effective training data for rare genera that occur only in specific locations. The stratified random split adopted here represents a pragmatic compromise that is widely used in remote sensing classification studies. Performance on truly independent test data from different regions or time periods may be lower than the reported metrics suggest, a limitation discussed further in Chapter 5.

Figure 3.5 shows the distribution of patches across genera for each split, confirming that the stratification achieves balanced representation.

[Figure 3.5: Class distribution across training, validation, and test splits. The bar chart shows the number of patches per genus in each split, demonstrating that stratified sampling preserves class proportions.]

**Class Weighting for Imbalanced Data**

Tree species distributions in temperate European forests are naturally imbalanced. Common genera like Picea (spruce) and Pinus (pine) dominate managed forests, while genera like Quercus (oak) or Fagus (beech) are less frequent in the dataset. This class imbalance poses a fundamental challenge for supervised learning: neural networks trained with standard cross-entropy loss tend to minimise overall error by optimising for majority classes, resulting in systematic underperformance on minority classes (He & Garcia, 2009; Johnson & Khoshgoftaar, 2019). In ecological applications, this behaviour is particularly problematic because rare species are often of greater conservation interest and their accurate classification is essential for biodiversity monitoring and forest management planning.

The literature on addressing class imbalance broadly distinguishes between data-level and algorithm-level approaches. Data-level methods modify the training distribution through oversampling minority classes (e.g., SMOTE; Chawla et al., 2002), undersampling majority classes, or generating synthetic samples through augmentation. While effective in many settings, these approaches alter the original data distribution and may introduce artifacts or require additional hyperparameter tuning. Algorithm-level methods instead modify the learning process itself, typically through cost-sensitive learning that assigns different penalties to different classes.

For this study, inverse frequency weighting is adopted. For a dataset with N total samples and n_c samples of class c, the weight for class c is computed as:

$$w_c = \frac{N}{C \times n_c}$$

where C is the number of classes. This weighting scheme assigns higher importance to rare classes during training by scaling the per-sample loss contributions, encouraging the model to learn discriminative features for minority genera rather than defaulting to majority class predictions. The weights are applied multiplicatively to the cross-entropy loss during backpropagation.

Alternative weighting schemes exist in the literature. The "effective number of samples" approach (Cui et al., 2019) accounts for diminishing marginal returns from additional samples of the same class, introducing a hyperparameter that controls the rate of saturation. Focal loss (Lin et al., 2017) takes a different approach by down-weighting easy examples regardless of class, thereby focusing learning on hard-to-classify samples. Class-balanced loss variants combine frequency-based weighting with focal loss mechanisms. For multi-label settings, asymmetric loss functions (Ridnik et al., 2021) have been proposed to handle the inherent positive-negative imbalance. In remote sensing land cover classification, inverse frequency weighting remains widely used due to its simplicity, interpretability, and demonstrated effectiveness (Cenggoro et al., 2017; Zhang et al., 2025). The computed weights are stored alongside the training labels for application during model training.

**Training Data Preparation**

The final preprocessing step converts the GeoTIFF stacks and filtered labels into arrays suitable for deep learning frameworks. For each split, the four-band patches are loaded and stacked into tensors of shape (N, 4, 6, 6), with remaining SAR nodata values interpolated using neighbourhood averaging and CHM nodata set to zero. Labels are converted to multi-hot vectors of length C, where each active genus receives a value of one; this representation supports both multi-label training and single-label evaluation using the primary label. The resulting arrays, class weights, and genus mappings are stored for direct loading during model training.


## 3.1.3 Dataset Statistics

This section presents the statistical characteristics of the final S1+CHM dataset.

**Band Value Distributions**

Figure 3.6 shows histograms of pixel values for each input band across training, validation, and test splits. The SAR bands exhibit heavy-tailed distributions characteristic of radar backscatter: VV values concentrate between -XX and -XX dB, VH values between -XX and -XX dB, and the VV/VH ratio centres around XX. The CHM distribution peaks near zero metres, corresponding to young stands, clearings, or non-forest areas, with a tail extending to the 95th percentile clipping threshold of XX m. The three splits display nearly identical distributions, confirming that stratified sampling preserved statistical properties across partitions.

[Figure 3.6: Histograms of band values by split. Each subplot shows the distribution of one input band (VV, VH, VV/VH, CHM) across all pixels in the training, validation, and test sets.]

Figure 3.7 presents aggregated histograms for each band across the complete dataset.

[Figure 3.7: Total histogram for each input band across the complete dataset.]

**Sample Visualisation**

Figure 3.8 displays randomly selected patches from each split. The SAR and CHM patterns are spatially coherent: forest areas exhibit higher VH backscatter and positive CHM values, while clearings show lower backscatter and near-zero CHM. No systematic artifacts or misalignments are apparent.

[Figure 3.8: Example patches from training, validation, and test sets. Each row shows one patch with its four input bands (VV, VH, VV/VH, CHM) displayed as greyscale images.]

Figure 3.9 provides additional four-band sample visualisations illustrating the range of forest types and height structures in the dataset.

[Figure 3.9: Additional four-band sample patches showing diverse forest conditions.]

**Class Distribution**

Figure 3.10 shows the distribution of primary labels across genera for each split. Picea and Pinus together account for approximately XX% of all samples, while several genera (e.g., Larix, Pseudotsuga) contain fewer than 200 patches each. The class proportions are consistent across all three partitions, with the test set maintaining the original TreeSatAI distribution.

[Figure 3.10: Distribution of primary labels across training, validation, and test splits. Each group represents one genus, with bars showing counts in each split.]

**Summary Statistics**

Table 3.1 summarises the final dataset characteristics.

| Property | Training | Validation | Test |
|----------|----------|------------|------|
| Number of patches | XX,XXX | X,XXX | X,XXX |
| Number of genera | 15 | 15 | 15 |
| Patch size (pixels) | 6 × 6 | 6 × 6 | 6 × 6 |
| Spatial resolution | 10 m | 10 m | 10 m |
| Input channels | 4 | 4 | 4 |
| Label format | Multi-hot | Multi-hot | Multi-hot |

Table 3.2 reports per-band statistics computed from the training set, used for z-score normalisation during model training.

| Band | Mean | Std | Min | 5th %ile | 95th %ile | Max |
|------|------|-----|-----|----------|-----------|-----|
| VV (dB) | | | | | | |
| VH (dB) | | | | | | |
| VV/VH | | | | | | |
| CHM (m) | | | | | | |

Table 3.3 reports the class weights computed from training set frequencies.

| Genus | Training samples | Weight |
|-------|------------------|--------|
| Picea | | |
| Pinus | | |
| ... | | |

The preprocessing pipeline produced XX,XXX usable patches with complete S1+CHM coverage and valid labels. The class imbalance ratio (largest to smallest class) is approximately XX:1. The resulting dataset preserves spatial alignment between SAR and CHM data and maintains the original TreeSatAI test split for benchmark comparability.


---

## References

Ahlswede, S., et al. (2023). TreeSatAI Benchmark Archive: A multi-sensor, multi-label dataset for tree species classification in remote sensing. Earth System Science Data, 15, 681-706.

Cenggoro, T. W., et al. (2017). Classification of imbalanced land-use/land-cover data using variational semi-supervised learning. In 2017 International Conference on Innovative and Creative Information Technology (ICITech), pp. 1-6. IEEE.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.

Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9268-9277.

Dostálová, A., et al. (2021). European wide forest classification based on Sentinel-1 data. Remote Sensing, 13(3), 337.

Earth Lab (2017). Create a Canopy Height Model With Lidar Data. University of Colorado Boulder. https://earthdatascience.org/courses/earth-analytics/lidar-raster-data-r/lidar-chm-dem-dsm/

GDAL (2024). gdal_fillnodata. GDAL Documentation. https://gdal.org/programs/gdal_fillnodata.html

GRASS Development Team (2024). r.fill.stats - Raster gap filling by interpolation. GRASS GIS Manual. https://grass.osgeo.org/grass78/manuals/r.fill.stats.html

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.

Johnson, J. M., & Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance. Journal of Big Data, 6(1), 27.

Karasiak, N., et al. (2021). Spatial dependence between training and test sets: another pitfall of classification accuracy assessment in remote sensing. Machine Learning, 111, 2715-2740.

Lang, N., et al. (2023). A high-resolution canopy height model of the Earth. Nature Ecology & Evolution, 7, 1778-1789.

Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2980-2988.

Moudrý, V., et al. (2024). Comparison of three global canopy height maps and their applicability to biodiversity modeling: Accuracy issues revealed. Ecosphere, 15(10), e70026.

Mulverhill, C., et al. (2022). Mapping Forest Canopy Height Across Large Areas by Upscaling ALS Estimates with Freely Available Satellite Data. Remote Sensing, 7(9), 12563-12587.

Næsset, E. (2002). Predicting forest stand characteristics with airborne scanning laser using a practical two-stage procedure and field data. Remote Sensing of Environment, 80(1), 88-99.

NEON (2023). Create a Canopy Height Model from Lidar-derived rasters in R. National Ecological Observatory Network. https://www.neonscience.org/resources/learning-hub/tutorials/create-chm-rasters-r

OpenForest4D (2024). What is a Canopy Height Model? https://openforest4d.org/what-is-a-canopy-height-model/

Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nature Communications, 11, 4540.

Potapov, P., et al. (2021). Mapping global forest canopy height through integration of GEDI and Landsat data. Remote Sensing of Environment, 253, 112165.

Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., & Zelnik-Manor, L. (2021). Asymmetric loss for multi-label classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 82-91.

Shao, G., et al. (2022). High-Resolution Canopy Height Model Generation and Validation Using USGS 3DEP LiDAR Data in Indiana, USA. Remote Sensing, 14(4), 935.

SkyTruth (2024). Extensive aerial LiDAR data enables canopy height mapping across the Central Appalachian mining region. https://skytruth.org/2024/12/extensive-aerial-lidar-data-enables-canopy-height-mapping-across-the-central-appalachian-mining-region/

Stereńczak, K., et al. (2018). Testing and evaluating different LiDAR-derived canopy height model generation methods for tree height estimation. International Journal of Applied Earth Observation and Geoinformation, 71, 132-143.

Tolan, J., et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. Remote Sensing of Environment, 300, 113888.

Zhang, Y., et al. (2025). A deep learning-based solution to the class imbalance problem in high-resolution land cover classification. Remote Sensing, 17(11), 1845.



## 3.2 Model Architecture

This research employs two categories of baseline models to establish reference performance levels for the classification and regression tasks. The primary baseline uses multilayer perceptrons (MLPs), following the architecture employed in the original TreeSatAI benchmark (Ahlswede et al., 2023), enabling direct comparison with published results. The secondary baseline uses U-Net architectures, selected for their established effectiveness in image-to-image tasks and their suitability for extension to multitask learning in subsequent experiments.


### 3.2.1 Multilayer Perceptron Baseline

The MLP baseline follows the architecture used in the TreeSatAI benchmark to ensure comparability with published results. The original TreeSatAI study evaluated multiple sensor modalities including aerial imagery, Sentinel-2 optical data, and Sentinel-1 SAR data. For aerial and Sentinel-2 imagery, the benchmark employed ResNet architectures pretrained on ImageNet, leveraging transfer learning from natural image domains. However, for Sentinel-1 data, the benchmark used MLPs rather than convolutional networks, as SAR backscatter characteristics differ fundamentally from optical imagery and pretrained models are not directly transferable (Ahlswede et al., 2023). Since this study focuses exclusively on Sentinel-1 SAR data for canopy height estimation, the MLP architecture provides the appropriate baseline for direct comparison with the TreeSatAI Sentinel-1 results.

MLPs represent a standard approach for patch-level classification when spatial relationships within the patch are not explicitly modelled (Goodfellow et al., 2016). For the small patch sizes in this study (6 × 6 pixels), MLP architectures remain competitive with convolutional approaches while providing a simpler baseline that treats each patch as a feature vector without imposing spatial priors.

**MLP Classifier**

The classification MLP receives the flattened input tensor and processes it through three hidden layers before producing class logits. Given an input patch of shape (C, H, W) where C = 3 channels (VV, VH, VV/VH), H = W = 6 pixels, the input is first flattened to a vector of dimension C × H × W = 108. The network architecture consists of:

- Input layer: 108 features (flattened 3 × 6 × 6 input)
- Hidden layer 1: 512 units, batch normalisation, ReLU activation, dropout
- Hidden layer 2: 512 units, batch normalisation, ReLU activation, dropout
- Hidden layer 3: 512 units, batch normalisation, ReLU activation, dropout
- Output layer: 15 units (one per genus)

Each hidden layer follows the pattern Linear → BatchNorm → ReLU → Dropout, which has become standard practice for stabilising training and preventing overfitting in deep networks (Ioffe & Szegedy, 2015). The output layer produces raw logits that are passed through a sigmoid function during inference to obtain per-class probabilities for multi-label classification. At evaluation time, a threshold of 0.5 is applied to convert probabilities to binary predictions.

**MLP Regressor**

The regression MLP shares the same hidden layer structure (three layers of 512 units each) but differs in its output configuration. The network outputs a vector of length H × W = 36, which is reshaped to a spatial map of shape (1, 6, 6). To ensure non-negative height predictions, the output passes through a softplus activation function, defined as softplus(x) = log(1 + exp(x)), which provides a smooth approximation to the ReLU function while avoiding dead gradients at zero (Dugas et al., 2001).

MLPs have been successfully applied to various remote sensing regression tasks, including vegetation parameter retrieval from satellite imagery. Studies on leaf area index (LAI) estimation have demonstrated that MLPs can effectively learn the inverse mapping from spectral observations to biophysical parameters when coupled with radiative transfer model simulations (Remote Sensing, 2025). Similarly, research on tree height-diameter relationships has shown that shallow MLPs achieve competitive performance with ensemble methods like random forests and gradient boosting for forest parameter estimation (Forests, 2025). These applications support the feasibility of MLP-based regression for canopy height estimation from SAR backscatter values.

Figure 3.X illustrates the MLP architecture for both classification and regression tasks.

[Figure 3.X: MLP baseline architecture. (a) Classifier: flattened input passes through three hidden layers of 512 units each to produce class logits. (b) Regressor: similar structure but outputs a spatial height map with softplus activation.]


### 3.2.2 U-Net Baseline

The U-Net architecture (Ronneberger et al., 2015) was originally developed for biomedical image segmentation and has since become a foundational architecture for dense prediction tasks in remote sensing (Stoian et al., 2019; Kattenborn et al., 2021). The key innovation of U-Net is the use of skip connections between encoder and decoder stages, which preserve fine-grained spatial information that would otherwise be lost during downsampling. This property makes U-Net particularly suitable for tasks requiring pixel-level outputs, such as canopy height estimation. U-Net has demonstrated strong performance across diverse remote sensing applications including land cover mapping, crop classification, and forest monitoring (Kattenborn et al., 2021).

**Challenges of Small Patch Sizes**

The patch dimensions in this study (6 × 6 pixels, corresponding to 60 m × 60 m ground coverage) present specific challenges for deep learning architectures. Most remote sensing studies employ substantially larger patch sizes, typically ranging from 64 × 64 to 256 × 256 pixels, or work with imagery covering kilometre-scale extents at metre-level resolution (Li et al., 2018; Wang et al., 2024). The small spatial extent of the patches used here means that the model primarily learns local texture and backscatter patterns rather than broader contextual features spanning multiple land cover types or forest stand boundaries.

This constraint has direct implications for network design. Standard U-Net architectures employ multiple downsampling stages that progressively reduce spatial dimensions by factors of 2, enabling the network to capture increasingly abstract features at coarser scales. However, with an input size of only 6 × 6 pixels, aggressive downsampling would rapidly eliminate spatial structure entirely. After just two downsampling operations with stride 2, the feature map would reduce to dimensions too small to preserve meaningful spatial relationships. Research on small-patch remote sensing classification has noted that pooling operations can cause representational bottlenecks when key features are minute or irregularly distributed (MDPI Remote Sensing, 2022), and that special architectural considerations are needed to prevent excessive information loss (Liang et al., 2021).

For the U-Net architecture in this study, the downsampling pathway is therefore carefully designed to balance two competing requirements: extracting hierarchical features through spatial compression while preserving sufficient spatial information for the decoder to reconstruct pixel-level predictions. The encoder uses only two downsampling stages, reducing the 6 × 6 input first to 3 × 3, then to a 1 × 1 bottleneck representation. This minimal depth ensures that spatial structure is not entirely collapsed before reaching the decoder, while still allowing the network to learn abstract representations at the bottleneck.

**Architectural Components**

The U-Net implementation uses three fundamental building blocks that progressively transform input features into task-specific predictions:

*DoubleConv Block*: Each encoding and decoding stage applies two consecutive 3 × 3 convolutions, each followed by batch normalisation and ReLU activation. This double convolution pattern, inherited from the original U-Net design, allows the network to learn increasingly complex feature combinations at each spatial scale. The first convolution in each block transforms the input channels to the target channel count, while the second convolution refines these features by learning nonlinear combinations. The learned weights in these layers capture local spatial patterns such as texture gradients, edge responses, and channel correlations that are diagnostic of forest structure.

*Down Block*: Downsampling is performed using strided convolutions rather than pooling operations. The strided convolution applies a learnable filter while simultaneously reducing spatial dimensions, allowing the network to learn which information to retain during compression rather than applying fixed pooling functions (Springenberg et al., 2015). The learned weights determine how neighbouring pixels are combined during downsampling, effectively learning task-specific spatial aggregation patterns. In the context of SAR imagery, these weights may learn to emphasise consistent backscatter patterns while suppressing speckle-like variations.

*Up Block*: Upsampling is performed using transposed convolutions that restore the spatial dimensions. The transposed convolution learns to distribute compressed feature information back across the spatial grid in a manner that supports accurate reconstruction. After upsampling, features are concatenated with the corresponding skip connection from the encoder, then processed by a DoubleConv block. This concatenation is essential for preserving spatial constraints and fine-grained features that would otherwise be lost during the compression-expansion cycle. The skip connections provide high-resolution spatial cues that guide the decoder in placing predicted values at correct locations.

**U-Net Classifier**

For patch-level multi-label classification, the goal differs from the pixel-wise segmentation task for which U-Net was originally designed. Rather than predicting a class label for each pixel independently, the classifier must produce a single set of genus probabilities for the entire patch. This distinction motivates a simplified variant that focuses on extracting discriminative features rather than preserving pixel-level spatial structure.

The classification U-Net follows the encoder pathway to extract hierarchical features but adapts the output mechanism for patch-level prediction. After the bottleneck, the decoder reconstructs a feature map at the original spatial resolution, and a 1 × 1 convolution produces a feature map of shape (num_classes, 6, 6), representing per-pixel class activations. Global average pooling is then applied across the spatial dimensions, yielding logits of shape (num_classes,). This pooling operation aggregates evidence from all spatial locations, which is appropriate for patch-level multi-label classification where the entire patch receives genus labels rather than individual pixels.

This approach of using encoder features followed by global pooling for classification is common in remote sensing applications where patch-level labels are available but pixel-level annotations are not. The TreeSatAI benchmark itself employs similar strategies for Sentinel data, using MLP or lightweight architectures that aggregate spatial information into patch-level predictions (Ahlswede et al., 2023). The decoder pathway, while included for architectural consistency with the regressor, primarily serves to refine spatial feature maps before pooling rather than producing pixel-wise outputs.

Dropout is applied after the first upsampling stage to provide regularisation. The output logits are passed through a sigmoid function during inference to obtain independent per-class probabilities.

**U-Net Regressor**

The regression U-Net requires full spatial output, as canopy height varies across the patch and must be predicted at each pixel location. This task fully exploits the U-Net architecture's strength in combining contextual understanding with spatial precision.

The encoder progressively compresses spatial dimensions while expanding feature depth:

- Initial DoubleConv: 3 → B channels (where B is the base channel count), spatial 6 × 6
- Down + DoubleConv: B → 2B channels, spatial 6 × 6 → 3 × 3
- Down + DoubleConv: 2B → 4B channels, spatial 3 × 3 → 1 × 1 (bottleneck)

The bottleneck representation at 1 × 1 spatial resolution captures the most abstract, globally-informed features of the input patch. At this stage, all spatial information has been compressed into channel-wise feature vectors that encode the overall characteristics of the patch.

The decoder mirrors the encoder, progressively restoring spatial resolution:

- Up: 4B → 2B channels, spatial 1 × 1 → 3 × 3, concatenate with encoder features
- DoubleConv: refine concatenated features
- Up: 2B → B channels, spatial 3 × 3 → 6 × 6, concatenate with encoder features
- DoubleConv: refine concatenated features

The concatenation at each decoder stage combines upsampled features (which carry contextual information from deeper layers) with skip connection features (which retain high-resolution spatial details from the encoder). This fusion is critical for the regression task: the contextual features inform the overall height distribution expected in the patch, while the spatial features guide precise localisation of height variations. Without skip connections, the decoder would need to reconstruct all spatial detail from the compressed bottleneck representation alone, leading to blurred or spatially imprecise predictions (Ronneberger et al., 2015; Wang et al., 2023).

A final 1 × 1 convolution produces a single-channel output of shape (1, 6, 6). The raw output passes through a softplus activation to ensure non-negativity, as canopy height cannot be negative. Unlike the classifier, no global pooling is applied; the output retains full spatial resolution, enabling pixel-wise height predictions.

Figure 3.X shows the U-Net architecture with channel dimensions at each stage.

[Figure 3.X: U-Net baseline architecture. The encoder (left) progressively reduces spatial dimensions while increasing channel depth. The decoder (right) restores spatial resolution using skip connections (horizontal arrows). The classifier applies global average pooling after the final decoder stage to produce class logits; the regressor outputs a spatial height map directly.]

**Channel Configuration**

The base channel count B is a hyperparameter that controls model capacity. Higher values increase the number of learnable parameters and representational capacity but also increase computational cost and risk of overfitting. The channel progression follows the pattern B → 2B → 4B through the encoder, which is standard practice in U-Net variants (Ronneberger et al., 2015). The specific value of B is determined through hyperparameter tuning for each task.

Table 3.X summarises the architectural specifications for both baseline model families.

| Model | Input Shape | Hidden/Base Channels | Output Shape | Parameters (approx.) |
|-------|-------------|---------------------|--------------|---------------------|
| MLP Classifier | (3, 6, 6) | 512, 512, 512 | (15,) | ~550K |
| MLP Regressor | (3, 6, 6) | 512, 512, 512 | (1, 6, 6) | ~550K |
| U-Net Classifier | (3, 6, 6) | B = 128 | (15,) | ~2M |
| U-Net Regressor | (3, 6, 6) | B = 256 | (1, 6, 6) | ~8M |


### 3.2.3 Design Rationale for Multitask Extension

The baseline models are designed with multitask learning extension in mind. The U-Net architecture is particularly suitable for this purpose because the encoder naturally provides shared representations that can serve multiple decoder heads. In subsequent experiments, a multitask U-Net variant shares the encoder while maintaining separate classification and regression decoders, as illustrated in Figure 3.X.

[Figure 3.X: Multitask U-Net architecture (for reference). The shared encoder feeds into two separate decoder branches: one for species classification and one for canopy height regression. Skip connections are duplicated for each decoder.]

The shared encoder design follows the hard parameter sharing paradigm, which has been shown to provide effective regularisation by forcing the encoder to learn representations useful for multiple tasks (Ruder, 2017; Vandenhende et al., 2021). This architectural choice establishes a foundation for investigating whether joint training on classification and regression improves performance compared to the single-task baselines described in this section.


---

## References (to add to existing)

Ahlswede, S., Schulz, C., Gava, C., Helber, P., Bischke, B., Förster, M., Arias, F., Hees, J., Demir, B., & Kleinschmit, B. (2023). TreeSatAI Benchmark Archive: a multi-sensor, multi-label dataset for tree species classification in remote sensing. Earth System Science Data, 15, 681-695.

Dugas, C., et al. (2001). Incorporating second-order functional knowledge for better option pricing. Advances in Neural Information Processing Systems, 13, 472-478.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448-456.

Kattenborn, T., et al. (2021). Review on convolutional neural networks (CNN) in vegetation remote sensing. ISPRS Journal of Photogrammetry and Remote Sensing, 173, 24-49.

Li, Y., Zhang, H., Xue, X., Jiang, Y., & Shen, Q. (2018). Deep learning for remote sensing image classification: A survey. WIREs Data Mining and Knowledge Discovery, 8(6), e1264.

Liang, X., et al. (2021). DRSNet: Novel architecture for small patch and low-resolution remote sensing image scene classification. International Journal of Applied Earth Observation and Geoinformation, 104, 102841.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 234-241. Springer.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098.

Springenberg, J. T., et al. (2015). Striving for simplicity: The all convolutional net. In ICLR Workshop.

Stoian, A., et al. (2019). Land cover maps production with high resolution satellite image time series and convolutional neural networks: Adaptations and limits for operational systems. Remote Sensing, 11(17), 1986.

Vandenhende, S., et al. (2021). Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7), 3614-3633.

Wang, Y., et al. (2023). A deep learning method for optimizing semantic segmentation accuracy of remote sensing images based on improved UNet. Scientific Reports, 13, 7600.

Wang, Z., et al. (2024). Scale-aware deep reinforcement learning for high resolution remote sensing imagery classification. ISPRS Journal of Photogrammetry and Remote Sensing, 208, 53-73.