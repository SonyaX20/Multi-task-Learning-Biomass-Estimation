# 2. Theoretical Background

This chapter provides the theoretical foundations for this thesis, covering three main areas. Section 2.1 examines SAR remote sensing for forest parameter estimation, including the physical basis of radar-vegetation interactions and existing approaches for height and species extraction. Section 2.2 reviews multitask learning principles and their application in remote sensing contexts. Section 2.3 discusses dataset availability for forest monitoring research and the challenges of obtaining multitask annotations.

## 2.1 SAR Imagery for Forest Parameter Estimation

Synthetic Aperture Radar has become a valuable tool for forest monitoring, offering capabilities that complement optical sensors. The key advantage of SAR is its active sensing mechanism: by transmitting microwave pulses and recording the backscattered signal, SAR systems can acquire imagery regardless of solar illumination or atmospheric conditions. This all-weather, day-and-night capability is particularly useful for monitoring forests in regions with frequent cloud cover, where optical data availability is limited (Moreira et al., 2013).

The interaction between SAR signals and forest canopies provides information about vegetation structure that optical sensors cannot directly capture. Depending on the wavelength, SAR penetrates vegetation to different depths: X-band signals interact primarily with leaves and small branches at the canopy surface, C-band (used by Sentinel-1) penetrates into the upper canopy structure, while L-band and P-band wavelengths can reach tree trunks and even the ground surface (NASA SAR Handbook, 2019). This penetration capability enables SAR to sense vertical forest structure, making it sensitive to parameters such as canopy height.

Research has demonstrated the potential of SAR data for forest canopy height estimation. Li et al. (2020) showed that coupling ICESat-2 LiDAR with Sentinel-1, Sentinel-2 and Landsat-8 data achieved correlation coefficients of 0.78 for canopy height prediction, with the addition of Sentinel-1 backscattering coefficients contributing positively to model performance. Similarly, Wang et al. (2022) found that synergizing ICESat-2, Sentinel-1, Sentinel-2 and topographic information using Random Forest achieved RMSE values of 3.15-3.37 m for different forest types in China. The VH and VV polarization channels from Sentinel-1 were validated as important predictors, with cross-polarized backscatter (VH) showing stronger sensitivity to vegetation structure due to its response to volume scattering.

More recently, deep learning approaches have advanced canopy height estimation. Lang et al. (2023) developed a probabilistic deep learning model that fused sparse GEDI LiDAR data with Sentinel-2 optical images, producing global canopy height maps at 10-m resolution with RMSE of 7.9 m. Schwartz et al. (2024) demonstrated the first application combining GEDI, Sentinel-1, and Sentinel-2 data in a fully convolutional neural network for canopy height regression, achieving MAE of 2.02 m in a French coniferous forest. Castro et al. (2024) found that combining seasonal Sentinel-1 and optical features with PALSAR data yielded R² of 0.72 and RMSE of 3.43 m for northern forests, noting that SAR data provides valuable structural insights complementary to optical spectral information.

For tree species classification from SAR, research has shown that texture features, temporal patterns, and polarimetric decomposition products provide discriminative information. The TreeSatAI benchmark (Ahlswede et al., 2023) demonstrated that Sentinel-1 data contributes to species classification when combined with optical imagery, though optical sensors typically achieve higher accuracy when used alone. Dostálová et al. (2021) achieved European-wide forest classification using Sentinel-1 data, demonstrating the potential of C-band SAR for vegetation type discrimination.

The C-band wavelength of Sentinel-1 presents both opportunities and challenges for forest applications. On one hand, the free availability, global coverage, and frequent revisit make Sentinel-1 data accessible for operational applications in a way that commercial SAR sensors are not. On the other hand, C-band has limited penetration into dense forest canopies compared to longer wavelength systems like L-band ALOS-2 PALSAR, which affects sensitivity to height in mature forests with closed canopies. Torres de Almeida et al. (2022) compared Sentinel-1 and Sentinel-2 for canopy height modeling and found that S2 outperformed S1, with S1 having higher error rates, potentially due to C-band's limited capacity to penetrate complex tropical forest canopies. Despite these limitations, C-band SAR has shown useful correlations with canopy height, particularly when temporal features or texture information are exploited.

The rationale for investigating SAR-only approaches relates to operational constraints. While optical-SAR fusion generally improves model performance, the requirement for cloud-free optical imagery limits how frequently predictions can be made. Approximately 67% of Earth's surface is covered by clouds at any given time, and in tropical forests where cloud cover persists throughout much of the year, waiting for clear optical scenes may delay monitoring by weeks or months. A robust SAR-based approach would enable continuous monitoring regardless of atmospheric conditions, supporting applications in forest change detection, degradation monitoring, and inventory updating where timeliness is important.

Furthermore, developing SAR-only methods has value for understanding what information SAR actually provides about forests. When SAR is always combined with optical data, it becomes difficult to isolate the contribution of radar features to the prediction. By investigating SAR-only models, this research aims to clarify the extent to which structural information in radar backscatter can support forest parameter extraction.

## 2.2 Multitask Learning in Remote Sensing

Multitask learning (MTL) is a machine learning paradigm where a single model is trained to perform multiple related tasks simultaneously, using shared representations to improve performance across all tasks. The underlying principle is that different tasks can share a common feature space, and by learning these representations jointly, the model can benefit from the regularization effect of training on multiple objectives, reducing overfitting and improving generalization (Caruana, 1997; Ruder, 2017).

The theoretical foundations of multitask learning are well established. Vandenhende et al. (2022) provided a comprehensive survey of MTL for dense prediction tasks, identifying that shared encoder architectures with task-specific decoders represent the dominant paradigm. The benefits include improved sample efficiency (learning from multiple supervision signals), implicit regularization (preventing overfitting to task-specific noise), and feature sharing (leveraging complementary information across tasks). However, they also noted that multitask learning does not always help: when tasks compete for model capacity or share little underlying structure, joint training can hurt performance.

In remote sensing, multitask learning has shown promise across several application domains. Srivastava et al. (2017) demonstrated joint height estimation and semantic labeling of monocular aerial images using CNNs, establishing that these geometrically related tasks benefit from shared representations. Liu et al. (2022) developed associative methods for simultaneously segmenting semantics and estimating height from remote sensing imagery, achieving improved performance over single-task baselines. More recently, RSMTMamba (Shen et al., 2024) proposed a Mamba-based MTL framework that simultaneously performs semantic segmentation, height estimation, and boundary detection, achieving state-of-the-art performance by exploiting cross-task features.

For vegetation applications specifically, multitask approaches have been applied to joint prediction of related biophysical parameters. Wang et al. (2024) developed SegCR, a multimodal and multitask network that simultaneously performs semantic segmentation and cloud removal from SAR-optical image pairs, demonstrating that high-level and low-level vision tasks can enhance each other through joint learning. Du et al. (2024) proposed MTCDN, a concatenated deep learning framework for multitask change detection of optical and SAR images, showing the benefits of end-to-end multitask training over sequential approaches.

The application of multitask learning to forest parameter extraction is motivated by the relationships among forest attributes. Tree species identity influences growth patterns, wood density, and maximum attainable height. A spruce tree develops differently from an oak tree, and knowing the species provides prior information about expected height ranges. Canopy height correlates with stand age while also providing geometric information relevant to species discrimination: the height profile, crown shape, and canopy texture differ among species in ways that complement textural differences visible in SAR backscatter. By training a unified model to predict species and height jointly, the shared encoder can learn features that capture these relationships, potentially improving predictions for both tasks.

The question of when multitask learning helps versus hurts performance is central to this research. Zhang and Yang (2018) surveyed multitask learning approaches and identified several factors that influence success: task relatedness (tasks should share underlying structure), data regime (MTL often helps more when individual tasks have limited data), and architectural design (the balance between shared and task-specific parameters matters). For forest parameter extraction, species classification and height estimation appear related through the biophysical properties of trees, but they differ in nature: one is a discrete classification problem, the other a continuous regression. Whether a shared encoder can effectively serve both objectives from SAR features specifically, and how to balance their training signals, are open questions.

Despite theoretical motivation, the specific application of multitask learning for extracting multiple forest parameters from SAR features remains underexplored in the literature. Most existing multitask remote sensing studies leverage optical imagery, either alone or in combination with SAR. This gap presents an opportunity to investigate whether the structural information encoded in SAR backscatter is sufficient to support multitask learning for forest attributes.

## 2.3 Dataset Availability in Forest Monitoring

The development of deep learning models for forest monitoring is constrained by data availability. Unlike natural image datasets which contain millions of labeled examples, remote sensing datasets for forest applications are typically smaller and more specialized. This scarcity arises from the cost and difficulty of collecting ground-truth labels, as forest inventory measurements require trained personnel and often involve accessing remote locations.

Existing benchmark datasets for forest remote sensing tend to focus on single tasks. Some datasets provide multi-sensor imagery with species labels but do not include annotations for canopy height. Others offer LiDAR-derived height information but lack species-level labels, or cover only small geographic areas. This fragmentation means that a model trained for species classification on one dataset cannot easily incorporate height predictions from another, as the spatial locations, sensor configurations, and forest types differ.

The TreeSatAI Benchmark Archive (Ahlswede et al., 2023) represents a notable contribution to this landscape. It provides over 50,000 image triplets (aerial, Sentinel-1, Sentinel-2) with labels for 20 European tree species derived from forest administration data in Lower Saxony, Germany. The dataset has enabled significant research on multi-sensor tree species classification using both traditional machine learning and deep learning methods. However, TreeSatAI focuses on species classification and does not include annotations for canopy height or other structural parameters, limiting its direct applicability for multitask learning research that combines classification and regression objectives.

For canopy height specifically, several global and regional products have been developed. Potapov et al. (2021) produced a 30-m resolution global forest height map using GEDI and Landsat data with tree ensemble methods, achieving mean error of 9.1 m. Lang et al. (2023) improved upon this with a 10-m resolution global canopy height model using deep learning on Sentinel-2 and GEDI data, achieving RMSE of 7.9 m. More recently, Tolan et al. (2024) achieved sub-meter resolution canopy height maps using DiNOv2 models with high-resolution Maxar imagery and airborne LiDAR. These products provide potential sources for deriving height labels to extend species classification datasets.

This situation motivates the exploration of dataset extension strategies. If canopy height estimates can be derived from auxiliary sources such as global canopy height products, regional LiDAR surveys, or forest inventory statistics, it becomes possible to augment existing species classification datasets for multitask learning. The quality and reliability of these derived labels, and their impact on multitask model performance, represent important questions. By investigating how TreeSatAI can be extended with height information and assessing the feasibility of similar extensions to other datasets, this research aims to establish practical pathways for creating multitask-ready resources without requiring expensive new field campaigns.

Additional datasets relevant to this research include PureForest (2024), currently the largest publicly available LiDAR dataset for species classification with detailed vertical structure information, and various national forest inventory databases that provide plot-level species and height measurements. The integration of these diverse data sources, while methodologically challenging due to differences in spatial resolution, temporal coverage, and measurement protocols, offers opportunities to create richer annotations for multitask learning research.

---

## References

Ahlswede, S., et al. (2023). TreeSatAI Benchmark Archive: A multi-sensor, multi-label dataset for tree species classification in remote sensing. Earth System Science Data, 15, 681-706.

Caruana, R. (1997). Multitask learning. Machine Learning, 28(1), 41-75.

Castro, J. B., et al. (2024). A deep learning approach to estimate canopy height and uncertainty by integrating seasonal optical, SAR and limited GEDI LiDAR data over northern forests. arXiv preprint arXiv:2410.18108.

Dostálová, A., et al. (2021). European wide forest classification based on Sentinel-1 data. Remote Sensing, 13(3), 337.

Du, Z., et al. (2024). Concatenated deep learning framework for multitask change detection of optical and SAR images. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17, 719-731.

Lang, N., et al. (2023). A high-resolution canopy height model of the Earth. Nature Ecology & Evolution, 7, 1778-1789.

Li, W., et al. (2020). High-resolution mapping of forest canopy height using machine learning by coupling ICESat-2 LiDAR with Sentinel-1, Sentinel-2 and Landsat-8 data. International Journal of Applied Earth Observation and Geoinformation, 92, 102163.

Liu, W., et al. (2022). Associatively segmenting semantics and estimating height from monocular remote-sensing imagery. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-17.

Moreira, A., et al. (2013). A tutorial on synthetic aperture radar. IEEE Geoscience and Remote Sensing Magazine, 1(1), 6-43.

Potapov, P., et al. (2021). Mapping global forest canopy height through integration of GEDI and Landsat data. Remote Sensing of Environment, 253, 112165.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098.

Schwartz, M., et al. (2024). High-resolution canopy height map in the Landes forest (France) based on GEDI, Sentinel-1, and Sentinel-2 data with a deep learning approach. International Journal of Applied Earth Observation and Geoinformation, 128, 103711.

Shen, X., et al. (2024). Learning cross-task features with Mamba for remote sensing image multitask prediction. IEEE Transactions on Geoscience and Remote Sensing.

Srivastava, S., Volpi, M., & Tuia, D. (2017). Joint height estimation and semantic labeling of monocular aerial images with CNNs. IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 5173-5176.

Tolan, J., et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. Remote Sensing of Environment, 300, 113888.

Torres de Almeida, D. R. A., et al. (2022). Canopy height mapping by Sentinel-1 and -2 satellite images, airborne LiDAR data, and machine learning. Remote Sensing, 14(16), 4112.

Vandenhende, S., et al. (2022). Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7), 3614-3633.

Wang, J., et al. (2022). Forest canopy height mapping by synergizing ICESat-2, Sentinel-1, Sentinel-2 and topographic information based on machine learning methods. Remote Sensing, 14(2), 364.

Wang, M., et al. (2024). SegCR: A multimodal and multitask complementary fusion network for remote sensing semantic segmentation and cloud removal. IEEE Transactions on Geoscience and Remote Sensing, 63, 1-15.

Zhang, Y., & Yang, Q. (2018). An overview of multi-task learning. National Science Review, 5(1), 30-43.