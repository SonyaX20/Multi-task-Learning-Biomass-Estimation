
# 1. Introduction
Forests play a central role in ecosystem preservation, and reliable methods for monitoring forest structure at large scales are essential for resource management. Traditional forest inventory relies on field surveys where trained personnel measure attributes like tree height, diameter, and species composition. These methods are accurate at the plot level but costly and impractical for regional or continental assessments.

Remote sensing offers a scalable alternative. Optical sensors such as Landsat and Sentinel-2 have been widely used for vegetation monitoring through spectral reflectance analysis. However, optical imagery is limited by cloud cover, with approximately 67% of Earth's surface obscured at any given time, and tropical forests experiencing persistent cloud conditions for much of the year. Synthetic Aperture Radar (SAR) addresses this limitation through active microwave sensing that operates regardless of weather or illumination conditions. The European Space Agency's Sentinel-1 mission provides freely available C-band SAR data with global coverage and six-day revisit times. Research has demonstrated that SAR backscatter contains information about forest vertical structure, with studies achieving correlation coefficients of 0.72-0.84 for canopy height prediction when combining Sentinel-1 with auxiliary data sources (Li et al., 2020; Wang et al., 2022).

Deep learning has advanced remote sensing image analysis, with convolutional neural networks and encoder-decoder architectures achieving strong performance for classification and regression tasks. Lang et al. (2023) produced a global canopy height map at 10-m resolution by fusing GEDI LiDAR with Sentinel-2 imagery using deep learning, demonstrating the potential of neural networks for large-scale forest parameter estimation. Most existing approaches, however, train separate models for individual forest parameters, overlooking relationships among attributes like species identity and canopy height. Multitask learning offers an alternative paradigm where a single model predicts multiple parameters simultaneously, potentially exploiting shared representations to improve accuracy across related tasks (Vandenhende et al., 2022).

A key challenge for developing multitask models is data availability. Existing benchmark datasets for forest remote sensing typically focus on single tasks: some provide species labels without height annotations, others offer LiDAR-derived heights without species information. This fragmentation limits research on multitask approaches that could exploit the interdependencies among forest attributes. The TreeSatAI benchmark (Ahlswede et al., 2023) provides Sentinel-1 imagery with species labels for German forests, covering 20 European tree species grouped into 15 genera across Lower Saxony. However, it does not include height information. Extending this established benchmark with canopy height labels derived from official German LiDAR elevation products represents a pathway to enable multitask learning research while maintaining data quality and spatial coverage.

This thesis investigates multitask learning for forest attribute estimation from Sentinel-1 SAR imagery. This thesis uses two datasets to address different aspects of multitask learning. The primary focus is on TreeSatAI-CHM, which extends the established TreeSatAI benchmark with canopy height labels derived from German LiDAR data. This dataset enables investigation of fundamental multitask learning challenges like loss scale mismatches, gradient conflicts, and parameter sharing strategies in a controlled environment with well-recognized benchmark data. Additionally, the iMAESTRO dataset provides comprehensive reference data for genus segmentation, canopy height, and biomass estimation, enabling testing of physics-aware deep learning through allometric constraints that link height and biomass.

This thesis addresses the following questions:

1. How can existing single-task forest remote sensing datasets be extended to support multitask learning?

2. Does multitask learning outperform single-task models for forest attribute estimation from SAR imagery?

3. How do shared encoders learn task-specific and common forest features when trained on multiple tasks simultaneously?

4. Where do gradient conflicts emerge between classification and regression tasks in multitask learning?

5. Can physics-aware constraints improve multitask learning for forest monitoring?

The main contributions of this thesis are:

- Extended the TreeSatAI benchmark with canopy height labels derived from official German LiDAR elevation products, creating TreeSatAI-CHM for multitask learning research.
- Developed multitask architectures for joint species classification and height regression from Sentinel-1 SAR imagery.
- Demonstrated physics-aware multitask learning by incorporating allometric constraints linking canopy height and biomass.
- Provided quantitative evidence of task interactions showing where tasks align (deeper layers) and conflict (early layers) in shared architectures.

The remainder of this thesis is organized as follows. Chapter 2 provides the theoretical background, covering SAR remote sensing for forest monitoring, multitask learning principles and applications, and dataset availability for forest parameter extraction. Chapter 3 describes the methodology, including dataset preparations and experimental design. The TreeSatAI-CHM dataset construction involves generating canopy height models from digital terrain and surface models, aligning them to the TreeSatAI patch grid. Model architectures include MLP and U-Net baselines, and a MultiTaskUNet with shared encoder and task-specific decoders for joint learning. The iMAESTRO dataset provides forest data with genus segmentation, canopy height, and biomass labels, enabling testing of allometric constraints. Chapter 4 presents experiments and results, focusing on TreeSatAI-CHM for understanding fundamental multitask learning dynamics, with iMAESTRO experiments demonstrating physics-aware deep learning. Chapter 5 concludes with key findings, connecting back to the thesis questions.

---

## References

Ahlswede, S., et al. (2023). TreeSatAI Benchmark Archive: A multi-sensor, multi-label dataset for tree species classification in remote sensing. Earth System Science Data, 15, 681-706.

Lang, N., et al. (2023). A high-resolution canopy height model of the Earth. Nature Ecology & Evolution, 7, 1778-1789.

Li, W., et al. (2020). High-resolution mapping of forest canopy height using machine learning by coupling ICESat-2 LiDAR with Sentinel-1, Sentinel-2 and Landsat-8 data. International Journal of Applied Earth Observation and Geoinformation, 92, 102163.

Vandenhende, S., et al. (2022). Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7), 3614-3633.

Wang, J., et al. (2022). Forest canopy height mapping by synergizing ICESat-2, Sentinel-1, Sentinel-2 and topographic information based on machine learning methods. Remote Sensing, 14(2), 364.