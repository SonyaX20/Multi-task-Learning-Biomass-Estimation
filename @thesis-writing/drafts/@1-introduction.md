
# 1. Introduction

Forests play a central role in the global carbon cycle, biodiversity conservation, and climate regulation. Reliable methods for monitoring forest structure at large scales are essential for forest management, conservation planning, and climate policy. Traditional forest inventory relies on field surveys where trained personnel measure attributes like tree height, diameter, and species composition. While accurate at the plot level, these methods are costly and impractical for regional or continental assessments.

Remote sensing offers a scalable alternative. Optical sensors such as Landsat and Sentinel-2 have been widely used for vegetation monitoring through spectral reflectance analysis. However, optical imagery is limited by cloud cover, with approximately 67% of Earth's surface obscured at any given time, and tropical forests experiencing persistent cloud conditions for much of the year. Synthetic Aperture Radar (SAR) addresses this limitation through active microwave sensing that operates regardless of weather or illumination conditions. The European Space Agency's Sentinel-1 mission provides freely available C-band SAR data with global coverage and six-day revisit times. Research has demonstrated that SAR backscatter contains information about forest vertical structure, with studies achieving correlation coefficients of 0.72-0.84 for canopy height prediction when combining Sentinel-1 with auxiliary data sources (Li et al., 2020; Wang et al., 2022).

Deep learning has advanced remote sensing image analysis, with convolutional neural networks and encoder-decoder architectures achieving strong performance for classification and regression tasks. Lang et al. (2023) produced a global canopy height map at 10-m resolution by fusing GEDI LiDAR with Sentinel-2 imagery using deep learning, demonstrating the potential of neural networks for large-scale forest parameter estimation. Most existing approaches, however, train separate models for individual forest parameters, overlooking relationships among attributes like species identity and canopy height. Multitask learning offers an alternative paradigm where a single model predicts multiple parameters simultaneously, potentially exploiting shared representations to improve accuracy across related tasks (Vandenhende et al., 2022).

A key challenge for developing multitask models is data availability. Existing benchmark datasets for forest remote sensing typically focus on single tasks: some provide species labels without height annotations, others offer LiDAR-derived heights without species information. This fragmentation limits research on multitask approaches that could exploit the interdependencies among forest attributes. Extending existing datasets with additional parameters derived from auxiliary sources represents a potential pathway to enable multitask learning research.

Building on these observations, this thesis addresses the following research questions:

- **RQ1:** How can existing single-task forest remote sensing datasets be extended to support multitask learning, and what are the quality implications of derived annotations?
- **RQ2:** Can a single deep learning model simultaneously predict tree species and canopy height from Sentinel-1 SAR imagery, and what architectural choices influence performance?
- **RQ3:** Does multitask learning improve prediction performance compared to single-task models, and under what conditions?
- **RQ4:** How do multi-scale feature fusion strategies and architectural components contribute to multitask model performance?

This thesis investigates multitask learning for the simultaneous extraction of tree species and canopy height from SAR imagery. Using Sentinel-1 data and building upon the TreeSatAI benchmark dataset (Ahlswede et al., 2023), this work develops a methodology for extending existing single-task datasets with canopy height labels, designs a multitask deep learning architecture with shared encoder and task-specific heads, establishes an experimental framework for training and evaluation, conducts systematic ablation and comparison studies, and derives practical recommendations for operational deployment.

The contributions of this thesis are:

- An extended multitask forest dataset with documented methodology for deriving height labels from auxiliary sources.
- A SAR-only multitask architecture for joint species classification and height estimation, addressing a gap in the literature.
- Systematic comparison of multitask versus single-task learning under controlled conditions, quantifying when joint learning is beneficial.
- Ablation-based insights identifying essential architectural components for SAR-based forest parameter extraction.
- Practical recommendations for data preparation, training strategies, and deployment considerations.

The remainder of this thesis is organized as follows. Chapter 2 provides the theoretical background, covering SAR remote sensing for forest monitoring, multitask learning principles and applications, and dataset availability for forest parameter extraction. Chapter 3 describes the methodology, including dataset extension procedures, model architecture, and experimental design. Chapter 4 presents experiments and results organized around the research questions. Chapter 5 discusses findings, limitations, and implications. Chapter 6 concludes with key findings and directions for future work.

---

## References

Ahlswede, S., et al. (2023). TreeSatAI Benchmark Archive: A multi-sensor, multi-label dataset for tree species classification in remote sensing. Earth System Science Data, 15, 681-706.

Lang, N., et al. (2023). A high-resolution canopy height model of the Earth. Nature Ecology & Evolution, 7, 1778-1789.

Li, W., et al. (2020). High-resolution mapping of forest canopy height using machine learning by coupling ICESat-2 LiDAR with Sentinel-1, Sentinel-2 and Landsat-8 data. International Journal of Applied Earth Observation and Geoinformation, 92, 102163.

Vandenhende, S., et al. (2022). Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7), 3614-3633.

Wang, J., et al. (2022). Forest canopy height mapping by synergizing ICESat-2, Sentinel-1, Sentinel-2 and topographic information based on machine learning methods. Remote Sensing, 14(2), 364.