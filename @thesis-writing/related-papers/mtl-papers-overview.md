# Literature Review: Multitask Learning

## 1. Fundamentals of Multitask Learning

Multitask learning (MTL) is a learning paradigm that improves generalization performance by leveraging domain-specific information contained in the training signals of related tasks. By learning multiple tasks simultaneously through a shared representation, what is learned for each task can help other tasks be learned more effectively. The fundamental premise, established by Caruana (1997), is that related tasks share commonalities that, when exploited, lead to improved data efficiency, reduced overfitting through shared representations, and faster learning by leveraging auxiliary information.

* Caruana, R. (1997). Multitask Learning. *Machine Learning*, 28, 41-75. [Foundational paper establishing MTL principles]
* Zhang, Y., & Yang, Q. (2021). A Survey on Multi-Task Learning. *IEEE Transactions on Knowledge and Data Engineering*, 34(12), 5586-5609. [Comprehensive survey categorizing MTL into feature learning, low-rank, task clustering, task relation learning, and decomposition approaches]

---

## 2. Architectural Approaches in Deep Multitask Learning

### 2.1 Hard Parameter Sharing

Hard parameter sharing remains the most prevalent approach to MTL in neural networks. The architecture typically consists of shared hidden layers between all tasks, while maintaining task-specific output layers. This approach has been theoretically shown to reduce the risk of overfitting by an order of N (where N is the number of tasks), as it becomes harder for the model to find a shared representation that fits all tasks simultaneously while also overfitting.

* Crawshaw, M. (2020). Multi-Task Learning with Deep Neural Networks: A Survey. *arXiv:2009.09796*. [Comprehensive survey on deep MTL covering architectures, optimization, and task relationships]
* Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., & Van Gool, L. (2021). Multi-Task Learning for Dense Prediction Tasks: A Survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), 3614-3633. [Seminal survey specifically addressing dense prediction tasks like segmentation and depth estimation]

For encoder-decoder architectures like U-Net, hard parameter sharing typically manifests as a shared encoder that branches into task-specific decoders. This design is particularly suited for your patch-wise classification and height regression tasks, as both tasks can benefit from shared low-level feature extraction while maintaining specialized prediction heads.

* Shi, H., et al. (2023). Deep Multitask Learning with Progressive Parameter Sharing. *ICCV 2023*. [Proposes progressively increasing parameter sharing during training to balance shared and task-specific learning]

### 2.2 Soft Parameter Sharing

In soft parameter sharing, each task has its own set of parameters, and the distance between parameters of different models is regularized to encourage similarity. Cross-stitch networks pioneered this approach by learning linear combinations of activations from multiple task-specific networks.

* Misra, I., Shrivastava, A., Gupta, A., & Hebert, M. (2016). Cross-Stitch Networks for Multi-Task Learning. *CVPR 2016*, 3994-4003. [Introduces cross-stitch units for soft parameter sharing]
* Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., & Chi, E. H. (2018). Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. *KDD 2018*, 1930-1939. [Proposes MMoE architecture for modeling task relationships]

### 2.3 Encoder-Focused vs. Decoder-Focused Architectures

Recent work has distinguished between encoder-focused models (EFM) and decoder-focused models (DFM). Encoder-focused approaches aim to learn richer feature representations by sharing information during encoding, while decoder-focused approaches propagate task-specific information during decoding.

* Vandenhende, S., et al. (2021). [Survey referenced above] demonstrates that decoder-focused architectures generally outperform encoder-focused ones for multi-task dense prediction in terms of overall performance.

For your U-Net-based approach combining classification and regression, the encoder-focused paradigm with task-specific decoder heads is the natural starting point, though you may benefit from exploring cross-task interaction mechanisms in the decoder.

---

## 3. Optimization Strategies for Multitask Learning

### 3.1 Loss Weighting Methods

A critical challenge in MTL is balancing contributions from different tasks. Naïve equal weighting often leads to suboptimal performance as tasks may learn at different rates or have losses of different magnitudes.

**Uncertainty-Based Weighting**: Kendall et al. (2018) proposed deriving task weights from homoscedastic uncertainty, automatically balancing tasks by learning noise parameters that scale inversely with each task's importance.

* Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR 2018*, 7482-7491. [Foundational work on uncertainty-based automatic task weighting]

**GradNorm**: Chen et al. (2018) introduced gradient normalization to balance gradient magnitudes across tasks, dynamically adjusting weights to equalize training rates.

* Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. *ICML 2018*, 794-803. [Proposes balancing gradients by learning task weights that normalize gradient magnitudes]

**Dynamic Weight Average (DWA)**: Liu et al. (2019) proposed a simpler approach that adjusts weights based on the rate of change of task losses, requiring no gradient computation for weight updates.

* Liu, S., Johns, E., & Davison, A. J. (2019). End-to-End Multi-Task Learning with Attention. *CVPR 2019*, 1871-1880. [Introduces MTAN architecture and DWA loss weighting]

For your classification-regression combination, uncertainty weighting is particularly relevant as it naturally handles different loss scales (cross-entropy for classification vs. L1/L2 for regression).

### 3.2 Gradient Manipulation Methods

When tasks have conflicting gradients (negative cosine similarity), standard gradient descent can lead to suboptimal solutions where one task dominates.

**PCGrad (Projecting Conflicting Gradients)**: Yu et al. (2020) proposed projecting conflicting gradients onto each other's normal planes to eliminate destructive interference.

* Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient Surgery for Multi-Task Learning. *NeurIPS 2020*, 5824-5836. [Introduces gradient projection to resolve task conflicts]

**CAGrad (Conflict-Averse Gradient Descent)**: Liu et al. (2021) extended this by finding update vectors that maximize the worst-case improvement across all tasks.

* Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. (2021). Conflict-Averse Gradient Descent for Multi-task Learning. *NeurIPS 2021*, 18878-18890. [Proposes optimization that avoids gradient conflicts while ensuring progress on all tasks]

**FAMO (Fast Adaptive Multitask Optimization)**: A recent advancement that ensures equal optimization rates across tasks without requiring computation of all task gradients at each step.

* Liu, B., Feng, Y., Stone, P., & Liu, Q. (2023). FAMO: Fast Adaptive Multitask Optimization. *arXiv:2306.03792*. [Efficient multitask optimizer with amortized gradient computation]

A critical finding from recent work suggests that sophisticated gradient manipulation methods may not always outperform simple scalarization with proper hyperparameter tuning:

* Xin, D., Ghorbani, B., Garg, A., Firat, O., & Gilmer, J. (2022). Do Current Multi-Task Optimization Methods in Deep Learning Even Help? *NeurIPS 2022*. [Questions whether complex MTL optimizers consistently outperform well-tuned baselines]

This suggests that for your experiments, you should establish strong baseline performance with uncertainty weighting before investing in more complex gradient manipulation.

---

## 4. Attention-Based Multitask Architectures for Dense Prediction

### 4.1 Multi-Task Attention Networks (MTAN)

MTAN introduced soft-attention modules for learning task-specific features from a global feature pool, achieving parameter efficiency while allowing tasks to selectively attend to shared features.

* Liu, S., Johns, E., & Davison, A. J. (2019). [Referenced above] [MTAN architecture with task-specific attention masks]

### 4.2 Transformer-Based Approaches

Recent work has leveraged transformer architectures for cross-task reasoning, using query-based mechanisms to facilitate task interaction.

* Xu, Y., Li, X., Yuan, H., Yang, Y., Zhang, J., Tong, Y., Zhang, L., & Tao, D. (2023). Multi-Task Learning with Multi-Query Transformer for Dense Prediction. *IEEE Transactions on Circuits and Systems for Video Technology*, 34(2), 1228-1240. [Uses task-specific queries and cross-task attention for dense prediction]

* Ye, H., & Xu, D. (2022). Inverted Pyramid Multi-Task Transformer for Dense Scene Understanding. *ECCV 2022*, 514-530. [Proposes efficient cross-task interaction at multiple resolutions]

* Lopes, I., Vu, T. H., & de Charette, R. (2023). Cross-Task Attention Mechanism for Dense Multi-Task Learning. *WACV 2023*. [Introduces explicit cross-task attention for information exchange]

For U-Net-based architectures, attention mechanisms can be incorporated into skip connections or decoder blocks to enable task-aware feature selection.

---

## 5. Task Relationships: Synergies and Conflicts

### 5.1 Understanding Task Interactions

Not all tasks benefit equally from joint training. Tasks can exhibit positive transfer (complementary information), negative transfer (conflicting representations), or neutral relationships.

* Standley, T., Zamir, A., Chen, D., Guibas, L., Malik, J., & Savarese, S. (2020). Which Tasks Should Be Learned Together in Multi-task Learning? *ICML 2020*, 9120-9132. [Systematic study of task groupings and their impact on performance]

Key finding: Task similarity doesn't guarantee beneficial joint learning. Network capacity, dataset size, and training dynamics all influence whether task combinations improve performance.

### 5.2 Classification and Regression Task Interactions

Your specific combination of classification (tree species) and regression (height estimation) tasks is particularly interesting. These tasks share spatial structure information but have different output characteristics:

- **Complementary aspects**: Both tasks benefit from learning forest structure, texture, and spatial patterns. Classification boundaries often correlate with height transitions between species.
- **Potential conflicts**: Classification focuses on discriminative boundaries between categories, while regression learns continuous mappings. The optimal feature representations may differ.

* Sener, O., & Koltun, V. (2018). Multi-Task Learning as Multi-Objective Optimization. *NeurIPS 2018*, 527-538. [Frames MTL as multi-objective optimization, providing theoretical grounding for understanding task trade-offs]

---

## 6. Multitask Learning in Remote Sensing Applications

### 6.1 Forest Parameter Estimation

Multitask learning has shown particular promise for forest monitoring, where related biophysical parameters share underlying representations from SAR and optical imagery.

* Chhapariya, K., Benoit, A., Buddhiraju, K. M., & Kumar, A. (2024). A Multitask Deep Learning Model for Classification and Regression of Hyperspectral Images: Application to the Large-scale Dataset. *arXiv:2407.16384*. [Directly relevant: combines classification of forest types with regression of continuous variables like height, biomass, and LAI using shared encoder architecture]

This work demonstrates that U-Net-style architectures can effectively handle simultaneous classification (three classes) and regression (ten continuous forest variables) on hyperspectral data, achieving competitive performance on both task types through hard parameter sharing.

### 6.2 Tree Species Classification with Satellite Data

* Ahlswede, S., Schulz, C., Gava, C., Helber, P., Bischke, B., Förster, M., Lemme, F., & Demir, B. (2023). TreeSatAI Benchmark Archive: A Multi-sensor, Multi-label Dataset for Tree Species Classification in Remote Sensing. *Earth System Science Data*, 15(2), 681-695. [Benchmark dataset combining aerial, Sentinel-1, and Sentinel-2 data for 20 European tree species]

This dataset provides insights into the challenges of tree species classification from satellite imagery and establishes baseline performance metrics that inform expectations for your classification task.

* Grabska-Szwagrzyk, E., Hostert, P., Pflugmacher, D., & Ostapowicz, K. (2023). Mapping Tree Species Diversity in Temperate Montane Forests Using Sentinel-1 and Sentinel-2 Imagery and Topography Data. *Remote Sensing of Environment*, 293, 113576. [Demonstrates that combining S1 SAR with S2 optical data improves tree species mapping]

### 6.3 Height and Biomass Estimation

* Zhang, W., Zhao, L., Li, Y., Shi, J., Yan, M., & Ji, Y. (2022). Forest Above-Ground Biomass Inversion Using Optical and SAR Images Based on a Multi-Step Feature Optimized Inversion Model. *Remote Sensing*, 14(7), 1608. [Multi-source approach to forest parameter estimation]

* Becker, A., Russo, S., Puliti, S., Lang, N., Schindler, K., & Wegner, J. D. (2023). Country-Wide Retrieval of Forest Structure from Optical and SAR Satellite Imagery with Deep Ensembles. *ISPRS Journal of Photogrammetry and Remote Sensing*, 195, 269-286. [Large-scale forest structure estimation combining optical and SAR]

### 6.4 Multitask Approaches for Forest Monitoring

* SaTHE (2025): A recent multitask architecture for tree height estimation combining optical and SAR data, producing both tree masks (segmentation) and height maps (regression) simultaneously.

* Tamiminia, H., Salehi, B., Mahdianpari, M., & Goulden, T. (2024). State-Wide Forest Canopy Height and Aboveground Biomass Map for New York with 10m Resolution, Integrating GEDI, Sentinel-1, and Sentinel-2 Data. *Ecological Informatics*, 79, 102404. [Demonstrates wall-to-wall mapping integrating multiple data sources]

---

## 7. Representation Learning and Feature Analysis in MTL

### 7.1 Understanding Shared Representations

A key question for your research is what representations are learned and how they benefit both tasks. Recent work has begun analyzing the internal representations of multitask networks.

* Zamir, A. R., Sax, A., Shen, W., Guibas, L. J., Malik, J., & Savarese, S. (2018). Taskonomy: Disentangling Task Transfer Learning. *CVPR 2018* (Best Paper). [Establishes transfer relationships between 26 tasks, providing empirical foundation for understanding task compatibility]

Key insight from Taskonomy: Semantic segmentation and depth estimation share transfer structure, suggesting that classification and height regression may similarly benefit from shared representations.

### 7.2 Encoder Representations in Dense Prediction

* Bruggemann, D., Kanakis, M., Obukhov, A., Georgoulis, S., & Van Gool, L. (2021). Exploring Relational Context for Multi-Task Dense Prediction. *ICCV 2021*, 15869-15878. [Analyzes how tasks relate through learned representations]

For SAR imagery specifically, the encoder must learn to extract both:
- **Structural information** (edges, textures, backscatter patterns) useful for classification
- **Geometric information** (height-related phase/intensity patterns) useful for regression

These may be complementary: forest structure often correlates with both species composition and height.

---

## 8. Practical Considerations and Recommendations

### 8.1 Architecture Selection for Your Task

For patch-wise classification and height regression using U-Net with Sentinel-1 SAR:

**Recommended starting architecture**: Hard parameter sharing with shared encoder and task-specific decoder heads
- Classification head: Final convolution → softmax for tree species classes
- Regression head: Final convolution → linear activation for height values

* Ji, N. H., Dong, H. Q., Meng, F. Y., et al. (2023). Semantic Segmentation and Depth Estimation Based on Residual Attention Mechanism. *Sensors*, 23(17), 7466. [Example of combining segmentation and depth estimation in a shared architecture]

### 8.2 Loss Function Configuration

Given your combination of classification and regression:

```
L_total = w_cls * L_classification + w_reg * L_regression
```

**Options for weight determination**:
1. Fixed weights with grid search
2. Uncertainty weighting (Kendall et al., 2018)
3. Dynamic Weight Average (Liu et al., 2019)

The regression loss scale typically differs significantly from cross-entropy, making automatic weighting methods particularly valuable.

### 8.3 Potential Task Synergies to Investigate

Based on the literature, you should analyze:

1. **Do class boundaries correlate with height discontinuities?** If species transitions co-occur with height changes, the shared encoder should learn edge-detection features beneficial to both tasks.

2. **Does height information improve classification?** Height stratification may help disambiguate species with similar SAR signatures but different canopy structures.

3. **Does species information improve height estimation?** Species-specific allometric relationships could inform height predictions if learned implicitly.

---

## 9. Summary of Key Papers by Category

### Foundational/Survey Papers
1. Caruana (1997) - MTL foundations
2. Crawshaw (2020) - Deep MTL survey
3. Vandenhende et al. (2021) - Dense prediction MTL survey
4. Zhang & Yang (2021) - Comprehensive MTL survey
5. Jun et al. (2024) - MTL survey spanning traditional to foundation models

### Architecture Papers
6. Misra et al. (2016) - Cross-stitch networks
7. Liu et al. (2019) - MTAN
8. Ma et al. (2018) - MMoE
9. Shi et al. (2023) - Progressive parameter sharing
10. Xu et al. (2023) - MQTransformer
11. Ye & Xu (2022) - InvPT

### Optimization Papers
12. Kendall et al. (2018) - Uncertainty weighting
13. Chen et al. (2018) - GradNorm
14. Yu et al. (2020) - PCGrad
15. Liu et al. (2021) - CAGrad
16. Sener & Koltun (2018) - Multi-objective MTL
17. Xin et al. (2022) - Critical evaluation of MTL optimizers

### Remote Sensing Applications
18. Ahlswede et al. (2023) - TreeSatAI dataset
19. Chhapariya et al. (2024) - Multitask classification/regression for forest
20. Grabska-Szwagrzyk et al. (2023) - Tree species mapping with S1/S2
21. Becker et al. (2023) - Forest structure from optical/SAR
22. Tamiminia et al. (2024) - Height/biomass mapping

### Task Relationship Analysis
23. Standley et al. (2020) - Task grouping study
24. Zamir et al. (2018) - Taskonomy

---

## 10. Direction for Your Experimental Design

Based on this review, your experimental framework should:

1. **Baseline establishment**: Train single-task models for classification and regression separately to establish performance bounds

2. **Simple MTL**: Implement hard parameter sharing with equal weights, then uncertainty weighting

3. **Architecture variants**: Compare encoder-only sharing vs. partial decoder sharing

4. **Gradient analysis**: Monitor gradient conflicts during training to assess task compatibility

5. **Representation analysis**: Visualize encoder features to understand what is being learned for each task

6. **Ablation studies**: Systematically remove components to quantify contributions

The literature suggests that careful baseline tuning may matter more than sophisticated optimization methods, but gradient manipulation techniques like CAGrad provide fallback options if task conflicts are severe.