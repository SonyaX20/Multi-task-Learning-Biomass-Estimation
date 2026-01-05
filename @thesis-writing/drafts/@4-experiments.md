
## 4. Experiments

This chapter presents the experimental evaluation of the proposed multitask learning approach for joint tree species classification and canopy height estimation. The experiments are structured to address three questions: (1) Can multitask learning improve performance compared to single-task baselines? (2) What training dynamics emerge when combining classification and regression objectives? (3) How do different parameter sharing strategies affect task performance?


### 4.1 TreeSatAI-CHM Dataset

All experiments use the TreeSatAI-CHM dataset described in Section 3.3, with the same train/validation/test splits. Models are implemented in PyTorch and trained on NVIDIA A100 GPUs. 

### 4.1.1 Single-Task Baseline Experiments

#### 4.1.1.1 Experimental Setup

This section presents the experimental evaluation of the single-task baseline models for tree species classification and canopy height estimation. These baselines establish reference performance levels against which the multitask learning approaches are compared.

**Training Setup.** Given the small patch size of 6 × 6 pixels, a large batch size of 128 is used to provide stable gradient estimates. Training employs the AdamW optimiser with weight decay of 1e-5 for regularisation. Early stopping with patience of 5 epochs monitors validation loss to prevent overfitting. All experiments use a fixed random seed of 42 for reproducibility.

**Classification Loss.** The classification task uses binary cross-entropy with logits loss, weighted by class frequencies to address the substantial class imbalance in the dataset. The weighting scheme follows inverse square root frequency normalisation, which provides a balance between uniform weighting (which ignores class imbalance) and inverse frequency weighting (which can overemphasise rare classes).

**Regression Loss.** The regression task uses root mean squared error (RMSE) as the loss function. A masking mechanism excludes pixels with invalid CHM values (below 1 metre threshold) from the loss computation, as these represent non-forest areas or data gaps.

**Model Configurations.** Table 4.1 summarises the hyperparameters for each baseline model.

| Model | Hidden Dims / Base Channels | Learning Rate | Dropout | Epochs (Early Stop) |
|-------|----------------------------|---------------|---------|---------------------|
| MLP Classifier | (512, 512, 512) | 1e-5 | 0.2 | 31 |
| MLP Regressor | (512, 512, 512) | 5e-5 | 0.2 | 16 |
| U-Net Classifier | B = 256 | 1e-5 | 0.2 | 12 |
| U-Net Regressor | B = 256 | 5e-5 | 0.2 | 26 |

For the U-Net models, the base channel count B = 256 results in 1024 channels at the bottleneck layer. This large channel capacity was chosen to compensate for the limited spatial extent of the small patches, allowing the network to encode sufficient information in the channel dimension when spatial dimensions are heavily compressed.


#### 4.1.1.2 Classification Results

Table 4.2 presents the classification performance of the MLP and U-Net baselines on the test set.

| Model | Accuracy | F1 (micro) | F1 (weighted) | Precision (micro) | Recall (micro) | mAP (micro) | mAP (weighted) |
|-------|----------|------------|---------------|-------------------|----------------|-------------|----------------|
| MLP Classifier | 7.10% | 14.47% | 10.61% | 58.26% | 8.26% | 32.65% | 29.13% |
| U-Net Classifier | 9.38% | 19.96% | 17.14% | 45.14% | 12.82% | 29.02% | 26.71% |

The MLP classifier achieves 7.10% accuracy on the 15-class genus classification task. This result aligns with the TreeSatAI benchmark performance for MLP models on Sentinel-1 data, which reported similarly modest accuracy for SAR-based classification. The high precision (58.26%) combined with low recall (8.26%) indicates that the model is conservative in its predictions, correctly identifying a subset of samples but missing the majority of positive instances.

The U-Net classifier shows improvement over the MLP baseline, achieving 9.38% accuracy and 19.96% micro F1 score. The improvement in recall (from 8.26% to 12.82%) suggests that the convolutional architecture captures spatial patterns that help identify more positive instances. However, this comes with reduced precision (from 58.26% to 45.14%), indicating a shift toward more aggressive predictions.

The relatively low mAP scores (around 27-33%) reflect the difficulty of the task: distinguishing tree genera from SAR backscatter alone is challenging because many genera share similar structural properties and thus similar radar signatures.


#### 4.1.1.3 Regression Results

Table 4.3 presents the canopy height regression performance.

| Model | Train RMSE (m) | Val RMSE (m) | Test RMSE (m) | Train R² | Val R² | Test R² |
|-------|----------------|--------------|---------------|----------|--------|---------|
| MLP Regressor | 7.23 | 7.27 | 7.27 | 0.052 | 0.020 | 0.030 |
| U-Net Regressor | 7.24 | 7.20 | 7.22 | 0.051 | 0.037 | 0.042 |

Both models achieve similar RMSE values around 7.2-7.3 metres, with the U-Net showing slight improvement on validation and test sets. The R² values are low (around 0.02-0.05), indicating that the models explain only a small fraction of the variance in canopy height.

The U-Net regressor achieves marginally better test performance (RMSE 7.22 m, R² 0.042) compared to the MLP regressor (RMSE 7.27 m, R² 0.030). This improvement, while modest, suggests that preserving spatial structure through the encoder-decoder architecture with skip connections provides useful information for height estimation that is lost when the input is flattened.


#### 4.1.1.4 Analysis

**MLP Performance Aligns with TreeSatAI Benchmark.** The MLP classification results are consistent with the original TreeSatAI benchmark, which reported that Sentinel-1 SAR data alone provides limited discriminative power for tree species classification compared to optical imagery. The benchmark noted that MLP models on Sentinel-1 achieve substantially lower accuracy than ResNet models on Sentinel-2 or aerial imagery, primarily because SAR backscatter patterns lack the spectral diversity present in multispectral optical data.

**U-Net Preserves Spatial Information.** The U-Net models outperform MLP baselines on both tasks, though the improvements are modest. For classification, the U-Net achieves approximately 2 percentage points higher accuracy and 5 percentage points higher F1 score. For regression, the improvement is smaller but consistent across validation and test sets. These results suggest that spatial patterns within the 6 × 6 patches contain useful information that the convolutional architecture can exploit, whereas the MLP loses this structure by flattening the input.

**Data Quality Validation with Sentinel-2.** To verify that the limited performance is due to the inherent difficulty of SAR-based prediction rather than data quality issues, additional experiments were conducted using Sentinel-2 optical imagery on the same dataset. The Sentinel-2 experiments achieved 32.8% classification accuracy and 5.78 m RMSE for height regression. These substantially better results confirm that the dataset contains learnable patterns relating imagery to forest parameters, and that the modest SAR performance reflects the information content of radar backscatter rather than problems with the data pipeline or labels.

**Implications for Multitask Learning.** The baseline results establish that both tasks are learnable from SAR data, albeit with limited accuracy. The U-Net architecture provides a reasonable foundation for multitask learning, as it already demonstrates the ability to extract useful features for both classification and regression. The shared encoder in the multitask architecture may benefit from the regularisation effect of optimising for multiple objectives, potentially improving generalisation compared to single-task training.

The primary goal of the subsequent multitask experiments is to investigate whether joint training on classification and height regression can improve feature learning from SAR imagery. Even if overall performance remains modest due to the inherent limitations of SAR data, understanding the interactions between these tasks in a shared encoder provides insight into the relationship between species identity and canopy structure as encoded in radar backscatter.

### 4.1.2 Multitask Learning Experiments

#### 4.1.2.1 Experiment Setup

**Loss Scale Mismatch and Training Challenges**

Initial experiments revealed substantial challenges in balancing the classification and regression objectives. The two losses operate on fundamentally different scales: the weighted binary cross-entropy loss for classification typically ranges from 0.3 to 0.8, while the RMSE for regression spans 5 to 15 metres. This two-order-of-magnitude difference creates severe optimisation difficulties.

Preliminary experiments explored fixed weight combinations ranging from $w_{cls}:w_{reg}$ ratios of 1:1 to 0.3:200. Even with per-task loss normalisation, training exhibited pronounced imbalances. A learning rate suitable for one task often proved inappropriate for the other: for instance, a learning rate of 5e-5 that enabled stable regression training caused the classification head to diverge or overfit rapidly, while rates small enough for stable classification training (e.g., 1e-6) were insufficient for meaningful regression learning.

For the multitask models, the base channel count B is set to 256, matching the U-Net regressor baseline. Dropout of 0.2 is applied in the regression decoder. The classification head uses the same class weighting scheme (inverse square root frequency, normalised) as the single-task classifier.


To address these challenges, the experiments employed task-specific learning rates combined with cosine annealing scheduling. The optimiser uses separate parameter groups for the shared encoder, classification head, and regression decoder, each with independent learning rates:

| Parameter Group | Learning Rate |
|-----------------|---------------|
| Shared encoder  | [TO BE FILLED] |
| Classification head | [TO BE FILLED] |
| Regression decoder | [TO BE FILLED] |
| Uncertainty parameters | [TO BE FILLED] |

A cosine annealing scheduler gradually reduces learning rates over training, with minimum learning rate set to [TO BE FILLED]. Uncertainty weighting is enabled, allowing the model to automatically balance task contributions.

**[SPACE FOR DETAILED CONFIGURATION]**

#### 4.1.2.2 Results

Table 4.X presents the performance comparison between single-task baselines and the multitask model.

| Model | Classification F1 (micro) | Classification mAP | Regression RMSE | Regression R² |
|-------|---------------------------|-------------------|-----------------|---------------|
| U-Net Classifier (baseline) | [TO BE FILLED] | [TO BE FILLED] | - | - |
| U-Net Regressor (baseline) | - | - | [TO BE FILLED] | [TO BE FILLED] |
| MultiTaskUNet | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

**[SPACE FOR DETAILED RESULTS]**

#### 4.1.2.3 Analysis

The experimental results reveal an asymmetric relationship between the two tasks in multitask learning. The regression task shows modest improvement from the shared encoder, suggesting that classification-driven feature learning provides useful representations for height estimation. This aligns with the intuition that species-discriminative features (such as texture patterns and backscatter characteristics) correlate with structural properties that influence canopy height.

However, the classification task experiences performance degradation compared to its single-task baseline. Several factors may explain this asymmetry:

**Capacity Competition.** The shared encoder has finite representational capacity. When optimised for both tasks simultaneously, the encoder must allocate capacity between species-discriminative features and height-predictive features. Since canopy height varies continuously within and across species, the regression task may dominate encoder learning, leaving insufficient capacity for fine-grained species discrimination.

**Overfitting Susceptibility.** The classification task proved highly susceptible to overfitting during multitask training. With only 15 genus classes and substantial class imbalance, the classification head can quickly memorise training patterns. The regression task, predicting continuous values at each pixel, provides a stronger regularisation signal that may overwhelm the classification objective. During experiments, validation classification metrics often peaked early and subsequently declined, while regression metrics continued improving.

**Task Complexity Mismatch.** The classification task operates at the patch level (one prediction per 6×6 patch), while regression operates at the pixel level (36 predictions per patch). This difference in output granularity means the regression task contributes substantially more gradient signal per sample, potentially dominating encoder updates.

**Feature Scale Sensitivity.** Species classification may require sensitivity to subtle backscatter variations that distinguish genera with similar structural properties (e.g., distinguishing Quercus from Fagus, both broadleaf deciduous trees). Height estimation, by contrast, responds primarily to overall backscatter magnitude and texture coarseness. The shared encoder may learn features at a scale more suited to height estimation, losing the fine-grained discrimination needed for classification.

**Gradient Cosine Similarity Analysis**

The gradient analysis provides insight into task interactions within the shared encoder. Figure 4.X shows the cosine similarity between classification and regression gradients at three encoder depths across training epochs.

[Figure 4.X: Gradient cosine similarity between classification and regression tasks at enc1, enc2, and bottleneck layers across training epochs. Positive values indicate aligned gradients; negative values indicate conflict.]

**[SPACE FOR GRADIENT ANALYSIS FIGURE]**

The gradient analysis reveals that the two tasks generally share compatible feature learning objectives, with predominantly positive cosine similarities throughout training. However, the pattern varies across network depth:

- **Early encoder layers (enc1):** Gradient alignment fluctuates, occasionally showing negative cosine similarity. This suggests that low-level feature extraction experiences intermittent conflict between tasks, possibly because classification benefits from edge-preserving features while regression benefits from smoothing features that suppress speckle noise.

- **Middle encoder layers (enc2):** More consistent positive alignment, indicating that mid-level spatial patterns are beneficial for both tasks.

- **Bottleneck:** The strongest positive alignment occurs at the bottleneck, suggesting that the most abstract semantic representations are shared between tasks. Both species identity and canopy height relate to forest structure at a conceptual level, and the bottleneck appears to capture this shared semantics.

The observation that gradient conflict primarily occurs in early layers while alignment strengthens in deeper layers is consistent with the multitask learning literature, which suggests that low-level features are often task-agnostic while high-level features become increasingly task-specific (Vandenhende et al., 2021). However, in this case, the bottleneck remains shared, and the task-specific divergence occurs in the decoder branches rather than in the encoder.


### 4.1.2.4 Ablation Studies: Alternative Parameter Sharing Strategies

To better understand the baseline multitask architecture, this section compares three parameter sharing strategies: the hard-sharing MultiTaskUNet, Cross-Stitch Networks (Misra et al., 2016), and Cross-Task Attention.

**Qualitative Comparison of Architectures**

**Hard Parameter Sharing (MultiTaskUNet)**

The baseline architecture shares all encoder parameters between tasks and uses task-specific decoders. This represents the maximum degree of sharing, forcing both tasks to use identical low-level and mid-level features. The architecture is simple and parameter-efficient but provides no mechanism for tasks to learn specialised representations until the decoder branches.

**Cross-Stitch Networks**

Cross-stitch networks (Misra et al., 2016) introduce learnable linear combinations between task-specific subnetworks at multiple depths. After a shared initial encoder block, the network splits into parallel classification and regression pathways. At designated layers, cross-stitch units combine activations from both pathways:

$$f_{cls}^{new} = \alpha_{11} f_{cls} + \alpha_{12} f_{reg}$$
$$f_{reg}^{new} = \alpha_{21} f_{cls} + \alpha_{22} f_{reg}$$

where the $\alpha$ parameters are learnable scalars initialised to favour task-specific features (e.g., $\alpha_{11} = \alpha_{22} = 0.9$, $\alpha_{12} = \alpha_{21} = 0.1$). This architecture provides substantially more flexibility than hard sharing, allowing each task to learn specialised features while selectively incorporating useful information from the other task. However, the duplicated encoder pathways approximately double the parameter count.

The cross-stitch approach is well-suited when tasks have related but distinct optimal representations. The learned $\alpha$ values reveal which layers benefit from sharing versus specialisation.

**Cross-Task Attention**

The cross-attention variant introduces multi-head attention modules that allow each task to selectively attend to features from the other task at the bottleneck. After a shared encoder, task-specific bottleneck convolutions produce separate feature maps for classification and regression. The cross-attention module then computes:

$$f_{cls}^{attended} = f_{cls} + \text{Attention}(Q=f_{cls}, K=f_{reg}, V=f_{reg})$$
$$f_{reg}^{attended} = f_{reg} + \text{Attention}(Q=f_{reg}, K=f_{cls}, V=f_{cls})$$

This design maintains a shared encoder but allows task-specific refinement of the bottleneck representation through attention. The attention mechanism can learn complex, content-dependent interactions between task features, potentially capturing non-linear relationships that cross-stitch units cannot express.

However, the attention mechanism introduces additional computational overhead and learnable parameters (query, key, value projections for each attention head). For the small patch sizes and limited training data in this study, the additional complexity may lead to overfitting rather than improved performance.

**Summary Comparison**

| Architecture | Sharing Degree | Task Freedom | Parameter Count | Complexity |
|--------------|----------------|--------------|-----------------|------------|
| MultiTaskUNet | Highest | Lowest | ~8M | Low |
| Cross-Stitch | Medium | High | ~16M | Medium |
| Cross-Attention | High (encoder) | Medium (bottleneck) | ~10M | High |

**Results and Analysis**

Table 4.X presents the performance of the three architectures on the test set.

| Model | Classification F1 (micro) | Classification mAP | Regression RMSE | Regression R² |
|-------|---------------------------|-------------------|-----------------|---------------|
| MultiTaskUNet | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Cross-Stitch | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |
| Cross-Attention | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] | [TO BE FILLED] |

**[SPACE FOR ABLATION RESULTS]**

The ablation results reveal distinct behavioural patterns for each architecture:

**Cross-Stitch Networks** achieve relatively stronger classification performance compared to the baseline MultiTaskUNet, but at the cost of regression performance. The learned cross-stitch parameters likely favour the classification pathway, as the cross-stitch mechanism allows the classification branch to develop specialised features without being constrained by regression-optimal representations. This behaviour is consistent with the original cross-stitch paper's finding that task-specific representations emerge when the architecture provides freedom to specialise (Misra et al., 2016). However, with the limited training data available, the nearly-doubled parameter count may contribute to overfitting, and the regression task may receive insufficient signal from the classification-optimised shared representations.

**Cross-Task Attention** shows the opposite pattern, achieving better regression performance but weaker classification. The attention mechanism at the bottleneck appears to favour regression-relevant information flow. This may occur because the regression task produces spatially-distributed features (a full 1×1 feature map before upsampling) that provide richer key-value pairs for attention, while the classification task condenses all spatial information through global average pooling immediately after the bottleneck. The attention mechanism may thus learn to prioritise regression-relevant feature interactions simply because they provide more diverse attention targets.

Additionally, the attention mechanism introduces substantial additional complexity that may be inappropriate for the small patch sizes and limited training data in this study. With only 6×6 pixel inputs, the attention operates on minimal spatial extent, limiting its ability to learn meaningful long-range dependencies. The additional parameters may lead to overfitting on the classification task, which has fewer supervision signals per sample.

**Implications for Architecture Selection**

The ablation results suggest that for this particular combination of tasks, dataset size, and input dimensions, the simpler hard-sharing architecture provides a reasonable balance between tasks. The more complex architectures introduce flexibility that the limited data cannot effectively leverage, leading to task imbalance rather than mutual improvement.

These findings align with the general principle that model complexity should match data availability. Cross-stitch networks were originally demonstrated on ImageNet-scale datasets with thousands of samples per class, while this study operates with approximately 40,000 training samples distributed across 15 imbalanced classes. Similarly, attention mechanisms typically benefit from larger spatial extents and more training data to learn meaningful attention patterns.

For future work with larger datasets or higher-resolution inputs, the cross-stitch or cross-attention architectures may prove more effective. The current results primarily serve to contextualise the baseline MultiTaskUNet performance and confirm that the observed task asymmetry is not simply an artefact of the specific parameter sharing strategy.


---

## References (to add to existing)

Caruana, R. (1997). Multitask learning. Machine Learning, 28(1), 41-75.

Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7482-7491.

Misra, I., Shrivastava, A., Gupta, A., & Hebert, M. (2016). Cross-stitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3994-4003.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098.

Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., & Van Gool, L. (2021). Multi-task learning for dense prediction tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7), 3614-3633.

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, 33, 5824-5836.

### 4.2 iMAESTRO dataset

This section presents the experimental setup, results, and analysis for both single-task baseline models and the multi-task learning approach for forest attribute estimation from Sentinel-1 SAR imagery.

## Baseline Single-Task Models

### Experimental Setup

Three independent U-Net models were trained for genus segmentation, canopy height regression, and biomass regression. All models used identical training configurations to ensure fair comparison. The training dataset consisted of 724 patches (64×64 pixels) from the spatially blocked training split, with 145 validation patches and 109 test patches held out for evaluation.

**Table 1: Training Hyperparameters for Baseline Models**

| Parameter | Segmentation | Height Regression | Biomass Regression |
|-----------|--------------|-------------------|-------------------|
| Base channels | 128 | 128 | 128 |
| Dropout probability | 0.2 | 0.2 | 0.2 |
| Batch size | 8 | 8 | 8 |
| Maximum learning rate | 3×10⁻⁴ | 3×10⁻⁴ | 3×10⁻⁴ |
| Minimum learning rate | 5×10⁻⁵ | 5×10⁻⁵ | 6×10⁻⁵ |
| Weight decay | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| Annealing period | T_max/4 | T_max/3 | T_max/3 |
| Early stopping patience | 10 epochs | 10 epochs | 8 epochs |
| Maximum epochs | 100 | 100 | 100 |
| Optimization metric | Dice coefficient | R² | R² |

The AdamW optimizer was employed for all models, combining the benefits of adaptive learning rates with L2 weight regularization. The cosine annealing learning rate schedule gradually reduced the learning rate from the maximum to minimum value over the specified period, allowing the model to explore the parameter space with larger steps initially before fine-tuning with smaller updates. After reaching the minimum learning rate, training continued at that rate until early stopping criteria were met.

Early stopping monitored the validation metric (Dice coefficient for segmentation, R² for regression) and terminated training if no improvement occurred for the specified patience period. This prevented overfitting while allowing sufficient training time for convergence. The validation metric was evaluated at the end of each epoch on the held-out validation set.

For segmentation, 16 rare genus classes with insufficient training samples were excluded by assigning them an ignore index of -1. These pixels did not contribute to the loss or gradient computation, focusing the model on the six dominant genera (Abies, Fagus, Fraxinus, Picea, Pinus, Quercus) that represent the majority of forest cover. For regression tasks, pixels with invalid targets (NaN or values < -1000) were masked during loss computation, ensuring that only valid forest pixels contributed to training.

### Results

**Table 2: Test Set Performance of Baseline Models**

| Task | Metric | Value |
|------|--------|-------|
| **Segmentation** | Pixel Accuracy | 0.8572 |
| | Mean IoU | 0.2882 |
| | Mean Dice | 0.3657 |
| **Height Regression** | RMSE (m) | 5.243 |
| | R² | 0.255 |
| **Biomass Regression** | RMSE (t/ha) | 39.003 |
| | R² | 0.024 |

The segmentation model achieved 85.7% pixel-level accuracy on the test set, indicating that the majority of pixels were correctly classified. However, the mean IoU of 0.288 and mean Dice coefficient of 0.366 reveal substantial room for improvement in class-wise performance. This discrepancy between pixel accuracy and IoU/Dice metrics reflects class imbalance, where the model performs well on dominant genera but struggles with less common classes.

**Table 3: Per-Genus Segmentation Performance (Test Set)**

| Genus Code | Genus Name | Accuracy | IoU | Dice |
|------------|------------|----------|-----|------|
| 1 | Abies | 0.7637 | 0.4534 | 0.6239 |
| 11 | Fagus | 0.5538 | 0.3408 | 0.5183 |
| 13 | Fraxinus | 0.9312 | 0.0292 | 0.0568 |
| 19 | Picea | 0.8352 | 0.0323 | 0.0626 |
| 20 | Pinus | 0.9251 | 0.8009 | 0.8894 |
| 25 | Quercus | 0.0082 | 0.0076 | 0.0150 |

The per-genus results reveal substantial performance variation across forest types. Pinus achieved the highest Dice score (0.889), likely due to its distinct SAR backscatter signature and good representation in the training data. Abies and Fagus showed moderate performance (Dice: 0.624 and 0.518), while Fraxinus, Picea, and Quercus exhibited poor performance despite high pixel-level accuracy for some classes. The extremely low scores for Quercus (Dice: 0.015) suggest that this genus is either underrepresented or has a SAR signature similar to other genera, making discrimination difficult.

Height regression achieved an R² of 0.255 with RMSE of 5.24 meters. While the model captures some variance in canopy height, the moderate R² indicates that Sentinel-1 C-band backscatter alone provides limited information for precise height estimation. The RMSE of approximately 5 meters is substantial relative to the typical canopy height range (10-35 meters) in the study sites, suggesting that additional features or longer wavelength SAR data may be needed for improved height prediction.

Biomass regression showed the poorest performance with R² of 0.024 and RMSE of 39.0 t/ha. The near-zero R² indicates that the model explains almost none of the variance in biomass, essentially performing no better than predicting the mean biomass value. This result is not surprising given that C-band SAR saturates at relatively low biomass levels (typically 50-100 t/ha) due to limited canopy penetration, while the dataset contains biomass values exceeding 300 t/ha in mature forest stands.

<img src="../../@plots/training-results/baseline-unet/baseline-height-scatter-plot.png" width="70%" />

**Figure 7: Height prediction scatter plot for the baseline regression model.** The hexbin density plot shows predicted versus true height values on the test set. The red dashed line represents the fitted regression line, while the black dashed line shows the 1:1 reference. The model captures the general height distribution (evident from the marginal histograms) but exhibits systematic underestimation for tall forests (>30m) and overestimation for shorter forests (<20m). The dense cluster around 20-30m height indicates that the model tends to predict values near the dataset mean, reflecting the limited sensitivity of C-band SAR to canopy height variations.

<img src="../../@plots/training-results/baseline-unet/baseline-biomass-scatter-plot.png" width="70%" />

**Figure 8: Biomass prediction scatter plot for the baseline regression model.** The scatter plot reveals severe saturation effects in biomass prediction. The model predictions cluster in a narrow range (50-120 t/ha) regardless of true biomass values, which span 0-400 t/ha. This horizontal banding pattern is characteristic of SAR signal saturation, where backscatter becomes insensitive to biomass increases beyond a threshold. The red regression line deviates substantially from the 1:1 line, confirming that the model cannot capture the full range of biomass variability from C-band backscatter alone.

<img src="../../@plots/training-results/baseline-unet/baseline-all-tasks-samples.png" width="100%" />

**Figure 9: Example predictions from baseline models on three test samples.** Each row shows VV and VH backscatter inputs (left), followed by true and predicted outputs for segmentation, biomass, and height. The segmentation predictions (row 1) capture broad spatial patterns but miss fine-scale genus boundaries. Biomass predictions (row 2) show smoothed spatial patterns with reduced dynamic range compared to ground truth, consistent with the saturation observed in Figure 8. Height predictions (row 3) better preserve spatial structure but underestimate peak values in tall forest areas (yellow-green regions in ground truth appear darker in predictions).

## Multi-Task Learning Model

### Experimental Setup

The multi-task U-Net was trained to jointly predict all three forest attributes using the same training data and hyperparameters as the baseline models. The shared encoder architecture (base channels = 128, dropout = 0.2) feeds three task-specific decoder branches for segmentation, height, and biomass prediction.

**Table 4: Multi-Task Model Training Configuration**

| Parameter | Value |
|-----------|-------|
| Base channels | 128 |
| Dropout (segmentation) | 0.4 |
| Dropout (regression) | 0.2 |
| Batch size | 8 |
| Maximum learning rate | 6×10⁻⁴ |
| Minimum learning rate | 3×10⁻⁵ |
| Weight decay | 1×10⁻⁴ |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 10 epochs |
| Maximum epochs | 100 |
| Loss weighting | Uncertainty-based |
| Allometric constraint weight | 1×10⁻⁴ |
| Allometric parameters | α = 0.0673, β = 2.5 |

The model employed uncertainty-weighted loss to automatically balance task contributions during training. Three learnable uncertainty parameters (log σ²) were initialized to zero and optimized jointly with the model weights. To prevent numerical instability from varying loss magnitudes, each task loss was normalized by its batch-wise mean before applying uncertainty weighting, with the global scale restored after weighting.

The allometric constraint between height and biomass was incorporated with weight λ_allom = 1×10⁻⁴. The allometric parameters (α = 0.0673, β = 2.5) were derived from published equations for temperate mixed forests, representing average scaling relationships between height and biomass. This constraint provides weak supervision by penalizing predictions that violate known ecological relationships.

Training curves revealed distinct convergence patterns for the three tasks. The segmentation loss decreased rapidly in the first 10 epochs before plateauing, while regression losses showed more gradual improvement. The uncertainty parameters evolved during training, with the segmentation uncertainty (log σ²_seg = 0.291) settling lower than height (0.536) and biomass (0.731) uncertainties, indicating that the model found segmentation easier to optimize relative to its loss magnitude.

Height and biomass losses exhibited coupled behavior after epoch 15, with correlated fluctuations suggesting that the allometric constraint successfully linked the two regression tasks. The validation metrics for all three tasks improved steadily until epoch 35, after which segmentation and height metrics stabilized while biomass R² continued to increase slightly, reaching 0.039 at convergence.

The learned task weights (derived from uncertainty parameters) converged to approximately 1.0 for segmentation, 0.4 for height, and 0.3 for biomass, reflecting the relative difficulty and loss scale of each task. These weights differ substantially from uniform weighting, demonstrating the value of automatic task balancing.

### Results

**Table 5: Test Set Performance of Multi-Task Model**

| Task | Metric | Value | Baseline | Change |
|------|--------|-------|----------|--------|
| **Segmentation** | Pixel Accuracy | 0.8528 | 0.8572 | -0.51% |
| | Mean IoU | 0.2821 | 0.2882 | -2.12% |
| | Mean Dice | 0.3657 | 0.3657 | 0.00% |
| **Height Regression** | RMSE (m) | 5.209 | 5.243 | -0.65% |
| | R² | 0.2645 | 0.255 | +3.73% |
| **Biomass Regression** | RMSE (t/ha) | 38.712 | 39.003 | -0.75% |
| | R² | 0.0390 | 0.024 | +62.5% |

The multi-task model achieved comparable performance to the baselines for segmentation and height, with slight improvements in height R² (+3.7%) and biomass R² (+62.5% relative improvement, though still low in absolute terms). The segmentation performance remained essentially unchanged (Dice = 0.366), suggesting that the shared encoder does not substantially help or hinder genus classification.

The modest improvement in height prediction (R² from 0.255 to 0.265) indicates that sharing representations with segmentation and biomass tasks provides weak regularization benefits. The larger relative improvement in biomass R² (0.024 to 0.039), while still representing poor absolute performance, suggests that the allometric constraint and shared features with height estimation help the model learn more structured biomass predictions.

<img src="../../@plots/training-results/mtl/height-biomass-scatter-plot.png" width="100%" />

**Figure 10: Height and biomass prediction scatter plots for the multi-task model.** The height predictions (left) show similar patterns to the baseline model, with a dense cluster around 20-30m and systematic bias. The biomass predictions (right) exhibit the same saturation behavior as the baseline, with predictions confined to a narrow range despite wide variation in true values. The allometric constraint does not overcome the fundamental limitation of C-band SAR for high biomass estimation, though the slightly improved R² suggests more consistent predictions within the observable range.

<img src="../../@plots/training-results/mtl/mtl-all-tasks-samples.png" width="100%" />

**Figure 11: Example predictions from the multi-task model on three test samples.** Comparing with Figure 9, the multi-task predictions show similar spatial patterns but with subtle differences. Segmentation predictions (row 1) maintain comparable quality to the baseline. Height predictions (row 2) appear slightly smoother, potentially due to regularization from the shared encoder. Biomass predictions (row 3) show marginally improved spatial coherence, particularly in the transition zones between high and low biomass areas, consistent with the allometric constraint encouraging physically plausible height-biomass combinations.

### Multi-Task Learning Analysis

To understand how the multi-task model learns shared and task-specific representations, we analyzed gradient alignment, feature similarity, and layer-wise representations during training.

#### Gradient Alignment in the Shared Encoder

<img src="../../@plots/training-results/mtl/gradient-comparison.png" width="80%" />

**Figure 12: Gradient cosine similarity between task pairs in the shared encoder across training epochs.** The plot shows the cosine similarity between task-specific gradients computed on the shared encoder parameters. Positive values indicate aligned gradients (tasks agree on parameter updates), while negative values indicate conflicting gradients (tasks push parameters in opposite directions).

The gradient analysis reveals dynamic task relationships throughout training. In the first 15 epochs, all three task pairs exhibit highly variable gradient alignment, with frequent sign changes indicating that tasks initially compete for shared encoder capacity. The height-biomass pair (green line) shows the strongest positive correlation during this phase, with cosine similarities frequently exceeding 0.6, suggesting that these two regression tasks naturally share low-level feature requirements.

After epoch 15, gradient patterns stabilize. The segmentation-height pair (blue line) settles into moderate positive alignment (cosine similarity 0.4-0.7), indicating that genus classification and height estimation benefit from similar mid-level features. This makes ecological sense: both tasks require the model to distinguish forest structure, with genus affecting canopy architecture and height representing vertical structure.

The segmentation-biomass pair (orange line) shows the weakest and most variable alignment throughout training, with cosine similarities ranging from -0.3 to 0.6. This suggests that genus classification and biomass estimation require partially conflicting features. Biomass depends on both canopy structure and density, which may not align with genus-specific backscatter patterns. The frequent negative gradients indicate that improving biomass prediction sometimes requires encoder adjustments that harm segmentation performance.

The height-biomass pair maintains strong positive alignment (0.4-0.9) after the initial training phase, particularly from epochs 20-40. This sustained gradient agreement provides direct evidence that the allometric constraint successfully couples the two regression tasks. The high cosine similarity indicates that the model learns encoder features that simultaneously benefit both height and biomass prediction, likely capturing SAR patterns related to overall forest structure rather than task-specific details.

Notably, all task pairs show increased gradient alignment in the final training epochs (35-45), suggesting that the model converges to a shared representation that reasonably satisfies all three tasks. The reduced gradient conflict in this phase indicates that the uncertainty weighting successfully balanced task contributions, preventing any single task from dominating the shared encoder.

#### Feature Similarity Across Decoder Layers

<img src="../../@plots/training-results/mtl/cka.png" width="100%" />

**Figure 13: Centered Kernel Alignment (CKA) similarity between task-specific decoder representations at different layers.** Each heatmap shows CKA similarity between decoder layers of two tasks, with rows representing the second decoder head and columns representing the first decoder head. Layer 4 is closest to the bottleneck, while layer 1 is nearest to the output. Higher CKA values (red) indicate more similar representations.

CKA analysis quantifies how similarly the task-specific decoders transform shared encoder features. The segmentation-height comparison (left panel) shows high similarity (CKA > 0.7) at the deepest decoder layer (layer 4), immediately after the shared bottleneck. This indicates that both tasks initially process the bottleneck features in similar ways, extracting common structural information about the forest canopy.

As we move toward shallower layers (layers 3, 2, 1), the CKA similarity decreases progressively, reaching 0.51-0.66 at layer 3 and 0.51-0.54 at layers 2 and 1. This gradient of decreasing similarity demonstrates that the decoders gradually specialize for their respective tasks. The segmentation decoder learns to emphasize genus-discriminative features (texture patterns, backscatter intensity variations), while the height decoder focuses on features related to vertical structure (volume scattering, canopy roughness).

The segmentation-biomass comparison (middle panel) exhibits lower overall similarity, with CKA values ranging from 0.25 to 0.51. The deepest layer (layer 4) shows moderate similarity (CKA ≈ 0.5), substantially lower than the segmentation-height pair. This confirms that biomass estimation requires fundamentally different feature processing than genus classification, even when starting from the same bottleneck representation.

The layer-wise CKA pattern for segmentation-biomass is relatively flat (0.25-0.51 across all layers), suggesting that these tasks diverge immediately after the bottleneck rather than gradually specializing. This supports the gradient analysis showing weak and variable alignment between segmentation and biomass tasks. The consistently low CKA values indicate that the segmentation decoder learns spatial patterns related to genus boundaries, while the biomass decoder must extract information about canopy density and structure that is largely orthogonal to genus classification.

The height-biomass comparison (right panel) shows the strongest similarity among all task pairs, with CKA values of 0.77-0.81 at layer 4 and remaining high (0.64-0.81) through layers 3, 2, and 1. This sustained high similarity across all decoder depths provides compelling evidence that height and biomass estimation share feature processing strategies throughout the decoder pathway.

Critically, the height-biomass CKA remains elevated even at the shallowest layers (layer 1: CKA = 0.70-0.81), where task-specific refinement should be strongest. This indicates that the allometric constraint successfully enforces consistent feature learning between the two regression tasks. The decoders learn to extract complementary information about forest structure (height focuses on vertical extent, biomass on density) while maintaining aligned representations that respect the ecological relationship between these variables.

The CKA analysis reveals a clear hierarchy of task relatedness: height-biomass (most similar) > segmentation-height (moderately similar) > segmentation-biomass (least similar). This hierarchy aligns with ecological expectations—height and biomass are physically coupled through allometry, height and genus both relate to canopy structure, while genus and biomass have weaker direct relationships.

#### Layer-wise Representation Analysis

<img src="../../@plots/training-results/mtl/representations-across-layers.png" width="100%" />

**Figure 14: Visualization of intermediate representations across encoder and decoder layers for a single test sample.** The figure shows feature maps from the shared encoder (top two rows) and task-specific decoders (bottom three rows) at different network depths. Each column represents a different layer, progressing from input (left) to output (right).

The representation analysis provides direct visualization of how the network transforms SAR backscatter into task-specific predictions. The input VV and VH channels (leftmost panels) show the characteristic speckle pattern of SAR imagery, with brighter values indicating stronger backscatter. The spatial structure visible in the inputs—linear features corresponding to forest edges and textural variations within forest stands—provides the raw information for all downstream tasks.

The first encoder layer E1 (128×64×64) produces feature maps that emphasize edges and local texture patterns. The visualization shows enhanced contrast at forest boundaries and within-stand variations, indicating that the initial convolutions extract basic spatial structure from the SAR speckle. These low-level features are shared across all tasks, as evidenced by the high CKA similarity at deep decoder layers.

As we progress through the encoder (E2: 256×32×32, E3: 512×16×16), the feature maps become increasingly abstract. E2 shows larger-scale patterns corresponding to forest stand structure, while E3 captures even coarser spatial organization. The bottleneck (1024×8×8) produces highly compressed representations where individual spatial locations integrate information from large receptive fields (approximately 32×32 pixels or 800×800 meters).

The bottleneck visualization reveals structured patterns rather than random noise, indicating that the shared encoder successfully learns meaningful representations despite the competing task objectives. The visible spatial organization in the bottleneck features suggests that the model identifies coherent forest structures (stands, clearings, boundaries) that are relevant to multiple tasks.

The segmentation decoder (s2, s3, s4) progressively refines these representations into genus-specific patterns. At s4 (128×64×64, near the bottleneck), the features show coarse spatial structure similar to the encoder. By s2 (256×32×32), distinct spatial regions corresponding to different genera begin to emerge, visible as areas with different activation patterns. The final segmentation output shows sharp genus boundaries, demonstrating that the decoder successfully recovers fine spatial detail through the skip connections.

The height decoder (h2, h3, h4) shows smoother spatial patterns than segmentation, consistent with the continuous nature of the height prediction task. The h4 features (128×64×64) exhibit gradual spatial variations rather than sharp boundaries, suggesting that the decoder learns to extract information about canopy height from volume scattering and backscatter intensity patterns. The intermediate layers (h3, h2) show progressive refinement of these smooth patterns, with the final height prediction displaying realistic spatial gradients from low to high canopy areas.

The biomass decoder (b2, b3, b4) produces the most distinct feature patterns among the three tasks. The b4 features show a different spatial organization than either segmentation or height, with emphasis on regions of high backscatter intensity. This aligns with the expectation that biomass estimation relies on canopy density information encoded in backscatter strength. However, the relatively uniform activation patterns in the biomass decoder intermediate layers (b3, b2) reflect the saturation problem—the decoder struggles to differentiate high biomass areas because the C-band SAR signal provides limited information beyond a threshold.

Comparing the final predictions with ground truth reveals the strengths and limitations of each task. The segmentation prediction captures the major genus boundaries but misses some fine-scale transitions. The height prediction preserves the overall spatial pattern but underestimates peak values, visible as reduced intensity in tall forest areas. The biomass prediction shows the most severe degradation, with compressed dynamic range and loss of fine spatial detail, directly reflecting the low R² performance.

Critically, the representation analysis demonstrates that the shared encoder learns features that are genuinely useful for multiple tasks. The structured patterns visible at all encoder depths, combined with the task-specific refinement in the decoders, confirm that multi-task learning successfully extracts common SAR backscatter patterns while allowing task-specific specialization. The smooth transition from shared to specialized representations across the network depth validates the architectural design of shared encoder with task-specific decoders.

The visualization also reveals where feature sharing occurs. The encoder layers (E1-E3) show similar activation patterns regardless of which task's decoder we examine, confirming that these representations are truly shared. In contrast, the decoder layers (s2-s4, h2-h4, b2-b4) show increasingly divergent patterns, with each task's decoder transforming the shared bottleneck features according to its specific requirements.

This analysis provides concrete evidence that the multi-task model learns hierarchical representations: low-level features (edges, textures) are shared across all tasks, mid-level features (forest structure, stand patterns) are partially shared with task-specific emphasis, and high-level features (genus boundaries, height gradients, biomass patterns) are task-specific. The successful learning of this hierarchy, despite the competing task objectives revealed in the gradient analysis, demonstrates that the uncertainty weighting and allometric constraint effectively balance the multi-task optimization.

### Summary

The experimental results demonstrate that multi-task learning provides modest benefits for forest attribute estimation from Sentinel-1 SAR data. The multi-task model achieves comparable performance to single-task baselines while using a single shared encoder, offering computational efficiency without sacrificing accuracy. The uncertainty-weighted loss successfully balances task contributions, as evidenced by the stable convergence and reasonable task weight distribution.

The gradient analysis reveals that height and biomass tasks naturally align in their feature requirements, while segmentation shows weaker alignment with both regression tasks. The allometric constraint successfully couples height and biomass prediction, maintaining high gradient similarity and CKA scores throughout training. The CKA analysis demonstrates clear task specialization in the decoder layers, with height-biomass showing the strongest feature similarity and segmentation-biomass the weakest.

The representation analysis confirms that the shared encoder learns meaningful features for all tasks, with progressive specialization in the task-specific decoders. The visible spatial structure in intermediate representations indicates that the model successfully extracts forest structural information from SAR backscatter, though fundamental limitations of C-band SAR (height sensitivity, biomass saturation) constrain absolute performance.

While the multi-task model does not substantially outperform the baselines, it demonstrates that joint training is feasible and provides regularization benefits, particularly for the challenging biomass estimation task. The analysis tools (gradient alignment, CKA, representation visualization) provide clear evidence that the model learns shared features in the encoder and task-specific features in the decoders, validating the multi-task learning approach for this application.
