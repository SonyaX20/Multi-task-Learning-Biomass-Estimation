## Experiment Objectives
Channel Attention in UNet. 

## Baseline U-Net Regression

### Table 1. UNet Regressopm Results Summary

| Exp  | RMSE   | R²     |
|------|--------|--------|
| exp0-unet-reg | 8.0268 | 0.0750 |
| exp1-unet-reg | 8.0405 | 0.0740 |
| exp2-unet-reg | 7.9597 | 0.0857 |
| exp3-unet-reg | 7.9528 | 0.0925 |



### Table 2. UNet Regression Configs and Results

| Exp  | Configs | RMSE   | R²     |
|------|---------|--------|--------|
| exp0-unet-reg | ch=128, lr=1e-5→1e-4, pat=10 | 8.0268 | 0.0750 |
| exp1-unet-reg | ch=256, lr=1e-5→1e-4, pat=10 | 8.0405 | 0.0740 |
| exp2-unet-reg |  ch=256, lr=1e-5→1e-4, pat=10, half T_max in scheduler | 7.9597 | 0.0857 |
| exp3-unet-reg | ch=128, lr=1e-5→1e-4, full T_max  | 7.9528 | 0.0925 |

## Baseline U-Net Classification

**exp0-unet-cls:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Abies | 0.04 | 0.01 | 0.01 | 120 |
| Acer | 0.14 | 0.01 | 0.02 | 319 |
| Alnus | 0.18 | 0.01 | 0.01 | 348 |
| Betula | 0.00 | 0.00 | 0.00 | 322 |
| Cleared | 0.47 | 0.30 | 0.36 | 567 |
| Fagus | 0.34 | 0.06 | 0.10 | 1110 |
| Fraxinus | 0.14 | 0.01 | 0.02 | 302 |
| Larix | 0.00 | 0.00 | 0.00 | 466 |
| Picea | 0.40 | 0.19 | 0.25 | 1100 |
| Pinus | 0.49 | 0.37 | 0.42 | 1167 |
| Populus | 0.00 | 0.00 | 0.00 | 55 |
| Prunus | 0.00 | 0.00 | 0.00 | 38 |
| Pseudotsuga | 0.16 | 0.04 | 0.07 | 462 |
| Quercus | 0.32 | 0.07 | 0.12 | 1194 |
| Tilia | 0.00 | 0.00 | 0.00 | 25 |
| **micro avg** | 0.41 | 0.13 | 0.20 | 7595 |
| **macro avg** | 0.18 | 0.07 | 0.09 | 7595 |
| **weighted avg** | 0.30 | 0.13 | 0.17 | 7595 |
| **samples avg** | 0.18 | 0.14 | 0.15 | 7595 |

**exp1-unet-cls:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Abies | 0.40 | 0.02 | 0.03 | 120 |
| Acer | 0.14 | 0.03 | 0.06 | 319 |
| Alnus | 0.09 | 0.06 | 0.07 | 348 |
| Betula | 0.11 | 0.04 | 0.06 | 322 |
| Cleared | 0.49 | 0.26 | 0.34 | 567 |
| Fagus | 0.28 | 0.30 | 0.29 | 1110 |
| Fraxinus | 0.08 | 0.03 | 0.04 | 302 |
| Larix | 0.12 | 0.11 | 0.12 | 466 |
| Picea | 0.30 | 0.26 | 0.28 | 1100 |
| Pinus | 0.44 | 0.40 | 0.42 | 1167 |
| Populus | 0.00 | 0.00 | 0.00 | 55 |
| Prunus | 0.00 | 0.00 | 0.00 | 38 |
| Pseudotsuga | 0.11 | 0.05 | 0.06 | 462 |
| Quercus | 0.28 | 0.25 | 0.26 | 1194 |
| Tilia | 0.00 | 0.00 | 0.00 | 25 |
| **micro avg** | 0.29 | 0.22 | 0.25 | 7595 |
| **macro avg** | 0.19 | 0.12 | 0.14 | 7595 |
| **weighted avg** | 0.27 | 0.22 | 0.24 | 7595 |
| **samples avg** | 0.27 | 0.23 | 0.24 | 7595 |

### Table 3. UNet Classification Results

| Exp  | Micro Precision | Micro Recall | Micro F1 | Micro mAP | Weighted Precision | Weighted Recall | Weighted F1 | Weighted mAP | Accuracy |
|------|-----------------|--------------|----------|-----------|--------------------|-----------------|--------------|--------------|---------:|
| exp0-unet-cls | 40.73% | 13.01% | 19.72% | 26.97% | 29.81% | 13.01% | 16.87% | 25.53% | 9.04% |
| exp1-unet-cls | 29.01% | 21.91% | 24.96% | 24.96% | 27.11% | 21.91% | 23.54% | 24.12% | 11.50% |

### Table 4. UNet Classification Experiment Details

| Exp | Config | Accuracy |
|-----|--------|----------|
| exp0-unet-cls | ch=128, lr=1e-5 | 9.04% |
| exp1-unet-cls | ch=256, lr=1e-5 | 11.50% |


## Baseline MLP Regression
### Table 5. MLP Regression Results Summary

| Exp  | RMSE   | R²     |
|------|--------|--------|
| exp0-mlp-reg | 7.9909 | 0.079 |

### Table 6. MLP Regression Configs and Results

| Exp  | Configs | RMSE   | R²     |
|------|---------|--------|--------|
| exp0-mlp-reg | lr=1e-5→1e-4, wd=1e-4, pat=10 | 7.99 | 0.079 |

## Baseline MLP Classification

**exp0-mlp-cls:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Abies | 0.00 | 0.00 | 0.00 | 120 |
| Acer | 0.00 | 0.00 | 0.00 | 319 |
| Alnus | 0.00 | 0.00 | 0.00 | 348 |
| Betula | 0.00 | 0.00 | 0.00 | 322 |
| Cleared | 0.75 | 0.10 | 0.18 | 567 |
| Fagus | 0.00 | 0.00 | 0.00 | 1110 |
| Fraxinus | 0.00 | 0.00 | 0.00 | 302 |
| Larix | 0.00 | 0.00 | 0.00 | 466 |
| Picea | 0.00 | 0.00 | 0.00 | 1100 |
| Pinus | 0.57 | 0.30 | 0.39 | 1167 |
| Populus | 0.00 | 0.00 | 0.00 | 55 |
| Prunus | 0.00 | 0.00 | 0.00 | 38 |
| Pseudotsuga | 0.00 | 0.00 | 0.00 | 462 |
| Quercus | 0.00 | 0.00 | 0.00 | 1194 |
| Tilia | 0.00 | 0.00 | 0.00 | 25 |
| **micro avg** | 0.59 | 0.05 | 0.10 | 7595 |
| **macro avg** | 0.09 | 0.03 | 0.04 | 7595 |
| **weighted avg** | 0.14 | 0.05 | 0.07 | 7595 |
| **samples avg** | 0.08 | 0.06 | 0.07 | 7595 |

### Table 7. MLP Classification Results

| Exp  | Micro Precision | Micro Recall | Micro F1 | Micro mAP | Weighted Precision | Weighted Recall | Weighted F1 | Weighted mAP | Accuracy |
|------|-----------------|--------------|----------|-----------|--------------------|-----------------|--------------|--------------|---------:|
| exp0-mlp-cls | 58.93%          | 5.39%        | 9.87%    | 31.54%    | 14.36%             | 5.39%           | 7.39%        | 28.17%       | 4.81%    |

## Comparison Tables

### Table 8. Classification Experiments Comparison

| Exp | Micro Precision | Micro Recall | Micro F1 | Micro mAP | Accuracy |
|-----|-----------------|--------------|----------|-----------|----------|
| exp0-unet-cls | 40.73% | 13.01% | 19.72% | 26.97% | 9.04% |
| exp1-unet-cls | 29.01% | 21.91% | 24.96% | 24.96% | 11.50% |
| exp0-mlp-cls | 58.93% | 5.39% | 9.87% | 31.54% | 4.81% |

### Table 9. sRegression Experiments Comparison

| Exp | RMSE | R² |
|-----|------|-----|
| exp0-unet-reg | 8.0268 | 0.0750 |
| exp1-unet-reg | 8.0405 | 0.0740 |
| exp2-unet-reg | 7.9597 | 0.0857 |
| exp3-unet-reg | 7.9528 | 0.0925 |
| exp0-mlp-reg | 7.9909 | 0.0790 |

## Conclusion

1. CosineAnnealingLR suits U-Net better than ExponentialLR (nature).
2. Bigger UNet of 256 base channels outperforms smaller UNet of 128 base channels.
3. CosineAnnealingLR suits regression, no scheduler for classification.