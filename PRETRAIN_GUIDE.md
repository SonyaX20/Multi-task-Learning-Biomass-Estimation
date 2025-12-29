# 🎓 重建式预训练 (Reconstruction Pretraining) 使用指南

## 📋 概述

实现了基于**Masked Autoencoder (MAE)**的自监督预训练方案，用于SAR遥感影像分类任务。

### 核心思想

```
第一阶段: Pretraining (无需标签)
├─ 随机mask 50%的图像patches
├─ 训练模型重建被mask的部分
└─ 学习有用的特征表示

第二阶段: Fine-tuning (使用标签)
├─ 加载预训练的encoder
├─ 添加分类头
└─ 在分类任务上fine-tune
```

---

## 🏗️ 架构设计

### 1. Masked Autoencoder (MAE)

```python
MaskedAutoencoder
├─ Patch Embedding: 6x6 → 3x3 patches (9 patches)
├─ Random Masking: 保留50%, mask 50%
├─ Encoder: 4-layer Transformer
│   └─ 只处理visible patches (节省计算)
├─ Decoder: 2-layer Transformer  
│   └─ 重建所有patches (包括masked)
└─ Reconstruction Head: 预测像素值
```

**关键特性**:
- ✅ 高mask比例 (50-75%) 强迫学习全局特征
- ✅ 轻量decoder 降低计算成本
- ✅ Transformer架构 捕捉长距离依赖

### 2. Classification Model (Fine-tuning)

```python
PretrainedClassifier
├─ Pretrained Encoder (从MAE加载)
│   ├─ Patch Embedding (frozen/trainable)
│   ├─ Positional Embedding
│   └─ Transformer Blocks
├─ Global Average Pooling
└─ Classification Head
    ├─ Dropout (0.3)
    ├─ Linear (embed_dim → embed_dim/2)
    ├─ GELU
    └─ Linear (embed_dim/2 → num_classes)
```

---

## 🚀 使用流程

### 步骤 1: 预训练 (Pretraining)

```bash
python pretrain_reconstruction.py
```

#### 输出
- `pretrained_mae.pth` - 预训练模型权重
- `pretrain_loss_curve.png` - 训练曲线
- `pretrain_viz/reconstruction_epoch_*.png` - 重建可视化

#### 预期效果
```
Epoch 50/50
  Train Loss: 0.45
  Val Loss: 0.48
  
✓ 模型学会了重建被mask的图像区域
```

---

### 步骤 2: Fine-tuning (分类)

```bash
python finetune_classification.py
```

#### 两种模式

**模式1: Feature Extraction** (冻结encoder)
```python
freeze_encoder=True
epochs=20
lr=5e-4  # 更大的学习率
```
- 更快
- 适合数据少的情况
- 防止过拟合

**模式2: Full Fine-tuning** (训练整个模型)
```python
freeze_encoder=False
epochs=30
lr=1e-4  # 较小的学习率
```
- 更好的性能
- 需要更多数据
- 推荐使用

#### 输出
- `finetuned_frozen.pth` - 冻结模式模型
- `finetuned_full.pth` - 全fine-tuning模型
- `finetune_results.png` - 训练曲线

---

## 📊 参数配置

### Pretraining 参数

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `mask_ratio` | 0.5 | Mask比例 | 0.5-0.75 |
| `embed_dim` | 128 | Embedding维度 | 64-256 |
| `encoder_depth` | 4 | Encoder层数 | 4-8 |
| `decoder_depth` | 2 | Decoder层数 | 1-4 |
| `num_heads` | 4 | 注意力头数 | 4-8 |
| `epochs` | 50 | 训练轮数 | 50-100 |
| `batch_size` | 128 | Batch大小 | 64-256 |
| `lr` | 1e-3 | 学习率 | 5e-4 to 2e-3 |

### Fine-tuning 参数

| 参数 | Frozen | Full | 说明 |
|------|--------|------|------|
| `freeze_encoder` | True | False | 是否冻结 |
| `epochs` | 20 | 30 | 训练轮数 |
| `lr` | 5e-4 | 1e-4 | 学习率 |
| `dropout` | 0.3 | 0.3 | Dropout率 |

---

## 💡 调优建议

### Pretraining 调优

#### 1. Mask Ratio

```python
# 轻度mask (学习局部特征)
mask_ratio=0.5

# 中度mask (推荐)
mask_ratio=0.6

# 重度mask (强迫学习全局特征)
mask_ratio=0.75
```

**选择建议**:
- 小图像(6x6) → 0.5-0.6
- 大图像 → 0.7-0.8
- 数据多 → 更高mask ratio

#### 2. Model Capacity

```python
# 小模型 (快速实验)
embed_dim=64, encoder_depth=2, decoder_depth=1

# 中等模型 (推荐)
embed_dim=128, encoder_depth=4, decoder_depth=2

# 大模型 (追求性能)
embed_dim=256, encoder_depth=6, decoder_depth=4
```

#### 3. Training Duration

```python
# 快速验证
epochs=20

# 标准训练 (推荐)
epochs=50

# 充分训练
epochs=100
```

**判断标准**:
- Val loss不再下降 → 可以停止
- 重建效果清晰 → 训练充分

---

### Fine-tuning 调优

#### 1. Freeze vs Full

| 场景 | 推荐模式 | 理由 |
|------|---------|------|
| 数据量 < 1000 | Frozen | 防止过拟合 |
| 数据量 1000-5000 | Frozen | 快速收敛 |
| 数据量 > 5000 | Full | 更好性能 |
| 目标任务差异大 | Full | 需要适应 |
| 计算资源有限 | Frozen | 更快 |

#### 2. Learning Rate

```python
# Feature Extraction (frozen)
lr = 5e-4  # 可以用更大的学习率

# Full Fine-tuning
lr = 1e-4  # 较小，避免破坏预训练权重
lr = 5e-5  # 更保守
```

#### 3. Epochs

```python
# Frozen模式
epochs = 15-20  # 收敛快

# Full模式
epochs = 30-50  # 需要更多时间
```

---

## 📈 预期改进

### 与从头训练对比

| 指标 | 从头训练 | Frozen | Full Fine-tune | 改进 |
|------|---------|--------|----------------|------|
| **验证准确率** | 35-40% | 42-47% | 48-55% | +20-40% |
| **测试mAP** | 0.25 | 0.30 | 0.35 | +20-40% |
| **收敛速度** | 30 epochs | 15 epochs | 25 epochs | 更快 |
| **数据效率** | 需要大量标注 | 中等 | 中等 | 更高 |
| **小数据性能** | 较差 | 好 | 很好 | 大幅提升 |

### 关键优势

1. **无需标注数据**: Pretraining阶段不需要标签
2. **更好的特征**: 学习到的表示更具泛化性
3. **数据效率**: 标注数据少时效果显著
4. **迁移学习**: 可以迁移到相关任务

---

## 🔬 工作原理

### 为什么Masked Autoencoding有效？

#### 1. 强迫全局理解
```
Mask 50%的patches
↓
模型必须理解整个图像的全局结构
↓
不能只依赖局部纹理
```

#### 2. 学习有用表示
```
重建任务
↓
编码器必须提取关键特征
↓
这些特征对下游任务有用
```

#### 3. 自监督学习
```
不需要人工标注
↓
可以使用大量无标注数据
↓
学习更通用的特征
```

---

## 🛠️ 可视化分析

### 重建质量检查

每10个epoch会生成重建可视化：

```
pretrain_viz/reconstruction_epoch_10.png
├─ Original: 原始图像
├─ Masked: 被mask的图像 (50%被遮挡)
└─ Reconstructed: 重建结果
```

**好的重建特征**:
- ✅ 主要结构清晰
- ✅ 纹理大致正确
- ✅ 边界位置准确
- ⚠️ 不需要像素级完美

---

## 📝 代码示例

### 完整工作流

```python
# ========== 1. Pretraining ==========
from pretrain_reconstruction import pretrain_mae

model, history = pretrain_mae(
    train_h5='data/train_data.h5',
    test_h5='data/test_data.h5',
    epochs=50,
    batch_size=128,
    mask_ratio=0.5,
    save_path='pretrained_mae.pth'
)

# ========== 2. Fine-tuning ==========
from finetune_classification import finetune_classifier

# 模式1: Frozen encoder
model_frozen, history = finetune_classifier(
    pretrained_path='pretrained_mae.pth',
    train_h5='data/train_data.h5',
    test_h5='data/test_data.h5',
    num_classes=15,
    epochs=20,
    freeze_encoder=True,
    save_path='finetuned_frozen.pth'
)

# 模式2: Full fine-tuning  
model_full, history = finetune_classifier(
    pretrained_path='pretrained_mae.pth',
    train_h5='data/train_data.h5',
    test_h5='data/test_data.h5',
    num_classes=15,
    epochs=30,
    freeze_encoder=False,
    save_path='finetuned_full.pth'
)
```

---

## 🎯 使用场景

### 适合的情况

✅ **标注数据少** (< 5000样本)
- Pretraining可以利用无标注数据
- Fine-tuning只需少量标注样本

✅ **数据分布相似** 
- Pretraining和downstream任务来自同一领域
- SAR影像 → SAR分类

✅ **计算资源充足**
- Pretraining需要额外时间
- 但长期来看更高效

✅ **需要迁移学习**
- 可以在一个大数据集上预训练
- 迁移到多个小数据集

### 不太适合的情况

⚠️ **标注数据充足** (> 50000样本)
- 从头训练可能就够了
- Pretraining收益有限

⚠️ **任务差异大**
- Pretraining数据和目标任务很不同
- 预训练权重帮助有限

⚠️ **时间紧迫**
- Pretraining需要额外时间
- 快速实验建议直接训练

---

## 🔍 故障排除

### 问题1: Pretraining loss不降

**可能原因**:
- Learning rate太大/太小
- Mask ratio太高
- Model capacity不足

**解决方案**:
```python
# 调整学习率
lr = 5e-4  # 降低

# 降低mask ratio
mask_ratio = 0.5  # 从0.75降低

# 增加模型容量
embed_dim = 256
encoder_depth = 6
```

---

### 问题2: 重建质量差

**可能原因**:
- 训练不充分
- Decoder太简单
- 数据预处理问题

**解决方案**:
```python
# 增加训练时间
epochs = 100

# 增强decoder
decoder_depth = 4

# 检查数据
# 确保没有NaN，数值范围合理
```

---

### 问题3: Fine-tuning效果不好

**可能原因**:
- Pretraining不充分
- Learning rate不合适
- Freeze/Full选择不当

**解决方案**:
```python
# 确保pretraining充分
# 检查重建效果

# 调整学习率
lr = 1e-4  # full fine-tuning
lr = 5e-4  # frozen encoder

# 尝试两种模式对比
freeze_encoder = True   # 试试frozen
freeze_encoder = False  # 试试full
```

---

### 问题4: 过拟合

**可能原因**:
- 数据太少
- Dropout太小
- Full fine-tuning too aggressive

**解决方案**:
```python
# 使用frozen模式
freeze_encoder = True

# 增加dropout
dropout = 0.5

# 减少epochs
epochs = 15

# 使用更多数据增强
# (可以在baseline_clf.py中添加)
```

---

## 📚 理论背景

### Masked Autoencoder (MAE)

**论文**: "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
**作者**: He et al., Facebook AI Research

**核心贡献**:
1. 高mask ratio (75%) 对vision很有效
2. 非对称encoder-decoder设计
3. 简单但强大的预训练方法

### 为什么对SAR有效？

1. **SAR特性**: 相干斑噪声，需要全局理解
2. **小图像**: 6x6图像，mask迫使学习全局
3. **有限标注**: SAR标注数据昂贵
4. **迁移性**: 学习的特征可迁移

---

## 📊 实验建议

### 实验计划

#### 基线实验
1. 从头训练 (baseline)
2. Frozen encoder
3. Full fine-tuning

#### 对比指标
- 验证准确率
- 测试mAP
- 收敛速度 (达到目标准确率的epochs)
- 小数据性能 (使用10%, 50%数据)

#### 消融实验
```python
# Mask ratio ablation
for mask_ratio in [0.5, 0.6, 0.75]:
    pretrain_mae(mask_ratio=mask_ratio)

# Model size ablation
for embed_dim in [64, 128, 256]:
    pretrain_mae(embed_dim=embed_dim)

# Pretraining epochs ablation
for epochs in [20, 50, 100]:
    pretrain_mae(epochs=epochs)
```

---

## ✅ 检查清单

### Pretraining 阶段

- [ ] 数据加载正常 (无NaN)
- [ ] 模型创建成功
- [ ] 训练loss持续下降
- [ ] 验证loss收敛
- [ ] 重建效果清晰
- [ ] 模型保存成功

### Fine-tuning 阶段

- [ ] 预训练模型加载成功
- [ ] Encoder权重正确加载
- [ ] 分类头初始化正常
- [ ] 训练/验证loss正常
- [ ] 准确率超过baseline
- [ ] 模型保存成功

---

## 🎓 总结

### 核心要点

1. **两阶段训练**: Pretraining (重建) + Fine-tuning (分类)
2. **自监督学习**: 无需标注数据预训练
3. **两种Fine-tune模式**: Frozen (快) vs Full (好)
4. **显著改进**: 特别是在小数据场景

### 推荐配置

```python
# Pretraining (推荐)
mask_ratio = 0.5
embed_dim = 128
encoder_depth = 4
epochs = 50

# Fine-tuning (推荐)
freeze_encoder = False  # Full fine-tuning
epochs = 30
lr = 1e-4
```

---

**创建时间**: 2025-10-29  
**方法**: Masked Autoencoder (MAE)  
**任务**: SAR影像分类  
**状态**: ✅ Ready to use


