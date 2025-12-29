# 🚀 预训练+分类 快速开始

## 一、两步完成预训练分类

### 步骤 1: 预训练 (Pretraining)

```bash
python pretrain_reconstruction.py
```

**做什么**: 训练Masked Autoencoder重建图像  
**需要**: 无标注数据即可  
**时间**: ~2-4小时 (50 epochs)  
**输出**: `pretrained_mae.pth`

---

### 步骤 2: 分类 (Fine-tuning)

```bash
python finetune_classification.py
```

**做什么**: 使用预训练模型进行分类  
**需要**: 标注数据  
**时间**: ~1-2小时 (30 epochs)  
**输出**: `finetuned_full.pth`

---

## 二、核心概念

### Masked Autoencoder (MAE)

```
原理: 
1. 随机mask 50%的图像patches
2. 只用visible patches训练encoder
3. Decoder重建所有patches
4. 学习有用的特征表示

好处:
✅ 无需标注数据
✅ 学习更好的特征
✅ 提升下游任务性能
```

### 两种Fine-tuning模式

#### 模式1: Frozen Encoder (特征提取)
```python
freeze_encoder=True
```
- ⚡ 更快 (15-20 epochs)
- 💾 防止过拟合
- 📊 适合数据少

#### 模式2: Full Fine-tuning (全模型训练)
```python
freeze_encoder=False
```
- 🎯 更好性能
- ⏱️ 需要更多时间 (30 epochs)
- 📈 推荐使用

---

## 三、预期效果

### 与从头训练对比

| 指标 | 从头训练 | 预训练+Fine-tune | 提升 |
|------|---------|-----------------|------|
| 验证准确率 | ~40% | **~50%** | +25% |
| 测试mAP | ~0.25 | **~0.35** | +40% |
| 收敛速度 | 30 epochs | **20 epochs** | 更快 |

---

## 四、文件说明

### 生成的文件

```
pretrain_reconstruction.py       # 预训练脚本
└─ pretrained_mae.pth           # 预训练模型 ✅

finetune_classification.py       # Fine-tuning脚本
├─ finetuned_frozen.pth         # Frozen模式模型
└─ finetuned_full.pth           # Full模式模型 ✅ 推荐

可视化输出:
├─ pretrain_viz/                # 重建可视化
├─ pretrain_loss_curve.png      # 预训练曲线
└─ finetune_results.png         # Fine-tune曲线
```

---

## 五、参数调整

### 快速调整 (pretrain_reconstruction.py)

```python
# 第548行左右
model, history = pretrain_mae(
    epochs=50,          # 训练轮数 (20-100)
    mask_ratio=0.5,     # Mask比例 (0.5-0.75)
    batch_size=128      # Batch大小 (64-256)
)
```

### 快速调整 (finetune_classification.py)

```python
# 第488行左右 - Frozen模式
finetune_classifier(
    epochs=20,              # 训练轮数
    freeze_encoder=True,    # 冻结encoder
)

# 第500行左右 - Full模式
finetune_classifier(
    epochs=30,              # 训练轮数
    freeze_encoder=False,   # 训练整个模型
)
```

---

## 六、常见问题

### Q1: Pretraining需要多久？

**A**: 
- CPU: 6-8小时
- GPU: 2-4小时
- Colab GPU: 1-2小时

### Q2: 必须先Pretrain吗？

**A**: 是的！必须先运行`pretrain_reconstruction.py`生成`pretrained_mae.pth`，然后才能运行fine-tuning。

### Q3: 可以只用一种模式吗？

**A**: 可以！代码会运行两种模式做对比，你可以注释掉不需要的：

```python
# 只运行Full Fine-tuning (推荐)
if __name__ == "__main__":
    # 注释掉Frozen模式
    # model_frozen, history_frozen = finetune_classifier(...)
    
    # 只保留Full模式
    model_finetune, history_finetune = finetune_classifier(
        freeze_encoder=False,
        ...
    )
```

### Q4: Pretraining loss应该降到多少？

**A**: 
- 初始: ~2.0-2.5
- 目标: ~0.4-0.6
- 最佳: ~0.3-0.5

### Q5: 怎么知道Pretraining好不好？

**A**: 查看重建图像：
```
pretrain_viz/reconstruction_epoch_50.png
```
- ✅ 主要结构清晰
- ✅ 大致纹理正确
- ⚠️ 不需要像素级完美

### Q6: Fine-tuning准确率多少算好？

**A**:
- 基线 (从头训练): 35-40%
- Frozen: 42-47%
- Full: 48-55%
- 目标: 超过基线10-15%

---

## 七、快速检查清单

### Pretraining ✅

- [ ] 数据路径正确
- [ ] Training loss下降到<0.5
- [ ] Validation loss收敛
- [ ] `pretrained_mae.pth`已生成
- [ ] 重建图像看起来合理

### Fine-tuning ✅

- [ ] `pretrained_mae.pth`存在
- [ ] 模型成功加载预训练权重
- [ ] 验证准确率 > 基线
- [ ] mAP > 基线
- [ ] 模型已保存

---

## 八、典型输出示例

### Pretraining 输出

```
======================================================================
Masked Autoencoder Pretraining
======================================================================

Dataset loaded:
  Train: 55,400 samples
  Val: 6,789 samples

Model created:
  Total parameters: 1,234,567
  Mask ratio: 50%

Epoch 50/50
  Train Loss: 0.4523
  Val Loss: 0.4812
  ✓ Best model saved! Val Loss: 0.4812
```

### Fine-tuning 输出

```
======================================================================
MODE 2: Full Fine-tuning (Trainable Encoder)
======================================================================

Pretrained model info:
  Epoch: 50
  Val Loss: 0.4812

Model info:
  Total parameters: 1,456,789
  Trainable parameters: 1,456,789
  Freeze encoder: False

Epoch 30/30
  Train Loss: 1.2345 | Train Acc: 55.67%
  Val Loss: 1.4567 | Val Acc: 52.34% | Val mAP: 0.3567
  ✓ Best model saved! mAP: 0.3567

Test Results:
  Accuracy: 53.21%
  mAP: 0.3645
```

---

## 九、最简使用流程

### 1. 准备数据
```
data/
├─ train_data.h5
├─ test_data.h5
└─ label_mapping.json
```

### 2. 运行预训练
```bash
python pretrain_reconstruction.py
```
☕ 休息2-4小时

### 3. 运行Fine-tuning
```bash
python finetune_classification.py
```
☕ 休息1-2小时

### 4. 查看结果
```bash
# 查看训练曲线
open finetune_results.png

# 查看重建效果
open pretrain_viz/reconstruction_epoch_50.png

# 查看预训练曲线
open pretrain_loss_curve.png
```

### 5. 使用模型
```python
import torch
from finetune_classification import PretrainedClassifier

# 加载模型
checkpoint = torch.load('finetuned_full.pth')
# ... 进行预测
```

---

## 十、性能优化技巧

### 加速Pretraining

```python
# 减少epochs (快速实验)
epochs=20

# 减小模型
embed_dim=64
encoder_depth=2

# 增大batch size
batch_size=256
```

### 提升Fine-tuning性能

```python
# 使用Full模式
freeze_encoder=False

# 增加训练时间
epochs=50

# 降低learning rate
lr=5e-5
```

---

## 十一、下一步

### 实验建议

1. **基线对比**: 先用baseline_clf.py训练从头开始的模型
2. **运行Pretrain**: 执行预训练获得特征
3. **对比结果**: 比较预训练vs从头训练的性能差异
4. **参数调优**: 根据结果调整mask_ratio, epochs等
5. **消融实验**: 测试不同配置的影响

### 进阶使用

- 调整mask_ratio观察影响
- 尝试不同的模型大小
- 实验不同的fine-tuning策略
- 可视化学习到的特征
- 迁移到其他SAR任务

---

## 📚 参考文档

- **完整指南**: `PRETRAIN_GUIDE.md`
- **代码注释**: 文件内有详细注释
- **Baseline**: `baseline_clf.py`

---

**创建时间**: 2025-10-29  
**适用**: SAR影像分类  
**方法**: Masked Autoencoder (MAE)  
**难度**: ⭐⭐⭐ (中等)


