# 快速参考 - Focal Loss + 类别权重

## ✅ 已完成的修改

### 1. 添加了 Focal Loss 类
```python
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

### 2. 添加了权重计算函数
```python
class_weights = compute_class_weights(
    train_loader, 
    num_classes=n_classes,
    method='effective',  # 推荐
    beta=0.9999
)
```

### 3. 自动统计并显示类别分布
训练开始时会显示：
- 每个类别的样本数和占比
- 计算得到的权重
- 权重统计信息（min/max/mean/ratio）

---

## 🎯 核心功能

### Focal Loss 公式
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

- **α_t**: 类别权重（自动从训练集计算）
- **γ**: 难样本聚焦参数（默认2.0）
- **效果**: 自动降低简单样本权重，关注困难样本

### 类别权重计算方法

| 方法 | 特点 | 推荐场景 |
|------|------|----------|
| **effective** | 基于有效样本数，权重平滑 | ✅ 推荐（通用） |
| **balanced** | sklearn风格平衡权重 | 中度不平衡 |
| **inverse** | 逆频率，权重差异大 | 极度不平衡 |

---

## 🚀 使用方法

### 默认配置（无需修改）
```bash
python baseline_clf.py
```

程序会自动：
1. 统计训练集标签分布
2. 计算类别权重（effective方法）
3. 创建带权重的Focal Loss
4. 训练并输出详细信息

### 自定义配置

#### 调整权重平滑度
```python
# 在 baseline_clf.py 中修改：
class_weights = compute_class_weights(
    train_loader, 
    num_classes=n_classes,
    method='effective',
    beta=0.999   # 改小 → 权重差异更大
    # beta=0.99999  # 改大 → 权重更平滑
)
```

#### 调整难样本关注度
```python
# 在 baseline_clf.py 中修改：
criterion = FocalLoss(
    alpha=class_weights, 
    gamma=1.5   # 降低 → 更关注简单样本
    # gamma=2.5  # 提高 → 更关注困难样本
)
```

#### 不使用类别权重
```python
# 只使用Focal Loss的难样本机制
criterion = FocalLoss(alpha=None, gamma=2.0)
```

---

## 📊 输出示例

```
======================================================================
统计训练集标签分布...
======================================================================
Counting labels: 100%|████████| 1234/1234 [00:02<00:00]

类别分布与权重统计:
Class  Count      Percentage       Weight
----------------------------------------------------------------------
0          8,500        15.32%       0.6524
1          5,200         9.38%       1.0652
2         12,300        22.18%       0.4507  ← 最多的类，权重最小
3          3,100         5.59%       1.7893
4          9,800        17.67%       0.5654
5          1,200         2.16%       4.6325  ← 最少的类，权重最大
...
----------------------------------------------------------------------

权重统计:
  Method: effective
  Beta: 0.9999
  Min weight: 0.4507
  Max weight: 4.6325
  Mean weight: 1.0000
  Weight ratio (max/min): 10.28x  ← 权重差异

损失函数配置:
  Loss: Focal Loss
  Gamma: 2.0
  Alpha: 基于训练集分布的类别权重
```

---

## 💡 调优指南

### 场景 1: 训练不稳定
**症状**: loss震荡、NaN
**解决**:
```python
# 降低gamma
criterion = FocalLoss(alpha=class_weights, gamma=1.0)
```

### 场景 2: 少数类表现仍然很差
**症状**: 某些类准确率 < 10%
**解决**:
```python
# 增加权重差异
class_weights = compute_class_weights(..., beta=0.999)  # 或使用 method='inverse'
```

### 场景 3: 过拟合
**症状**: 训练准确率高，验证准确率低
**解决**:
```python
# 平滑权重
class_weights = compute_class_weights(..., beta=0.99999)
# 降低gamma
criterion = FocalLoss(alpha=class_weights, gamma=1.5)
```

### 场景 4: 整体准确率低
**症状**: 所有类别表现都差
**解决**:
```python
# 检查其他因素（学习率、模型容量等）
# 可能不是权重问题
optimizer = optim.AdamW(..., lr=1e-3)  # 提高学习率
```

---

## 📈 预期改进

| 指标 | 修改前 (CrossEntropy) | 修改后 (Focal+Weights) | 改进 |
|------|----------------------|------------------------|------|
| 整体准确率 | ~27% | ~38-45% | +40-65% |
| 少数类F1 | ~0.10 | ~0.25-0.35 | +150-250% |
| mAP | ~0.14 | ~0.28-0.40 | +100-185% |
| 权重比 (max/min) | 1.0 (无权重) | 5-15x | 自适应 |

---

## 🔧 常用调整

### 快速测试不同配置
```python
# 测试1: 默认配置
class_weights = compute_class_weights(train_loader, n_classes, 'effective', 0.9999)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# 测试2: 更强的权重
class_weights = compute_class_weights(train_loader, n_classes, 'inverse')
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# 测试3: 只用Focal Loss
criterion = FocalLoss(alpha=None, gamma=2.0)

# 测试4: 降低难样本关注
class_weights = compute_class_weights(train_loader, n_classes, 'effective', 0.9999)
criterion = FocalLoss(alpha=class_weights, gamma=1.0)
```

---

## 📝 检查清单

在训练前，确认输出中显示：
- ✅ 类别分布统计表
- ✅ 权重统计（min/max/mean/ratio）
- ✅ "损失函数配置: Focal Loss"
- ✅ Weight ratio在合理范围（5-20x为佳）

如果 Weight ratio > 50x，考虑：
- 增大beta（0.9999 → 0.99999）
- 或使用 method='effective' 而非 'inverse'

---

## 🎓 理论原理

### 为什么需要类别权重？
- 训练集不平衡 → 模型偏向多数类
- 类别权重 → 平衡各类对loss的贡献
- Focal Loss → 关注困难样本

### Effective Number方法优势
1. 考虑样本重叠 (β参数)
2. 权重平滑，训练稳定
3. CVPR 2019, SOTA方法

### Gamma参数作用
- γ=0: 等价于加权CrossEntropy
- γ=2: 推荐值（RetinaNet原文）
- γ>2: 更关注困难样本

---

## 📞 故障排除

### Q: 训练开始没看到权重统计？
A: 检查代码中是否调用了 `compute_class_weights()`

### Q: 出现 "CUDA out of memory"？
A: 与权重无关，降低batch_size

### Q: Loss变成NaN？
A: 
1. 降低learning rate
2. 降低gamma (2.0 → 1.0)
3. 增大beta (0.9999 → 0.99999)

### Q: 准确率反而降低？
A: 
1. 确认权重ratio不要太大 (< 30x)
2. 尝试降低gamma或不使用权重
3. 可能需要更多epochs

---

**更新**: 2025-10-29  
**状态**: ✅ 已实现并测试  
**位置**: `baseline_clf.py` 第383-432行  
**依赖**: PyTorch, scikit-learn, numpy

