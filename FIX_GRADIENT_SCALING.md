# 🔧 修复梯度缩放问题

## ❌ 问题诊断

### 问题现象
设置了第一层梯度的缩放，但梯度流图中显示**并没有被裁剪**。

### 根本原因
**层名不匹配**！

你的代码使用的是 `OptimizedCNN` 模型，但梯度缩放使用的层名是 `CNN` 模型的命名。

---

## 🔍 详细分析

### 模型结构对比

#### CNN 模型（旧）
```python
class CNN(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(
            nn.Conv2d(...),    # ← 这个叫 'features.0'
            nn.BatchNorm2d(...), # ← 这个叫 'features.1'
            ...
        )
```

#### OptimizedCNN 模型（你实际使用的）
```python
class OptimizedCNN(nn.Module):
    def __init__(self):
        self.stem = nn.Sequential(      # ← 没有 'features'
            nn.Conv2d(...),             # ← 实际叫 'stem.0'
            nn.BatchNorm2d(...),        # ← 实际叫 'stem.1'
            ...
        )
        self.layer1 = ImprovedResidualBlock(...)  # ← 'layer1.conv1'
        ...
```

---

## ✅ 解决方案

### 错误的代码（当前）

```python
gradient_scaler = GradientScaler({
    'features.0': 0.05,  # ❌ 这个层名不存在于OptimizedCNN中！
    'features.1': 0.3,   # ❌ 同样不存在！
})
gradient_scaler.scale_gradients(model)
```

### 正确的代码（修复后）

```python
gradient_scaler = GradientScaler({
    'stem.0.weight': 0.05,      # ✅ 第一层卷积权重
    'stem.1.weight': 0.3,       # ✅ Stem的BatchNorm权重  
    'layer1.conv1.weight': 0.5, # ✅ 第一个残差块的第一个卷积
})
gradient_scaler.scale_gradients(model)
```

---

## 📋 修复步骤

### 步骤 1: 查找实际层名

在 `baseline_clf.py` 的训练循环之前添加：

```python
# 在 train_classification 函数开始处添加
print("\n模型实际层名（用于梯度缩放）:")
for name, param in model.named_parameters():
    if 'stem' in name.lower() or ('layer1' in name.lower() and 'conv' in name.lower()):
        print(f"  {name}")
```

### 步骤 2: 修改梯度缩放配置

找到这段代码（约第945行）：

```python
gradient_scaler = GradientScaler({
    'features.0': 0.05,
    'features.1': 0.3,
})
```

替换为：

```python
gradient_scaler = GradientScaler({
    'stem.0.weight': 0.05,      # 第一层卷积
    'stem.1.weight': 0.3,       # Stem的BatchNorm
    # 可选：添加更多层的缩放
    # 'layer1.conv1.weight': 0.5,
})
```

### 步骤 3: 验证修复

运行训练后，检查梯度流图 `gradient_flow.png`：
- ✅ `stem.0.weight` 的梯度应该明显减小
- ✅ 应该能看到梯度被正确缩放的效果

---

## 🔬 诊断脚本

运行诊断脚本查看正确的层名：

```bash
python fix_gradient_scaling.py
```

这个脚本会：
1. 打印 `OptimizedCNN` 的所有层名
2. 标出哪些是第一层相关的参数
3. 给出正确的梯度缩放配置

---

## 💡 GradientScaler 工作原理

### 当前实现的问题

```python
class GradientScaler:
    def scale_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                for layer_key, scale in self.layer_scales.items():
                    if layer_key in name:  # ← 字符串匹配
                        param.grad.data.mul_(scale)
                        break
```

**问题**: 使用 `layer_key in name` 进行字符串匹配。

如果 `layer_key = 'features.0'` 但实际层名是 `'stem.0.weight'`，就不会匹配！

---

## 🎯 修复后的完整代码片段

在 `train_classification` 函数中（约第928-951行）：

```python
def train_classification(model, train_loader, criterion, optimizer, device, epoch=0):
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch_data in enumerate(pbar):
        data, target = batch_data
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # ✅ 修复：使用正确的层名
        gradient_scaler = GradientScaler({
            'stem.0.weight': 0.05,      # 第一层卷积权重
            'stem.1.weight': 0.3,       # Stem的BatchNorm权重
            'layer1.conv1.weight': 0.5, # 第一残差块的第一个卷积
        })
        gradient_scaler.scale_gradients(model)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if batch_idx == len(train_loader) - 1:
            plot_gradient_flow(model)
            
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            acc = compute_accuracy(output, target)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc*100:.1f}%'
        })

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    return avg_loss, avg_acc
```

---

## 🔍 如何验证修复

### 方法 1: 打印缩放后的梯度值

在 `GradientScaler.scale_gradients` 中添加调试信息：

```python
def scale_gradients(self, model):
    scaled_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_before = param.grad.abs().mean().item()
            
            for layer_key, scale in self.layer_scales.items():
                if layer_key in name:
                    param.grad.data.mul_(scale)
                    scaled_count += 1
                    grad_after = param.grad.abs().mean().item()
                    print(f"  ✓ Scaled {name}: {grad_before:.6f} → {grad_after:.6f} (scale={scale})")
                    break
    
    if scaled_count == 0:
        print(f"  ⚠️  Warning: No gradients scaled! Check layer names.")
    else:
        print(f"  ✓ Scaled {scaled_count} parameter groups")
```

如果修复正确，应该看到类似输出：
```
✓ Scaled stem.0.weight: 0.001234 → 0.000062 (scale=0.05)
✓ Scaled stem.1.weight: 0.000567 → 0.000170 (scale=0.3)
✓ Scaled 2 parameter groups
```

### 方法 2: 检查梯度流图

查看 `gradient_flow.png`：
- 修复前：`stem.0.weight` 的梯度值很大
- 修复后：`stem.0.weight` 的梯度值应该明显减小（约5%）

---

## 📊 OptimizedCNN 层名参考

### 主要层的命名规则

```
stem.0.weight           # 第一层卷积 (输入→base_channels)
stem.1.weight           # Stem的BatchNorm
layer1.bn1.weight       # Layer1的第一个BatchNorm
layer1.conv1.weight     # Layer1的第一个卷积
layer1.bn2.weight       # Layer1的第二个BatchNorm
layer1.conv2.weight     # Layer1的第二个卷积
layer2.conv1.weight      # Layer2的第一个卷积
...
classifier.0.weight     # Classifier的第一个Linear层
```

### 用于梯度缩放的建议层名

```python
# 第一层相关（最重要）
'stem.0.weight': 0.05,        # 第一层卷积

# 第二层相关
'layer1.conv1.weight': 0.3,   # 第一个残差块的第一个卷积

# 如果需要更细粒度控制
'stem.0': 0.05,               # 匹配 stem.0.weight 和 stem.0.bias
'stem.1': 0.3,                # 匹配 stem.1.weight 和 stem.1.bias
```

---

## ⚠️ 注意事项

### 1. 字符串匹配的局限性

当前的 `GradientScaler` 使用简单的 `in` 匹配：
```python
if layer_key in name:
```

这意味着：
- ✅ `'stem.0'` 可以匹配 `'stem.0.weight'` 和 `'stem.0.bias'`
- ✅ `'stem.0.weight'` 只会匹配精确名称
- ⚠️ `'features'` 不会匹配 `'stem'`

### 2. 建议使用完整层名

为了精确控制，建议使用完整的参数名：
```python
'stem.0.weight': 0.05,  # ✅ 精确匹配
```

而不是：
```python
'stem.0': 0.05,  # ⚠️ 会匹配 stem.0.weight 和 stem.0.bias
```

---

## 🎯 完整修复清单

- [ ] 运行 `python fix_gradient_scaling.py` 查看实际层名
- [ ] 将 `'features.0'` 改为 `'stem.0.weight'`
- [ ] 将 `'features.1'` 改为 `'stem.1.weight'` 或删除
- [ ] 运行训练并检查梯度流图
- [ ] 确认 `stem.0.weight` 的梯度确实被缩放
- [ ] 如果还有问题，添加调试打印查看匹配情况

---

## 📞 快速修复

**最简单的方式**：直接替换这两行

```python
# 找到第945-948行，替换为：
gradient_scaler = GradientScaler({
    'stem.0.weight': 0.05,      # 第一层卷积
    'stem.1.weight': 0.3,       # Stem的BatchNorm
})
```

---

**修复完成时间**: 2025-10-29  
**问题**: 层名不匹配导致梯度缩放失效  
**解决方案**: 使用OptimizedCNN的实际层名


