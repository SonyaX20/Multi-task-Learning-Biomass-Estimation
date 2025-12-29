# -*- coding: utf-8 -*-
"""
修复梯度缩放问题 - 诊断脚本

问题：梯度缩放使用的层名与实际模型不匹配
原因：OptimizedCNN的层名是 'stem.0' 而不是 'features.0'
"""

import torch
import torch.nn as nn

# 复制OptimizedCNN的定义（从baseline_clf.py）
class ImprovedResidualBlock(nn.Module):
    """PreActivation 残差块"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        residual = self.shortcut(x)
        out = out + residual
        return out


class OptimizedCNN(nn.Module):
    """优化后的 CNN - 针对梯度流问题"""
    def __init__(self, in_channels=2, num_classes=15, base_channels=64, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels, momentum=0.01),
            nn.GELU()
        )
        self.layer1 = ImprovedResidualBlock(
            base_channels, base_channels*2,
            stride=1, dropout=dropout
        )
        self.layer2 = ImprovedResidualBlock(
            base_channels*2, base_channels*4,
            stride=2, dropout=dropout * 0.7
        )
        self.layer3 = ImprovedResidualBlock(
            base_channels*4, base_channels*8,
            stride=1, dropout=dropout * 0.5
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_channels*8, base_channels*4),
            nn.BatchNorm1d(base_channels*4, momentum=0.01),
            nn.GELU(),
            nn.Dropout(dropout * 1.2),
            nn.Linear(base_channels*4, base_channels*2),
            nn.BatchNorm1d(base_channels*2, momentum=0.01),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels*2, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def print_model_layers(model):
    """打印模型的所有层名，特别是卷积层"""
    print("\n" + "="*70)
    print("OptimizedCNN 模型层名列表")
    print("="*70)
    print("\n所有卷积层和第一层的参数:")
    print("-"*70)
    
    conv_layers = []
    first_layers = []
    
    for name, param in model.named_parameters():
        if 'conv' in name.lower() or 'stem' in name.lower():
            print(f"{name:<50} shape: {tuple(param.shape)}")
            if 'stem' in name.lower() or 'layer1.conv1' in name.lower():
                conv_layers.append(name)
                if len(first_layers) < 3:  # 前3个重要层
                    first_layers.append(name)
    
    print("\n" + "="*70)
    print("推荐的梯度缩放配置:")
    print("="*70)
    print("\n当前错误的配置（使用CNN的层名）:")
    print("  'features.0': 0.05  # ❌ 不存在！这是CNN模型的层名")
    print("  'features.1': 0.3   # ❌ 不存在！")
    
    print("\n正确的配置（使用OptimizedCNN的实际层名）:")
    if conv_layers:
        print(f"  '{first_layers[0] if first_layers else 'stem.0.weight'}': 0.05,  # ✓ 第一层卷积")
        print(f"  '{first_layers[1] if len(first_layers) > 1 else 'layer1.conv1.weight'}': 0.3,  # ✓ 第二层")
    
    print("\n所有可用的第一层相关参数:")
    for i, layer in enumerate(first_layers[:5]):
        print(f"  {i+1}. {layer}")
    
    print("\n" + "="*70)
    print("修复后的代码应该是:")
    print("="*70)
    print("""
# 修复前（错误）:
gradient_scaler = GradientScaler({
    'features.0': 0.05,  # ❌ 不匹配OptimizedCNN
    'features.1': 0.3,
})

# 修复后（正确）:
gradient_scaler = GradientScaler({
    'stem.0.weight': 0.05,      # ✓ 第一层卷积权重
    'stem.1.weight': 0.3,       # ✓ Stem的BatchNorm权重
    'layer1.conv1.weight': 0.5, # ✓ 第一个残差块的第一个卷积
})
    """)


if __name__ == "__main__":
    # 创建模型
    model = OptimizedCNN(in_channels=2, num_classes=15, base_channels=32, dropout=0.2)
    
    # 打印所有层名
    print("\n完整模型结构:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只打印叶子节点
            print(f"{name}: {type(module).__name__}")
    
    # 打印关键层名用于梯度缩放
    print_model_layers(model)
    
    # 测试一个forward pass以获得梯度
    print("\n" + "="*70)
    print("测试梯度名称（模拟训练时的梯度）:")
    print("="*70)
    
    x = torch.randn(2, 2, 6, 6)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    print("\n有梯度的层（可用于缩放）:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'stem' in name.lower() or 'layer1' in name.lower():
                grad_norm = param.grad.abs().mean().item()
                print(f"  {name:<50} grad_mean: {grad_norm:.6f}")
    
    # 清理
    model.zero_grad()


