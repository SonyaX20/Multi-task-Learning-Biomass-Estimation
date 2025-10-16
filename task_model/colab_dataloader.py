"""
用于Google Colab的数据加载模块
便于在Colab中按cell逐步运行
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json


# ============================================================================
# 1. 加载HDF5数据和元数据
# ============================================================================

def load_h5_info(h5_path):
    """
    加载HDF5文件的基本信息
    
    Args:
        h5_path: HDF5文件路径
    
    Returns:
        info字典包含数据集的基本信息
    """
    with h5py.File(h5_path, 'r') as h5f:
        info = {
            'n_samples': h5f.attrs['n_samples'],
            'n_bands': h5f.attrs['n_bands'],
            'height': h5f.attrs['height'],
            'width': h5f.attrs['width'],
            'n_classes': h5f.attrs['n_classes'],
            'datasets': list(h5f.keys())
        }
    
    print(f"HDF5文件信息: {h5_path}")
    print(f"  样本数: {info['n_samples']}")
    print(f"  波段数: {info['n_bands']}")
    print(f"  图像尺寸: {info['height']} x {info['width']}")
    print(f"  类别数: {info['n_classes']}")
    print(f"  数据集: {info['datasets']}")
    
    return info


def load_label_mapping(mapping_path):
    """加载标签映射"""
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    print(f"标签映射:")
    print(f"  类别数: {mapping['n_classes']}")
    print(f"  示例标签: {list(mapping['label_to_idx'].items())[:5]}")
    
    return mapping


# ============================================================================
# 2. 数据统计函数（可选）
# ============================================================================

def compute_data_stats(h5_path, n_samples=1000):
    """
    计算数据集的统计信息（用于了解数据分布）
    
    Args:
        h5_path: HDF5文件路径
        n_samples: 采样数量
    
    Returns:
        stats: 包含每个波段统计信息的字典
    """
    print(f"计算数据统计信息 (采样 {n_samples} 个样本)...")
    
    with h5py.File(h5_path, 'r') as h5f:
        images = h5f['images']
        n_bands = h5f.attrs['n_bands']
        total_samples = h5f.attrs['n_samples']
        
        # 采样
        n_samples = min(n_samples, total_samples)
        indices = np.random.choice(total_samples, n_samples, replace=False)
        
        stats = {}
        band_names = ['CHM', 'VV', 'VH', 'VV/VH']
        
        for band_idx in range(n_bands):
            band_values = []
            nan_count = 0
            
            for idx in indices:
                band_data = images[idx][band_idx]
                nan_count += np.isnan(band_data).sum()
                valid_data = band_data[~np.isnan(band_data)]
                if len(valid_data) > 0:
                    band_values.extend(valid_data.flatten())
            
            if len(band_values) > 0:
                values = np.array(band_values)
                stats[band_names[band_idx]] = {
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'nan_count': int(nan_count)
                }
            else:
                stats[band_names[band_idx]] = {
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'nan_count': int(nan_count)
                }
        
        # 打印结果
        print("\n数据统计信息:")
        for name, stat in stats.items():
            if stat['mean'] is not None:
                print(f"  {name:8s}: min={stat['min']:8.2f}, max={stat['max']:8.2f}, "
                      f"mean={stat['mean']:8.2f}, std={stat['std']:8.2f}, "
                      f"NaN={stat['nan_count']:8d}")
            else:
                print(f"  {name:8s}: 无有效数据")
    
    return stats


# ============================================================================
# 3. Dataset类（支持回归任务）
# ============================================================================

class RegressionH5Dataset(Dataset):
    """
    回归任务数据集
    - 输入: VV + VH (Band 1, 2)
    - 输出: CHM (Band 0)
    - 保留NaN值，使用masked loss处理
    """
    
    def __init__(self, h5_path):
        """
        Args:
            h5_path: HDF5文件路径
        """
        self.h5f = h5py.File(h5_path, 'r')
        self.images = self.h5f['images']
        self.n_samples = self.images.shape[0]
        
        print(f"Regression Dataset loaded: {h5_path}")
        print(f"  Samples: {self.n_samples}")
        print(f"  Input: VV + VH (2 channels)")
        print(f"  Output: CHM (1 channel)")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 读取数据（保留NaN值）
        img = self.images[idx].astype(np.float32).copy()  # (4, H, W)
        
        # Band 0: CHM (目标)
        # Band 1: VV (输入)
        # Band 2: VH (输入)
        # Band 3: VV/VH (不使用)
        
        input_bands = img[1:3]  # (2, H, W) - VV和VH
        target_band = img[0:1]  # (1, H, W) - CHM
        
        # 转换为tensor
        input_tensor = torch.from_numpy(input_bands)
        target_tensor = torch.from_numpy(target_band)
        
        return input_tensor, target_tensor
    
    def close(self):
        """关闭HDF5文件"""
        if hasattr(self, 'h5f'):
            self.h5f.close()
    
    def __del__(self):
        self.close()


# ============================================================================
# 4. 创建DataLoader（回归任务）
# ============================================================================

def create_dataloaders(train_h5, val_h5, batch_size=32, num_workers=2):
    """
    创建训练和验证DataLoader（回归任务：VV+VH -> CHM）
    
    注意：
        - 输入: VV + VH (2 channels)
        - 输出: CHM (1 channel)
        - 保留NaN值，使用masked loss处理
    
    Args:
        train_h5: 训练集HDF5路径
        val_h5: 验证集HDF5路径（使用测试集作为验证集）
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader
    """
    # 创建数据集
    train_dataset = RegressionH5Dataset(train_h5)
    val_dataset = RegressionH5Dataset(val_h5)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader创建完成 (回归任务):")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} batches")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_loader)} batches")
    print(f"  ⚠️  使用masked loss处理NaN值")
    
    return train_loader, val_loader


# ============================================================================
# 5. Masked Loss函数
# ============================================================================

class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss - 忽略NaN值的均方误差损失
    """
    
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 (batch, 1, H, W)
            target: 目标值 (batch, 1, H, W)
        
        Returns:
            loss: masked MSE loss
        """
        # 创建mask：非NaN且非Inf的位置
        mask = ~torch.isnan(target) & ~torch.isinf(target) & ~torch.isnan(pred) & ~torch.isinf(pred)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 只计算有效位置的loss
        loss = ((pred[mask] - target[mask]) ** 2).mean()
        
        return loss


class MaskedRMSELoss(nn.Module):
    """
    Masked RMSE Loss - 忽略NaN值的均方根误差损失
    """
    
    def __init__(self):
        super(MaskedRMSELoss, self).__init__()
        self.mse_loss = MaskedMSELoss()
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        return torch.sqrt(mse + 1e-8)  # 添加小值避免梯度消失


def compute_masked_rmse(pred, target):
    """
    计算masked RMSE（用于评估）
    
    Args:
        pred: 预测值
        target: 目标值
    
    Returns:
        rmse: float
    """
    mask = ~torch.isnan(target) & ~torch.isinf(target) & ~torch.isnan(pred) & ~torch.isinf(pred)
    
    if mask.sum() == 0:
        return 0.0
    
    mse = ((pred[mask] - target[mask]) ** 2).mean()
    rmse = torch.sqrt(mse).item()
    
    return rmse


def compute_masked_r2(pred, target):
    """
    计算masked R² score（用于评估）
    
    Args:
        pred: 预测值
        target: 目标值
    
    Returns:
        r2: float
    """
    mask = ~torch.isnan(target) & ~torch.isinf(target) & ~torch.isnan(pred) & ~torch.isinf(pred)
    
    if mask.sum() == 0:
        return 0.0
    
    pred_valid = pred[mask]
    target_valid = target[mask]
    
    # R² = 1 - SS_res / SS_tot
    ss_res = ((target_valid - pred_valid) ** 2).sum()
    ss_tot = ((target_valid - target_valid.mean()) ** 2).sum()
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    
    return r2.item()


# ============================================================================
# 6. UNet模型定义（回归任务）
# ============================================================================

def conv_block(in_ch, out_ch):
    """卷积块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class RegressionUNet(nn.Module):
    """
    UNet用于回归任务
    输入: (batch, 2, H, W) - VV + VH
    输出: (batch, 1, H, W) - CHM预测
    """
    
    def __init__(self, in_channels=2, out_channels=1):
        super(RegressionUNet, self).__init__()
        
        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        # Output - 回归输出（CHM）
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output - 直接输出feature map（回归任务）
        out = self.out(d1)
        
        return out


def create_model(in_channels=2, out_channels=1):
    """创建回归模型并打印信息"""
    model = RegressionUNet(in_channels=in_channels, out_channels=out_channels)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"回归模型创建完成:")
    print(f"  输入通道: {in_channels} (VV + VH)")
    print(f"  输出通道: {out_channels} (CHM)")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return model


# ============================================================================
# 7. 训练辅助函数（回归任务）
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch - 回归任务
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: Masked RMSE loss
        optimizer: 优化器
        device: 设备
    
    Returns:
        epoch_loss: 平均损失
        epoch_rmse: RMSE
        epoch_r2: R²
    """
    model.train()
    running_loss = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    n_batches = 0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 检查loss是否为NaN
        if torch.isnan(loss):
            print("警告: Loss为NaN，跳过此batch")
            continue
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算RMSE和R²
        rmse = compute_masked_rmse(outputs, targets)
        r2 = compute_masked_r2(outputs, targets)
        
        total_rmse += rmse
        total_r2 += r2
        n_batches += 1
    
    epoch_loss = running_loss / n_batches if n_batches > 0 else 0
    epoch_rmse = total_rmse / n_batches if n_batches > 0 else 0
    epoch_r2 = total_r2 / n_batches if n_batches > 0 else 0
    
    return epoch_loss, epoch_rmse, epoch_r2


def evaluate(model, val_loader, criterion, device):
    """
    评估模型 - 回归任务
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: Masked RMSE loss
        device: 设备
    
    Returns:
        epoch_loss: 平均损失
        epoch_rmse: RMSE
        epoch_r2: R²
    """
    model.eval()
    running_loss = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if not torch.isnan(loss):
                running_loss += loss.item()
            
            # 计算RMSE和R²
            rmse = compute_masked_rmse(outputs, targets)
            r2 = compute_masked_r2(outputs, targets)
            
            total_rmse += rmse
            total_r2 += r2
            n_batches += 1
    
    epoch_loss = running_loss / n_batches if n_batches > 0 else 0
    epoch_rmse = total_rmse / n_batches if n_batches > 0 else 0
    epoch_r2 = total_r2 / n_batches if n_batches > 0 else 0
    
    return epoch_loss, epoch_rmse, epoch_r2


# ============================================================================
# 8. 历史记录和可视化
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """
    绘制训练历史
    
    Args:
        history: 包含训练历史的字典
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (RMSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # RMSE
    axes[1].plot(epochs, history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
    axes[1].plot(epochs, history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Training and Validation RMSE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].plot(epochs, history['train_r2'], 'b-', label='Train R²', linewidth=2)
    axes[2].plot(epochs, history['val_r2'], 'r-', label='Val R²', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('R² Score', fontsize=12)
    axes[2].set_title('Training and Validation R²', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    plt.show()
    
    return fig


# ============================================================================
# 7. 可视化函数
# ============================================================================

def visualize_sample(images, labels, idx=0, label_mapping=None, figsize=(16, 4)):
    """
    可视化一个样本的4个波段
    
    Args:
        images: tensor, shape (batch, 4, H, W)
        labels: tensor, shape (batch,)
        idx: 要可视化的样本索引
        label_mapping: 标签映射字典（可选）
        figsize: 图像大小
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 获取单个样本
    img = images[idx].cpu().numpy()  # (4, H, W)
    label = labels[idx].item()
    
    # 波段名称
    band_names = ['CHM (Canopy Height)', 'VV (Vertical-Vertical)', 
                  'VH (Vertical-Horizontal)', 'VV/VH (Ratio)']
    
    # 创建子图
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    for i in range(4):
        band = img[i]
        
        # 处理NaN值用于显示
        band_display = band.copy()
        nan_mask = np.isnan(band_display)
        
        # 显示图像
        im = axes[i].imshow(band_display, cmap='viridis')
        axes[i].set_title(f'{band_names[i]}\n(Band {i})', fontsize=10)
        axes[i].axis('off')
        
        # 添加colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # 显示统计信息
        valid_data = band[~nan_mask]
        if len(valid_data) > 0:
            stats_text = f'Min: {valid_data.min():.2f}\n'
            stats_text += f'Max: {valid_data.max():.2f}\n'
            stats_text += f'Mean: {valid_data.mean():.2f}\n'
            stats_text += f'NaN: {nan_mask.sum()}/{band.size}'
        else:
            stats_text = 'All NaN'
        
        axes[i].text(0.02, 0.98, stats_text, 
                    transform=axes[i].transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 获取标签名称
    if label_mapping and 'idx_to_label' in label_mapping:
        label_name = label_mapping['idx_to_label'].get(str(label), f'Class {label}')
    else:
        label_name = f'Class {label}'
    
    plt.suptitle(f'Sample {idx} - Label: {label_name} ({label})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_batch(images, labels, n_samples=4, label_mapping=None):
    """
    可视化batch中的多个样本
    
    Args:
        images: tensor, shape (batch, 4, H, W)
        labels: tensor, shape (batch,)
        n_samples: 要显示的样本数量
        label_mapping: 标签映射字典（可选）
    """
    import matplotlib.pyplot as plt
    
    batch_size = images.shape[0]
    n_samples = min(n_samples, batch_size)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    band_names = ['CHM', 'VV', 'VH', 'VV/VH']
    
    for sample_idx in range(n_samples):
        img = images[sample_idx].cpu().numpy()
        label = labels[sample_idx].item()
        
        # 获取标签名称
        if label_mapping and 'idx_to_label' in label_mapping:
            label_name = label_mapping['idx_to_label'].get(str(label), f'Class {label}')
        else:
            label_name = f'Class {label}'
        
        for band_idx in range(4):
            ax = axes[sample_idx, band_idx]
            band = img[band_idx]
            
            # 显示
            im = ax.imshow(band, cmap='viridis')
            
            if sample_idx == 0:
                ax.set_title(band_names[band_idx], fontsize=10)
            
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 第一列显示样本标签
            if band_idx == 0:
                ax.text(-0.1, 0.5, f'{label_name}', 
                       transform=ax.transAxes, rotation=90,
                       fontsize=10, verticalalignment='center',
                       fontweight='bold')
    
    plt.suptitle(f'Batch Visualization ({n_samples} samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_band_histograms(images, idx=0):
    """
    绘制一个样本4个波段的直方图
    
    Args:
        images: tensor, shape (batch, 4, H, W)
        idx: 样本索引
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    img = images[idx].cpu().numpy()
    band_names = ['CHM', 'VV', 'VH', 'VV/VH']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i in range(4):
        band = img[i]
        valid_data = band[~np.isnan(band)]
        
        if len(valid_data) > 0:
            axes[i].hist(valid_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{band_names[i]}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # 显示统计信息
            stats_text = f'Min: {valid_data.min():.2f}\n'
            stats_text += f'Max: {valid_data.max():.2f}\n'
            stats_text += f'Mean: {valid_data.mean():.2f}\n'
            stats_text += f'Std: {valid_data.std():.2f}'
            
            axes[i].text(0.02, 0.98, stats_text,
                        transform=axes[i].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[i].text(0.5, 0.5, 'All NaN', 
                        transform=axes[i].transAxes,
                        ha='center', va='center', fontsize=14)
    
    plt.suptitle(f'Sample {idx} - Band Histograms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# 9. 使用说明（回归任务：VV+VH -> CHM）
# ============================================================================

"""
在Google Colab中使用本模块进行回归训练 (VV+VH预测CHM):

# Cell 1: 安装依赖
!pip install h5py -q

# Cell 2: 挂载Google Drive并复制文件
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/tree_species_unet/*.h5 .
!cp /content/drive/MyDrive/tree_species_unet/*.json .
!cp /content/drive/MyDrive/tree_species_unet/colab_dataloader.py .

# Cell 3: 导入模块
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from colab_dataloader import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Cell 4: 创建DataLoader（回归任务：VV+VH -> CHM）
# 使用测试集作为验证集
train_loader, val_loader = create_dataloaders(
    train_h5='train_data.h5',
    val_h5='test_data.h5',  # 使用测试集作为验证集
    batch_size=32,
    num_workers=2
)

# Cell 5: 测试数据加载
for inputs, targets in train_loader:
    print(f"Input shape: {inputs.shape}")    # (batch, 2, H, W) - VV+VH
    print(f"Target shape: {targets.shape}")  # (batch, 1, H, W) - CHM
    print(f"Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"Input NaN: {torch.isnan(inputs).sum().item()}")
    print(f"Target NaN: {torch.isnan(targets).sum().item()}")
    break

# Cell 6: 创建回归模型
model = create_model(in_channels=2, out_channels=1)
model = model.to(device)

# Cell 7: 设置训练参数（使用Masked RMSE Loss）
criterion = MaskedRMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5, verbose=True
)

# Cell 8: 训练循环（带历史记录）
num_epochs = 50
best_rmse = float('inf')

# 历史记录
history = {
    'train_loss': [],
    'train_rmse': [],
    'train_r2': [],
    'val_loss': [],
    'val_rmse': [],
    'val_r2': []
}

print("="*80)
print("开始训练：回归任务 (VV+VH -> CHM)")
print("="*80)

for epoch in range(num_epochs):
    # 训练
    train_loss, train_rmse, train_r2 = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # 验证
    val_loss, val_rmse, val_r2 = evaluate(
        model, val_loader, criterion, device
    )
    
    # 学习率调整（基于验证RMSE）
    scheduler.step(val_rmse)
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_rmse'].append(train_rmse)
    history['train_r2'].append(train_r2)
    history['val_loss'].append(val_loss)
    history['val_rmse'].append(val_rmse)
    history['val_r2'].append(val_r2)
    
    # 保存最佳模型
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }, 'best_model.pth')
        print(f"✓ Saved best model (RMSE: {val_rmse:.4f}, R²: {val_r2:.4f})")
    
    # 打印进度
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train: Loss={train_loss:.4f}, RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
    print(f"  Val:   Loss={val_loss:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}")
    print(f"  Best RMSE: {best_rmse:.4f}")
    print("-"*80)

print(f"\\n训练完成! 最佳RMSE: {best_rmse:.4f}")

# Cell 9: 绘制训练历史
plot_training_history(history, save_path='training_history.png')

# Cell 10: 加载最佳模型并最终评估
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

val_loss, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)

print("\\n"+"="*80)
print("最终评估结果:")
print("="*80)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation R²:   {val_r2:.4f}")
print("="*80)


# ============================================================================
# 说明:
# ============================================================================

本代码实现了回归任务：使用VV和VH两个SAR波段预测CHM（冠层高度模型）

关键特性:
1. 使用Masked Loss函数（MaskedRMSELoss）自动忽略NaN值
2. 评估指标：RMSE和R² score
3. 自动保存最佳模型（基于最低验证RMSE）
4. 绘制训练历史（Loss、RMSE、R²的3张图）
5. 使用测试集作为验证集

数据流:
- 输入: Band 1 (VV) + Band 2 (VH) -> (batch, 2, H, W)
- 输出: Band 0 (CHM) -> (batch, 1, H, W)
- Band 3 (VV/VH) 不使用

模型: UNet架构，适合像素级回归任务
损失: Masked RMSE - 自动忽略NaN像素
"""

