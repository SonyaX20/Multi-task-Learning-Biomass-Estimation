# -*- coding: utf-8 -*-
"""
使用预训练模型进行分类 (Fine-tuning)

流程:
1. 加载预训练的MAE encoder
2. 添加分类头
3. Fine-tune整个模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import json
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Colab环境配置
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/data/"
except:
    BASE_DIR = "/Users/siyux1927/local/thesis0926/data/"

"""### 导入预训练模型组件"""

# 从pretrain_reconstruction.py导入必要的类
from pretrain_reconstruction import PatchEmbedding, PretrainedEncoder, MaskedAutoencoder

"""### Classification Dataset"""

class H5ClassificationDataset(Dataset):
    """分类数据集：加载图像和标签"""
    def __init__(self, h5_path: str):
        self.h5f = h5py.File(h5_path, 'r')
        self.images = self.h5f['images']
        self.labels = self.h5f['labels']
        self.valid_indices = self._filter_nan_samples()
        self.n_samples = len(self.valid_indices)
        
        print(f"Classification Dataset: {h5_path}")
        print(f"  Total samples: {len(self.labels)}")
        print(f"  Valid samples: {self.n_samples}")
    
    def _filter_nan_samples(self):
        valid_indices = []
        for idx in range(len(self.h5f['labels'])):
            img = self.images[idx].astype(np.float32)
            input_bands = img[1:3]  # VV, VH
            if not np.isnan(input_bands).any():
                valid_indices.append(idx)
        return valid_indices
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img = self.images[actual_idx].astype(np.float32).copy()
        input_bands = img[1:3]  # VV, VH
        label = int(self.labels[actual_idx])
        
        vv = torch.from_numpy(input_bands[0])  # [H, W]
        vh = torch.from_numpy(input_bands[1])  # [H, W]
        image = torch.stack([vv, vh], dim=0)   # [2, H, W]
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
    def close(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()
    
    def __del__(self):
        self.close()


"""### Classification Model with Pretrained Encoder"""

class PretrainedClassifier(nn.Module):
    """
    使用预训练encoder的分类器
    
    架构:
    - Pretrained Encoder: 从MAE加载
    - Classification Head: 新增的分类层
    """
    def __init__(self, pretrained_encoder, num_patches, embed_dim, num_classes, 
                 freeze_encoder=False, dropout=0.3):
        super().__init__()
        
        self.encoder = pretrained_encoder
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # 是否冻结encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("  Encoder frozen (feature extraction mode)")
        else:
            print("  Encoder trainable (fine-tuning mode)")
        
        # Classification head
        # 方案1: Global average pooling + MLP
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        # Encoder
        features = self.encoder(x)  # [B, N, D]
        
        # Global average pooling
        pooled = features.mean(dim=1)  # [B, D]
        
        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits


def load_pretrained_encoder(checkpoint_path, device):
    """
    从预训练checkpoint加载encoder
    
    Args:
        checkpoint_path: 预训练模型路径
        device: 设备
    
    Returns:
        encoder: PretrainedEncoder
        config: 模型配置
    """
    print(f"\nLoading pretrained model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Pretrained model info:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    # 重建MAE模型
    mae_model = MaskedAutoencoder(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_channels=config['in_channels'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        decoder_depth=config['decoder_depth'],
        num_heads=config['num_heads'],
        mask_ratio=config['mask_ratio']
    )
    
    # 加载权重
    mae_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 提取encoder
    encoder = mae_model.get_encoder()
    
    print(f"  Encoder extracted successfully!")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Num patches: {mae_model.num_patches}")
    
    return encoder, config


"""### Training Functions"""

def compute_accuracy(pred, target):
    pred_labels = torch.argmax(pred, dim=1)
    correct = (pred_labels == target).float()
    return correct.sum().item() / target.size(0)


def compute_map(pred, target, num_classes):
    probs = torch.softmax(pred, dim=1).cpu().numpy()
    target_np = target.cpu().numpy()
    target_onehot = np.eye(num_classes)[target_np]
    
    aps = []
    for i in range(num_classes):
        if target_onehot[:, i].sum() > 0:
            try:
                ap = average_precision_score(target_onehot[:, i], probs[:, i])
                aps.append(ap)
            except:
                continue
    
    return np.mean(aps) if len(aps) > 0 else 0.0


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        acc = compute_accuracy(outputs, labels)
        
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc*100:.1f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_acc = 0
    all_preds = []
    all_targets = []
    n_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = compute_accuracy(outputs, labels)
            
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
            
            all_preds.append(outputs)
            all_targets.append(labels)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc*100:.1f}%'
            })
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mAP = compute_map(all_preds, all_targets, num_classes)
    
    return total_loss / n_batches, total_acc / n_batches, mAP


"""### Main Fine-tuning Script"""

def finetune_classifier(pretrained_path, train_h5, test_h5, num_classes,
                       epochs=30, batch_size=128, freeze_encoder=False,
                       save_path='finetuned_classifier.pth'):
    """
    使用预训练模型进行分类fine-tuning
    
    Args:
        pretrained_path: 预训练模型路径
        train_h5: 训练数据
        test_h5: 测试数据
        num_classes: 类别数
        epochs: fine-tuning轮数
        batch_size: batch大小
        freeze_encoder: 是否冻结encoder
        save_path: 保存路径
    """
    print(f"\n{'='*70}")
    print("Fine-tuning Pretrained Model for Classification")
    print(f"{'='*70}")
    
    # 加载数据
    train_dataset = H5ClassificationDataset(train_h5)
    test_dataset = H5ClassificationDataset(test_h5)
    
    # 划分训练/验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    torch.manual_seed(42)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_subset):,} samples")
    print(f"  Val: {len(val_subset):,} samples")
    print(f"  Test: {len(test_dataset):,} samples")
    
    # 加载预训练encoder
    encoder, config = load_pretrained_encoder(pretrained_path, device)
    encoder = encoder.to(device)
    
    # 创建分类模型
    model = PretrainedClassifier(
        pretrained_encoder=encoder,
        num_patches=9,  # 3x3 patches
        embed_dim=config['embed_dim'],
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Freeze encoder: {freeze_encoder}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 如果encoder冻结，使用更大的学习率
    lr = 5e-4 if freeze_encoder else 1e-4
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr:.0e}")
    print(f"  Batch size: {batch_size}")
    
    # 训练
    best_val_map = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_map': []
    }
    
    print(f"\n{'='*70}")
    print("Starting Fine-tuning")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        
        val_loss, val_acc, val_map = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_map'].append(val_map)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val mAP: {val_map:.4f}")
        
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_map': val_map,
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"  ✓ Best model saved! mAP: {best_val_map:.4f}")
    
    # 测试集评估
    print(f"\n{'='*70}")
    print("Evaluating on Test Set")
    print(f"{'='*70}")
    
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_map = evaluate(
        model, test_loader, criterion, device, num_classes
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  mAP: {test_map:.4f}")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot([x*100 for x in history['train_acc']], label='Train')
    axes[0, 1].plot([x*100 for x in history['val_acc']], label='Val')
    axes[0, 1].set_title('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['val_map'])
    axes[1, 0].set_title('Validation mAP')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5, 
                   f"Best Validation mAP: {best_val_map:.4f}\n"
                   f"Test Accuracy: {test_acc*100:.2f}%\n"
                   f"Test mAP: {test_map:.4f}\n"
                   f"Freeze Encoder: {freeze_encoder}",
                   fontsize=14, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('finetune_results.png', dpi=150)
    plt.show()
    
    return model, history


# 主函数
if __name__ == "__main__":
    # 加载label mapping
    with open(BASE_DIR + 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    n_classes = label_mapping['n_classes']
    
    # Fine-tuning (两种模式对比)
    
    # 模式1: Feature Extraction (冻结encoder)
    print("\n" + "="*70)
    print("MODE 1: Feature Extraction (Frozen Encoder)")
    print("="*70)
    
    model_frozen, history_frozen = finetune_classifier(
        pretrained_path='pretrained_mae.pth',
        train_h5=BASE_DIR + 'train_data.h5',
        test_h5=BASE_DIR + 'test_data.h5',
        num_classes=n_classes,
        epochs=20,
        batch_size=128,
        freeze_encoder=True,
        save_path='finetuned_frozen.pth'
    )
    
    # 模式2: Full Fine-tuning (训练整个模型)
    print("\n" + "="*70)
    print("MODE 2: Full Fine-tuning (Trainable Encoder)")
    print("="*70)
    
    model_finetune, history_finetune = finetune_classifier(
        pretrained_path='pretrained_mae.pth',
        train_h5=BASE_DIR + 'train_data.h5',
        test_h5=BASE_DIR + 'test_data.h5',
        num_classes=n_classes,
        epochs=30,
        batch_size=128,
        freeze_encoder=False,
        save_path='finetuned_full.pth'
    )


