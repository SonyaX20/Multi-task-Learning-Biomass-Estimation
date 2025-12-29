# -*- coding: utf-8 -*-
"""
重建式预训练 (Reconstruction-based Pretraining)
使用 Masked Autoencoder 进行自监督学习

策略:
1. Pretraining: 随机mask输入patches，重建原始图像
2. Fine-tuning: 使用预训练的encoder进行分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Colab环境配置
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/data/"
except:
    BASE_DIR = "/Users/siyux1927/local/thesis0926/data/"

"""### Dataset for Pretraining (无需标签)"""

class H5PretrainDataset(Dataset):
    """预训练数据集：只加载图像，不需要标签"""
    def __init__(self, h5_path: str):
        self.h5f = h5py.File(h5_path, 'r')
        self.images = self.h5f['images']
        self.valid_indices = self._filter_nan_samples()
        self.n_samples = len(self.valid_indices)
        
        print(f"Pretrain Dataset: {h5_path}")
        print(f"  Total samples: {len(self.images)}")
        print(f"  Valid samples: {self.n_samples}")
    
    def _filter_nan_samples(self):
        valid_indices = []
        for idx in range(len(self.h5f['images'])):
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
        
        vv = torch.from_numpy(input_bands[0])  # [H, W]
        vh = torch.from_numpy(input_bands[1])  # [H, W]
        image = torch.stack([vv, vh], dim=0)   # [2, H, W]
        
        return image  # 只返回图像，不返回标签
    
    def close(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()
    
    def __del__(self):
        self.close()

"""### Masked Autoencoder Model"""

class PatchEmbedding(nn.Module):
    """将图像分割成patches并嵌入"""
    def __init__(self, img_size=6, patch_size=2, in_channels=2, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 9 patches for 6x6 image
        
        # 使用卷积进行patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, C, H, W] = [B, 2, 6, 6]
        x = self.proj(x)  # [B, embed_dim, 3, 3]
        x = x.flatten(2)  # [B, embed_dim, 9]
        x = x.transpose(1, 2)  # [B, 9, embed_dim]
        return x


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for SAR images
    
    架构:
    - Encoder: 将visible patches编码
    - Decoder: 重建所有patches (包括masked)
    - Mask策略: 随机mask 50-75%的patches
    """
    def __init__(self, img_size=6, patch_size=2, in_channels=2, 
                 embed_dim=128, encoder_depth=4, decoder_depth=2,
                 num_heads=4, mlp_ratio=4.0, mask_ratio=0.5):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Encoder: Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # Decoder: 更轻量的Transformer
        decoder_embed_dim = embed_dim // 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=num_heads // 2,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, 
                                      patch_size * patch_size * in_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
    
    def random_masking(self, x, mask_ratio):
        """
        随机mask patches
        Args:
            x: [B, N, D]
            mask_ratio: mask的比例
        Returns:
            x_masked: [B, N*(1-mask_ratio), D] - 保留的patches
            mask: [B, N] - binary mask (0: keep, 1: remove)
            ids_restore: [B, N] - 用于恢复原始顺序
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # 生成随机噪声用于shuffle
        noise = torch.rand(B, N, device=x.device)
        
        # 排序获得shuffle索引
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前len_keep个
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, 
                               index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 生成binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """
        Encoder forward
        Args:
            x: [B, C, H, W]
        Returns:
            latent: [B, N_visible, D]
            mask: [B, N]
            ids_restore: [B, N]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Encoder
        latent = self.encoder(x)
        
        return latent, mask, ids_restore
    
    def forward_decoder(self, latent, ids_restore):
        """
        Decoder forward
        Args:
            latent: [B, N_visible, D]
            ids_restore: [B, N]
        Returns:
            pred: [B, N, patch_size^2 * C]
        """
        # Embed tokens
        x = self.decoder_embed(latent)
        
        # Append mask tokens
        B, N_visible, D = x.shape
        N = self.num_patches
        mask_tokens = self.mask_token.repeat(B, N - N_visible, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # [B, N, D]
        
        # Unshuffle
        x_full = torch.gather(x_full, dim=1, 
                             index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Add positional embedding
        x_full = x_full + self.decoder_pos_embed
        
        # Decoder
        x = self.decoder(x_full)
        
        # Prediction
        pred = self.decoder_pred(x)
        
        return pred
    
    def forward_loss(self, imgs, pred, mask):
        """
        计算重建损失 (只在masked patches上)
        Args:
            imgs: [B, C, H, W]
            pred: [B, N, patch_size^2 * C]
            mask: [B, N] (0: visible, 1: masked)
        """
        # Patchify target
        target = self.patchify(imgs)  # [B, N, patch_size^2 * C]
        
        # 归一化到每个patch
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        
        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        
        # 只计算masked patches的loss
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def patchify(self, imgs):
        """
        将图像转换为patches
        Args:
            imgs: [B, C, H, W]
        Returns:
            patches: [B, N, patch_size^2 * C]
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = w = H // p
        
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum('bchpwq->bhwpqc', x)
        patches = x.reshape(B, h * w, p * p * C)
        
        return patches
    
    def unpatchify(self, patches):
        """
        将patches还原为图像
        Args:
            patches: [B, N, patch_size^2 * C]
        Returns:
            imgs: [B, C, H, W]
        """
        B, N, _ = patches.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        
        x = patches.reshape(B, h, w, p, p, self.in_channels)
        x = torch.einsum('bhwpqc->bchpwq', x)
        imgs = x.reshape(B, self.in_channels, h * p, w * p)
        
        return imgs
    
    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass
        Args:
            imgs: [B, C, H, W]
            mask_ratio: mask比例，如果为None使用self.mask_ratio
        Returns:
            loss, pred, mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask
    
    def get_encoder(self):
        """获取encoder用于下游分类任务"""
        return PretrainedEncoder(self.patch_embed, self.pos_embed, 
                                self.encoder, self.num_patches)


class PretrainedEncoder(nn.Module):
    """从MAE提取的encoder，用于分类"""
    def __init__(self, patch_embed, pos_embed, encoder, num_patches):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.encoder = encoder
        self.num_patches = num_patches
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [B, num_patches, embed_dim]
        """
        x = self.patch_embed(x)  # [B, N, D]
        x = x + self.pos_embed   # Add positional embedding
        x = self.encoder(x)       # [B, N, D]
        return x


"""### Pretraining Functions"""

def train_pretrain_epoch(model, dataloader, optimizer, device, epoch):
    """预训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}")
    
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        loss, pred, mask = model(images)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/n_batches:.4f}'
        })
    
    return total_loss / n_batches


def visualize_reconstruction(model, dataloader, device, epoch, save_dir='./pretrain_viz'):
    """可视化重建效果"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # 获取一个batch
        images = next(iter(dataloader))
        images = images.to(device)
        
        # 只取前4个样本
        images = images[:4]
        
        # Forward
        loss, pred, mask = model(images)
        
        # Unpatchify predictions
        pred_imgs = model.unpatchify(pred)
        
        # 创建可视化
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        
        for i in range(4):
            # Original
            axes[i, 0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title(f'Original (VV)')
            axes[i, 0].axis('off')
            
            # Masked
            masked_img = images[i].clone()
            mask_2d = mask[i].reshape(3, 3).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            masked_img[:, mask_2d == 1] = 0
            axes[i, 1].imshow(masked_img[0].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title(f'Masked ({model.mask_ratio*100:.0f}%)')
            axes[i, 1].axis('off')
            
            # Reconstructed
            axes[i, 2].imshow(pred_imgs[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 2].set_title('Reconstructed')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_epoch_{epoch+1}.png', dpi=150)
        plt.close()
        
        print(f"Visualization saved to {save_dir}/reconstruction_epoch_{epoch+1}.png")


"""### Main Pretraining Script"""

def pretrain_mae(train_h5, test_h5, epochs=50, batch_size=128, 
                 mask_ratio=0.5, save_path='pretrained_mae.pth'):
    """
    预训练Masked Autoencoder
    
    Args:
        train_h5: 训练数据路径
        test_h5: 验证数据路径
        epochs: 训练轮数
        batch_size: batch大小
        mask_ratio: mask比例
        save_path: 保存路径
    """
    print(f"\n{'='*70}")
    print("Masked Autoencoder Pretraining")
    print(f"{'='*70}")
    
    # 创建数据集
    train_dataset = H5PretrainDataset(train_h5)
    val_dataset = H5PretrainDataset(test_h5)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    
    # 创建模型
    model = MaskedAutoencoder(
        img_size=6,
        patch_size=2,
        in_channels=2,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=2,
        num_heads=4,
        mask_ratio=mask_ratio
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Mask ratio: {mask_ratio*100:.0f}%")
    print(f"  Num patches: {model.num_patches}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # 训练
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\n{'='*70}")
    print("Starting Pretraining")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_pretrain_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for images in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                loss, _, _ = model(images)
                val_loss += loss.item()
                n_batches += 1
        
        val_loss = val_loss / n_batches
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,
                'config': {
                    'img_size': 6,
                    'patch_size': 2,
                    'in_channels': 2,
                    'embed_dim': 128,
                    'encoder_depth': 4,
                    'decoder_depth': 2,
                    'num_heads': 4,
                    'mask_ratio': mask_ratio
                }
            }, save_path)
            print(f"  ✓ Best model saved! Val Loss: {best_loss:.4f}")
        
        # 每10个epoch可视化一次
        if (epoch + 1) % 10 == 0:
            visualize_reconstruction(model, val_loader, device, epoch)
    
    print(f"\n{'='*70}")
    print("Pretraining Complete!")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretraining Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pretrain_loss_curve.png', dpi=150)
    plt.show()
    
    return model, history


# 运行预训练
if __name__ == "__main__":
    model, history = pretrain_mae(
        train_h5=BASE_DIR + 'train_data.h5',
        test_h5=BASE_DIR + 'test_data.h5',
        epochs=50,
        batch_size=128,
        mask_ratio=0.5,
        save_path='pretrained_mae.pth'
    )


