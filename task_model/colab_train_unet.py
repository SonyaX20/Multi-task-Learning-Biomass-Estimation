"""
Google Colab训练脚本
使用UNet进行树种分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 导入我们的dataloader
from colab_dataloader import create_dataloaders, SimpleUNet, load_label_mapping


class UNetTrainer:
    """UNet训练器"""
    
    def __init__(self, model, train_loader, test_loader, label_mapping, 
                 device='cuda', learning_rate=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.label_mapping = label_mapping
        self.device = device
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5, verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rate': []
        }
        
        self.best_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{running_loss/(pbar.n+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, n_epochs=50, save_dir='./models'):
        """训练模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*80)
        print("开始训练")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {n_epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Classes: {self.label_mapping['n_classes']}")
        print("="*80 + "\n")
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 评估
            test_loss, test_acc, preds, labels = self.evaluate()
            
            # 学习率调整
            self.scheduler.step(test_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['learning_rate'].append(current_lr)
            
            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_acc': test_acc,
                    'train_acc': train_acc
                }, save_dir / 'best_model.pth')
                print(f"✓ Saved best model (acc: {test_acc:.2f}%)")
            
            # 打印结果
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Best Acc: {self.best_acc:.2f}%")
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✓ Restored best model (acc: {self.best_acc:.2f}%)")
        
        # 保存训练历史
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def plot_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['test_loss'], label='Test Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Test Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['test_acc'], label='Test Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Test Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
        plt.show()
    
    def generate_classification_report(self, save_dir='./models'):
        """生成分类报告"""
        save_dir = Path(save_dir)
        
        print("\nGenerating classification report...")
        _, _, preds, labels = self.evaluate()
        
        # 获取标签名称
        idx_to_label = self.label_mapping['idx_to_label']
        label_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
        
        # 分类报告
        report = classification_report(
            labels, preds, 
            target_names=label_names,
            output_dict=True
        )
        
        # 保存报告
        with open(save_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印报告
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=label_names))
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                   xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_dir / 'confusion_matrix.png'}")
        plt.show()
        
        return report


def main():
    """主训练函数"""
    
    # ========== 配置 ==========
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # ========== 加载数据 ==========
    print("\nLoading data...")
    train_loader, test_loader, train_ds, test_ds = create_dataloaders(
        train_h5_path='train_data.h5',
        test_h5_path='test_data.h5',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize=True,
        handle_nan='zero'
    )
    
    # 加载标签映射
    label_mapping = load_label_mapping('label_mapping.json')
    n_classes = label_mapping['n_classes']
    
    print(f"\nDataset info:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # ========== 创建模型 ==========
    print("\nCreating model...")
    model = SimpleUNet(in_channels=4, n_classes=n_classes)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ========== 训练 ==========
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        label_mapping=label_mapping,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    history = trainer.train(n_epochs=NUM_EPOCHS, save_dir='./models')
    
    # ========== 可视化和评估 ==========
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    # 绘制训练历史
    trainer.plot_history('models/training_history.png')
    
    # 生成分类报告
    trainer.generate_classification_report('models')
    
    # 关闭数据集
    train_ds.close()
    test_ds.close()
    
    print("\n✓ All done!")
    print(f"Best test accuracy: {trainer.best_acc:.2f}%")
    print(f"Models saved to: ./models/")


if __name__ == '__main__':
    main()

