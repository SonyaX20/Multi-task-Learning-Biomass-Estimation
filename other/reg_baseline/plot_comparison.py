import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_results(base_dir):
    """
    遍历 reg_baseline 下的所有子文件夹，加载 training_results.json
    返回一个字典，键为实验名称，值为数据字典
    """
    results = {}
    base_path = Path(base_dir)
    
    # 遍历所有子文件夹
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            json_path = subdir / "training_results.json"
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        results[subdir.name] = data
                        print(f"成功加载: {subdir.name}")
                except Exception as e:
                    print(f"加载 {subdir.name} 时出错: {e}")
    
    return results

def extract_metrics(results):
    """
    从结果中提取 train_loss, val_loss, r_2 (val_r2)
    返回一个字典，包含所有实验的指标
    """
    metrics = {}
    
    for exp_name, data in results.items():
        history = data.get('history', {})
        
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_r2 = history.get('val_r2', [])
        
        metrics[exp_name] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'r_2': val_r2
        }
        
        print(f"{exp_name}: train_loss={len(train_loss)} epochs, "
              f"val_loss={len(val_loss)} epochs, r_2={len(val_r2)} epochs")
    
    return metrics

def plot_single_metric(metrics, metric_key, file_name_prefix, ylabel, title_name, output_path, max_epoch=80, zoom_start=20, zoom_end=80):
    """
    绘制单个指标的对比图（完整图和放大图）
    
    Args:
        metrics: 指标数据字典
        metric_key: 指标键名（如 'train_loss', 'val_loss'）
        file_name_prefix: 文件名前缀（如 'train_loss', 'r2'）
        ylabel: Y轴标签
        title_name: 图表标题名称
        output_path: 输出路径
        max_epoch: 最大显示的epoch数（默认80）
        zoom_start: 放大图的起始epoch（默认20）
        zoom_end: 放大图的结束epoch（默认80）
    """
    # 1. 绘制完整图（截取到max_epoch）
    plt.figure(figsize=(12, 6))
    for exp_name, data in metrics.items():
        values = data[metric_key]
        # 截取到max_epoch
        values_truncated = values[:max_epoch]
        epochs = range(1, len(values_truncated) + 1)
        plt.plot(epochs, values_truncated, label=exp_name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title_name} Comparison (Epochs 1-{max_epoch})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_epoch + 1)
    plt.tight_layout()
    plt.savefig(output_path / f'{file_name_prefix}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {file_name_prefix}_comparison.png")
    plt.close()
    
    # 2. 绘制放大图（epoch zoom_start 到 zoom_end）
    plt.figure(figsize=(12, 6))
    for exp_name, data in metrics.items():
        values = data[metric_key]
        # 截取到zoom_end
        values_truncated = values[:zoom_end]
        # 只取zoom_start到zoom_end之间的数据
        if len(values_truncated) >= zoom_start:
            values_zoomed = values_truncated[zoom_start-1:zoom_end]
            epochs = range(zoom_start, zoom_start + len(values_zoomed))
            plt.plot(epochs, values_zoomed, label=exp_name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title_name} Comparison (Epochs {zoom_start}-{zoom_end})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(zoom_start - 1, zoom_end + 1)
    plt.tight_layout()
    plt.savefig(output_path / f'{file_name_prefix}_comparison_epoch{zoom_start}_{zoom_end}.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {file_name_prefix}_comparison_epoch{zoom_start}_{zoom_end}.png")
    plt.close()

def plot_comparison(metrics, output_dir):
    """
    绘制3张对比图：train_loss, val_loss, r_2
    每张图都有两个版本：完整图（1-80 epoch）和放大图（20-80 epoch）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 定义要绘制的指标
    # 格式: (metric_key, file_name_prefix, ylabel, title_name)
    metrics_to_plot = [
        ('train_loss', 'train_loss', 'Train Loss', 'Train Loss'),
        ('val_loss', 'val_loss', 'Validation Loss', 'Validation Loss'),
        ('r_2', 'r2', 'R² Score', 'R² Score')
    ]
    
    # 绘制每个指标的两张图
    for metric_key, file_name_prefix, ylabel, title_name in metrics_to_plot:
        plot_single_metric(metrics, metric_key, file_name_prefix, ylabel, title_name, output_path, 
                          max_epoch=80, zoom_start=20, zoom_end=80)
    
    print(f"\n所有图表已保存到: {output_path}")

def main():
    # 设置基础目录
    base_dir = Path(__file__).parent  # reg_baseline 目录
    
    print("=" * 50)
    print("开始加载训练结果...")
    print("=" * 50)
    
    # 加载所有实验结果
    results = load_training_results(base_dir)
    
    if not results:
        print("未找到任何 training_results.json 文件！")
        return
    
    print(f"\n找到 {len(results)} 个实验结果")
    print("=" * 50)
    
    # 提取指标
    print("\n提取指标数据...")
    metrics = extract_metrics(results)
    
    # 绘制对比图
    print("\n开始绘制对比图...")
    plot_comparison(metrics, base_dir / "comparison_plots")
    
    print("\n" + "=" * 50)
    print("完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()

