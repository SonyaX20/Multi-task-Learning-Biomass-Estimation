"""
Visualization module for classification evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from inference import get_class_names


def plot_confusion_matrix(cm, label_mapping=None, num_classes=None, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: confusion matrix (numpy array)
        label_mapping: optional label mapping dict
        num_classes: number of classes
        title: plot title
    """
    if num_classes is None:
        num_classes = cm.shape[0]
    
    class_names = get_class_names(num_classes, label_mapping)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_per_class_accuracy_and_counts(metrics, save_path=None):
    """
    Plot per-class accuracy (line) and sample counts (bars).
    
    Args:
        metrics: dict from compute_per_class_metrics()
        save_path: optional path to save figure
    """
    cm = metrics['confusion_matrix']
    class_acc = metrics['class_accuracy']
    class_counts = metrics['class_counts']
    pred_counts = metrics['pred_counts']
    class_names = metrics['class_names']
    
    num_classes = len(class_names)
    
    # Bar chart parameters
    bar_width = 0.35
    group_width = bar_width * 2
    group_spacing = 0.2
    
    # Calculate x positions
    x_positions = []
    for i in range(num_classes):
        start_pos = i * (group_width + group_spacing)
        x_positions.append([start_pos, start_pos + bar_width])
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(max(14, num_classes * 0.9), 7))
    
    # Plot bars (true and pred counts)
    colors_bar = ['#3498db', '#e74c3c']  # true: blue, pred: red
    for i in range(num_classes):
        values = [class_counts[i], pred_counts[i]]
        for j, (val, color) in enumerate(zip(values, colors_bar)):
            ax1.bar(
                x_positions[i][j], val, bar_width, color=color, alpha=0.7,
                label='True' if i == 0 and j == 0 else ('Pred' if i == 0 and j == 1 else '')
            )
    
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Sample Count', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks([(pos[0] + pos[1]) / 2 for pos in x_positions])
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend(['True', 'Pred'], loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Create second y-axis for accuracy line
    ax2 = ax1.twinx()
    line_x_positions = [(pos[0] + pos[1]) / 2 for pos in x_positions]
    
    # Plot accuracy line
    ax2.plot(
        line_x_positions, class_acc * 100, 'o-',
        color='#2ecc71', linewidth=2.5, markersize=8,
        label='Accuracy', zorder=5
    )
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.set_ylim([0, 105])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.title('Per-Class Accuracy and Sample Counts', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

