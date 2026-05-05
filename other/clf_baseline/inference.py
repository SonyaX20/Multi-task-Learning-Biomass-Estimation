"""
Inference module for classification model evaluation.
Core function: inference on test set with clean, minimal code.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def inference(model, loader, criterion, device):
    """
    Core inference function: evaluate model on test set.
    
    Args:
        model: PyTorch model
        loader: DataLoader for test set
        criterion: Loss function
        device: Device to run inference on
    
    Returns:
        dict with keys:
            - predictions: np.array of predicted class indices
            - labels: np.array of true class indices
            - logits: torch.Tensor of model outputs
            - avg_loss: float, average loss
            - avg_acc: float, average accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_logits = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference")
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            logits = model(data)
            loss = criterion(logits, target)
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == target).sum().item()
            
            # Accumulate statistics
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size
            
            # Store results
            all_logits.append(logits.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Update progress bar
            batch_acc = correct / batch_size
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{batch_acc*100:.1f}%'
            })
    
    # Aggregate results
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    all_logits = torch.cat(all_logits, dim=0)
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'logits': all_logits,
        'avg_loss': avg_loss,
        'avg_acc': avg_acc
    }


def print_evaluation_summary(results, label_mapping=None):
    """
    Print evaluation summary.
    
    Args:
        results: dict returned from inference()
        label_mapping: optional label mapping dict with 'idx_to_label' key
    """
    predictions = results['predictions']
    labels = results['labels']
    avg_loss = results['avg_loss']
    avg_acc = results['avg_acc']
    
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Overall Loss:     {avg_loss:.4f}")
    print(f"Overall Accuracy: {avg_acc*100:.2f}%")
    print(f"Total Samples:    {len(labels):,}")
    print(f"{'='*60}\n")


def get_class_names(num_classes, label_mapping=None):
    """Get class names from label mapping."""
    if label_mapping and 'idx_to_label' in label_mapping:
        idx_to_label = label_mapping['idx_to_label']
        if isinstance(list(idx_to_label.keys())[0], str):
            idx_to_label = {int(k): v for k, v in idx_to_label.items()}
        return [idx_to_label.get(i, f'Class_{i}') for i in range(num_classes)]
    return [f'Class_{i}' for i in range(num_classes)]


def compute_per_class_metrics(predictions, labels, num_classes, label_mapping=None):
    """
    Compute per-class metrics.
    
    Returns:
        dict with confusion matrix and per-class statistics
    """
    cm = confusion_matrix(labels, predictions)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    class_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    
    class_names = get_class_names(num_classes, label_mapping)
    
    return {
        'confusion_matrix': cm,
        'class_accuracy': class_acc,
        'class_counts': class_counts,
        'pred_counts': pred_counts,
        'class_names': class_names
    }


def print_per_class_analysis(metrics):
    """Print per-class analysis."""
    cm = metrics['confusion_matrix']
    class_acc = metrics['class_accuracy']
    class_counts = metrics['class_counts']
    pred_counts = metrics['pred_counts']
    class_names = metrics['class_names']
    
    print(f"{'Class':<20} {'Accuracy':>10} {'True':>8} {'Pred':>8} {'Correct':>8}")
    print(f"{'-'*60}")
    
    for i in range(len(class_acc)):
        true_count = int(class_counts[i])
        pred_count = int(pred_counts[i])
        correct_count = int(cm[i, i])
        accuracy = class_acc[i] * 100
        
        print(f"{class_names[i]:<20} {accuracy:>9.2f}% {true_count:>8} {pred_count:>8} {correct_count:>8}")
    
    print(f"{'-'*60}")
    print(f"Mean Per-Class Acc:    {class_acc.mean()*100:.2f}%")
    print(f"Std Per-Class Acc:     {class_acc.std()*100:.2f}%")
    
    best_idx = class_acc.argmax()
    worst_idx = class_acc.argmin()
    print(f"\nBest Performing Class:  {class_names[best_idx]:<20} ({class_acc[best_idx]*100:.2f}%)")
    print(f"Worst Performing Class: {class_names[worst_idx]:<20} ({class_acc[worst_idx]*100:.2f}%)")


def print_classification_report(predictions, labels, label_mapping=None):
    """Print sklearn classification report."""
    num_classes = len(np.unique(labels))
    class_names = get_class_names(num_classes, label_mapping)
    
    print("\nClassification Report:")
    print(classification_report(
        labels, predictions,
        target_names=class_names,
        zero_division=0
    ))

