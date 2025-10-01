"""
Metrics utilities for model evaluation.
"""

import torch
import numpy as np
from typing import Tuple, List
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate top-1 accuracy."""
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Calculate top-k accuracy."""
    _, top_k_pred = outputs.topk(k, 1, True, True)
    correct = top_k_pred.eq(targets.view(-1, 1).expand_as(top_k_pred)).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def precision_recall_f1(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score."""
    _, predicted = outputs.max(1)
    
    # Convert to numpy for sklearn
    y_true = targets.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    # Calculate metrics using sklearn
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    
    return precision, recall, f1


class MetricsTracker:
    """Track and compute metrics during training/evaluation."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        self.predictions.append(outputs.detach().cpu())
        self.targets.append(targets.detach().cpu())
        self.losses.append(loss)
    
    def compute(self) -> dict:
        """Compute all metrics."""
        if not self.predictions:
            return {}
        
        # Concatenate all batches
        all_outputs = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Calculate metrics
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy(all_outputs, all_targets),
            'top5_accuracy': top_k_accuracy(all_outputs, all_targets, k=5)
        }
        
        # Add precision, recall, F1
        precision, recall, f1 = precision_recall_f1(all_outputs, all_targets, self.num_classes)
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.predictions:
            return np.array([])
        
        all_outputs = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        _, predicted = all_outputs.max(1)
        return confusion_matrix(all_targets.numpy(), predicted.numpy())
    
    def plot_confusion_matrix(self, class_names: List[str] = None, save_path: str = None):
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix()
        
        if cm.size == 0:
            return
        
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names[:50] if class_names else None,  # Limit for readability
            yticklabels=class_names[:50] if class_names else None
        )
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class PerClassMetrics:
    """Track per-class metrics."""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.class_correct = torch.zeros(self.num_classes)
        self.class_total = torch.zeros(self.num_classes)
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Update per-class metrics."""
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets)
        
        for i in range(self.num_classes):
            class_mask = targets == i
            if class_mask.sum() > 0:
                self.class_correct[i] += correct[class_mask].sum().item()
                self.class_total[i] += class_mask.sum().item()
    
    def compute(self) -> dict:
        """Compute per-class accuracies."""
        class_accuracies = {}
        
        for i in range(self.num_classes):
            if self.class_total[i] > 0:
                accuracy = 100.0 * self.class_correct[i] / self.class_total[i]
                class_accuracies[self.class_names[i]] = accuracy.item()
            else:
                class_accuracies[self.class_names[i]] = 0.0
        
        return class_accuracies
    
    def get_worst_classes(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the worst performing classes."""
        class_accuracies = self.compute()
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1])
        return sorted_classes[:n]
    
    def get_best_classes(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the best performing classes."""
        class_accuracies = self.compute()
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[:n]


def calculate_model_flops(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate FLOPs for model (requires thop package).
    """
    try:
        from thop import profile
        dummy_input = torch.randn(1, *input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops
    except ImportError:
        print("thop package not available for FLOP calculation")
        return 0
