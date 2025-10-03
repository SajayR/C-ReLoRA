"""
Evaluation script for DinoV2 LoRA models.
"""

import os
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
from data.imagenet_dataset import create_imagenet_dataloaders
from models.dinov2_lora import create_dinov2_lora_model, DinoV2WithClassificationHead
from configs.config_manager import ConfigManager
from utils.metrics import MetricsTracker, PerClassMetrics
from utils.training_utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained DinoV2 LoRA models."""
    
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        
        # Setup data
        _, self.val_loader = create_imagenet_dataloaders(
            data_root=config.data.dataset_path,
            batch_size=config.data.batch_size,
            image_size=config.data.image_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )
        
        # Initialize metrics trackers
        self.metrics_tracker = MetricsTracker(config.model.num_classes)
        self.per_class_metrics = PerClassMetrics(config.model.num_classes)
    
    def load_model(self, checkpoint_path: str, model_type: str = "lora") -> nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_type: Either "lora" or "baseline"
        
        Returns:
            Loaded model
        """
        if model_type == "lora":
            # Convert target_modules to regular list if it's a ListConfig
            target_modules = self.config.lora.target_modules
            if hasattr(target_modules, 'to_container'):
                target_modules = target_modules.to_container()
            elif not isinstance(target_modules, list):
                target_modules = list(target_modules)
                
            model = create_dinov2_lora_model(
                model_name=self.config.model.name,
                num_classes=self.config.model.num_classes,
                lora_r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=target_modules
            )
        else:
            model = DinoV2WithClassificationHead(
                model_name=self.config.model.name,
                num_classes=self.config.model.num_classes,
                freeze_backbone=False
            )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        return model
    
    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate a model on the validation set.
        
        Args:
            model: Model to evaluate
            save_results: Whether to save detailed results
        
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        self.metrics_tracker.reset()
        self.per_class_metrics.reset()
        
        start_time = time.time()
        
        logger.info("Starting evaluation...")
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Update metrics
            self.metrics_tracker.update(outputs, targets, loss.item())
            self.per_class_metrics.update(outputs, targets)
            
            # Update progress bar
            if batch_idx % 100 == 0:
                current_metrics = self.metrics_tracker.compute()
                progress_bar.set_postfix({
                    'acc': f"{current_metrics.get('accuracy', 0):.2f}%",
                    'loss': f"{current_metrics.get('loss', 0):.4f}"
                })
        
        eval_time = time.time() - start_time
        
        # Compute final metrics
        final_metrics = self.metrics_tracker.compute()
        per_class_acc = self.per_class_metrics.compute()
        
        # Add timing info
        final_metrics['eval_time'] = eval_time
        final_metrics['samples_per_second'] = len(self.val_loader.dataset) / eval_time
        
        # Add per-class statistics
        class_accuracies = list(per_class_acc.values())
        final_metrics['mean_class_accuracy'] = np.mean(class_accuracies)
        final_metrics['std_class_accuracy'] = np.std(class_accuracies)
        final_metrics['min_class_accuracy'] = np.min(class_accuracies)
        final_metrics['max_class_accuracy'] = np.max(class_accuracies)
        
        logger.info(f"Evaluation completed in {eval_time:.2f}s")
        logger.info(f"Top-1 Accuracy: {final_metrics['accuracy']:.2f}%")
        logger.info(f"Top-5 Accuracy: {final_metrics['top5_accuracy']:.2f}%")
        logger.info(f"Mean Class Accuracy: {final_metrics['mean_class_accuracy']:.2f}%")
        
        if save_results:
            self._save_detailed_results(final_metrics, per_class_acc)
        
        return final_metrics
    
    def _save_detailed_results(self, metrics: Dict[str, Any], per_class_acc: Dict[str, float]):
        """Save detailed evaluation results."""
        results_dir = Path("./evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save overall metrics
        metrics_path = results_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save per-class accuracies
        per_class_path = results_dir / f"per_class_acc_{timestamp}.json"
        with open(per_class_path, 'w') as f:
            json.dump(per_class_acc, f, indent=2)
        
        # Save confusion matrix plot
        cm_path = results_dir / f"confusion_matrix_{timestamp}.png"
        self.metrics_tracker.plot_confusion_matrix(save_path=str(cm_path))
        
        # Plot per-class accuracy distribution
        self._plot_per_class_distribution(per_class_acc, results_dir / f"class_acc_dist_{timestamp}.png")
        
        logger.info(f"Detailed results saved to {results_dir}")
    
    def _plot_per_class_distribution(self, per_class_acc: Dict[str, float], save_path: Path):
        """Plot per-class accuracy distribution."""
        accuracies = list(per_class_acc.values())
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(accuracies, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Number of Classes')
        plt.title('Distribution of Per-Class Accuracies')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(accuracies)
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy Statistics')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def compare_models(
    config,
    model_paths: Dict[str, str],
    device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models.
    
    Args:
        config: Configuration object
        model_paths: Dictionary of model_name -> checkpoint_path
        device: Device to run evaluation on
    
    Returns:
        Dictionary of model_name -> metrics
    """
    evaluator = ModelEvaluator(config, device)
    results = {}
    
    for model_name, checkpoint_path in model_paths.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating {model_name}")
        logger.info(f"{'='*50}")
        
        # Determine model type from name
        model_type = "lora" if "lora" in model_name.lower() else "baseline"
        
        try:
            # Load and evaluate model
            model = evaluator.load_model(checkpoint_path, model_type)
            metrics = evaluator.evaluate_model(model, save_results=True)
            results[model_name] = metrics
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a comparison table of model results."""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Table header
    header = f"{'Model':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Loss':<10} {'Time (s)':<12}"
    print(header)
    print("-" * len(header))
    
    # Table rows
    for model_name, metrics in results.items():
        if "error" in metrics:
            print(f"{model_name:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<12}")
        else:
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<12.2f} "
                  f"{metrics['top5_accuracy']:<12.2f} "
                  f"{metrics['loss']:<10.4f} "
                  f"{metrics['eval_time']:<12.2f}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DinoV2 LoRA models")
    parser.add_argument("--config", type=str, default="default", help="Config name")
    parser.add_argument("--checkpoint", type=str, help="Single checkpoint to evaluate")
    parser.add_argument("--compare", type=str, nargs="+", help="Multiple checkpoints to compare")
    parser.add_argument("--compare-names", type=str, nargs="+", help="Names for comparison models")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup logging
    setup_logging("./logs")
    
    if args.checkpoint:
        # Single model evaluation
        evaluator = ModelEvaluator(config, device)
        model = evaluator.load_model(args.checkpoint, "lora")
        metrics = evaluator.evaluate_model(model)
        
        print(f"\nEvaluation Results:")
        print(f"Top-1 Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Evaluation Time: {metrics['eval_time']:.2f}s")
        
    elif args.compare:
        # Multiple model comparison
        if args.compare_names and len(args.compare_names) != len(args.compare):
            raise ValueError("Number of model names must match number of checkpoints")
        
        model_names = args.compare_names or [f"Model_{i+1}" for i in range(len(args.compare))]
        model_paths = dict(zip(model_names, args.compare))
        
        results = compare_models(config, model_paths, device)
        print_comparison_table(results)
        
        # Save comparison results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        comparison_path = output_dir / f"model_comparison_{int(time.time())}.json"
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comparison results saved to {comparison_path}")
    
    else:
        logger.error("Please provide either --checkpoint or --compare arguments")


if __name__ == "__main__":
    main()
