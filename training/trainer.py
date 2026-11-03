"""
Training module for TrashCAN segmentation - Fixed version for Mask R-CNN
"""
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
import time
import numpy as np
from tqdm import tqdm
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import copy
import gc

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.model import TrashCANModel
    from data.dataset import get_dataloaders
except ImportError as e:
    print(f"❌ Import error in trainer.py: {e}")
    raise

class Trainer:
    """Trainer class for TrashCAN segmentation model - Fixed for Mask R-CNN"""
    
    def __init__(self, model, train_loader, val_loader, device=None, cfg=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        if cfg is None:
            raise ValueError("Trainer membutuhkan resolved experiment config")
        self.config = cfg
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config.STEP_SIZE,
            gamma=self.config.GAMMA
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_metrics': [],  # Changed from val_loss to val_metrics
            'learning_rate': [],
            'coco_metrics': []
        }
        
        # Create output directory
        self.output_dir = Path(
            getattr(
                self.config,
                "RUN_DIR",
                Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for child in ("checkpoints", "metrics", "predictions", "benchmarks"):
            (self.output_dir / child).mkdir(parents=True, exist_ok=True)
        self.evaluation_count = 0
        self.save_run_metadata()
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            
            # Sum all losses
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += losses.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses.item(),
                'avg_loss': avg_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Clear cache every 100 batches to prevent memory issues
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self):
        """Evaluate bounding boxes on the configured internal validation split."""
        from evaluation.detection_evaluator import DetectionEvaluator, target_class_ap

        self.evaluation_count += 1
        evaluation_dir = self.output_dir / "metrics" / f"evaluation_{self.evaluation_count:03d}"
        category_map = getattr(self.config, "CATEGORY_MAP", None)
        if category_map is None:
            category_map = {
                row["id"]: row["name"]
                for row in self.val_loader.dataset.coco.dataset["categories"]
            }
        evaluator = DetectionEvaluator(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            annotation_file=getattr(self.config, "EVAL_ANNOTATION_JSON", self.config.VAL_JSON),
            category_map=category_map,
            output_dir=evaluation_dir,
            target_class=getattr(self.config, "TARGET_CLASS", "trash_plastic"),
            score_threshold=getattr(self.config, "EVAL_SCORE_THRESHOLD", 0.0),
        )
        metrics = evaluator.evaluate()
        metric_record = {
            **metrics["overall"],
            "target_class": metrics["target_class"],
            "critical_false_negatives": metrics["critical_false_negatives"],
            "primary_metric_name": f"AP_{getattr(self.config, 'TARGET_CLASS', 'trash_plastic')}",
            "primary_metric": target_class_ap(metrics),
            "num_images": metrics["num_images"],
            "num_predictions": metrics["num_predictions"],
            "artifact_dir": str(evaluation_dir),
        }
        self.history["coco_metrics"].append(metric_record)
        return metric_record["primary_metric"]
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        best_metric = float('-inf')
        best_model = None
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate (using COCO metrics)
            if getattr(self.config, "SKIP_VALIDATION", False):
                val_metric = None
                print("Validation skipped for this smoke test")
            else:
                val_metric = self.validate()
            self.history['val_metrics'].append(val_metric)
            
            # Update learning rate
            self.scheduler.step()
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"\nTrain Loss: {train_loss:.4f}")
            if val_metric is not None:
                print(f"Validation AP {getattr(self.config, 'TARGET_CLASS', 'trash_plastic')}: {val_metric:.4f}")
            print(f"Learning Rate: {self.history['learning_rate'][-1]:.6f}")
            
            # Save checkpoint based on mAP (higher is better)
            if val_metric is not None and val_metric > best_metric:
                best_metric = val_metric
                best_model = copy.deepcopy(self.model.state_dict())
                self.model.save(self.output_dir / 'checkpoints' / 'best_model.pth')
                print(f"✅ Saved best model with target AP: {best_metric:.4f}")
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self.model.save(self.output_dir / 'checkpoints' / f'model_epoch_{epoch}.pth')
                print(f"💾 Saved checkpoint for epoch {epoch}")
            
            # Save history
            self.save_history()
        
        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        # Final evaluation
        if getattr(self.config, "SKIP_VALIDATION", False):
            print("\nSkipping final detailed evaluation")
            final_metrics = None
        else:
            print("\nPerforming final detailed evaluation...")
            final_metrics = self.validate()  # This will run COCO evaluation again
        
        # Save final model
        self.model.save(self.output_dir / 'checkpoints' / 'final_model.pth')
        
        # Print final metrics
        if self.history['coco_metrics']:
            latest_metrics = self.history['coco_metrics'][-1]
            print("\nFinal COCO Metrics:")
            for key, value in latest_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        return self.history
    
    def save_history(self):
        """Save training history"""
        history_file = self.output_dir / 'metrics' / 'history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        with (self.output_dir / 'metrics' / 'epoch_metrics.csv').open(
            'w', newline='', encoding='utf-8'
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=['epoch', 'train_loss', 'val_AP', 'learning_rate']
            )
            writer.writeheader()
            for index, train_loss in enumerate(self.history['train_loss']):
                writer.writerow({
                    'epoch': index + 1,
                    'train_loss': train_loss,
                    'val_AP': self.history['val_metrics'][index] if index < len(self.history['val_metrics']) else None,
                    'learning_rate': self.history['learning_rate'][index] if index < len(self.history['learning_rate']) else None,
                })

    def save_run_metadata(self):
        """Save run-level metadata for reproducibility."""
        metadata = {
            'dataset_version': getattr(self.config, 'DATASET_VERSION', None),
            'data_fraction': getattr(self.config, 'DATA_FRACTION', None),
            'num_classes': getattr(self.config, 'NUM_CLASSES', None),
            'batch_size': getattr(self.config, 'BATCH_SIZE', None),
            'learning_rate': getattr(self.config, 'LEARNING_RATE', None),
            'num_epochs': getattr(self.config, 'NUM_EPOCHS', None),
            'input_width': getattr(self.config, 'INPUT_WIDTH', None),
            'input_height': getattr(self.config, 'INPUT_HEIGHT', None),
            'device': str(self.device),
            'trainer': self.__class__.__name__,
            'backbone': getattr(self.config, 'BACKBONE', 'resnet50'),
            'backbone_provenance': getattr(self.config, 'BACKBONE_PROVENANCE', None),
            'split_manifest_sha256': getattr(self.config, 'SPLIT_MANIFEST_SHA256', None),
            'target_class': getattr(self.config, 'TARGET_CLASS', None),
            'resolved_config': getattr(self.config, 'RESOLVED_CONFIG', None),
        }
        with open(self.output_dir / 'run_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def plot_history(self):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True)
        
        # mAP plot
        ax = axes[0, 1]
        if self.history['val_metrics']:
            ax.plot(self.history['val_metrics'], label='Validation mAP', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('Validation mAP')
            ax.legend()
            ax.grid(True)
        
        # Learning rate plot
        ax = axes[1, 0]
        ax.plot(self.history['learning_rate'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True)
        
        # COCO metrics bar plot
        ax = axes[1, 1]
        if self.history['coco_metrics']:
            latest_metrics = self.history['coco_metrics'][-1]
            metrics_to_plot = ['AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100']
            values = [latest_metrics.get(m, 0) for m in metrics_to_plot]
            
            bars = ax.bar(metrics_to_plot, values, color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow'])
            ax.set_xlabel('Metric')
            ax.set_ylabel('Value')
            ax.set_title('Final COCO Metrics')
            ax.grid(True, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()

def main():
    """Main training function"""
    from configs import get_config
    from utils.experiment import resolve_experiment_config

    config = resolve_experiment_config(get_config("auto"), mode="train")
    
    # Get dataloaders
    train_loader, val_loader, _, _ = get_dataloaders(config)
    
    # Create model
    model = TrashCANModel(cfg=config)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, cfg=config)
    
    # Train
    history = trainer.train()
    
    # Plot history
    trainer.plot_history()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Results saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()
