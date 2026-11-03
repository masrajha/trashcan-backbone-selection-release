import torch
import torch.nn as nn
import time
from tqdm import tqdm
import gc
from pathlib import Path
from datetime import datetime
import json
import csv
import numpy as np
import random
import hashlib

from utils.checkpoint import load_checkpoint

class RTX3050Trainer:
    def __init__(self, model, train_loader, val_loader=None, device='cuda', cfg=None):
        """
        Optimized trainer for RTX 3050 6GB Laptop GPU
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        if cfg is None:
            raise ValueError("RTX3050Trainer membutuhkan resolved experiment config")
        self.cfg = cfg

        # Setup for RTX 3050
        deterministic = bool(getattr(self.cfg, "DETERMINISTIC", True))
        allow_tf32 = bool(getattr(self.cfg, "ALLOW_TF32", False))
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        # Mixed precision
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(getattr(self.cfg, "USE_AMP", True) and torch.cuda.is_available())
        )

        # Gradient accumulation for stability
        self.accumulation_steps = int(
            getattr(self.cfg, "GRADIENT_ACCUMULATION_STEPS", 2)
        )

        # Memory optimization
        self.empty_cache_every = 50  # batches

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=getattr(self.cfg, "LEARNING_RATE", 0.001),
            momentum=getattr(self.cfg, "MOMENTUM", 0.9),
            weight_decay=getattr(self.cfg, "WEIGHT_DECAY", 0.0005)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=getattr(self.cfg, "STEP_SIZE", 10),
            gamma=getattr(self.cfg, "GAMMA", 0.1)
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_metrics': [],
            'learning_rate': [],
            'coco_metrics': [],
            'epoch_duration_seconds': [],
            'epoch_peak_allocated_mb': [],
            'epoch_peak_reserved_mb': [],
        }

        # Create output directory
        self.output_dir = Path(
            getattr(
                self.cfg,
                "RUN_DIR",
                Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for child in ("checkpoints", "metrics", "predictions", "benchmarks"):
            (self.output_dir / child).mkdir(parents=True, exist_ok=True)
        self.evaluation_count = 0
        self.start_epoch = 1
        self.best_metric = float('-inf')
        self.resume_metadata = None
        self.save_run_metadata()

    def _model_state_dict(self):
        return self.model.model.state_dict() if hasattr(self.model, 'model') else self.model.state_dict()

    def _load_model_state_dict(self, state_dict):
        target = self.model.model if hasattr(self.model, 'model') else self.model
        target.load_state_dict(state_dict)

    def _resume_identity(self, total_epochs):
        return {
            'run_id': getattr(self.cfg, 'RUN_ID', None),
            'backbone': getattr(self.cfg, 'BACKBONE', None),
            'seed': getattr(self.cfg, 'RANDOM_SEED', None),
            'split_manifest_sha256': getattr(self.cfg, 'SPLIT_MANIFEST_SHA256', None),
            'phase3_training_protocol_sha256': getattr(
                self.cfg, 'PHASE3_TRAINING_PROTOCOL_SHA256', None
            ),
            'total_epochs': int(total_epochs),
        }

    @staticmethod
    def _numpy_rng_state():
        state = np.random.get_state()
        return {
            'bit_generator': state[0],
            'state': state[1].tolist(),
            'position': state[2],
            'has_gauss': state[3],
            'cached_gaussian': state[4],
        }

    @staticmethod
    def _restore_numpy_rng_state(state):
        np.random.set_state((
            state['bit_generator'],
            np.asarray(state['state'], dtype=np.uint32),
            state['position'], state['has_gauss'], state['cached_gaussian'],
        ))

    @staticmethod
    def _loader_generator_state(loader):
        generator = getattr(loader, 'generator', None) if loader is not None else None
        return generator.get_state() if generator is not None else None

    @staticmethod
    def _restore_loader_generator_state(loader, state):
        generator = getattr(loader, 'generator', None) if loader is not None else None
        if state is not None:
            if generator is None:
                raise ValueError('DataLoader checkpoint memiliki generator state tetapi runtime tidak')
            generator.set_state(state)
    def save_training_state(self, path, epoch, total_epochs):
        """Atomically save everything required after a completed epoch."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format': 'trashcan_training_state_v1',
            'completed_epoch': int(epoch),
            'best_metric': float(self.best_metric),
            'identity': self._resume_identity(total_epochs),
            'model_state_dict': self._model_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history,
            'evaluation_count': int(self.evaluation_count),
            'data_order_state': {
                'train_generator': self._loader_generator_state(
                    getattr(self, 'train_loader', None)
                ),
                'val_generator': self._loader_generator_state(
                    getattr(self, 'val_loader', None)
                ),
            },
            'rng_state': {
                'python': random.getstate(),
                'numpy': self._numpy_rng_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        temporary = path.with_name(path.name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    def resume_from(self, path, total_epochs, reason=None):
        """Restore a full-state checkpoint after strict identity validation."""
        path = Path(path).resolve()
        checkpoint = load_checkpoint(path, map_location='cpu')
        if checkpoint.get('format') != 'trashcan_training_state_v1':
            raise ValueError(
                'Checkpoint bukan full training state; model-only checkpoint tidak dapat di-resume'
            )
        expected = self._resume_identity(total_epochs)
        actual = checkpoint.get('identity', {})
        mismatches = {
            key: {'checkpoint': actual.get(key), 'runtime': value}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f'Identitas resume tidak cocok: {mismatches}')
        completed_epoch = int(checkpoint['completed_epoch'])
        history = checkpoint['history']
        if len(history.get('train_loss', [])) != completed_epoch:
            raise ValueError('History checkpoint tidak konsisten dengan completed_epoch')
        if completed_epoch >= int(total_epochs):
            raise ValueError('Checkpoint sudah mencapai total epoch yang diminta')
        self._load_model_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = history
        self.evaluation_count = int(checkpoint['evaluation_count'])
        self.best_metric = float(checkpoint['best_metric'])
        data_order_state = checkpoint.get('data_order_state', {})
        self._restore_loader_generator_state(
            getattr(self, 'train_loader', None), data_order_state.get('train_generator')
        )
        self._restore_loader_generator_state(
            getattr(self, 'val_loader', None), data_order_state.get('val_generator')
        )
        random.setstate(checkpoint['rng_state']['python'])
        self._restore_numpy_rng_state(checkpoint['rng_state']['numpy'])
        torch.set_rng_state(checkpoint['rng_state']['torch_cpu'])
        if torch.cuda.is_available() and checkpoint['rng_state']['torch_cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_state']['torch_cuda'])
        self.start_epoch = completed_epoch + 1
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        self.resume_metadata = {
            'checkpoint': str(path),
            'checkpoint_sha256': digest.hexdigest(),
            'completed_epoch': completed_epoch,
            'next_epoch': self.start_epoch,
            'reason': reason or 'infrastructure_interruption',
            'resumed_at': datetime.now().isoformat(),
            'identity': actual,
        }
        with (self.output_dir / 'resume_metadata.json').open('w', encoding='utf-8') as handle:
            json.dump(self.resume_metadata, handle, indent=2)
        print(f'Resume tervalidasi: epoch {self.start_epoch} dari {total_epochs}')
        return completed_epoch
    def _compute_loss(self, loss_output):
        """Compute total loss from various output formats"""
        if isinstance(loss_output, dict):
            return sum(l for l in loss_output.values())
        elif isinstance(loss_output, (list, tuple)):
            return sum(loss_output)
        elif torch.is_tensor(loss_output):
            return loss_output
        else:
            raise ValueError(f"Unsupported loss output type: {type(loss_output)}")

    def train_epoch_fast(self, epoch):
        """Ultra-optimized training for RTX 3050"""
        self.model.train()
        total_loss = 0
        batch_count = 0

        # Progress bar with timing
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        self.optimizer.zero_grad(set_to_none=True)  # Faster zero_grad

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to GPU with non-blocking
            images = [img.to(self.device, non_blocking=True) for img in images]
            targets = [{k: v.to(self.device, non_blocking=True)
                       for k, v in t.items()} for t in targets]

            # Mixed precision forward
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                loss_output = self.model(images, targets)
                loss = self._compute_loss(loss_output)

            # Backward with scaling
            self.scaler.scale(loss).backward()

            # Update weights
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(self.accumulation_steps)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # Update stats
            total_loss += loss.item()
            batch_count += 1
            avg_loss = total_loss / batch_count

            # Update progress
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'avg': f'{avg_loss:.3f}',
                'mem': f'{torch.cuda.memory_allocated()/1e9:.2f}GB'
            })

            # Clear cache periodically
            if batch_idx % self.empty_cache_every == 0:
                torch.cuda.empty_cache()
                gc.collect()

        remainder = batch_count % self.accumulation_steps
        if remainder:
            self.scaler.unscale_(self.optimizer)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(remainder)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return total_loss / batch_count

    @torch.no_grad()
    def validate(self):
        """Evaluate bounding boxes on the internal validation split."""
        if self.val_loader is None:
            return 0.0

        from evaluation.detection_evaluator import DetectionEvaluator, target_class_ap

        self.evaluation_count += 1
        evaluation_dir = self.output_dir / "metrics" / f"evaluation_{self.evaluation_count:03d}"
        category_map = getattr(self.cfg, "CATEGORY_MAP", None)
        if category_map is None:
            category_map = {
                row["id"]: row["name"]
                for row in self.val_loader.dataset.coco.dataset["categories"]
            }
        evaluator = DetectionEvaluator(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            annotation_file=getattr(self.cfg, "EVAL_ANNOTATION_JSON", self.cfg.VAL_JSON),
            category_map=category_map,
            output_dir=evaluation_dir,
            target_class=getattr(self.cfg, "TARGET_CLASS", "trash_plastic"),
            score_threshold=getattr(self.cfg, "EVAL_SCORE_THRESHOLD", 0.0),
        )
        metrics = evaluator.evaluate()
        metric_record = {
            **metrics["overall"],
            "target_class": metrics["target_class"],
            "critical_false_negatives": metrics["critical_false_negatives"],
            "primary_metric_name": f"AP_{getattr(self.cfg, 'TARGET_CLASS', 'trash_plastic')}",
            "primary_metric": target_class_ap(metrics),
            "num_images": metrics["num_images"],
            "num_predictions": metrics["num_predictions"],
            "artifact_dir": str(evaluation_dir),
        }
        self.history["coco_metrics"].append(metric_record)
        return metric_record["primary_metric"]

    def train(self, num_epochs=30, resume_from=None, resume_reason=None):
        """Main training loop"""
        print(f"Starting RTX 3050 optimized training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Batch accumulation: {self.accumulation_steps} steps")

        if resume_from:
            self.resume_from(resume_from, num_epochs, resume_reason)
        cuda_metrics = torch.cuda.is_available() and torch.device(self.device).type == 'cuda'
        run_started = time.perf_counter()

        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            if cuda_metrics:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_started = time.perf_counter()

            # Train
            train_loss = self.train_epoch_fast(epoch)
            self.history['train_loss'].append(train_loss)

            # Validate periodically to save time on early smoke tests.
            validate_every = getattr(self.cfg, "VALIDATE_EVERY", 2)
            skip_validation = getattr(self.cfg, "SKIP_VALIDATION", False)
            if skip_validation:
                val_metric = None
                self.history['val_metrics'].append(None)
                print("Validation skipped for this smoke test")
            elif self.val_loader and (epoch % validate_every == 0 or epoch == num_epochs):
                val_metric = self.validate()
                self.history['val_metrics'].append(val_metric)
                print(f"Validation AP {getattr(self.cfg, 'TARGET_CLASS', 'trash_plastic')}: {val_metric:.4f}")
            else:
                val_metric = None
                self.history['val_metrics'].append(None)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Print progress
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save best model berdasarkan bbox AP validasi internal.
            if val_metric is not None and val_metric > self.best_metric:
                self.best_metric = val_metric
                self.save_model(self.output_dir / 'checkpoints' / 'best_model.pth')
                print(
                    f"✅ Saved best model with AP "
                    f"{getattr(self.cfg, 'TARGET_CLASS', 'trash_plastic')}: {self.best_metric:.4f}"
                )

            if cuda_metrics:
                torch.cuda.synchronize(self.device)
                peak_allocated_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
                peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / 1024**2
            else:
                peak_allocated_mb = 0.0
                peak_reserved_mb = 0.0
            epoch_duration_seconds = time.perf_counter() - epoch_started
            self.history['epoch_duration_seconds'].append(epoch_duration_seconds)
            self.history['epoch_peak_allocated_mb'].append(peak_allocated_mb)
            self.history['epoch_peak_reserved_mb'].append(peak_reserved_mb)
            print(
                f"Epoch resources: {epoch_duration_seconds:.2f}s | "
                f"peak allocated {peak_allocated_mb:.1f} MB | "
                f"peak reserved {peak_reserved_mb:.1f} MB"
            )

            # Persist an atomic full-state checkpoint after every completed epoch.
            self.save_history()
            self.save_training_state(
                self.output_dir / 'checkpoints' / 'latest_training_state.pth',
                epoch,
                num_epochs,
            )
            if epoch % 5 == 0:
                self.save_model(self.output_dir / 'checkpoints' / f'model_epoch_{epoch}.pth')
                self.save_training_state(
                    self.output_dir / 'checkpoints' / f'training_state_epoch_{epoch}.pth',
                    epoch,
                    num_epochs,
                )
                print(f"Saved model and full training state for epoch {epoch}")

        # Save final model dengan format yang sama
        self.save_model(self.output_dir / 'checkpoints' / 'final_model.pth')

        # Plot history
        self.plot_history()

        if cuda_metrics:
            torch.cuda.synchronize(self.device)
        run_duration_seconds = time.perf_counter() - run_started
        resource_summary = {
            'schema_version': 1,
            'device': str(self.device),
            'gpu_name': torch.cuda.get_device_name(self.device) if cuda_metrics else None,
            'cuda_metrics_available': cuda_metrics,
            'num_epochs': len(self.history['train_loss']),
            'total_duration_seconds': float(sum(self.history['epoch_duration_seconds'])),
            'execution_segment_duration_seconds': run_duration_seconds,
            'resume': self.resume_metadata,
            'mean_epoch_duration_seconds': (
                float(np.mean(self.history['epoch_duration_seconds']))
                if self.history['epoch_duration_seconds'] else 0.0
            ),
            'max_epoch_peak_allocated_mb': max(
                self.history['epoch_peak_allocated_mb'], default=0.0
            ),
            'max_epoch_peak_reserved_mb': max(
                self.history['epoch_peak_reserved_mb'], default=0.0
            ),
            'epochs': [
                {
                    'epoch': index + 1,
                    'duration_seconds': self.history['epoch_duration_seconds'][index],
                    'peak_allocated_mb': self.history['epoch_peak_allocated_mb'][index],
                    'peak_reserved_mb': self.history['epoch_peak_reserved_mb'][index],
                    'train_loss': self.history['train_loss'][index],
                }
                for index in range(len(self.history['train_loss']))
            ],
        }
        with (self.output_dir / 'metrics' / 'resource_usage.json').open(
            'w', encoding='utf-8'
        ) as handle:
            json.dump(resource_summary, handle, indent=2)
        self.save_history()

        return self.history

    def save_model(self, path):
        """Save model state dengan format yang kompatibel"""
        # PERBAIKAN: Simpan state_dict dari MaskRCNN (bukan wrapper)
        # Cek jika model adalah TrashCANModel wrapper
        if hasattr(self.model, 'model'):
            # Ambil state_dict dari MaskRCNN yang sebenarnya
            model_state_dict = self.model.model.state_dict()
            # Ambil jumlah kelas dari model
            num_classes = self.model.model.roi_heads.box_predictor.cls_score.out_features - 1  # Kurangi background
        else:
            # Model langsung MaskRCNN
            model_state_dict = self.model.state_dict()
            num_classes = self.model.roi_heads.box_predictor.cls_score.out_features - 1

        save_dict = {
            'model_state_dict': model_state_dict,  # State dict dari MaskRCNN langsung
            'num_classes': num_classes,
            'config': {
                'num_classes': num_classes,
                'dataset_version': getattr(self.cfg, 'DATASET_VERSION', None),
                'backbone': getattr(self.cfg, 'BACKBONE', 'resnet50'),
                'phase4_fixed_fusion': bool(getattr(self.cfg, 'PHASE4_FIXED_FUSION', False)),
                'phase4_protocol_sha256': getattr(self.cfg, 'PHASE4_PROTOCOL_SHA256', None),
                'backbone_pretraining_scope': getattr(self.cfg, 'BACKBONE_PRETRAINING_SCOPE', 'backbone_only'),
                'backbone_provenance': getattr(self.cfg, 'BACKBONE_PROVENANCE', None),
                'phase3_stage': getattr(self.cfg, 'PHASE3_STAGE', None),
                'phase3_training_protocol_sha256': getattr(
                    self.cfg, 'PHASE3_TRAINING_PROTOCOL_SHA256', None
                ),
                'data_fraction': getattr(self.cfg, 'DATA_FRACTION', None),
                'split_manifest_sha256': getattr(self.cfg, 'SPLIT_MANIFEST_SHA256', None),
                'category_map': getattr(self.cfg, 'CATEGORY_MAP', None),
                'target_class': getattr(self.cfg, 'TARGET_CLASS', None),
                'target_category_id': getattr(self.cfg, 'TARGET_CATEGORY_ID', None),
                'primary_metric': f"AP_{getattr(self.cfg, 'TARGET_CLASS', 'trash_plastic')}",
                'source_annotation_sha256': (
                    getattr(self.cfg, 'SPLIT_MANIFEST', {}) or {}
                ).get('source_annotation_sha256'),
                'resolved_config': getattr(self.cfg, 'RESOLVED_CONFIG', None),
            }
        }
        torch.save(save_dict, path)
        print(f"💾 Model saved to: {path}")

    def save_run_metadata(self):
        """Save run-level metadata for reproducibility."""
        if self.cfg is None:
            return

        metadata = {
            'dataset_version': getattr(self.cfg, 'DATASET_VERSION', None),
            'data_fraction': getattr(self.cfg, 'DATA_FRACTION', None),
            'num_classes': getattr(self.cfg, 'NUM_CLASSES', None),
            'batch_size': getattr(self.cfg, 'BATCH_SIZE', None),
            'learning_rate': getattr(self.cfg, 'LEARNING_RATE', None),
            'num_epochs': getattr(self.cfg, 'NUM_EPOCHS', None),
            'input_width': getattr(self.cfg, 'INPUT_WIDTH', None),
            'input_height': getattr(self.cfg, 'INPUT_HEIGHT', None),
            'device': str(self.device),
            'trainer': self.__class__.__name__,
            'split_manifest_sha256': getattr(self.cfg, 'SPLIT_MANIFEST_SHA256', None),
            'target_class': getattr(self.cfg, 'TARGET_CLASS', None),
            'resolved_config': getattr(self.cfg, 'RESOLVED_CONFIG', None),
        }
        with open(self.output_dir / 'run_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_history(self):
        """Save training history"""
        history_file = self.output_dir / 'metrics' / 'history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        with (self.output_dir / 'metrics' / 'epoch_metrics.csv').open(
            'w', newline='', encoding='utf-8'
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    'epoch', 'train_loss', 'val_AP', 'learning_rate',
                    'epoch_duration_seconds', 'peak_allocated_mb',
                    'peak_reserved_mb',
                ],
            )
            writer.writeheader()
            for index, train_loss in enumerate(self.history['train_loss']):
                writer.writerow({
                    'epoch': index + 1,
                    'train_loss': train_loss,
                    'val_AP': self.history['val_metrics'][index] if index < len(self.history['val_metrics']) else None,
                    'learning_rate': self.history['learning_rate'][index] if index < len(self.history['learning_rate']) else None,
                    'epoch_duration_seconds': self.history['epoch_duration_seconds'][index] if index < len(self.history['epoch_duration_seconds']) else None,
                    'peak_allocated_mb': self.history['epoch_peak_allocated_mb'][index] if index < len(self.history['epoch_peak_allocated_mb']) else None,
                    'peak_reserved_mb': self.history['epoch_peak_reserved_mb'][index] if index < len(self.history['epoch_peak_reserved_mb']) else None,
                })

    def plot_history(self):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.history['train_loss']) + 1)

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Loss plot
            ax = axes[0]
            ax.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')

            # Plot bbox AP hanya untuk epoch yang menjalankan validasi internal.
            val_epochs = [i+1 for i, v in enumerate(self.history['val_metrics']) if v is not None]
            val_metrics = [v for v in self.history['val_metrics'] if v is not None]
            if val_metrics:
                ax.plot(val_epochs, val_metrics, label='Validation bbox AP', marker='s')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss / AP')
            ax.set_title('Training Loss and Validation AP')
            ax.legend()
            ax.grid(True)

            # Learning rate plot
            ax = axes[1]
            ax.plot(epochs, self.history['learning_rate'], marker='^', color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'training_history.png', dpi=150)
            plt.close()
            print(f"📈 Training history plot saved")

        except ImportError:
            print("⚠️  matplotlib not available, skipping plot")
