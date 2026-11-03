"""
Learning rate scheduler configurations for TrashCAN training
Supports multiple scheduling strategies with warmup and cooldown
"""

import torch
import math
import warnings
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CosineAnnealingWarmRestarts,
    CyclicLR, OneCycleLR
)
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum

class SchedulerType(Enum):
    """Enumeration of supported scheduler types"""
    STEP = "step"
    MULTISTEP = "multistep"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    CYCLIC = "cyclic"
    ONE_CYCLE = "one_cycle"
    CUSTOM = "custom"

class SchedulerFactory:
    """Factory class for creating learning rate schedulers"""
    
    # Preset configurations for different scenarios
    PRESETS = {
        # Standard Mask R-CNN training (COCO style)
        'maskrcnn_standard': {
            'type': 'multistep',
            'milestones': [16, 22],
            'gamma': 0.1,
            'warmup_epochs': 1,
            'warmup_factor': 0.001
        },
        
        # RTX 3050 optimized training
        'rtx3050_optimized': {
            'type': 'cosine',
            'T_max': 30,  # Total epochs
            'eta_min': 1e-6,
            'warmup_epochs': 3,
            'warmup_factor': 0.01
        },
        
        # Fast convergence with OneCycleLR
        'onecycle_fast': {
            'type': 'one_cycle',
            'max_lr': 0.01,
            'total_steps': 1000,  # Estimated steps
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
            'cycle_momentum': True,
            'warmup_epochs': 0  # OneCycle has built-in warmup
        },
        
        # Fine-tuning with ReduceLROnPlateau
        'finetune_plateau': {
            'type': 'reduce_lr_on_plateau',
            'mode': 'max',  # For mAP
            'factor': 0.5,
            'patience': 5,
            'threshold': 0.0001,
            'min_lr': 1e-6,
            'warmup_epochs': 2,
            'warmup_factor': 0.1
        },
        
        # Cyclic learning rates for exploration
        'cyclic_exploration': {
            'type': 'cyclic',
            'base_lr': 1e-4,
            'max_lr': 1e-3,
            'step_size_up': 200,
            'step_size_down': 200,
            'mode': 'triangular',
            'gamma': 1.0,
            'scale_mode': 'cycle',
            'cycle_momentum': True
        },
        
        # Cosine annealing with warm restarts
        'cosine_restarts': {
            'type': 'cosine_warm_restarts',
            'T_0': 10,  # First restart
            'T_mult': 2,  # Multiply T_0 each restart
            'eta_min': 1e-6,
            'warmup_epochs': 2,
            'warmup_factor': 0.01
        }
    }
    
    @classmethod
    def create_scheduler(cls, optimizer, total_epochs: int, 
                        total_steps: Optional[int] = None,
                        preset: str = 'maskrcnn_standard',
                        custom_params: Optional[Dict[str, Any]] = None):
        """
        Create scheduler based on preset or custom parameters
        
        Args:
            optimizer: PyTorch optimizer
            total_epochs: Total number of training epochs
            total_steps: Total number of training steps (for step-based schedulers)
            preset: Name of preset configuration
            custom_params: Custom parameters to override preset
        
        Returns:
            SchedulerWrapper: Wrapped scheduler with additional functionality
        """
        # Get base configuration
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")
        
        config = cls.PRESETS[preset].copy()
        
        # Override with custom parameters if provided
        if custom_params:
            config.update(custom_params)
        
        # Get scheduler type
        scheduler_type = SchedulerType(config.pop('type', 'multistep'))
        
        # Extract warmup parameters
        warmup_epochs = config.pop('warmup_epochs', 0)
        warmup_factor = config.pop('warmup_factor', 0.001)
        
        # Create base scheduler
        if scheduler_type == SchedulerType.STEP:
            scheduler = StepLR(optimizer, **config)
        elif scheduler_type == SchedulerType.MULTISTEP:
            scheduler = MultiStepLR(optimizer, **config)
        elif scheduler_type == SchedulerType.EXPONENTIAL:
            scheduler = ExponentialLR(optimizer, **config)
        elif scheduler_type == SchedulerType.COSINE:
            # Use total_epochs for T_max
            if 'T_max' not in config:
                config['T_max'] = total_epochs
            scheduler = CosineAnnealingLR(optimizer, **config)
        elif scheduler_type == SchedulerType.COSINE_WARM_RESTARTS:
            scheduler = CosineAnnealingWarmRestarts(optimizer, **config)
        elif scheduler_type == SchedulerType.REDUCE_LR_ON_PLATEAU:
            scheduler = ReduceLROnPlateau(optimizer, **config)
        elif scheduler_type == SchedulerType.CYCLIC:
            scheduler = CyclicLR(optimizer, **config)
        elif scheduler_type == SchedulerType.ONE_CYCLE:
            # For OneCycleLR, we need total_steps
            if total_steps is None:
                raise ValueError("total_steps required for OneCycleLR")
            config['total_steps'] = total_steps
            scheduler = OneCycleLR(optimizer, **config)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Create wrapper with warmup
        wrapper = SchedulerWrapper(
            scheduler=scheduler,
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            warmup_epochs=warmup_epochs,
            warmup_factor=warmup_factor,
            total_epochs=total_epochs
        )
        
        return wrapper
    
    @classmethod
    def get_learning_rate_schedule(cls, optimizer, epoch: int, 
                                  schedule: List[Dict[str, Any]]) -> float:
        """
        Manual learning rate schedule based on epoch
        
        Args:
            optimizer: PyTorch optimizer
            epoch: Current epoch
            schedule: List of schedule entries
                Each entry: {'epoch': X, 'lr': Y} or {'epoch': X, 'multiplier': Z}
        
        Returns:
            Current learning rate
        """
        schedule = sorted(schedule, key=lambda x: x['epoch'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        for entry in schedule:
            if epoch >= entry['epoch']:
                if 'lr' in entry:
                    current_lr = entry['lr']
                elif 'multiplier' in entry:
                    current_lr *= entry['multiplier']
        
        return current_lr

class WarmupScheduler:
    """Warmup scheduler for gradual learning rate increase"""
    
    def __init__(self, optimizer, warmup_epochs: int, 
                 warmup_factor: float = 0.001, 
                 after_scheduler=None):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            warmup_factor: Starting learning rate multiplier
            after_scheduler: Main scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.after_scheduler = after_scheduler
        self.finished = False
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize warmup learning rates
        self.warmup_lrs = [
            base_lr * warmup_factor for base_lr in self.base_lrs
        ]
        
        # Set initial learning rates
        self._set_lrs(self.warmup_lrs)
        
        self.epoch = 0
    
    def step(self, epoch=None, metrics=None):
        """Step the scheduler"""
        if epoch is None:
            epoch = self.epoch + 1
        
        self.epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            current_lrs = [
                warmup_lr + (base_lr - warmup_lr) * alpha
                for warmup_lr, base_lr in zip(self.warmup_lrs, self.base_lrs)
            ]
            self._set_lrs(current_lrs)
        else:
            # Warmup finished, switch to main scheduler
            if not self.finished:
                self.finished = True
                # Reset to base learning rates
                self._set_lrs(self.base_lrs)
            
            # Use main scheduler if provided
            if self.after_scheduler is not None:
                if isinstance(self.after_scheduler, ReduceLROnPlateau):
                    if metrics is None:
                        warnings.warn("ReduceLROnPlateau requires metrics, but none provided")
                    else:
                        self.after_scheduler.step(metrics, epoch)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epochs)
    
    def _set_lrs(self, lrs):
        """Set learning rates for all parameter groups"""
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get scheduler state"""
        state = {
            'warmup_epochs': self.warmup_epochs,
            'warmup_factor': self.warmup_factor,
            'finished': self.finished,
            'epoch': self.epoch,
            'base_lrs': self.base_lrs,
            'warmup_lrs': self.warmup_lrs
        }
        
        if self.after_scheduler is not None:
            state['after_scheduler'] = self.after_scheduler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_factor = state_dict['warmup_factor']
        self.finished = state_dict['finished']
        self.epoch = state_dict['epoch']
        self.base_lrs = state_dict['base_lrs']
        self.warmup_lrs = state_dict['warmup_lrs']
        
        if 'after_scheduler' in state_dict and self.after_scheduler is not None:
            self.after_scheduler.load_state_dict(state_dict['after_scheduler'])

class SchedulerWrapper:
    """Wrapper around scheduler for additional functionality"""
    
    def __init__(self, scheduler, optimizer, scheduler_type: SchedulerType,
                 warmup_epochs: int = 0, warmup_factor: float = 0.001,
                 total_epochs: int = 50):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.total_epochs = total_epochs
        
        # Create warmup scheduler if needed
        if warmup_epochs > 0:
            self.scheduler = WarmupScheduler(
                optimizer, warmup_epochs, warmup_factor, scheduler
            )
        
        # Track history
        self.history = {
            'epoch': [],
            'learning_rate': [],
            'metrics': []  # For ReduceLROnPlateau
        }
        
        self.current_epoch = 0
    
    def step(self, epoch=None, metrics=None):
        """Step the scheduler"""
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        
        # Store current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['epoch'].append(self.current_epoch)
        self.history['learning_rate'].append(current_lr)
        
        if metrics is not None:
            self.history['metrics'].append(metrics)
        
        # Step the scheduler
        if isinstance(self.scheduler, WarmupScheduler):
            self.scheduler.step(self.current_epoch, metrics)
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            if metrics is None:
                warnings.warn("ReduceLROnPlateau requires metrics")
            else:
                self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_schedule(self) -> List[float]:
        """Get learning rate schedule for all epochs"""
        schedule = []
        
        # Simulate schedule
        temp_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=1.0)
        
        if self.warmup_epochs > 0:
            warmup_scheduler = WarmupScheduler(
                temp_optimizer, self.warmup_epochs, self.warmup_factor, self.scheduler
            )
            
            for epoch in range(self.total_epochs):
                warmup_scheduler.step(epoch)
                schedule.append(temp_optimizer.param_groups[0]['lr'])
        else:
            temp_scheduler = self._create_temp_scheduler(temp_optimizer)
            
            for epoch in range(self.total_epochs):
                temp_scheduler.step()
                schedule.append(temp_optimizer.param_groups[0]['lr'])
        
        return schedule
    
    def _create_temp_scheduler(self, optimizer):
        """Create a temporary scheduler for simulation"""
        if self.scheduler_type == SchedulerType.STEP:
            return StepLR(optimizer, **self.scheduler.__dict__)
        elif self.scheduler_type == SchedulerType.COSINE:
            return CosineAnnealingLR(optimizer, T_max=self.total_epochs)
        # Add other scheduler types as needed
        else:
            return StepLR(optimizer, step_size=30, gamma=0.1)
    
    def plot_schedule(self, save_path: Optional[str] = None):
        """Plot learning rate schedule"""
        import matplotlib.pyplot as plt
        
        schedule = self.get_schedule()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(schedule) + 1), schedule, 'b-', linewidth=2)
        
        # Mark warmup phase
        if self.warmup_epochs > 0:
            plt.axvspan(1, self.warmup_epochs, alpha=0.2, color='green', label='Warmup')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'Learning Rate Schedule ({self.scheduler_type.value})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def state_dict(self):
        """Get scheduler state"""
        return {
            'scheduler': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else {},
            'scheduler_type': self.scheduler_type.value,
            'warmup_epochs': self.warmup_epochs,
            'warmup_factor': self.warmup_factor,
            'total_epochs': self.total_epochs,
            'current_epoch': self.current_epoch,
            'history': self.history
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        if hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(state_dict['scheduler'])
        
        self.scheduler_type = SchedulerType(state_dict['scheduler_type'])
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_factor = state_dict['warmup_factor']
        self.total_epochs = state_dict['total_epochs']
        self.current_epoch = state_dict['current_epoch']
        self.history = state_dict['history']

class LearningRateFinder:
    """Learning rate finder for optimal learning rate selection"""
    
    @staticmethod
    def find_optimal_lr(model, train_loader, criterion, 
                       optimizer_class, lr_range=(1e-7, 10),
                       num_iterations=100, smooth_window=10):
        """
        Find optimal learning rate using LR range test
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            criterion: Loss function
            optimizer_class: Optimizer class (e.g., torch.optim.SGD)
            lr_range: Range of learning rates to test
            num_iterations: Number of iterations for test
            smooth_window: Window size for smoothing loss
        
        Returns:
            Dict with results and suggested learning rate
        """
        import numpy as np
        
        # Store original model state
        original_state = model.state_dict().copy()
        
        # Create optimizer with arbitrary learning rate
        optimizer = optimizer_class(model.parameters(), lr=lr_range[0])
        
        # Learning rate multiplier
        lr_multiplier = (lr_range[1] / lr_range[0]) ** (1 / num_iterations)
        
        # Track results
        lrs = []
        losses = []
        smoothed_losses = []
        
        # Get batch
        data_iter = iter(train_loader)
        
        print("Running learning rate finder...")
        for iteration in range(num_iterations):
            try:
                # Get next batch
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)
            
            # Update learning rate
            lr = lr_range[0] * (lr_multiplier ** iteration)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Store results
            lrs.append(lr)
            losses.append(loss.item())
            
            # Smooth losses
            if len(losses) >= smooth_window:
                smoothed = np.convolve(losses, np.ones(smooth_window)/smooth_window, mode='valid')
                smoothed_losses.extend(smoothed[-1:])
            else:
                smoothed_losses.append(loss.item())
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration:3d}/{num_iterations}: LR={lr:.2e}, Loss={loss.item():.4f}")
        
        # Restore original model state
        model.load_state_dict(original_state)
        
        # Find optimal learning rate (minimum of smoothed loss gradient)
        if len(smoothed_losses) > 1:
            # Calculate gradients
            gradients = np.gradient(smoothed_losses)
            
            # Find point with steepest negative gradient
            min_gradient_idx = np.argmin(gradients)
            optimal_lr = lrs[min_gradient_idx]
        else:
            optimal_lr = lr_range[0] * 10  # Default heuristic
        
        return {
            'learning_rates': lrs,
            'losses': losses,
            'smoothed_losses': smoothed_losses,
            'optimal_lr': optimal_lr,
            'suggested_lr_range': (optimal_lr / 10, optimal_lr * 3)
        }

# Example usage
if __name__ == "__main__":
    print("Testing SchedulerFactory...")
    
    # Create a dummy optimizer
    import torch.nn as nn
    dummy_model = nn.Sequential(nn.Linear(10, 2))
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)
    
    # Create scheduler using preset
    scheduler = SchedulerFactory.create_scheduler(
        dummy_optimizer,
        total_epochs=30,
        preset='rtx3050_optimized'
    )
    
    print(f"Created scheduler: {scheduler.scheduler_type.value}")
    print(f"Warmup epochs: {scheduler.warmup_epochs}")
    
    # Test stepping
    for epoch in range(5):
        scheduler.step(epoch)
        print(f"Epoch {epoch}: LR = {scheduler.get_learning_rate():.6f}")
    
    # Plot schedule
    scheduler.plot_schedule()
    
    print("\nTesting LearningRateFinder (mock)...")
    print("Note: Actual LR finder requires real data and model training")