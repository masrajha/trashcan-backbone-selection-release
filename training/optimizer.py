"""
Optimizer configurations for TrashCAN training
Supports multiple optimizers with configurable parameters
"""

import torch
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW
from typing import Dict, Any, Optional, Union

class OptimizerFactory:
    """Factory class for creating optimizers with different configurations"""
    
    # Preset configurations for different scenarios
    PRESETS = {
        # For Mask R-CNN with ResNet50 backbone (COCO pretrained)
        'maskrcnn_coco': {
            'type': 'sgd',
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'nesterov': True,
            'dampening': 0.0
        },
        
        # For RTX 3050 training with higher batch size
        'rtx3050_optimized': {
            'type': 'sgd',
            'lr': 0.0005,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'nesterov': True,
            'dampening': 0.0
        },
        
        # For fine-tuning with AdamW (transformer models)
        'adamw_finetune': {
            'type': 'adamw',
            'lr': 0.0001,
            'betas': (0.9, 0.999),
            'weight_decay': 0.01,
            'eps': 1e-8
        },
        
        # For fast convergence
        'adam_fast': {
            'type': 'adam',
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'weight_decay': 0.0,
            'eps': 1e-8
        },
        
        # For stable training with gradient clipping
        'stable_training': {
            'type': 'sgd',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'nesterov': False,
            'dampening': 0.0,
            'max_grad_norm': 1.0
        }
    }
    
    @classmethod
    def create_optimizer(cls, model, preset: str = 'maskrcnn_coco', 
                         custom_params: Optional[Dict[str, Any]] = None):
        """
        Create optimizer based on preset or custom parameters
        
        Args:
            model: PyTorch model with parameters
            preset: Name of preset configuration
            custom_params: Custom parameters to override preset
        
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        # Get base configuration
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")
        
        config = cls.PRESETS[preset].copy()
        
        # Override with custom parameters if provided
        if custom_params:
            config.update(custom_params)
        
        # Get optimizer type
        optimizer_type = config.pop('type', 'sgd').lower()
        
        # Filter out gradient clipping parameter
        max_grad_norm = config.pop('max_grad_norm', None)
        
        # Get parameters to optimize (filter out those that don't require gradients)
        params = [p for p in model.parameters() if p.requires_grad]
        
        # Create optimizer
        if optimizer_type == 'sgd':
            optimizer = SGD(params, **config)
        elif optimizer_type == 'adam':
            optimizer = Adam(params, **config)
        elif optimizer_type == 'adamw':
            optimizer = AdamW(params, **config)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Add gradient clipping info to optimizer
        if max_grad_norm is not None:
            optimizer.max_grad_norm = max_grad_norm
        
        return optimizer
    
    @classmethod
    def get_lr_multipliers(cls, model, backbone_lr: float = 1.0,
                          rpn_lr: float = 1.0, roi_lr: float = 1.0):
        """
        Create parameter groups with different learning rates for different parts
        
        Args:
            model: Mask R-CNN model
            backbone_lr: Multiplier for backbone parameters
            rpn_lr: Multiplier for RPN parameters
            roi_lr: Multiplier for ROI head parameters
        
        Returns:
            List of parameter groups
        """
        param_groups = []
        
        # Extract model parts if it's a Mask R-CNN
        if hasattr(model, 'backbone'):
            # Backbone parameters
            backbone_params = []
            for name, param in model.backbone.named_parameters():
                if param.requires_grad:
                    backbone_params.append(param)
            
            if backbone_params:
                param_groups.append({
                    'params': backbone_params,
                    'lr_multiplier': backbone_lr
                })
        
        if hasattr(model, 'rpn'):
            # RPN parameters
            rpn_params = []
            for name, param in model.rpn.named_parameters():
                if param.requires_grad:
                    rpn_params.append(param)
            
            if rpn_params:
                param_groups.append({
                    'params': rpn_params,
                    'lr_multiplier': rpn_lr
                })
        
        if hasattr(model, 'roi_heads'):
            # ROI heads parameters
            roi_params = []
            for name, param in model.roi_heads.named_parameters():
                if param.requires_grad:
                    roi_params.append(param)
            
            if roi_params:
                param_groups.append({
                    'params': roi_params,
                    'lr_multiplier': roi_lr
                })
        
        # Any remaining parameters
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and not any(
                param is p for group in param_groups for p in group['params']
            ):
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr_multiplier': 1.0
            })
        
        return param_groups

class GradientManager:
    """Utility class for gradient management and optimization"""
    
    @staticmethod
    def clip_gradients(optimizer, max_norm: float = 1.0, norm_type: float = 2.0):
        """Clip gradients to prevent exploding gradients"""
        if hasattr(optimizer, 'max_grad_norm'):
            max_norm = optimizer.max_grad_norm
        
        parameters = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        
        if len(parameters) > 0:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    
    @staticmethod
    def check_gradients(model, verbose: bool = False):
        """Check gradient statistics for debugging"""
        grad_stats = {
            'total_params': 0,
            'params_with_grad': 0,
            'grad_norms': [],
            'grad_means': [],
            'grad_stds': []
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_stats['total_params'] += param.numel()
                
                if param.grad is not None:
                    grad_stats['params_with_grad'] += param.numel()
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    
                    grad_stats['grad_norms'].append(grad_norm)
                    grad_stats['grad_means'].append(grad_mean)
                    grad_stats['grad_stds'].append(grad_std)
                    
                    if verbose and 'conv' in name or 'weight' in name:
                        print(f"{name:50} | Norm: {grad_norm:.6f} | Mean: {grad_mean:.6f} | Std: {grad_std:.6f}")
        
        if grad_stats['params_with_grad'] > 0:
            avg_grad_norm = sum(grad_stats['grad_norms']) / len(grad_stats['grad_norms'])
            print(f"\nGradient Statistics:")
            print(f"  Parameters with gradients: {grad_stats['params_with_grad']:,} / {grad_stats['total_params']:,}")
            print(f"  Average gradient norm: {avg_grad_norm:.6f}")
            print(f"  Max gradient norm: {max(grad_stats['grad_norms']):.6f}")
            print(f"  Min gradient norm: {min(grad_stats['grad_norms']):.6f}")
        
        return grad_stats
    
    @staticmethod
    def apply_weight_decay(optimizer, weight_decay: float = 0.0005):
        """Apply weight decay to all parameters except biases and batch norm"""
        decay = []
        no_decay = []
        
        for name, param in optimizer.param_groups[0]['params'][0].named_parameters():
            if not param.requires_grad:
                continue
                
            # Don't apply weight decay to biases and batch norm parameters
            if len(param.shape) == 1 or name.endswith('.bias') or 'bn' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        # Recreate optimizer with separate parameter groups
        param_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        
        # Get original optimizer settings
        orig_config = optimizer.defaults
        
        # Create new optimizer
        if isinstance(optimizer, SGD):
            new_optimizer = SGD(param_groups, **orig_config)
        elif isinstance(optimizer, Adam):
            new_optimizer = Adam(param_groups, **orig_config)
        elif isinstance(optimizer, AdamW):
            new_optimizer = AdamW(param_groups, **orig_config)
        else:
            raise ValueError(f"Unsupported optimizer type: {type(optimizer)}")
        
        return new_optimizer

class OptimizerWrapper:
    """Wrapper around optimizer for additional functionality"""
    
    def __init__(self, optimizer, model=None):
        self.optimizer = optimizer
        self.model = model
        self.state = {
            'step': 0,
            'loss_history': [],
            'grad_norm_history': [],
            'learning_rate_history': []
        }
    
    def step(self, closure=None):
        """Perform optimization step with gradient clipping"""
        # Clip gradients if max_grad_norm is set
        if hasattr(self.optimizer, 'max_grad_norm'):
            GradientManager.clip_gradients(self.optimizer, self.optimizer.max_grad_norm)
        
        # Store current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.state['learning_rate_history'].append(current_lr)
        
        # Perform optimization step
        if closure is not None:
            loss = self.optimizer.step(closure)
        else:
            self.optimizer.step()
            loss = None
        
        self.state['step'] += 1
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero out gradients (more efficient with set_to_none=True)"""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def multiply_learning_rate(self, factor: float):
        """Multiply learning rate by factor"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
    
    def decay_learning_rate(self, decay_factor: float = 0.1):
        """Decay learning rate by factor"""
        self.multiply_learning_rate(decay_factor)
    
    def get_gradient_norm(self) -> float:
        """Calculate gradient norm"""
        total_norm = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def log_gradient_statistics(self):
        """Log gradient statistics for monitoring"""
        grad_norm = self.get_gradient_norm()
        self.state['grad_norm_history'].append(grad_norm)
        
        print(f"[Step {self.state['step']}] LR: {self.get_learning_rate():.6f}, Grad Norm: {grad_norm:.4f}")
        
        # Check for gradient explosion/vanishing
        if grad_norm > 1000:
            print(f"⚠️  Warning: Gradient norm is very high: {grad_norm:.2f}")
        elif grad_norm < 1e-6:
            print(f"⚠️  Warning: Gradient norm is very low: {grad_norm:.2e}")
        
        return grad_norm

def create_optimizer_for_model(model, config: Dict[str, Any]) -> OptimizerWrapper:
    """
    High-level function to create optimizer with wrapper
    
    Args:
        model: PyTorch model
        config: Configuration dictionary with optimizer settings
    
    Returns:
        OptimizerWrapper: Wrapped optimizer with additional functionality
    """
    optimizer_type = config.get('type', 'sgd')
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0005)
    
    if optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', False)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,
                       weight_decay=weight_decay, nesterov=nesterov)
    elif optimizer_type == 'adam':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        optimizer = Adam(model.parameters(), lr=lr, betas=betas,
                        weight_decay=weight_decay, eps=eps)
    elif optimizer_type == 'adamw':
        betas = config.get('betas', (0.9, 0.999))
        eps = config.get('eps', 1e-8)
        optimizer = AdamW(model.parameters(), lr=lr, betas=betas,
                         weight_decay=weight_decay, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Apply gradient clipping if specified
    if 'max_grad_norm' in config:
        optimizer.max_grad_norm = config['max_grad_norm']
    
    return OptimizerWrapper(optimizer, model)

# Example usage
if __name__ == "__main__":
    # Test the optimizer factory
    print("Testing OptimizerFactory...")
    
    # Create a dummy model
    import torch.nn as nn
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3),
        nn.ReLU(),
        nn.Linear(64 * 30 * 30, 10)  # Assuming 30x30 feature map
    )
    
    # Create optimizer using preset
    optimizer = OptimizerFactory.create_optimizer(dummy_model, preset='maskrcnn_coco')
    print(f"Created optimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Create optimizer with custom parameters
    custom_optimizer = OptimizerFactory.create_optimizer(
        dummy_model,
        preset='maskrcnn_coco',
        custom_params={'lr': 0.0001, 'weight_decay': 0.001}
    )
    print(f"\nCustom optimizer LR: {custom_optimizer.param_groups[0]['lr']}")
    
    # Test gradient manager
    print("\nTesting GradientManager...")
    dummy_input = torch.randn(1, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (1,))
    criterion = nn.CrossEntropyLoss()
    
    output = dummy_model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    
    GradientManager.check_gradients(dummy_model, verbose=True)