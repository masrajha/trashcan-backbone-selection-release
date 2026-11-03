"""
GPU utilities for TrashCAN pipeline
"""
import torch
import gc
from contextlib import contextmanager

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon
        print("✅ Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (GPU not available)")
    
    return device

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

@contextmanager
def gpu_memory_saver():
    """Context manager to save GPU memory"""
    try:
        yield
    finally:
        clear_gpu_cache()

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")