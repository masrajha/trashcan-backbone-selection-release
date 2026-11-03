# NEW: configs/high_res_config.py - Untuk akurasi maksimal
from .base_config import BaseConfig

class HighResolutionConfig(BaseConfig):
    """Configuration for maximum accuracy (slower training)"""
    DATASET_VERSION = "material"
    DATA_FRACTION = 1.0
    RANDOM_SEED = 42
    NUM_CLASSES = 16
    PRETRAINED = True
    
    # Maximum resolution that fits in VRAM
    MIN_SIZE = 1024
    MAX_SIZE = 1024
    
    # Smaller batch size for high resolution
    BATCH_SIZE = 2
    
    # Longer training for convergence
    NUM_EPOCHS = 100
    
    # Learning rate schedule
    LEARNING_RATE = 0.0005
    STEP_SIZE = 20
    GAMMA = 0.5
    
    # More frequent validation
    VALIDATE_EVERY = 1
    
    # Strong data augmentation
    AUGMENTATION_STRENGTH = "high"
