# NEW: configs/base_config.py - Base class untuk inheritance
import torch
from pathlib import Path

class BaseConfig:
    """Base configuration with sensible defaults"""
    DATASET_VERSION = "material"
    DATA_FRACTION = 1.0
    RANDOM_SEED = 42
    DETERMINISTIC = True
    ALLOW_TF32 = False

    NUM_CLASSES = 16
    PRETRAINED = True
    BACKBONE = "resnet50"
    BACKBONE_PRETRAINING_SCOPE = "backbone_only"

    INPUT_WIDTH = 800
    INPUT_HEIGHT = 800
    MIN_SIZE = 800
    MAX_SIZE = 1333

    NUM_EPOCHS = 30
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_SIZE = 10
    GAMMA = 0.1

    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False
    VALIDATE_EVERY = 1
    SKIP_VALIDATION = False
    USE_AMP = False
    
    # Hardware detection
    @property
    def DEVICE(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @property
    def GPU_NAME(self):
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
    
    # Dynamic paths
    @property
    def ROOT_DIR(self):
        return Path(__file__).parent.parent
    
    @property
    def ORIGINAL_DATA_DIR(self):
        return self.ROOT_DIR / "original_data"
    
    @property  
    def COCO_DIR(self):
        return self.ROOT_DIR / f"{self.DATASET_VERSION}_version"
    
    @property
    def BATCH_SIZE(self):
        return getattr(self, "_batch_size_override", 2)

    @property
    def INPUT_SIZE(self):
        return (self.INPUT_HEIGHT, self.INPUT_WIDTH)

    @property
    def TRAIN_DIR(self):
        return self.COCO_DIR / "train"

    @property
    def VAL_DIR(self):
        return self.COCO_DIR / "val"

    @property
    def TRAIN_JSON(self):
        return self.COCO_DIR / "instances_train_trashcan.json"

    @property
    def VAL_JSON(self):
        return self.COCO_DIR / "instances_val_trashcan.json"
